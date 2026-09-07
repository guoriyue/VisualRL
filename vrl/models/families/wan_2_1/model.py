"""Wan 2.1 t2v diffusers-backed model.

Diffusion implementation for Wan T2V on the RL path. The generation helper
flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does only
one transformer forward (with optional batched CFG concat) and returns noise
predictions.

Per-family ``WanT2VSamplingState`` is private to this file. The engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Differences from SD3:
- ``prompt_embeds`` only (no pooled/CLIP); transformer signature lacks
  ``pooled_projections``.
- 5D latents ``[B, C, T, H, W]`` (Wan VAE temporal axis).
- VAE decode applies Wan-specific per-channel ``latents_mean`` /
  ``latents_std`` denormalization (over ``z_dim``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.families.wan_2_1.config import (
    normalize_wan_boundary_ratio,
    normalize_wan_trainable_transformers,
    wan_topology_from_build,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.peft_adapter import disable_adapter_on, load_trainable_lora_adapter
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusionModelBase,
    GuidedDiffusionSamplingStateBase,
    ReplayRolloutStubs,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.models.weight_utils import load_weights_into, require_weights_for

logger = logging.getLogger(__name__)


class _PipelineOffloadState(Enum):
    """One committed Wan hook lifecycle state."""

    MODEL = "model"
    SEQUENTIAL = "sequential"
    BROKEN = "broken"


def _batch_align_tensor(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            "Wan I2V conditioning tensor has incompatible batch dimension: "
            f"{tuple(value.shape)} for batch_size={batch_size}",
        )
    return value.expand(batch_size, *value.shape[1:])


@dataclass
class WanT2VSamplingState(GuidedDiffusionSamplingStateBase):
    """Private Wan T2V sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    do_cfg: bool
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None


@dataclass
class WanI2VSamplingState(GuidedDiffusionSamplingStateBase):
    """Private Wan I2V sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    image_embeds: torch.Tensor | None
    condition: torch.Tensor
    do_cfg: bool
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None


class WanT2VDiffusersModel(
    LoraModelMixin,
    DiffusersPipelineModelBase,
    DiffusionBackboneRunnerBase,
):
    """Diffusers-backed Wan 2.1 T2V model (1.3B variant).

    Implements the backbone-runner protocol itself (batched CFG over the T5
    text embeds); the I2V subclass overrides ``build_branch`` with its
    channel-wise conditioning concat.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map Wan transformer kwargs into the shared backbone contract."""
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
        )

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        trainable_transformers: Any = None,
        expert_lifecycle_profiling: bool = False,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        self._pipeline_offload: _PipelineOffloadState | None = None
        self.transformer_2 = getattr(pipeline, "transformer_2", None)
        self._boundary_ratio = normalize_wan_boundary_ratio(
            _config_value(getattr(pipeline, "config", None), "boundary_ratio"),
            field_name="pipeline boundary_ratio",
        )
        self._trainable_transformer_names = normalize_wan_trainable_transformers(
            trainable_transformers,
            dual_stage=self._boundary_ratio is not None,
        )
        self._expert_lifecycle_profiling = bool(expert_lifecycle_profiling)
        self._last_expert_name: str | None = None
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer
        self.pipeline.transformer_2 = transformer

    @property
    def boundary_ratio(self) -> float | None:
        return self._boundary_ratio

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> WanT2VDiffusersModel:
        """Load the diffusers WanPipeline + freeze non-trainable modules."""
        boundary_ratio, trainable_transformers = wan_topology_from_build(build)
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(
            build.model_name_or_path,
            torch_dtype=build.parameter_dtype,
            **build.pretrained_kwargs,
        )
        _validate_wan_pipeline(
            pipeline,
            task="Wan T2V",
            expected_boundary_ratio=boundary_ratio,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        offload_mode = _resolve_wan_offload_mode(build)
        _stage_eager_wan_modules(
            pipeline,
            build,
            offload_mode=offload_mode,
            eager_module_dtypes={
                "vae": torch.float32,
                "text_encoder": build.parameter_dtype,
            },
        )
        return cls(
            pipeline=pipeline,
            device=build.device,
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=bool(
                (build.model_config or {}).get("expert_lifecycle_profiling", False),
            ),
        )

    # Empty training adapters must initially preserve base Wan output.
    _lora_default_init_weights = True

    def _wan_transformers(self) -> dict[str, Any]:
        modules = {"transformer": self.transformer}
        if self.transformer_2 is not None:
            modules["transformer_2"] = self.transformer_2
        return modules

    def apply_full_finetune(self, build: ModelBuild) -> None:
        for module in self.trainable_modules.values():
            module.requires_grad_(True)
            if getattr(build, "defer_trainable_device_move", False):
                # FSDP owns the CPU-to-device transition and normalizes storage
                # dtype immediately before sharding. Moving or duplicating the
                # full parameter set here defeats block-wise FSDP construction.
                continue
            if _resolve_wan_offload_mode(build) == "none":
                module.to(self.device, dtype=build.parameter_dtype)
            else:
                # Accelerate installs its hooks after this method. Normalize the
                # CPU storage now so trainer payloads and rollout parameters have
                # one exact dtype without prematurely occupying the whole GPU.
                module.to(dtype=build.parameter_dtype)

    def apply_lora(self, build: ModelBuild) -> None:
        """Attach LoRA to the configured Wan trainable transformer(s)."""

        from peft import LoraConfig, get_peft_model

        lora_path = build.lora_path
        names = self._trainable_transformer_names
        if lora_path and len(names) != 1:
            raise ValueError(
                "model.lora.path can only resume one Wan transformer; set "
                "model.trainable_transformers to a single transformer name",
            )

        lora_config = build.lora
        if lora_config is None:
            raise ValueError("LoRA runtime build requires model.lora configuration")

        for name in names:
            transformer = self._wan_transformers()[name]
            transformer.requires_grad_(False)
            if not _defer_wan_trainable_device_move(build):
                transformer.to(self.device)
            if lora_path:
                wrapped = load_trainable_lora_adapter(
                    transformer,
                    lora_path,
                    expected_rank=lora_config["rank"],
                    expected_alpha=lora_config["alpha"],
                    expected_dropout=lora_config.get("dropout", 0.0),
                    expected_target_modules=lora_config["target_modules"],
                    # Same construction dtype as the fresh-create branch below.
                    autocast_adapter_dtype=False,
                )
                wrapped.set_adapter("default")
            else:
                cfg = LoraConfig(
                    r=lora_config["rank"],
                    lora_alpha=lora_config["alpha"],
                    lora_dropout=lora_config.get("dropout", 0.0),
                    init_lora_weights=lora_config.get(
                        "init_lora_weights",
                        self._lora_default_init_weights,
                    ),
                    target_modules=lora_config["target_modules"],
                )
                # Adapters must be created in the transformer's resolved dtype,
                # not peft's default fp32 upcast: the FSDP actor policy stores
                # every trainer parameter in the resolved low precision (update
                # precision comes from the fp32-master optimizer on all
                # topologies), so an fp32 adapter here makes the rollout
                # worker's runtime dtype diverge from the FSDP trainer's synced
                # payload — load_trainable_state then fails its strict dtype
                # check (measured on the hpsv3 fsdp 4-rank smoke, 2026-08-16).
                # One construction dtype keeps single-process, FSDP, and worker
                # replicas byte-compatible at weight sync.
                wrapped = get_peft_model(transformer, cfg, autocast_adapter_dtype=False)
            self._set_wan_transformer(name, wrapped)

    @property
    def uses_pipeline_cpu_offload(self) -> bool:
        """Whether this model is configured for Accelerate-owned residency."""

        return self._pipeline_offload is not None

    @property
    def pipeline_cpu_offload_healthy(self) -> bool:
        """Whether configured pipeline hooks are installed and reusable."""

        return self._pipeline_offload is not _PipelineOffloadState.BROKEN

    def apply_generation_offload(self, build: ModelBuild) -> None:
        """Install pipeline offload hooks after LoRA has changed the module tree."""

        mode = _resolve_wan_offload_mode(build)
        state = self._pipeline_offload
        if state is _PipelineOffloadState.BROKEN:
            raise RuntimeError("Wan pipeline CPU offload is not reusable")
        loaded_mode = "none" if state is None else state.value
        if state is not None and mode != loaded_mode:
            raise RuntimeError(
                "Wan pipeline offload mode changed during rollout construction: "
                f"loaded={loaded_mode!r}, final={mode!r}",
            )
        if mode == "none":
            return
        target = _PipelineOffloadState(mode)
        if state is target:
            return
        # Initial installation is transactional too. A partial hook graph must
        # never be published as a reusable rollout model.
        self._pipeline_offload = _PipelineOffloadState.BROKEN
        _enable_wan_pipeline_offload(self.pipeline, self.device, mode=mode)
        self._pipeline_offload = target

    @property
    def policy_cores(self) -> dict[str, Any]:
        """Both experts, not just ``self.transformer`` and not ``trainable_modules``.

        A dual-stage config that trains only one expert still SAMPLES through
        both, so every optimization pass must reach both: otherwise the two
        halves of one trajectory run at different precisions and the
        rollout-vs-replay drift differs across the boundary.
        """

        return self._wan_transformers()

    def set_module_root(self, name: str, module: Any) -> None:
        """Route a replaced expert back to both the attribute and the pipeline.

        One override serves every caller: rollout compile walks all experts,
        FSDP/DDP walks only the trained ones, and both land here.
        """

        self._set_wan_transformer(name, module)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        modules = self._wan_transformers()
        return {name: modules[name] for name in self._trainable_transformer_names}

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for module in self.trainable_modules.values():
                stack.enter_context(disable_adapter_on(module))
            yield

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        validated = self._validate_wan_trainable_state(state_dict)
        if not self.uses_pipeline_cpu_offload:
            return self._load_validated_wan_trainable_state(validated)

        return self._with_pipeline_cpu_offload_suspended(
            lambda: self._load_validated_wan_trainable_state(validated),
            operation="trainable weight sync",
        )

    def reset_pipeline_cpu_offload(self) -> None:
        """Return every component to CPU and reinstall fresh streaming hooks.

        Accelerate post-forward hooks do not run when a module forward raises,
        and model-level offload deliberately leaves the final component on the
        execution device. Cycling Diffusers' public hook API is the common
        phase-parking and error-recovery boundary for both cases.
        """

        if not self.uses_pipeline_cpu_offload:
            return
        self._with_pipeline_cpu_offload_suspended(
            lambda: None,
            operation="pipeline residency reset",
        )

    def _with_pipeline_cpu_offload_suspended(
        self,
        operation_fn: Callable[[], Any],
        *,
        operation: str,
    ) -> Any:
        """Run one CPU-resident operation and transactionally re-arm hooks."""

        if not self.pipeline_cpu_offload_healthy:
            raise RuntimeError(
                "Wan pipeline CPU offload is not reusable: its hook lifecycle "
                "is incomplete or previously failed",
            )

        state = self._pipeline_offload
        if state is None:
            return operation_fn()
        mode = state.value
        # Publish the transition before touching hooks. Partial removal, hook
        # reinstall, or weight mutation must leave one terminal state.
        self._pipeline_offload = _PipelineOffloadState.BROKEN

        remove_hooks = getattr(self.pipeline, "remove_all_hooks", None)
        if not callable(remove_hooks):
            raise RuntimeError(
                f"Wan pipeline CPU offload requires pipeline.remove_all_hooks() for {operation}",
            )

        try:
            remove_hooks()
        except BaseException as remove_error:
            raise RuntimeError(
                f"Wan pipeline CPU offload hook removal failed during {operation}; "
                "the rollout model is no longer reusable",
            ) from remove_error

        result: Any = None
        operation_error: BaseException | None = None
        try:
            result = operation_fn()
        except BaseException as error:
            operation_error = error

        try:
            _enable_wan_pipeline_offload(
                self.pipeline,
                self.device,
                mode=mode,
            )
        except BaseException as reinstall_error:
            detail = f"Wan pipeline CPU offload hook reinstall failed during {operation}"
            if operation_error is not None:
                detail += f" after operation failure {operation_error!r}"
            raise RuntimeError(
                f"{detail}; the rollout model is no longer reusable",
            ) from reinstall_error

        if operation_error is not None:
            # A state-dict copy can fail after mutating an arbitrary prefix. Fresh
            # hooks make teardown safe, but they cannot make partially updated
            # policy weights valid. Never publish this model as reusable.
            raise RuntimeError(
                f"Wan {operation} failed after pipeline hooks were removed; "
                "the rollout model may contain partial state and is no longer reusable",
            ) from operation_error

        self._pipeline_offload = state
        return result

    def _load_validated_wan_trainable_state(
        self,
        validated: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        results = {}
        for name, module in self.trainable_modules.items():
            target = getattr(module, "_orig_mod", module)
            prefixed = {f"{name}.{key}": value for key, value in validated[name].items()}
            results[name] = load_weights_into(target, prefixed, prefix=name)
        return results

    def validate_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        """Reject partial dual-expert payloads before a weight-sync ACK."""

        self._validate_wan_trainable_state(state_dict)

    def _validate_wan_trainable_state(
        self,
        state_dict: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        state = dict(state_dict)
        modules = self.trainable_modules
        accepted_prefixes = tuple(f"{name}." for name in modules)
        unexpected = sorted(key for key in state if not key.startswith(accepted_prefixes))
        if unexpected:
            raise ValueError(
                f"{type(self).__name__}: unexpected trainable state keys {unexpected[:5]}",
            )
        validated: dict[str, dict[str, Any]] = {}
        for name, module in modules.items():
            prefix = f"{name}."
            module_state = {key: value for key, value in state.items() if key.startswith(prefix)}
            validated[name] = require_weights_for(module, module_state, prefix=name)
        return validated

    def _set_wan_transformer(self, name: str, transformer: Any) -> None:
        if name == "transformer":
            self._set_transformer(transformer)
            return
        if name == "transformer_2":
            self._set_transformer_2(transformer)
            return
        raise ValueError(f"unknown Wan transformer name: {name!r}")

    def _transformer_for_timestep(
        self,
        state: WanT2VSamplingState | WanI2VSamplingState,
        timestep: torch.Tensor,
    ) -> tuple[str, Any, float]:
        if state.boundary_ratio is None:
            return "transformer", self.transformer, state.guidance_scale
        if _uses_low_noise_transformer(
            timestep,
            boundary_ratio=state.boundary_ratio,
            num_train_timesteps=state.num_train_timesteps,
        ):
            transformer_2 = self.transformer_2
            if transformer_2 is None:
                raise RuntimeError("Wan dual-stage sampling requires transformer_2")
            return "transformer_2", transformer_2, state.guidance_scale_2 or state.guidance_scale
        return "transformer", self.transformer, state.guidance_scale

    def _forward_wan_backbone(
        self,
        state: WanT2VSamplingState | WanI2VSamplingState,
        timestep: torch.Tensor,
        *,
        extra_builder: Callable[[torch.dtype], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        bsz = state.latents.shape[0]
        expert_name, transformer, guidance_scale = self._transformer_for_timestep(
            state,
            timestep,
        )
        td = _module_dtype(transformer)
        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(timestep, bsz).to(
            device=latent_input.device,
            dtype=td,
        )
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        with self._expert_lifecycle_trace(expert_name, timestep):
            output = DiffusionBackboneCaller(
                transformer,
                self,
            )(
                DiffusionBackboneInput(
                    hidden_states=latent_input,
                    timestep=timestep_batch,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    do_cfg=state.do_cfg,
                    output_dtype=td,
                    extra=extra_builder(td) if extra_builder is not None else {},
                ),
            )
        return output.as_dict()

    @contextlib.contextmanager
    def _expert_lifecycle_trace(
        self,
        expert_name: str,
        timestep: torch.Tensor,
    ) -> Iterator[None]:
        if not self._expert_lifecycle_profiling or not torch.cuda.is_available():
            yield
            return

        device = torch.device(self.device)
        if device.type != "cuda":
            yield
            return
        torch.cuda.synchronize(device)
        switched = self._last_expert_name not in {None, expert_name}
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        before_free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.cuda.nvtx.range(f"wan.expert.{expert_name}"):
            yield
        torch.cuda.synchronize(device)
        after_free, _ = torch.cuda.mem_get_info(device)
        modules = self._wan_transformers()
        inactive_name = "transformer_2" if expert_name == "transformer" else "transformer"
        record = {
            "event": "wan_expert_lifecycle",
            "expert": expert_name,
            "timestep": float(torch.as_tensor(timestep).reshape(-1)[0].item()),
            "switched": switched,
            "allocated_before_bytes": before_allocated,
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
            "reserved_before_bytes": before_reserved,
            "reserved_after_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "physical_used_before_bytes": total - before_free,
            "physical_used_after_bytes": total - after_free,
            "active_cuda_parameter_bytes": _cuda_parameter_bytes(modules[expert_name]),
            "inactive_cuda_parameter_bytes": _cuda_parameter_bytes(modules[inactive_name]),
        }
        logger.info("WAN_EXPERT_LIFECYCLE %s", json.dumps(record, sort_keys=True))
        self._last_expert_name = expert_name

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Wan's T5 text encoder.

        Returns ``prompt_embeds`` and the matching ``negative_prompt_embeds``
        when CFG is active. Wan does not use a pooled CLIP embedding.
        """
        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanT2VSamplingState:
        """Build the per-request SamplingState for a Wan T2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        guidance_scale = request.guidance_scale
        guidance_scale_2 = _resolve_guidance_scale_2(guidance_scale, self._boundary_ratio)
        do_cfg = _uses_cfg(guidance_scale, guidance_scale_2)

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Wan prepare_latents signature:
        # (batch, channels, height, width, num_frames, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            generator,
            None,
        )

        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            guidance_scale_2=guidance_scale_2,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(pipe.scheduler),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: WanT2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan T2V transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE.

        Timestep shape convention (mirrors SD3 model):
        - rollouts: ``state.timesteps`` is 1-D ``[T]``; we expand a scalar to ``[B]``.
        - eval/training: collector packs per-sample timestep as ``[B]``; expand is a no-op.
        """
        t = state.timesteps[step_idx]
        return self._forward_wan_backbone(state, t)

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "guidance_scale_2": state.guidance_scale_2,
            "cfg": state.do_cfg,
            "boundary_ratio": state.boundary_ratio,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanT2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        # Pack as [1, B] so forward_step's state.timesteps[0] returns [B].
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and _uses_cfg(
                batch_context["guidance_scale"],
                batch_context.get("guidance_scale_2"),
            ),
            guidance_scale_2=batch_context.get("guidance_scale_2"),
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=batch_context.get("num_train_timesteps"),
        )

    # -- decode_latents ------------------------------------------------

    def encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode clean target video into Wan's normalized latent domain.

        ``video`` is ``[B,C,T,H,W]`` in ``[0,1]`` at the training geometry.
        This is the exact inverse of :meth:`decode_latents`'s per-channel
        denormalization and uses the deterministic posterior mode so an offline
        SFT shard is reproducible.
        """

        pipe = self.pipeline
        x = (video.to(device=self.device, dtype=pipe.vae.dtype) * 2.0) - 1.0
        with torch.no_grad():
            encoded = torch.cat(
                [pipe.vae.encode(sample.unsqueeze(0)).latent_dist.mode() for sample in x],
                dim=0,
            )
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(encoded.device, encoded.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1,
            pipe.vae.config.z_dim,
            1,
            1,
            1,
        ).to(encoded.device, encoded.dtype)
        return (encoded - latents_mean) * latents_std

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 5D latents -> video [B, C, T, H, W] via Wan VAE.

        Applies Wan-specific per-channel denormalization using the VAE's
        ``latents_mean`` / ``latents_std`` over the ``z_dim`` channel axis.
        """
        pipe = self.pipeline

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            x = batch.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1
            ).to(x.device, x.dtype)
            return x / latents_std + latents_mean

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda batch: pipe.vae.decode(
                    batch,
                    return_dict=False,
                )[0],
                postprocess=lambda video: pipe.video_processor.postprocess_video(
                    video,
                    output_type="pt",
                ),
                output_layout="video_btchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class WanT2VReplayModel(ReplayRolloutStubs, WanT2VDiffusersModel):
    """Replay-only Wan model that owns no text encoder, VAE, or pipeline."""

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        device: Any = None,
        transformer_2: Any = None,
        boundary_ratio: float | None = None,
        trainable_transformers: Any = None,
    ) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self._scheduler = scheduler
        self._device = device
        self._boundary_ratio = boundary_ratio
        self._trainable_transformer_names = normalize_wan_trainable_transformers(
            trainable_transformers,
            dual_stage=boundary_ratio is not None,
        )
        self._expert_lifecycle_profiling = False
        self._last_expert_name = None
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    def prepare_replay(self, build: ModelBuild) -> None:
        """Finish the multi-transformer replay setup with the build in hand.

        The generic replay loader constructs only the primary transformer +
        scheduler; Wan2.2 dual-stage checkpoints carry a second transformer
        whose presence is decided by the canonical build topology. Late-load it
        here and apply the already-normalized policy ownership.
        """
        boundary_ratio, trainable_transformers = wan_topology_from_build(build)
        if boundary_ratio is not None and self.transformer_2 is None:
            from vrl.models.loader import load_diffusers_transformer

            self.transformer_2 = load_diffusers_transformer(
                build,
                "WanTransformer3DModel",
                subfolder="transformer_2",
            )
        self._boundary_ratio = boundary_ratio
        self._trainable_transformer_names = trainable_transformers
        self._expert_lifecycle_profiling = bool(
            (build.model_config or {}).get("expert_lifecycle_profiling", False),
        )
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("WanT2VReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return None


class WanI2VDiffusersModel(WanT2VDiffusersModel):
    """Diffusers-backed Wan 2.1 I2V model (14B 480P variant)."""

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """I2V branch: channel-wise conditioning concat + CLIP image embeds.

        - Hidden states are ``cat([noise_latents, condition], dim=1)`` so the
          transformer sees both the noisy frames and the (mask + first-frame
          latent) conditioning channel-wise concatenated. The condition tensor
          is produced once per request by ``pipe.prepare_latents`` and rides
          ``request.extra``.
        - ``encoder_hidden_states_image`` is the CLIP visual feature from
          ``pipe.encode_image``; cond/uncond share the same tensor object so
          the batched-CFG packer concatenates it along dim 0 alongside
          ``prompt_embeds``.

        Wan 2.2 5B's expand_timesteps mode (mask-blended hidden states) remains
        a separate future upgrade path; the A14B dual-stage routing selects
        ``transformer`` / ``transformer_2`` outside this method.
        """
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds

        extra = request.extra
        condition = _batch_align_tensor(
            extra["condition"],
            request.hidden_states.shape[0],
        )
        # Channel-wise concat (mimics WanImageToVideoPipeline `__call__`:
        # `latent_model_input = torch.cat([latents, condition], dim=1)`).
        hidden_states = torch.cat(
            [request.hidden_states, condition.to(request.hidden_states.dtype)],
            dim=1,
        )

        extra_kwargs: dict[str, torch.Tensor] = {}
        image_embeds = extra.get("image_embeds")
        if image_embeds is not None:
            extra_kwargs["encoder_hidden_states_image"] = _batch_align_tensor(
                image_embeds,
                request.hidden_states.shape[0],
            )

        return DiffusionBranch(
            hidden_states=hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=extra_kwargs,
        )

    @classmethod
    def from_build(cls, build: ModelBuild) -> WanI2VDiffusersModel:
        """Load WanImageToVideoPipeline + freeze generation-only modules."""
        boundary_ratio, trainable_transformers = wan_topology_from_build(build)
        from diffusers import WanImageToVideoPipeline

        pipeline = WanImageToVideoPipeline.from_pretrained(
            build.model_name_or_path,
            torch_dtype=build.parameter_dtype,
            **build.pretrained_kwargs,
        )
        _validate_wan_pipeline(
            pipeline,
            task="Wan I2V",
            expected_boundary_ratio=boundary_ratio,
        )
        pipeline.set_progress_bar_config(disable=True)

        for module_name in ("vae", "text_encoder", "image_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)

        offload_mode = _resolve_wan_offload_mode(build)
        _stage_eager_wan_modules(
            pipeline,
            build,
            offload_mode=offload_mode,
            eager_module_dtypes={
                "vae": torch.float32,
                "text_encoder": build.parameter_dtype,
                "image_encoder": build.parameter_dtype,
            },
        )
        return cls(
            pipeline=pipeline,
            device=build.device,
            trainable_transformers=trainable_transformers,
            expert_lifecycle_profiling=bool(
                (build.model_config or {}).get("expert_lifecycle_profiling", False),
            ),
        )

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode Wan I2V text and optional CLIP image conditioning."""

        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        td = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        reference_image = kwargs.get("reference_image")
        image_embeds = kwargs.get("image_embeds")
        if image_embeds is None and reference_image is not None:
            image_embeds = self._encode_image_embeds(reference_image)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "image_embeds": image_embeds,
            "reference_image": reference_image,
        }

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanI2VSamplingState:
        """Build the per-request SamplingState for a Wan I2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        reference_image = kwargs.get("reference_image", encoded.get("reference_image"))
        if reference_image is None:
            raise ValueError("Wan I2V sampling requires a reference_image")

        guidance_scale = request.guidance_scale
        guidance_scale_2 = _resolve_guidance_scale_2(guidance_scale, self._boundary_ratio)
        do_cfg = _uses_cfg(guidance_scale, guidance_scale_2)

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        batch_size = prompt_embeds.shape[0]
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        image = pipe.video_processor.preprocess(
            reference_image,
            height=request.height,
            width=request.width,
        ).to(device, dtype=torch.float32)

        num_channels_latents = int(pipe.vae.config.z_dim)
        with torch.no_grad():
            latents, condition = pipe.prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                request.height,
                request.width,
                request.frame_count,
                torch.float32,
                device,
                generator,
                None,
                None,
            )

        image_embeds = encoded.get("image_embeds")
        if image_embeds is None:
            image_embeds = self._encode_image_embeds(reference_image)
        image_embeds = _align_optional_batch(image_embeds, batch_size)

        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            condition=condition,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            guidance_scale_2=guidance_scale_2,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(pipe.scheduler),
        )

    def forward_step(
        self,
        state: WanI2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan I2V transformer forward + optional batched CFG."""

        t = state.timesteps[step_idx]
        return self._forward_wan_backbone(
            state,
            t,
            extra_builder=lambda td: {
                "condition": state.condition.to(td),
                "image_embeds": (
                    None if state.image_embeds is None else state.image_embeds.to(td)
                ),
            },
        )

    def export_batch_context(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "guidance_scale_2": state.guidance_scale_2,
            "cfg": state.do_cfg,
            "conditioning": "reference_image",
            "boundary_ratio": state.boundary_ratio,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "image_embeds": state.image_embeds,
            "condition": state.condition,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanI2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            image_embeds=replay_tensors.get("image_embeds"),
            condition=replay_tensors["condition"],
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and _uses_cfg(
                batch_context["guidance_scale"],
                batch_context.get("guidance_scale_2"),
            ),
            guidance_scale_2=batch_context.get("guidance_scale_2"),
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=batch_context.get("num_train_timesteps"),
        )

    def _encode_image_embeds(self, reference_image: Any) -> torch.Tensor | None:
        if not _wan_i2v_uses_image_embeds(self.transformer):
            return None
        with torch.no_grad():
            image_embeds = self.pipeline.encode_image(reference_image, self.device)
        return image_embeds.to(self.transformer.dtype)


class WanI2VReplayModel(WanT2VReplayModel, WanI2VDiffusersModel):
    """Replay-only Wan I2V model that owns no text, image, VAE, or pipeline modules.

    Base order is the whole design: ``WanT2VReplayModel`` (first) supplies the
    replay plumbing -- constructor, dual-stage ``prepare_replay``, the
    pipeline-less ``pipeline`` / ``scheduler`` / ``_set_transformer`` -- and
    ``WanI2VDiffusersModel`` (second) supplies the I2V forward math
    (``forward_step`` / ``restore_eval_state`` / ``export_replay_tensors``),
    which the C3 order puts ahead of the T2V versions. Pinned in
    tests/models/steps/denoise/test_family_mro.py.
    """


def _align_optional_batch(value: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[:1] == (batch_size,):
        return value
    if value.shape[:1] != (1,):
        raise ValueError(
            "Wan I2V image_embeds batch dimension is incompatible with rollout "
            f"batch_size={batch_size}: {tuple(value.shape)}",
        )
    return value.repeat(batch_size, *([1] * (value.ndim - 1)))


def _wan_i2v_uses_image_embeds(transformer: Any) -> bool:
    config = _transformer_config(transformer)
    if isinstance(config, dict):
        return config.get("image_dim") is not None
    return getattr(config, "image_dim", None) is not None


def _transformer_config(transformer: Any) -> Any:
    config = getattr(transformer, "config", None)
    if isinstance(config, dict) and "image_dim" in config:
        return config
    if config is not None and hasattr(config, "image_dim"):
        return config
    base_model = getattr(transformer, "base_model", None)
    inner = getattr(base_model, "model", base_model)
    config = getattr(inner, "config", None)
    if config is not None:
        return config
    return getattr(transformer, "config", None)


def _validate_wan_pipeline(
    pipeline: Any,
    *,
    task: str,
    expected_boundary_ratio: float | None,
) -> None:
    if bool(_config_value(pipeline.config, "expand_timesteps", False)):
        raise NotImplementedError(
            f"{task} RL does not yet support expand_timesteps pipelines.",
        )
    loaded_boundary_ratio = normalize_wan_boundary_ratio(
        _config_value(pipeline.config, "boundary_ratio"),
        field_name="pipeline boundary_ratio",
    )
    if loaded_boundary_ratio != expected_boundary_ratio:
        raise ValueError(
            f"{task} pipeline boundary_ratio disagrees with the canonical ModelBuild: "
            f"pipeline={loaded_boundary_ratio!r}, build={expected_boundary_ratio!r}",
        )
    if loaded_boundary_ratio is not None and getattr(pipeline, "transformer_2", None) is None:
        raise ValueError(f"{task} dual-stage pipeline is missing transformer_2")


def _resolve_wan_offload_mode(build: ModelBuild) -> str:
    """Return normalized rollout residency without affecting replay builds."""

    extra = build.model_config or {}
    legacy = sorted(
        key
        for key in ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
        if key in extra
    )
    if legacy:
        raise ValueError(
            f"removed Wan model config key(s): {', '.join('model.' + key for key in legacy)}; "
            "use model.offload_mode='none', 'model', or 'sequential'",
        )
    rollout = getattr(build, "rollout", None)
    return str(getattr(rollout, "pipeline_offload_mode", "none"))


def _defer_wan_trainable_device_move(build: ModelBuild) -> bool:
    """Keep the full transformer on CPU for FSDP or pipeline CPU offload."""

    return bool(
        getattr(build, "defer_trainable_device_move", False)
        or _resolve_wan_offload_mode(build) != "none"
    )


def _stage_eager_wan_modules(
    pipeline: Any,
    build: ModelBuild,
    *,
    offload_mode: str,
    eager_module_dtypes: Mapping[str, torch.dtype],
) -> None:
    """Apply frozen-module dtypes before optional pipeline offload hooks."""

    for module_name, dtype in eager_module_dtypes.items():
        module = getattr(pipeline, module_name, None)
        if module is not None:
            # Diffusers loads the whole pipeline at the actor dtype. Wan's VAE
            # must still decode in fp32, including when Accelerate will later
            # stream it from CPU. A dtype-only conversion preserves CPU
            # residency; the no-offload path additionally stages the module on
            # the execution device.
            if offload_mode == "none":
                module.to(build.device, dtype=dtype)
            else:
                module.to(dtype=dtype)


def _enable_wan_pipeline_offload(
    pipeline: Any,
    device: Any,
    *,
    mode: str,
) -> None:
    """Install Accelerate hooks after adapters and wrappers are final."""

    # Diffusers exposes two mutually exclusive accelerate hooks here:
    # sequential streams per layer and is the 32 GB Wan I2V escape hatch; model
    # offload stages full modules and only works when the transformer fits.
    gpu_id = getattr(device, "index", None) or 0
    if mode == "sequential":
        pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
        return
    if mode == "model":
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)


def _resolve_guidance_scale_2(
    guidance_scale: float,
    boundary_ratio: float | None,
) -> float | None:
    # Wan2.2 dual-expert stages share a single guidance scale: the second
    # (low-noise) expert mirrors the first. No config/request producer sets a
    # per-expert override, so the second stage always mirrors ``guidance_scale``.
    if boundary_ratio is None:
        return None
    return guidance_scale


def _uses_cfg(guidance_scale: float, guidance_scale_2: float | None) -> bool:
    return guidance_scale > 1.0 or (guidance_scale_2 is not None and guidance_scale_2 > 1.0)


def _uses_low_noise_transformer(
    timestep: torch.Tensor,
    *,
    boundary_ratio: float,
    num_train_timesteps: int | None,
) -> bool:
    if num_train_timesteps is None:
        raise ValueError("Wan dual-stage routing requires scheduler.config.num_train_timesteps")
    boundary_timestep = float(boundary_ratio) * int(num_train_timesteps)
    values = torch.as_tensor(timestep).detach().to(dtype=torch.float32)
    low_noise = values < boundary_timestep
    first = bool(low_noise.reshape(-1)[0].item())
    if bool((low_noise != first).any().item()):
        raise ValueError(
            "Wan dual-stage replay batch crosses the transformer boundary within "
            "one forward_step; batch samples must share the same denoise step",
        )
    return first


def _scheduler_num_train_timesteps(scheduler: Any) -> int | None:
    value = _config_value(getattr(scheduler, "config", None), "num_train_timesteps")
    return None if value is None else int(value)


def _module_dtype(module: Any) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(module.parameters()).dtype
    except StopIteration as exc:
        raise RuntimeError(
            f"{type(module).__name__} has no parameters to infer dtype",
        ) from exc


def _cuda_parameter_bytes(module: Any) -> int:
    """Bytes of this expert's parameter shards currently resident on CUDA."""

    total = 0
    for parameter in module.parameters():
        local = getattr(parameter, "_local_tensor", parameter)
        if local.device.type == "cuda":
            total += local.numel() * local.element_size()
    return total


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


__all__ = [
    "WanI2VDiffusersModel",
    "WanI2VReplayModel",
    "WanI2VSamplingState",
    "WanT2VDiffusersModel",
    "WanT2VReplayModel",
    "WanT2VSamplingState",
]
