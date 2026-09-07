"""Cosmos3 Omni vision generator — diffusers-backed model for flow-matching GRPO RL.

Wraps the ``Cosmos3OmniPipeline`` released in diffusers 0.39 (0.40+ required by the pin). Cosmos3 is an omni MoT
(Mixture-of-Transformers): one ``Cosmos3OmniTransformer`` carries a Qwen3-VL
causal reasoner stream + a bidirectional diffusion generation stream. We train
ONLY the vision-generation stream as an RL policy; sound/action towers are off.

Two design facts that drive this file (verified against diffusers 0.39 source):

1. **Strictly batch=1.** ``Cosmos3OmniPipeline`` packs one sample at a time
   (``vision_tokens=[one [1,C,T,H,W] latent]``; all packed-static index tensors
   sized for a single sample). The batch executor pins ``samples_per_generation_batch=1``;
   ``forward_step`` therefore runs the single sample, no dim-0 batch.

2. **forward_step returns the RAW rectified-flow velocity as ``noise_pred`` —
   NOT divided by sigma.** Cosmos3's ``UniPCMultistepScheduler`` keeps sigmas in
   ``[0,1]`` → ``sde_step_with_logprob`` auto-detects ``edm_domain=False`` and
   consumes the velocity directly (flow-grpo branch). This is the OPPOSITE of
   predict2's EDM ``(hidden-combined)/sigma`` noise estimate; returning a raw
   velocity through an EDM scheduler (or vice-versa) yields garbage logprobs.

The per-step transformer call needs a large interleaved "packed_static" (text +
vision segments + joint mRoPE positions). The diffusers pipeline assembles it
INLINE in ``__call__`` (no reusable method), so ``prepare_sampling`` replicates
that assembly once (step-invariant) by calling the pipeline's segment builders;
``forward_step`` only splices the per-step ``vision_tokens`` + ``vision_timesteps``.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.execution.sample_batches import SampleAlignedValues
from vrl.generation.types import DenoiseRequest
from vrl.models.families.cosmos import CosmosReplayForward
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusionModelBase,
    GuidedDiffusionSamplingStateBase,
    ReplayRolloutStubs,
)
from vrl.models.steps.denoise.common import align_replay_tensor
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_FPS = 24
_DEFAULT_GUIDANCE = 7.0


@dataclass
class Cosmos3SamplingState(GuidedDiffusionSamplingStateBase):
    """Per-rollout sampling state. The packed_static dicts are step-invariant;
    forward_step splices the live latents + per-step timestep each call.

    ``latents`` is the live x_t, [1, C, T, H, W] fp32; ``timesteps`` is
    ``scheduler.timesteps`` (len == num_steps); ``scheduler`` is a
    UniPCMultistepScheduler (sigmas in [0,1] -> flow domain).
    """

    cond_packed_static: dict[str, Any]  # everything EXCEPT vision_tokens/vision_timesteps
    uncond_packed_static: dict[str, Any]
    vision_condition_mask: torch.Tensor  # [latent_t, 1, 1] — 0 noisy, 1 conditioned
    num_noisy_vision_tokens: int
    do_cfg: bool
    height: int
    width: int
    num_frames: int
    fps: int = _DEFAULT_FPS
    # raw cond/uncond input-id lists, kept for replay reassembly (§replay).
    cond_input_ids: list[int] = field(default_factory=list)
    uncond_input_ids: list[int] = field(default_factory=list)


class Cosmos3Model(CosmosReplayForward, LoraModelMixin, DiffusersPipelineModelBase):
    """Cosmos3 Omni T2V generator wrapped for the vrl diffusion RL seam."""

    # ---- properties (mirror predict2_5) ----
    @classmethod
    def from_build(cls, build: ModelBuild) -> Cosmos3Model:
        # Lazy: the optional cosmos extra must not be imported at module load.
        from diffusers import Cosmos3OmniPipeline

        kwargs: dict[str, Any] = {
            "torch_dtype": build.parameter_dtype,
            **build.revision_kwargs,
        }
        # enable_safety_checker=False avoids the cosmos_guardrail import/dep in dev.
        pipeline = Cosmos3OmniPipeline.from_pretrained(
            build.model_name_or_path,
            enable_safety_checker=False,
            **kwargs,
        )
        torch.set_grad_enabled(True)
        pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "vae"):
            pipeline.vae.requires_grad_(False)
            pipeline.vae.to(build.device, dtype=torch.float32)
        # No separate text encoder: the joint transformer consumes raw Qwen2 ids.
        pipeline.transformer.to(build.device, dtype=build.parameter_dtype)
        logger.info(
            "loaded Cosmos3 omni generator %s",
            kv(path=build.model_name_or_path, device=build.device, dtype=build.parameter_dtype),
        )
        return cls(pipeline=pipeline, device=build.device)

    def _lora_dtype(self, build: ModelBuild) -> Any | None:
        # The transformer is already cast at load (from_build .to(dtype));
        # skip the mixin's default pre-wrap dtype cast.
        del build
        return None

    # ---- encode ----
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        guidance_scale = float(kwargs.get("guidance_scale", _DEFAULT_GUIDANCE))
        text = prompt if isinstance(prompt, str) else prompt[0]
        neg = negative_prompt[0] if isinstance(negative_prompt, list) else negative_prompt
        cond_ids, uncond_ids = self.pipeline.tokenize_prompt(
            text,
            neg,
            num_frames=int(kwargs.get("num_frames", 93)),
            height=int(kwargs.get("height", 704)),
            width=int(kwargs.get("width", 1280)),
            fps=float(kwargs.get("fps", _DEFAULT_FPS)),
        )
        return {
            "cond_input_ids": cond_ids,
            "uncond_input_ids": uncond_ids,
            "guidance_scale": guidance_scale,
        }

    # ---- sampling-state assembly (the step-invariant packed_static) ----
    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Cosmos3SamplingState:
        del kwargs
        pipe = self.pipeline
        device = self.device
        guidance_scale = float(encoded["guidance_scale"])
        do_cfg = guidance_scale > 1.0
        fps = request.fps or _DEFAULT_FPS
        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # (1) initial noisy latents + condition mask. T2V => image/video None =>
        #     mask all-zero (no conditioned frames). Unpack BY POSITION (the type
        #     annotation is a stale 11-tuple; the real return is 12 fields).
        prepared = pipe.prepare_latents(
            image=None,
            video=None,
            num_frames=request.frame_count,
            height=request.height,
            width=request.width,
            fps=float(fps),
            generator=generator,
            device=device,
            dtype=pipe.transformer.dtype,
        )
        latents = prepared[0]
        vision_condition_mask = prepared[5]

        # (2) text segments (one per cond/uncond). The temporal mRoPE offset is
        #     shared between text and vision (siblings, not sequential).
        cond_text = pipe._prepare_text_segment(encoded["cond_input_ids"], device)
        uncond_text = pipe._prepare_text_segment(encoded["uncond_input_ids"], device)
        mrope_offset = cond_text["vision_start_temporal_offset"]

        # (3) vision segments. has_image_condition recomputed from the mask.
        cond_idx = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
        has_img = bool(cond_idx.numel() > 0)
        cond_frames = [int(i) for i in cond_idx.tolist()] or None
        cond_vision = pipe._prepare_vision_segment(
            latents,
            has_img,
            mrope_offset,
            float(fps),
            curr=cond_text["und_len"],
            device=device,
            condition_frame_indexes=cond_frames,
        )
        uncond_vision = pipe._prepare_vision_segment(
            latents,
            has_img,
            mrope_offset,
            float(fps),
            curr=uncond_text["und_len"],
            device=device,
            condition_frame_indexes=cond_frames,
        )

        cond_static = _assemble_packed_static(cond_text, cond_vision)
        uncond_static = _assemble_packed_static(uncond_text, uncond_vision)

        return Cosmos3SamplingState(
            latents=latents,
            timesteps=pipe.scheduler.timesteps,
            scheduler=pipe.scheduler,
            cond_packed_static=cond_static,
            uncond_packed_static=uncond_static,
            vision_condition_mask=vision_condition_mask,
            num_noisy_vision_tokens=int(cond_vision["num_noisy_vision_tokens"]),
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            fps=fps,
            cond_input_ids=list(encoded["cond_input_ids"]),
            uncond_input_ids=list(encoded["uncond_input_ids"]),
        )

    # ---- the per-step transformer call -> flow velocity ----
    def forward_step(
        self,
        state: Cosmos3SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        device = state.latents.device
        dtype = self.transformer.dtype
        timestep = float(state.timesteps[step_idx].item())
        vision_tokens = state.latents.to(device=device, dtype=dtype)
        vision_timesteps = torch.full(
            (state.num_noisy_vision_tokens,),
            timestep,
            device=device,
        )

        def _run(pack: dict[str, Any]) -> torch.Tensor:
            preds_vision, _sound, _action = self.transformer(
                input_ids=pack["input_ids"],
                text_indexes=pack["text_indexes"],
                position_ids=pack["position_ids"],
                und_len=pack["und_len"],
                sequence_length=pack["sequence_length"],
                vision_tokens=[vision_tokens],
                vision_token_shapes=pack["vision_token_shapes"],
                vision_sequence_indexes=pack["vision_sequence_indexes"],
                vision_mse_loss_indexes=pack["vision_mse_loss_indexes"],
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=pack["vision_noisy_frame_indexes"],
                # diffusers 0.40 wraps the three lists in Cosmos3OmniTransformerOutput by default.
                return_dict=False,
            )
            velocity, _s, _a = self.pipeline._mask_velocity_predictions(
                preds_vision,
                None,
                [state.vision_condition_mask],
            )
            return velocity

        cond = _run(state.cond_packed_static)
        if state.do_cfg:
            uncond = _run(state.uncond_packed_static)
            combined = uncond + state.guidance_scale * (cond - uncond)
        else:
            uncond = torch.zeros_like(cond)
            combined = cond
        # RAW flow velocity -> sde_step_with_logprob (UniPC sigmas in [0,1]).
        return {
            "noise_pred": combined.to(torch.float32),
            "noise_pred_cond": cond,
            "noise_pred_uncond": uncond,
        }

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Denorm with the vae latents_mean/inv_std then decode (mirrors the
        pipeline's final-step decode)."""
        pipe = self.pipeline
        in_dtype = latents.dtype
        vdtype = pipe.vae.dtype
        mean = pipe._vae_latents_mean.to(device=latents.device, dtype=vdtype)
        inv_std = pipe._vae_latents_inv_std.to(device=latents.device, dtype=vdtype)
        z_raw = latents.to(vdtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(z_raw).sample.to(in_dtype)
        return pipe.video_processor.postprocess_video(decoded, output_type="pt")[0]

    # ---- replay export/restore (batch=1) ----
    def export_batch_context(self, state: Cosmos3SamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
            "num_frames": state.num_frames,
            "fps": state.fps,
            "num_noisy_vision_tokens": state.num_noisy_vision_tokens,
        }

    def export_replay_tensors(self, state: Cosmos3SamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": None,
            "negative_prompt_embeds": None,
            "prompt_attention_mask": None,
            "pooled_prompt_embeds": None,
            "latents_clean": state.latents.detach(),
            "cond_input_ids": SampleAlignedValues((state.cond_input_ids,)),
            "uncond_input_ids": SampleAlignedValues((state.uncond_input_ids,)),
            "vision_condition_mask": align_replay_tensor(
                state.vision_condition_mask,
                state.latents.shape[0],
            ),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Cosmos3SamplingState:
        del step_idx  # cosmos indexes sigmas[step_idx] live in forward_step
        pipe = self.pipeline
        device = self.device
        fps = int(batch_context.get("fps", _DEFAULT_FPS))
        vision_condition_mask = replay_tensors["vision_condition_mask"]
        cond_input_ids = _single_sample_input_ids(
            replay_tensors["cond_input_ids"],
            name="cond_input_ids",
        )
        uncond_input_ids = _single_sample_input_ids(
            replay_tensors["uncond_input_ids"],
            name="uncond_input_ids",
        )
        cond_text = pipe._prepare_text_segment(cond_input_ids, device)
        uncond_text = pipe._prepare_text_segment(uncond_input_ids, device)
        mrope_offset = cond_text["vision_start_temporal_offset"]
        cond_idx = torch.nonzero(vision_condition_mask[:, 0, 0] > 0, as_tuple=False).flatten()
        has_img = bool(cond_idx.numel() > 0)
        cond_frames = [int(i) for i in cond_idx.tolist()] or None
        cond_vision = pipe._prepare_vision_segment(
            latents,
            has_img,
            mrope_offset,
            float(fps),
            curr=cond_text["und_len"],
            device=device,
            condition_frame_indexes=cond_frames,
        )
        uncond_vision = pipe._prepare_vision_segment(
            latents,
            has_img,
            mrope_offset,
            float(fps),
            curr=uncond_text["und_len"],
            device=device,
            condition_frame_indexes=cond_frames,
        )
        return Cosmos3SamplingState(
            latents=latents,
            timesteps=pipe.scheduler.timesteps,
            scheduler=pipe.scheduler,
            cond_packed_static=_assemble_packed_static(cond_text, cond_vision),
            uncond_packed_static=_assemble_packed_static(uncond_text, uncond_vision),
            vision_condition_mask=vision_condition_mask,
            num_noisy_vision_tokens=int(cond_vision["num_noisy_vision_tokens"]),
            guidance_scale=float(batch_context["guidance_scale"]),
            do_cfg=bool(batch_context["cfg"]),
            height=int(batch_context["height"]),
            width=int(batch_context["width"]),
            num_frames=int(batch_context["num_frames"]),
            fps=fps,
            cond_input_ids=cond_input_ids,
            uncond_input_ids=uncond_input_ids,
        )


def _single_sample_input_ids(value: Any, *, name: str) -> list[int]:
    """Unwrap one ragged replay row at Cosmos3's enforced batch-one boundary."""

    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(f"Cosmos3 replay {name} must contain exactly one sample row")
    row = value[0]
    if not isinstance(row, (list, tuple)):
        raise TypeError(f"Cosmos3 replay {name} row must be a token-id sequence")
    return [int(token_id) for token_id in row]


class Cosmos3ReplayModel(ReplayRolloutStubs, Cosmos3Model):
    """Trainer-side replay model: transformer + scheduler + the pipeline's segment
    builders, but no generation pipeline. It still needs the pipeline's
    ``_prepare_text_segment`` / ``_prepare_vision_segment`` / ``_mask_velocity_predictions``
    to rebuild packed_static, so it holds a weights-free pipeline shell."""

    def __init__(self, *, pipeline_shell: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        object.__setattr__(self, "_pipeline", pipeline_shell)
        self.transformer = pipeline_shell.transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def scheduler(self) -> Any:
        return self._scheduler


def _assemble_packed_static(text: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    """Merge text+vision segments into the transformer-facing packed_static, MINUS
    the step-varying ``vision_tokens`` / ``vision_timesteps`` (spliced per step).
    Mirrors the inline assembly in ``Cosmos3OmniPipeline.__call__``."""
    position_ids = torch.cat([text["text_mrope_ids"], vision["vision_mrope_ids"]], dim=1)
    return {
        **text,
        **vision,
        "position_ids": position_ids,
        "sequence_length": text["und_len"] + vision["num_vision_tokens"],
    }


__all__ = ["Cosmos3Model", "Cosmos3ReplayModel", "Cosmos3SamplingState"]
