"""SD 3.5 t2i diffusers-backed model.

Diffusion implementation for Stable Diffusion 3.5-Medium image generation.
The generation helper flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does only
one transformer forward (with optional batched CFG concat) and returns noise
predictions.

Per-family ``SD3SamplingState`` is private to this file — engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Timestep shape convention used by ``forward_step``:
- During rollouts ``timesteps`` is a 1-D tensor ``[T]`` of scheduler
  timesteps; ``state.timesteps[step_idx]`` is a scalar that we expand to
  ``[B]`` for the transformer call.
- During eval/training the collector pre-builds a ``SD3SamplingState``
  whose ``timesteps`` is a ``[1, B]`` tensor (per-sample timestep at the
  selected denoise step) and calls ``forward_step(state, 0, ...)``;
  ``state.timesteps[0]`` is then ``[B]`` and ``forward_step``'s
  ``t.expand(bsz)`` is a no-op (because the source already has shape
  ``[B]`` — ``Tensor.expand`` accepts equal sizes).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDiffusionSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import (
    LoraModelMixin,
    require_lora_for_previous_policy_adapter,
)


@dataclass
class SD3SamplingState(GuidedDiffusionSamplingStateBase):
    """Private SD3 sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None
    do_cfg: bool

    def __post_init__(self) -> None:
        # SD3.5 conditions on an embed AND a pooled embed, so the CFG
        # invariant DiffusionBackboneInput proves for negative_prompt_embeds
        # has a pooled twin that only this family knows about. Proving it here
        # is why the uncond branch reads the key directly, like its cond twin.
        if self.do_cfg and self.negative_pooled_prompt_embeds is None:
            raise ValueError(
                "SD3.5 CFG requires negative_pooled_prompt_embeds; "
                "do_cfg=True was paired with None",
            )


class SD3_5Model(
    VaeDecodeMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    DiffusionBackboneRunnerBase,
):
    """Diffusers-backed SD 3.5 t2i model.

    The model implements the backbone-runner protocol itself (``cfg_mode`` /
    ``build_branch``): "how to call MY transformer" is model knowledge, so
    ``forward_step`` passes ``self`` to the shared CFG caller.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "StableDiffusion3Pipeline"
    _frozen_encoder_names = ("text_encoder", "text_encoder_2", "text_encoder_3")
    # T5-XXL plus two CLIP encoders still fit beside the 2B/8B MMDiT; keep them
    # on-device (no CPU offload dance like Qwen-Image's 15 GB VL).
    _prompt_encoder_on_cpu = False

    @classmethod
    def from_build(cls, build: ModelBuild) -> SD3_5Model:
        """Reject the previous-adapter config before paying the pipeline load."""
        require_lora_for_previous_policy_adapter(build)
        return super().from_build(build)

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via SD3's three text encoders (T5 + 2x CLIP).

        Returns prompt_embeds (joint T5+CLIP sequence), pooled_prompt_embeds
        (CLIP pooled), and their negative counterparts when CFG is active.
        """
        max_seq = kwargs.get("max_sequence_length", 128)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=neg,
            negative_prompt_2=neg,
            negative_prompt_3=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        pooled_prompt_embeds = pooled_prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> SD3SamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
        # SD3 prepare_latents: (batch, channels, height, width, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        do_cfg = request.guidance_scale > 1.0

        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
        )

    # -- backbone runner protocol ---------------------------------------

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map SD3 transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            pooled = request.extra["pooled_prompt_embeds"]
        else:
            embeds = request.negative_prompt_embeds
            pooled = request.extra["negative_pooled_prompt_embeds"]
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={"pooled_projections": pooled},
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SD3SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """SD3 transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # SD3 timestep is broadcast across batch as the raw float (not /1000).
        # If t is already shape [B] (eval path packs timesteps as [1, B]),
        # Tensor.expand(bsz) is a no-op on the equal-sized dim.
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        pooled_prompt_embeds = state.pooled_prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        negative_pooled_prompt_embeds = (
            None
            if state.negative_pooled_prompt_embeds is None
            else state.negative_pooled_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
                extra={
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
        }

    def export_replay_tensors(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into trajectory replay tensors.

        ``latents_clean`` is the final (fully denoised) latent, captured at
        decode time: the x0 the forward-process objectives (DiffusionNFT,
        V-GRPO) regress toward. GRPO ignores it.
        """
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "negative_pooled_prompt_embeds": state.negative_pooled_prompt_embeds,
            "latents_clean": state.latents.detach(),
        }

    def diffusion_nft_prepare_transformer_input(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        timestep: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Raw ``SD3Transformer2DModel`` kwargs for the forward-process objectives.

        DiffusionNFT and V-GRPO call ``transformer(**inputs)[0]`` on the
        unwrapped transformer with a noised clean latent, so this returns
        exactly the conditional branch ``build_branch`` maps in ``forward_step``:
        the latent, the raw ``[0, 1000]`` timestep, the T5+CLIP sequence embeds
        and the pooled CLIP projection. Guidance is not a transformer input for
        SD3 (CFG is a two-branch combine the objectives never run).
        """
        del kwargs
        if pooled_prompt_embeds is None:
            raise ValueError("SD3.5 forward-process input requires pooled_prompt_embeds")
        td = self._transformer_dtype()
        bsz = int(latents.shape[0])
        timestep_batch = expand_batch_timestep(timestep, bsz).to(device=latents.device, dtype=td)
        return {
            "hidden_states": latents.to(td),
            "timestep": timestep_batch,
            "encoder_hidden_states": prompt_embeds.to(td),
            "pooled_projections": pooled_prompt_embeds.to(td),
            "return_dict": False,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> SD3SamplingState:
        """Rebuild SD3SamplingState from a batch slice for the eval forward path.

        Packs timesteps as ``[1, B]`` so ``state.timesteps[0]`` is ``[B]`` —
        matches the eval-path convention documented in the class docstring.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            pooled_prompt_embeds=replay_tensors["pooled_prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=replay_tensors.get(
                "negative_pooled_prompt_embeds",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
        )


class SD3_5ReplayModel(DiffusersReplayModelBase, SD3_5Model):
    """Replay-only SD3.5 model that owns no prompt encoders, VAE, or pipeline.

    ``DiffusersReplayModelBase`` precedes ``SD3_5Model`` in the MRO, so the ctor
    and the non-syncing ``_set_transformer`` both resolve to the replay base.
    """


__all__ = ["SD3SamplingState", "SD3_5Model", "SD3_5ReplayModel"]
