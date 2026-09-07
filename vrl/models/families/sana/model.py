"""SANA t2i diffusers-backed model.

Diffusion implementation for NVIDIA SANA (linear-attention DiT + DC-AE).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

SANA specifics vs SD3 (the reference family):
- Single Gemma-2 text encoder: ``encode_prompt`` returns a SEQUENCE embed plus
  an attention mask and NO pooled vector; the mask threads through the
  transformer as ``encoder_attention_mask`` on both CFG branches.
- Latents are UNPACKED ``[B, 32, H/32, W/32]`` (DC-AE, 32x compression — vs
  the 8x KL-VAE of SD3/FLUX). No packing anywhere; ``decode_latents`` is a
  plain ``latents / scaling_factor`` decode (DC-AE has no shift_factor).
- TRUE classifier-free guidance with both branches padded to the same
  ``max_sequence_length`` (300), so the branches batch into one forward
  (``batched_cfg``, like SD3 — unlike Qwen-Image's separate branches).
- The transformer multiplies the timestep by ``config.timestep_scale``
  (1.0 on current checkpoints; respected for parity with SanaPipeline).
- ``complex_human_instruction`` (SANA's CHI prompt template) is always
  disabled: VRL fixes it to None so RL datasets control their prompts.
  diffusers' ``SanaPipeline.__call__`` defaults it ON, so GPU parity runs must
  pass the same (disabled) CHI value on both sides.
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
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    EncoderAttentionMaskRunnerBase,
    MaskedPromptCollectorMixin,
    MaskedPromptSamplingState,
    VaeDecodeMixin,
    expand_batch_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin


@dataclass
class SanaSamplingState(MaskedPromptSamplingState):
    """Private SANA sampling state. Engine MUST NOT introspect."""


class SanaModel(
    VaeDecodeMixin,
    MaskedPromptCollectorMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    EncoderAttentionMaskRunnerBase,
):
    """Diffusers-backed SANA t2i model.

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (Gemma-2 encode at ``max_sequence_length``), so
    they pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"
    sampling_state_cls = SanaSamplingState

    _pipeline_classname = "SanaPipeline"
    _frozen_encoder_names = ("text_encoder",)
    # Gemma-2-2B is small enough to co-reside with the 1.6B DiT; keep it
    # on-device (no CPU offload dance like Qwen-Image's 15 GB VL).
    _prompt_encoder_on_cpu = False

    # -- backend ownership (called by runtime, not by collectors) -------

    @staticmethod
    def _apply_fp16_saturation_clamp(transformer: Any) -> None:
        """Reapply SANA's fp16 attention saturation on a non-fp16 transformer.

        The published fp16 checkpoint was calibrated WITH diffusers'
        ``SanaLinearAttnProcessor2_0`` output clip to the fp16 range, but that
        clip is dtype-conditional (``if original_dtype == torch.float16``).
        Running the same weights in fp32/bf16 skips it, and the un-saturated
        linear-attention outputs corrupt every image. It must be reapplied at
        each linear-attention layer (not the transformer's final output), so it
        rides ``set_attn_processor`` rather than ``forward_step``. Verified
        2026-07-18: the official pipeline in fp32 reproduces the corruption, and
        fp32 plus this clamp matches fp16 quality
        (outputs/quality_preflight/sana_fp32_probe).
        """

        from diffusers.models.attention_processor import SanaLinearAttnProcessor2_0

        class _SaturatedLinearAttnProcessor(SanaLinearAttnProcessor2_0):
            def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
                return super().__call__(*args, **kwargs).clip(-65504, 65504)

        transformer.set_attn_processor(
            {
                name: (
                    _SaturatedLinearAttnProcessor()
                    if type(processor) is SanaLinearAttnProcessor2_0
                    else processor
                )
                for name, processor in transformer.attn_processors.items()
            },
        )

    @classmethod
    def from_build(cls, build: ModelBuild) -> SanaModel:
        """Shared pipeline load, then SANA's two deviations: flow-match scheduler + fp16 clamp.

        The freeze / placement sequence (frozen encoder at the rollout prompt
        dtype, DC-AE in fp32 for decode fidelity) is the shared loader's; only
        what follows is SANA-specific.
        """
        model = super().from_build(build)
        pipeline = model.pipeline
        # SANA is rectified-flow native; diffusers ships DPMSolverMultistep for
        # fast inference, but flow-matching GRPO's per-step SDE log-prob needs a
        # FlowMatchEuler scheduler on BOTH sides. The replay bundle already loads
        # FlowMatchEuler (build.py, no scheduler_classname); the rollout was still
        # on DPMSolver, so rollout timesteps never matched replay's and
        # index_for_timestep(t) returned empty at the first-step parity check.
        # Swap rollout to FlowMatchEuler for per-step log-prob. SANA's shipped
        # DPM config calls this value ``flow_shift``; FlowMatch calls it ``shift``.
        # Passing it explicitly preserves the checkpoint's shift=3 instead of
        # silently accepting FlowMatch's default shift=1 (the color-block bug).
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = dict(pipeline.scheduler.config)
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler_config,
            shift=float(scheduler_config.get("flow_shift", 1.0)),
        )
        if build.parameter_dtype != torch.float16:
            cls._apply_fp16_saturation_clamp(pipeline.transformer)
        return model

    def prepare_replay(self, build: ModelBuild) -> None:
        """Replay forwards need the same non-fp16 saturation clamp as rollout."""
        if build.parameter_dtype != torch.float16:
            self._apply_fp16_saturation_clamp(self.transformer)

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Gemma-2 (sequence embeds + attention mask, no pooled).

        Returns the conditional embeds/mask and, when CFG is active, the
        unconditional embeds/mask (SANA's uncond default is the empty string).
        """
        max_seq = kwargs.get("max_sequence_length", 300)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg,
            num_images_per_prompt=1,
            device=self.device,
            max_sequence_length=max_seq,
            # VRL intentionally disables SANA's CHI template so RL datasets own
            # their prompts. Pin to None so an upstream default flip cannot
            # silently re-enable it.
            complex_human_instruction=None,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(td)
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> SanaSamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_attention_mask = encoded.get("negative_prompt_attention_mask")

        # Static flow-shift schedule (SANA has no dynamic shifting).
        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
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

        do_cfg = request.guidance_scale > 1.0 and negative_prompt_embeds is not None

        return SanaSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SanaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """SANA transformer forward + optional batched CFG."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # SanaPipeline multiplies the raw timestep by config.timestep_scale.
        timestep_scale = float(
            getattr(self.transformer.config, "timestep_scale", 1.0),
        )
        # Keep the scheduler timestep in fp32, exactly as SanaPipeline does.
        # Transformer inputs/weights remain at the native FP16 role dtype;
        # the time embedding owns its internal conversion.
        timestep_batch = expand_batch_timestep(t, bsz).to(latent_input.device) * timestep_scale
        negative_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=negative_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                # SanaPipeline promotes each transformer branch before CFG.
                # Keeping this fp32 also feeds the protected scheduler/log-prob
                # path without a lossy fp16 round trip.
                output_dtype=torch.float32,
                extra={
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "negative_encoder_attention_mask": state.negative_prompt_attention_mask,
                },
            ),
        )
        return output.as_dict()


class SanaReplayModel(DiffusersReplayModelBase, SanaModel):
    """Replay-only SANA model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["SanaModel", "SanaReplayModel", "SanaSamplingState"]
