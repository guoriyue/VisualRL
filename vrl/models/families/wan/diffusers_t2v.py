"""Diffusers Wan 1.3B T2V model with step-level denoising runtime.

Provides both monolithic ``generate()`` for serving and step-level
``denoise_init`` / ``predict_noise`` / ``decode_vae`` for RL training.
"""

from __future__ import annotations

import gc
import random
import sys
from typing import Any

from vrl.models.base import ModelResult, VideoGenerationModel, VideoGenerationRequest


class DiffusersWanT2VModel(VideoGenerationModel):
    """Diffusers-based Wan T2V model (targets Wan2.1-T2V-1.3B-Diffusers)."""

    model_family = "wan-diffusers-t2v"

    def __init__(
        self,
        *,
        pipeline: Any,  # diffusers.WanPipeline (already loaded)
        device: Any = None,
    ) -> None:
        self.pipeline = pipeline
        self._device = device

    @property
    def device(self) -> Any:
        if self._device is not None:
            return self._device
        return self.pipeline.device

    async def load(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {
            "name": "wan-diffusers-t2v-model",
            "family": self.model_family,
            "device": str(self.device),
        }

    # -- encode_text ---------------------------------------------------

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        pipe = self.pipeline
        device = self.device

        do_cfg = request.guidance_scale > 1.0
        max_seq = request.extra.get("max_sequence_length", 512)

        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=device,
        )
        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        return ModelResult(
            state_updates={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "pipeline": pipe,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(request.prompt.split())),
            },
        )

    # -- step-level denoising for RL training --------------------------

    async def denoise_init(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> Any:
        """Set up per-step denoising state for Wan T2V.

        Returns a ``DenoiseLoopState`` with ``DiffusersDenoiseState``.
        """
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        pipe = self.pipeline
        device = self.device

        prompt_embeds = state["prompt_embeds"]
        negative_prompt_embeds = state.get("negative_prompt_embeds")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            None,  # generator
            None,  # latents
        )

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)

        ms = DiffusersDenoiseState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            pipeline=pipe,
            seed=seed,
            model_family="wan-diffusers-t2v",
        )

        return DenoiseLoopState(
            current_step=0,
            total_steps=request.num_steps,
            model_state=ms,
        )

    async def predict_noise(
        self,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Forward pass using the pipeline's own transformer."""
        return self._predict_noise_impl(
            self.pipeline.transformer, denoise_state, step_idx,
        )

    def _predict_noise_with_model(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Training path: forward using externally-provided model (e.g. LoRA'd)."""
        return self._predict_noise_impl(model, denoise_state, step_idx)

    def _predict_noise_impl(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan T2V transformer forward + optional CFG."""
        import torch

        ms = denoise_state.model_state
        t = ms.timesteps[step_idx]
        batch_size = ms.latents.shape[0]
        transformer_dtype = ms.prompt_embeds.dtype

        latent_input = ms.latents.to(transformer_dtype)
        timestep_batch = t.expand(batch_size)

        # Forward pass: cond
        noise_pred_cond = model(
            hidden_states=latent_input,
            timestep=timestep_batch,
            encoder_hidden_states=ms.prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred_cond = noise_pred_cond.to(ms.prompt_embeds.dtype)

        # CFG: uncond pass
        if ms.do_cfg:
            noise_pred_uncond = model(
                hidden_states=latent_input,
                timestep=timestep_batch,
                encoder_hidden_states=ms.negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_pred_uncond + ms.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }

    # -- monolithic decode / generate ----------------------------------

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Decode latents -> video using Wan VAE normalization."""
        import torch

        pipe = self.pipeline
        latents = state["latents"]

        latents_for_decode = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_for_decode = latents_for_decode / latents_std + latents_mean
        video = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # [B, T, C, H, W] -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)

        return ModelResult(
            state_updates={"video": video},
            outputs={"video_shape": list(video.shape)},
        )

    async def decode_vae_for_latents(self, latents: Any) -> Any:
        """Decode raw latents -> video tensor [B, C, T, H, W].

        Standalone method for collector use (doesn't need request/state).
        """
        import torch

        pipe = self.pipeline

        latents_for_decode = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_for_decode = latents_for_decode / latents_std + latents_mean
        video = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # [B, T, C, H, W] -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        return video

    async def generate(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Monolithic generation: denoise_init → loop → decode_vae."""
        import torch

        denoise_loop = await self.denoise_init(request, state)
        ms = denoise_loop.model_state
        transformer_dtype = ms.prompt_embeds.dtype

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(denoise_loop.total_steps):
                    fwd = await self.predict_noise(denoise_loop, step_idx)
                    ms.latents = ms.scheduler.step(
                        fwd["noise_pred"], ms.timesteps[step_idx], ms.latents,
                        return_dict=False,
                    )[0]
                    denoise_loop.current_step = step_idx + 1

        state["latents"] = ms.latents
        result = await self.decode_vae(request, state)

        pipe = self.pipeline
        pipe.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        return result
