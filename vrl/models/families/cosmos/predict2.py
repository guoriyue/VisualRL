"""Diffusers Cosmos Predict2 executor."""

from __future__ import annotations

import gc
import hashlib
import random
import sys
import time
from typing import Any

from vrl.models.families.cosmos.variants import CosmosLocalExecutor, CosmosVariant
from vrl.models.base import ModelResult, VideoGenerationRequest

_MODEL_ID_MAP: dict[tuple[str, str], str] = {
    ("video2world", "2B"): "nvidia/Cosmos-Predict2-2B-Video2World",
    ("video2world", "14B"): "nvidia/Cosmos-Predict2-14B-Video2World",
    ("text2image", "0.6B"): "nvidia/Cosmos-Predict2-0.6B-Text2Image",
    ("text2image", "2B"): "nvidia/Cosmos-Predict2-2B-Text2Image",
    ("text2image", "14B"): "nvidia/Cosmos-Predict2-14B-Text2Image",
}

_T2I_VARIANTS = frozenset({CosmosVariant.PREDICT2_TEXT2IMAGE})


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class _PassthroughSafetyChecker:
    """No-op safety checker."""

    def to(self, device: Any) -> _PassthroughSafetyChecker:
        return self

    def check_text_safety(self, prompt: str) -> bool:
        return True

    def check_video_safety(self, video: Any) -> Any:
        return video


class DiffusersCosmosPredict2Executor(CosmosLocalExecutor):
    """Diffusers Predict2 executor (Video2World + Text2Image)."""

    execution_mode = "diffusers_cosmos_predict2"

    def __init__(
        self,
        *,
        variant: CosmosVariant,
        model_size: str = "2B",
        model_id_or_path: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
    ) -> None:
        self.variant = variant
        self.model_size = model_size
        self.model_id_or_path = model_id_or_path
        self.device_id = device_id
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling

        self._modules_loaded = False
        self._torch: Any = None
        self._np: Any = None
        self._pil_image: Any = None
        self._v2w_pipeline_cls: Any = None
        self._t2i_pipeline_cls: Any = None

        self._pipeline: Any = None
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}

    # -- module loading ------------------------------------------------

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        import numpy as np
        import torch
        from diffusers import Cosmos2TextToImagePipeline, Cosmos2VideoToWorldPipeline
        from PIL import Image

        self._torch = torch
        self._np = np
        self._pil_image = Image
        self._v2w_pipeline_cls = Cosmos2VideoToWorldPipeline
        self._t2i_pipeline_cls = Cosmos2TextToImagePipeline
        self._modules_loaded = True

    def _resolve_model_id(self) -> str:
        """Resolve HF model ID."""
        if self.model_id_or_path is not None:
            return self.model_id_or_path
        variant_key = self.variant.value.replace("predict2_", "")
        key = (variant_key, self.model_size)
        if key not in _MODEL_ID_MAP:
            raise ValueError(
                f"Unknown Cosmos Predict2 model: variant={self.variant.value}, "
                f"model_size={self.model_size}. "
                f"Valid combinations: {sorted(_MODEL_ID_MAP.keys())}"
            )
        return _MODEL_ID_MAP[key]

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        self._load_modules()
        torch = self._torch
        model_id = self._resolve_model_id()
        dtype = getattr(torch, self.dtype)
        pipeline_cls = (
            self._t2i_pipeline_cls
            if self.variant in _T2I_VARIANTS
            else self._v2w_pipeline_cls
        )
        import diffusers.pipelines.cosmos.pipeline_cosmos2_text2image as _t2i_mod
        import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _v2w_mod

        _orig_v2w = _v2w_mod.CosmosSafetyChecker
        _orig_t2i = _t2i_mod.CosmosSafetyChecker
        _v2w_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
        _t2i_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
        try:
            pipeline = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
        finally:
            _v2w_mod.CosmosSafetyChecker = _orig_v2w
            _t2i_mod.CosmosSafetyChecker = _orig_t2i
        pipeline.set_progress_bar_config(disable=True)
        if self.enable_cpu_offload:
            pipeline.enable_sequential_cpu_offload(gpu_id=self.device_id)
        if self.enable_vae_tiling:
            pipeline.vae.enable_tiling()
        self._pipeline = pipeline
        return pipeline

    async def load(self) -> None:
        self._load_modules()

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        self._load_modules()
        pipeline = self._get_pipeline()

        prompt = request.prompt or ""
        negative_prompt = request.negative_prompt or ""
        cache_key = _stable_hash(
            f"{self.variant.value}|{self.model_size}|{prompt}|{negative_prompt}"
        )

        cache_hit = cache_key in self._prompt_cache
        if cache_hit:
            prompt_embeds, negative_prompt_embeds = self._prompt_cache[cache_key]
            device = self._torch.device(f"cuda:{self.device_id}")
            prompt_embeds = prompt_embeds.to(device)
            negative_prompt_embeds = negative_prompt_embeds.to(device)
        else:
            encode_result = pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                do_classifier_free_guidance=True,
                device=self._torch.device(f"cuda:{self.device_id}"),
            )
            prompt_embeds = encode_result[0]
            negative_prompt_embeds = encode_result[1]
            self._prompt_cache[cache_key] = (
                prompt_embeds.cpu(),
                negative_prompt_embeds.cpu(),
            )

        return ModelResult(
            state_updates={
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "pipeline": pipeline,
            },
            runtime_state_updates={
                "prompt_cache_key": cache_key,
                "prompt_cache_hit": cache_hit,
                "variant": self.variant.value,
                "model_size": self.model_size,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(prompt.split())),
                "negative_prompt_used": bool(negative_prompt),
                "prompt_cache_hit": cache_hit,
            },
            notes=[
                f"Cosmos Predict2 encode_text completed for {self.variant.value}.",
                f"Prompt cache {'hit' if cache_hit else 'miss'} (key={cache_key}).",
            ],
            cache_hit=cache_hit,
        )

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        if self.variant in _T2I_VARIANTS:
            return ModelResult(
                state_updates={"has_reference": False},
                outputs={"reference_count": 0},
                notes=["Text2Image variant — no reference conditioning needed."],
            )

        self._load_modules()
        if not request.references:
            raise ValueError(
                "Video2World variant requires at least one reference "
                "image/video in request.references"
            )
        reference_path = request.references[0]
        reference_image = self._pil_image.open(reference_path).convert("RGB")

        return ModelResult(
            state_updates={
                "reference_image": reference_image,
                "has_reference": True,
            },
            runtime_state_updates={
                "reference_path": reference_path,
            },
            outputs={"reference_count": len(request.references)},
            notes=["Video2World reference image loaded for conditioning."],
        )

    async def generate(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        pipeline = state["pipeline"]
        torch = self._torch

        if request.seed is not None:
            seed = request.seed
            seed_policy = "explicit"
        else:
            seed = random.randint(0, sys.maxsize)
            seed_policy = "randomized"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        kwargs: dict[str, Any] = {
            "prompt_embeds": state["prompt_embeds"],
            "negative_prompt_embeds": state["negative_prompt_embeds"],
            "height": request.height,
            "width": request.width,
            "num_inference_steps": request.num_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "output_type": "np",
            "return_dict": True,
        }

        if state.get("reference_image") is not None:
            kwargs["image"] = state["reference_image"]

        started = time.perf_counter()
        pipeline_output = pipeline(**kwargs)
        elapsed_s = round(time.perf_counter() - started, 4)

        frames = pipeline_output.frames[0]

        pipeline.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        return ModelResult(
            state_updates={
                "video_frames": frames,
                "seed": seed,
            },
            runtime_state_updates={
                "frame_shape": list(frames.shape),
                "seed": seed,
                "seed_policy": seed_policy,
                "elapsed_s": elapsed_s,
            },
            outputs={
                "num_steps": request.num_steps,
                "guidance_scale": request.guidance_scale,
                "seed": seed,
                "elapsed_s": elapsed_s,
            },
            notes=[
                f"Cosmos Predict2 denoise completed in {elapsed_s}s "
                f"(steps={request.num_steps}, seed={seed}, policy={seed_policy}).",
            ],
        )

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        frames = self._np.asarray(state["video_frames"])
        return ModelResult(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "decoded_frame_count": int(frames.shape[0]),
                "decoded_spatial_size": [int(frames.shape[2]), int(frames.shape[1])],
            },
            outputs={"fps": request.fps or 16},
            notes=[
                "VAE decode stage is a passthrough — frames already decoded by diffusers pipeline."
            ],
        )

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        np = self._np
        frames = np.asarray(state["video_frames"])
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
        return ModelResult(
            state_updates={
                "video_frames": frames,
                "output_fps": request.fps or 16,
            },
            runtime_state_updates={
                "frame_count": int(frames.shape[0]),
                "output_fps": request.fps or 16,
            },
            outputs={"frame_count": int(frames.shape[0])},
            notes=["Postprocess normalized frames to uint8."],
        )

    # -- Step-level denoising for RL training ----------------------------

    async def denoise_init(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> Any:
        """Set up per-step denoising state for Cosmos Predict2.

        Returns a ``DenoiseLoopState`` whose ``model_state`` is a
        ``DiffusersDenoiseState`` populated with Cosmos-specific fields.
        """
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        pipeline = self._get_pipeline()
        device = torch.device(f"cuda:{self.device_id}")

        # Text conditioning must already be in state (from encode_text)
        prompt_embeds = state["prompt_embeds"]
        negative_prompt_embeds = state.get("negative_prompt_embeds")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        # Scheduler
        pipeline.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        num_channels_latents = pipeline.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        # Reference image for Video2World
        reference_image = state.get("reference_image")
        if reference_image is not None:
            video_input = pipeline.video_processor.preprocess_video(
                reference_image,
                height=request.height,
                width=request.width,
            ).to(device, dtype=pipeline.vae.dtype)
        else:
            video_input = torch.zeros(
                batch_size, 3, 1, request.height, request.width,
                device=device, dtype=pipeline.vae.dtype,
            )

        # Cosmos2 prepare_latents returns 6-tuple
        latents_result = pipeline.prepare_latents(
            video=video_input,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            do_classifier_free_guidance=do_cfg,
            dtype=torch.float32,
            device=device,
            generator=None,
            latents=None,
        )
        latents = latents_result[0]
        init_latents = latents_result[1]
        cond_indicator = latents_result[2]
        uncond_indicator = latents_result[3]
        cond_mask = latents_result[4]
        uncond_mask = latents_result[5]

        sigma_data = pipeline.scheduler.config.sigma_data
        sigma_conditioning = 0.0001

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)

        ms = DiffusersDenoiseState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipeline.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            pipeline=pipeline,
            init_latents=init_latents,
            cond_indicator=cond_indicator,
            uncond_indicator=uncond_indicator,
            cond_mask=cond_mask,
            uncond_mask=uncond_mask,
            fps=request.fps or 16,
            sigma_data=sigma_data,
            sigma_conditioning=sigma_conditioning,
            seed=seed,
            model_family="cosmos-predict2",
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
            self._get_pipeline().transformer, denoise_state, step_idx,
        )

    def _predict_noise_with_model(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Training path: forward using an externally-provided model (e.g. LoRA'd)."""
        return self._predict_noise_impl(model, denoise_state, step_idx)

    def _predict_noise_impl(
        self,
        model: Any,
        denoise_state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        """Cosmos Predict2 transformer forward + optional CFG.

        Handles cond_indicator blending and condition_mask kwargs.
        """
        import torch

        ms = denoise_state.model_state
        t = ms.timesteps[step_idx]
        batch_size = ms.latents.shape[0]
        transformer_dtype = ms.prompt_embeds.dtype

        # Build conditioning latent (reference frames blended with noise)
        cond_latent = (
            ms.init_latents * (1 - ms.sigma_conditioning)
            + ms.sigma_conditioning * torch.randn_like(ms.init_latents)
        )
        cond_latent_input = (
            ms.cond_indicator * cond_latent
            + (1 - ms.cond_indicator) * ms.latents.to(transformer_dtype)
        )

        # Forward pass: cond
        noise_pred_cond = model(
            hidden_states=cond_latent_input.to(transformer_dtype),
            timestep=t.expand(batch_size),
            encoder_hidden_states=ms.prompt_embeds,
            fps=ms.fps,
            condition_mask=ms.cond_mask,
            return_dict=False,
        )[0]
        noise_pred_cond = noise_pred_cond.to(ms.prompt_embeds.dtype)

        # CFG: uncond pass
        if ms.do_cfg:
            uncond_latent_input = (
                ms.uncond_indicator * cond_latent
                + (1 - ms.uncond_indicator) * ms.latents.to(transformer_dtype)
            )
            noise_pred_uncond = model(
                hidden_states=uncond_latent_input.to(transformer_dtype),
                timestep=t.expand(batch_size),
                encoder_hidden_states=ms.negative_prompt_embeds,
                fps=ms.fps,
                condition_mask=ms.uncond_mask,
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

    async def decode_vae_for_latents(self, latents: Any) -> Any:
        """Decode raw latents -> video tensor with Cosmos normalization.

        Returns video tensor [B, C, T, H, W].
        """
        import torch

        pipeline = self._get_pipeline()
        sigma_data = pipeline.scheduler.config.sigma_data

        latents_for_decode = latents.to(pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_std = (
            torch.tensor(pipeline.vae.config.latents_std)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        # Cosmos2 decode: z_raw = z_norm * std / sigma_data + mean
        latents_for_decode = latents_for_decode * latents_std / sigma_data + latents_mean
        video = pipeline.vae.decode(latents_for_decode, return_dict=False)[0]
        video = pipeline.video_processor.postprocess_video(video, output_type="pt")
        # [B, T, C, H, W] -> [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)
        return video

    def describe(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "executor": self.__class__.__name__,
            "variant": self.variant.value,
            "model_size": self.model_size,
            "model_id": self._resolve_model_id(),
            "device_id": self.device_id,
            "dtype": self.dtype,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_vae_tiling": self.enable_vae_tiling,
            "pipeline_loaded": self._pipeline is not None,
            "prompt_cache_size": len(self._prompt_cache),
        }
