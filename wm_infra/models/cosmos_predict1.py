"""Diffusers-based Cosmos Predict1 executor (Text2World + Video2World).

Owns the model forward logic for Cosmos Predict1 variants using HuggingFace
diffusers ``CosmosTextToWorldPipeline`` and ``CosmosVideoToWorldPipeline``.

Predict2 is explicitly deferred — constructing with a Predict2 variant raises
``NotImplementedError``.
"""

from __future__ import annotations

import gc
import hashlib
import random
import sys
import time
from typing import Any

from wm_infra.models.cosmos_adapter import CosmosLocalExecutor, CosmosVariant
from wm_infra.models.video_generation import (
    StageResult,
    VideoGenerationRequest,
)

# Cosmos Predict1 HuggingFace model ID map
_MODEL_ID_MAP: dict[tuple[str, str], str] = {
    ("text2world", "7B"): "nvidia/Cosmos-1.0-Diffusion-7B-Text2World",
    ("text2world", "14B"): "nvidia/Cosmos-1.0-Diffusion-14B-Text2World",
    ("video2world", "7B"): "nvidia/Cosmos-1.0-Diffusion-7B-Video2World",
    ("video2world", "14B"): "nvidia/Cosmos-1.0-Diffusion-14B-Video2World",
}

# Variants that use the Text2World pipeline
_T2W_VARIANTS = frozenset({CosmosVariant.PREDICT1_TEXT2WORLD})


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class DiffusersCosmosPredict1Executor(CosmosLocalExecutor):
    """Diffusers-based in-process executor for Cosmos Predict1 variants.

    Supports both Text2World and Video2World through a single class — the
    ``variant`` determines which diffusers pipeline class is loaded.
    """

    execution_mode = "diffusers_cosmos_predict1"

    def __init__(
        self,
        *,
        variant: CosmosVariant,
        model_size: str = "7B",
        model_id_or_path: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
    ) -> None:
        if variant == CosmosVariant.PREDICT2_VIDEO2WORLD:
            raise NotImplementedError(
                "Cosmos Predict2 is not yet implemented. "
                "Use a Predict1 variant (predict1_text2world or predict1_video2world)."
            )
        self.variant = variant
        self.model_size = model_size
        self.model_id_or_path = model_id_or_path
        self.device_id = device_id
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling

        # Lazy module refs — populated by _load_modules()
        self._modules_loaded = False
        self._torch: Any = None
        self._np: Any = None
        self._pil_image: Any = None
        self._t2w_pipeline_cls: Any = None
        self._v2w_pipeline_cls: Any = None

        # Cached state
        self._pipeline: Any = None
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}

    # -- module loading ------------------------------------------------

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        import numpy as np
        import torch
        from diffusers import CosmosTextToWorldPipeline, CosmosVideoToWorldPipeline
        from PIL import Image

        self._torch = torch
        self._np = np
        self._pil_image = Image
        self._t2w_pipeline_cls = CosmosTextToWorldPipeline
        self._v2w_pipeline_cls = CosmosVideoToWorldPipeline
        self._modules_loaded = True

    def _resolve_model_id(self) -> str:
        """Return the HuggingFace model ID for the current variant/size."""
        if self.model_id_or_path is not None:
            return self.model_id_or_path
        variant_key = self.variant.value.replace("predict1_", "")
        key = (variant_key, self.model_size)
        if key not in _MODEL_ID_MAP:
            raise ValueError(
                f"Unknown Cosmos Predict1 model: variant={self.variant.value}, "
                f"model_size={self.model_size}. "
                f"Valid combinations: {sorted(_MODEL_ID_MAP.keys())}"
            )
        return _MODEL_ID_MAP[key]

    def _get_pipeline(self) -> Any:
        """Lazy-load and cache the diffusers pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        self._load_modules()
        torch = self._torch
        model_id = self._resolve_model_id()
        dtype = getattr(torch, self.dtype)
        pipeline_cls = (
            self._t2w_pipeline_cls
            if self.variant in _T2W_VARIANTS
            else self._v2w_pipeline_cls
        )
        pipeline = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
        pipeline.set_progress_bar_config(disable=True)
        if self.enable_cpu_offload:
            pipeline.enable_sequential_cpu_offload(gpu_id=self.device_id)
        if self.enable_vae_tiling:
            pipeline.vae.enable_tiling()
        self._pipeline = pipeline
        return pipeline

    # -- executor interface: load --------------------------------------

    async def load(self) -> None:
        self._load_modules()

    # -- stage: encode_text --------------------------------------------

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
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
            # Cache on CPU to save GPU memory
            self._prompt_cache[cache_key] = (
                prompt_embeds.cpu(),
                negative_prompt_embeds.cpu(),
            )

        return StageResult(
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
                f"Cosmos Predict1 encode_text completed for {self.variant.value}.",
                f"Prompt cache {'hit' if cache_hit else 'miss'} (key={cache_key}).",
            ],
            cache_hit=cache_hit,
        )

    # -- stage: encode_conditioning ------------------------------------

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        if self.variant in _T2W_VARIANTS:
            # Text2World has no reference conditioning
            return StageResult(
                state_updates={"has_reference": False},
                outputs={"reference_count": 0},
                notes=["Text2World variant — no reference conditioning needed."],
            )

        # Video2World — load reference image/video
        self._load_modules()
        if not request.references:
            raise ValueError(
                "Video2World variant requires at least one reference "
                "image/video in request.references"
            )
        reference_path = request.references[0]
        reference_image = self._pil_image.open(reference_path).convert("RGB")

        return StageResult(
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

    # -- stage: denoise ------------------------------------------------

    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        pipeline = state["pipeline"]
        torch = self._torch

        # Seed handling
        if request.seed is not None:
            seed = request.seed
            seed_policy = "explicit"
        else:
            seed = random.randint(0, sys.maxsize)
            seed_policy = "randomized"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # Build pipeline kwargs
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

        # Video2World passes reference image
        if state.get("reference_image") is not None:
            kwargs["image"] = state["reference_image"]

        started = time.perf_counter()
        pipeline_output = pipeline(**kwargs)
        elapsed_s = round(time.perf_counter() - started, 4)

        frames = pipeline_output.frames[0]

        # Cleanup
        pipeline.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        return StageResult(
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
                f"Cosmos Predict1 denoise completed in {elapsed_s}s "
                f"(steps={request.num_steps}, seed={seed}, policy={seed_policy}).",
            ],
        )

    # -- stage: decode_vae (passthrough) -------------------------------

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        # VAE decode already happened inside the pipeline (output_type="np")
        frames = self._np.asarray(state["video_frames"])
        return StageResult(
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

    # -- stage: postprocess --------------------------------------------

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        np = self._np
        frames = np.asarray(state["video_frames"])
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)
        return StageResult(
            state_updates={
                "video_frames": frames,
                "output_fps": request.fps or 16,
                "_pipeline_output": frames,
            },
            runtime_state_updates={
                "frame_count": int(frames.shape[0]),
                "output_fps": request.fps or 16,
            },
            outputs={"frame_count": int(frames.shape[0])},
            notes=["Postprocess normalized frames to uint8."],
        )

    # -- describe ------------------------------------------------------

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
