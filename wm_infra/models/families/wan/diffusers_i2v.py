"""Diffusers-based Wan 2.2 Image-to-Video generation model.

Owns the model forward logic: pipeline loading, text encoding,
conditioning, diffusion, VAE decode, and postprocess.
"""

from __future__ import annotations

import gc
import random
import sys
from pathlib import Path
from typing import Any

from wm_infra.models.base import VideoGenerationModel
from wm_infra.models.families.wan.shared import resolve_wan_reference_path, stable_hash
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class DiffusersWanI2VModel(VideoGenerationModel):
    """Diffusers-based Wan I2V model."""

    model_family = "wan-diffusers-i2v"

    def __init__(
        self,
        *,
        default_model_dir: str | Path,
        device_id: int = 0,
        default_dtype: str = "bfloat16",
    ) -> None:
        self.default_model_dir = Path(default_model_dir)
        self.device_id = device_id
        self.default_dtype = default_dtype
        self._modules_loaded = False
        self._torch = None
        self._np = None
        self._pil_image = None
        self._pipeline_cls = None
        self._pipelines: dict[str, Any] = {}
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}
        self._conditioning_cache: dict[str, Any] = {}

    async def load(self) -> None:
        self._load_modules()

    def describe(self) -> dict[str, Any]:
        return {
            "name": "wan-diffusers-i2v-model",
            "family": self.model_family,
            "model_dir": str(self.default_model_dir),
            "device": f"cuda:{self.device_id}",
            "task_types": ["image_to_video"],
        }

    # -- module loading ------------------------------------------------

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        import numpy as np
        import torch
        from diffusers import WanImageToVideoPipeline
        from PIL import Image

        self._np = np
        self._pil_image = Image
        self._pipeline_cls = WanImageToVideoPipeline
        self._torch = torch
        self._modules_loaded = True

    def _device(self) -> Any:
        return self._torch.device(f"cuda:{self.device_id}")

    # -- pipeline lookup -----------------------------------------------

    def _resolve_model_dir(self, request: VideoGenerationRequest) -> Path:
        if request.task_type != "image_to_video":
            raise NotImplementedError("Diffusers Wan model only supports image_to_video requests")
        if request.ckpt_dir:
            ckpt_path = Path(request.ckpt_dir)
            if (ckpt_path / "model_index.json").exists():
                return ckpt_path
        if not self.default_model_dir.exists():
            raise FileNotFoundError(
                f"Wan diffusers model_dir does not exist: {self.default_model_dir}"
            )
        return self.default_model_dir

    def _create_pipeline(self, model_dir: Path) -> Any:
        self._load_modules()
        torch = self._torch
        dtype = getattr(torch, self.default_dtype)
        pipeline = self._pipeline_cls.from_pretrained(str(model_dir), torch_dtype=dtype)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_sequential_cpu_offload()
        pipeline.vae.enable_tiling()
        pipeline.vae.enable_slicing()
        return pipeline

    def _get_pipeline(self, model_dir: Path) -> Any:
        cache_key = str(model_dir)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is None:
            pipeline = self._create_pipeline(model_dir)
            self._pipelines[cache_key] = pipeline
        return pipeline

    # -- stages --------------------------------------------------------

    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        self._load_modules()
        model_dir = self._resolve_model_dir(request)
        pipeline = self._get_pipeline(model_dir)
        prompt_hash = stable_hash(
            f"{request.model_name}|{request.prompt}|{request.negative_prompt}"
        )
        cache_key = (
            f"{model_dir}|{prompt_hash}|cfg={int(request.guidance_scale > 1.0)}|"
            f"max_seq={request.extra.get('max_sequence_length', 512)}"
        )

        return StageResult(
            state_updates={
                "pipeline": pipeline,
                "model_dir": str(model_dir),
            },
            runtime_state_updates={
                "task_key": "i2v-A14B-diffusers",
                "checkpoint_dir": str(model_dir),
                "prompt_cache_key": cache_key,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(request.prompt.split())),
                "negative_prompt_used": bool(request.negative_prompt),
            },
            notes=[
                f"Prompt encode was scheduled against diffusers Wan I2V from {model_dir}.",
                "The actual T5 forward stays inside the verified diffusers pipeline offload path.",
            ],
        )

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        self._load_modules()
        reference_path = resolve_wan_reference_path(request.references[0])
        cache_key = (
            f"{state['model_dir']}|{reference_path}|"
            f"{request.width}x{request.height}|frames={request.frame_count}"
        )
        cache_hit = cache_key in self._conditioning_cache
        if cache_hit:
            reference_image = self._conditioning_cache[cache_key].copy()
        else:
            reference_image = self._pil_image.open(reference_path).convert("RGB")
            self._conditioning_cache[cache_key] = reference_image.copy()

        return StageResult(
            state_updates={
                "reference_image": reference_image,
            },
            runtime_state_updates={
                "reference_path": reference_path,
                "conditioning_size": [request.width, request.height],
            },
            outputs={"reference_count": len(request.references)},
            notes=["Reference image was staged for diffusers Wan I2V generation reuse."],
            cache_hit=cache_hit,
        )

    async def denoise(self, request: VideoGenerationRequest, state: dict[str, Any]) -> StageResult:
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
        guidance_scale_2 = (
            request.guidance_scale
            if request.high_noise_guidance_scale is None
            else request.high_noise_guidance_scale
        )
        pipeline_output = pipeline(
            image=state["reference_image"],
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            guidance_scale_2=guidance_scale_2,
            generator=generator,
            output_type="np",
            return_dict=True,
            max_sequence_length=request.extra.get("max_sequence_length", 512),
        )
        frames = pipeline_output.frames[0]
        pipeline.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()

        return StageResult(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "frame_shape": list(frames.shape),
                "seed": seed,
                "seed_policy": seed_policy,
                "guide_scale_pair": [request.guidance_scale, guidance_scale_2],
            },
            outputs={
                "solver": pipeline.scheduler.__class__.__name__,
                "num_steps": request.num_steps,
                "guidance_scale": [request.guidance_scale, guidance_scale_2],
            },
            notes=[
                "Diffusers Wan I2V completed denoise and decode through the pipeline-managed offload path.",
            ],
        )

    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        frames = self._np.asarray(state["video_frames"])
        return StageResult(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "decoded_frame_count": int(frames.shape[0]),
                "decoded_spatial_size": [int(frames.shape[2]), int(frames.shape[1])],
            },
            outputs={"fps": request.fps or 16},
            notes=[
                "VAE decode stage reused diffusers pipeline output from the verified offload path."
            ],
        )

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        frames = self._np.asarray(state["video_frames"])
        if frames.dtype != self._np.uint8:
            frames = self._np.clip(frames * 255.0, 0.0, 255.0).astype(self._np.uint8)
        return StageResult(
            state_updates={
                "video_frames": frames,
                "output_fps": request.fps or 16,
            },
            runtime_state_updates={
                "frame_count": len(frames),
                "output_fps": request.fps or 16,
            },
            outputs={"frame_count": len(frames)},
            notes=["Postprocess converted decoded tensors into numpy video frames."],
        )
