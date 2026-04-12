"""Native Cosmos Predict2.5 executor."""

from __future__ import annotations

import gc
import hashlib
import logging
import math
import os
import random
import sys
import time
from typing import Any

from vrl.models.families.cosmos.variants import CosmosLocalExecutor, CosmosVariant
from vrl.models.base import ModelResult, VideoGenerationRequest

logger = logging.getLogger(__name__)

_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no "
    "motion, motion blur, over-saturation, shaky footage, low resolution, grainy "
    "texture, pixelated images, poorly lit areas, underexposed and overexposed "
    "scenes, poor color balance, washed out colors, choppy sequences, jerky "
    "movements, low frame rate, artifacting, color banding, unnatural transitions, "
    "outdated special effects, fake elements, unconvincing visuals, poorly edited "
    "content, jump cuts, visual noise, and flickering. Overall, the video is of "
    "poor quality."
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]




def _resize_input(video: Any, resolution: list[int]) -> Any:
    """Resize-and-crop to target (H, W)."""
    import torchvision.transforms.functional as TF

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (
        int(math.ceil(scaling_ratio * orig_h)),
        int(math.ceil(scaling_ratio * orig_w)),
    )
    video = TF.resize(video, resizing_shape)
    return TF.center_crop(video, resolution)


def _read_and_process_image(
    img_path: str,
    resolution: list[int],
    num_video_frames: int,
    torch: Any,
) -> Any:
    """Image → video tensor (1, C, T, H, W)."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    img = Image.open(img_path)
    img_tensor = TF.to_tensor(img).unsqueeze(0)  # (1, C, H, W)
    # First frame is the image; rest are zeros.
    vid = torch.cat(
        [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
        dim=0,
    )
    vid = (vid * 255.0).to(torch.uint8)
    vid = _resize_input(vid, resolution)
    return vid.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)




class NativeCosmosPredict25Executor(CosmosLocalExecutor):
    """Native Predict2.5 executor."""

    execution_mode = "native_cosmos_predict25"

    def __init__(
        self,
        *,
        variant: CosmosVariant,
        model_size: str = "2B",
        model_id_or_path: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
        model_name: str | None = None,
    ) -> None:
        self.variant = variant
        self.model_size = model_size
        self.model_id_or_path = model_id_or_path
        self.device_id = device_id
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.model_name = model_name or f"{model_size}/post-trained"

        self._model: Any = None
        self._config: Any = None
        self._torch: Any = None
        self._np: Any = None
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}

    def _ensure_model(self) -> Any:
        """Lazy-load model from checkpoint."""
        if self._model is not None:
            return self._model

        import numpy as np
        import torch

        self._torch = torch
        self._np = np

        from cosmos_predict2._src.predict2.utils.model_loader import (
            load_model_from_checkpoint,
        )
        from cosmos_predict2.config import MODEL_CHECKPOINTS, MODEL_KEYS

        model_key = MODEL_KEYS[self.model_name]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        ckpt_path = self.model_id_or_path or checkpoint.s3.uri
        experiment_name = checkpoint.experiment

        if model_key.distilled:
            config_file = (
                "cosmos_predict2/_src/predict2/distill/configs/registry_predict2p5.py"
            )
        else:
            config_file = (
                "cosmos_predict2/_src/predict2/configs/video2world/config.py"
            )

        experiment_opts: list[str] = []
        if model_key.distilled:
            experiment_opts.append("model.config.init_student_with_teacher=False")

        if self.enable_cpu_offload:
            os.environ["COSMOS_PREDICT2_OFFLOAD_DIT"] = "1"

        model_device = None if self.enable_cpu_offload else "cuda"
        model, config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=experiment_opts,
            to_device=model_device,
        )

        if self.enable_cpu_offload:
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to("cpu")
        else:
            model.net.to("cuda")

        if self.enable_cpu_offload:
            if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
                model.tokenizer.encoder = model.tokenizer.encoder.to("cpu")
            if hasattr(model.tokenizer, "decoder") and model.tokenizer.decoder is not None:
                model.tokenizer.decoder = model.tokenizer.decoder.to("cpu")
            torch.cuda.empty_cache()

        self._model = model
        self._config = config
        logger.info(
            "Predict2.5 model loaded: %s (offload=%s)", self.model_name, self.enable_cpu_offload
        )
        return model

    def _compute_text_embeddings(self, prompt: str) -> Any:
        """Compute T5 embeddings."""
        model = self._model
        if model.text_encoder is not None:
            return model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [prompt], "images": None},
                input_caption_key="ai_caption",
            )
        from cosmos_predict2._src.predict2.inference.get_t5_emb import (
            get_text_embedding,
        )
        return get_text_embedding(prompt)

    def _offload_text_encoder(self) -> None:
        if not self.enable_cpu_offload:
            return
        model = self._model
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to("cpu")
            self._torch.cuda.empty_cache()

    def _load_diffusion_to_gpu(self) -> None:
        if not self.enable_cpu_offload:
            return
        model = self._model
        model.net = model.net.to("cuda")
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to("cuda")
        self._torch.cuda.empty_cache()

    def _offload_diffusion_to_cpu(self) -> None:
        if not self.enable_cpu_offload:
            return
        model = self._model
        model.net = model.net.to("cpu")
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to("cpu")

    def _load_tokenizer_encoder_to_gpu(self) -> None:
        if not self.enable_cpu_offload:
            return
        tok = self._model.tokenizer
        if hasattr(tok, "encoder") and tok.encoder is not None:
            tok.encoder = tok.encoder.to("cuda")
        self._torch.cuda.empty_cache()

    def _offload_tokenizer_encoder(self) -> None:
        if not self.enable_cpu_offload:
            return
        tok = self._model.tokenizer
        if hasattr(tok, "encoder") and tok.encoder is not None:
            tok.encoder = tok.encoder.to("cpu")
        self._torch.cuda.empty_cache()

    def _load_tokenizer_decoder_to_gpu(self) -> None:
        if not self.enable_cpu_offload:
            return
        tok = self._model.tokenizer
        if hasattr(tok, "decoder") and tok.decoder is not None:
            tok.decoder = tok.decoder.to("cuda")
        self._torch.cuda.empty_cache()

    def _offload_tokenizer_decoder(self) -> None:
        if not self.enable_cpu_offload:
            return
        tok = self._model.tokenizer
        if hasattr(tok, "decoder") and tok.decoder is not None:
            tok.decoder = tok.decoder.to("cpu")
        self._torch.cuda.empty_cache()

    def _build_data_batch(
        self,
        video: Any,
        t5_embeds: Any,
        neg_t5_embeds: Any,
        num_conditional_frames: int,
    ) -> dict[str, Any]:
        torch = self._torch
        _, _, _, H, W = video.shape

        data_batch: dict[str, Any] = {
            "dataset_name": "video_data",
            "video": video,
            "action": None,
            "fps": torch.randint(16, 32, (1,)).float(),
            "padding_mask": torch.zeros(1, 1, H, W),
            "num_conditional_frames": num_conditional_frames,
            "t5_text_embeddings": t5_embeds,
            "neg_t5_text_embeddings": neg_t5_embeds,
        }
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        return data_batch

    async def load(self) -> None:
        self._ensure_model()

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        model = self._ensure_model()
        torch = self._torch

        prompt = request.prompt or ""
        negative_prompt = request.negative_prompt or _DEFAULT_NEGATIVE_PROMPT
        cache_key = _stable_hash(
            f"predict25|{self.model_size}|{prompt}|{negative_prompt}"
        )

        cache_hit = cache_key in self._prompt_cache
        if cache_hit:
            t5_cpu, neg_t5_cpu = self._prompt_cache[cache_key]
            t5_embeds = t5_cpu.to("cuda")
            neg_t5_embeds = neg_t5_cpu.to("cuda")
        else:
            t5_embeds = self._compute_text_embeddings(prompt)
            neg_t5_embeds = self._compute_text_embeddings(negative_prompt)
            self._prompt_cache[cache_key] = (
                t5_embeds.detach().cpu(),
                neg_t5_embeds.detach().cpu(),
            )

        self._offload_text_encoder()

        return ModelResult(
            state_updates={
                "t5_embeds": t5_embeds,
                "neg_t5_embeds": neg_t5_embeds,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
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
                f"Predict2.5 T5 encoding {'(cache hit)' if cache_hit else 'completed'}.",
            ],
            cache_hit=cache_hit,
        )

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        model = self._ensure_model()
        torch = self._torch

        model_required_frames = model.tokenizer.get_pixel_num_frames(
            model.config.state_t
        )

        if not request.references:
            h, w = request.height, request.width
            vid_input = torch.zeros(1, 3, model_required_frames, h, w, dtype=torch.uint8)
            num_cond_frames = 0
            return ModelResult(
                state_updates={
                    "vid_input": vid_input,
                    "num_conditional_frames": num_cond_frames,
                    "model_required_frames": model_required_frames,
                },
                outputs={"reference_count": 0},
                notes=["No reference — text-to-world zero tensor prepared."],
            )

        reference_path = request.references[0]
        resolution = [request.height, request.width]
        ext = os.path.splitext(reference_path)[1].lower()

        if ext in _IMAGE_EXTENSIONS:
            vid_input = _read_and_process_image(
                reference_path, resolution, model_required_frames, torch
            )
            num_cond_frames = 1
        else:
            raise ValueError(
                f"Unsupported reference extension: {ext}. "
                f"Supported: {sorted(_IMAGE_EXTENSIONS)}"
            )

        return ModelResult(
            state_updates={
                "vid_input": vid_input,
                "num_conditional_frames": num_cond_frames,
                "model_required_frames": model_required_frames,
            },
            runtime_state_updates={"reference_path": reference_path},
            outputs={"reference_count": len(request.references)},
            notes=[f"Reference image loaded and preprocessed ({ext})."],
        )

    async def generate(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        model = self._ensure_model()
        torch = self._torch

        if request.seed is not None:
            seed = request.seed
            seed_policy = "explicit"
        else:
            seed = random.randint(0, sys.maxsize)
            seed_policy = "randomized"

        vid_input = state["vid_input"]
        num_cond_frames = state["num_conditional_frames"]
        t5_embeds = state["t5_embeds"]
        neg_t5_embeds = state["neg_t5_embeds"]

        data_batch = self._build_data_batch(
            video=vid_input,
            t5_embeds=t5_embeds,
            neg_t5_embeds=neg_t5_embeds,
            num_conditional_frames=num_cond_frames,
        )

        self._load_tokenizer_encoder_to_gpu()
        self._load_diffusion_to_gpu()

        started = time.perf_counter()
        generate_fn = (
            model.generate_samples_from_batch_lora
            if getattr(model.config, "use_lora", False)
            else model.generate_samples_from_batch
        )
        latent_samples = generate_fn(
            data_batch,
            n_sample=1,
            guidance=int(request.guidance_scale),
            seed=seed,
            is_negative_prompt=True,
            num_steps=request.num_steps,
        )
        elapsed_s = round(time.perf_counter() - started, 4)

        self._offload_diffusion_to_cpu()
        self._offload_tokenizer_encoder()

        latent_shape = list(latent_samples.shape) if hasattr(latent_samples, "shape") else None

        return ModelResult(
            state_updates={
                "latent_samples": latent_samples,
                "seed": seed,
            },
            runtime_state_updates={
                "latent_shape": latent_shape,
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
                f"Predict2.5 diffusion completed in {elapsed_s}s "
                f"(steps={request.num_steps}, seed={seed}, policy={seed_policy}).",
            ],
        )

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        model = self._ensure_model()
        torch = self._torch

        latent_samples = state["latent_samples"]

        self._load_tokenizer_decoder_to_gpu()
        if isinstance(latent_samples, list):
            video_chunks = [model.decode(chunk) for chunk in latent_samples]
            video = torch.cat(video_chunks, dim=3)  # concat along T
        else:
            video = model.decode(latent_samples)

        self._offload_tokenizer_decoder()
        video = (1.0 + video[0]) / 2.0

        gc.collect()
        torch.cuda.empty_cache()

        return ModelResult(
            state_updates={"video_tensor": video},
            runtime_state_updates={
                "decoded_frame_count": int(video.shape[1]),
                "decoded_spatial_size": [int(video.shape[3]), int(video.shape[2])],
            },
            outputs={"fps": request.fps or 16},
            notes=["VAE decode completed: latent → pixel."],
        )

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        torch = self._torch

        video = state["video_tensor"]  # (C, T, H, W) in [0, 1]
        frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        frames = frames.permute(1, 2, 3, 0).cpu().numpy()

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
            notes=["Postprocess: float → uint8 numpy."],
        )

    def describe(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "executor": self.__class__.__name__,
            "variant": self.variant.value,
            "model_size": self.model_size,
            "model_name": self.model_name,
            "device_id": self.device_id,
            "dtype": self.dtype,
            "enable_cpu_offload": self.enable_cpu_offload,
            "model_loaded": self._model is not None,
            "prompt_cache_size": len(self._prompt_cache),
        }
