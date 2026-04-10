"""Official Wan 2.2 video generation model.

Owns the model forward logic: pipeline loading, text encoding,
conditioning, diffusion, VAE decode, and postprocess.
"""

from __future__ import annotations

import gc
import importlib
import math
import random
import sys
from pathlib import Path
from typing import Any

from wm_infra.models.base import VideoGenerationModel
from wm_infra.models.families.wan.shared import _stable_hash, resolve_wan_reference_path
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


def _clone_tensor_list_to_cpu(tensors: list[Any]) -> list[Any]:
    return [tensor.detach().cpu() for tensor in tensors]


def _move_tensor_list_to_device(tensors: list[Any], device: Any) -> list[Any]:
    return [tensor.to(device) for tensor in tensors]


def _clone_tensor_to_cpu(tensor: Any) -> Any:
    return None if tensor is None else tensor.detach().cpu()


def _move_tensor_to_device(tensor: Any, device: Any, dtype: Any | None = None) -> Any:
    if tensor is None:
        return None
    kwargs = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


class OfficialWanModel(VideoGenerationModel):
    """Official Wan2.2 in-process model."""

    model_family = "wan-official"

    def __init__(
        self,
        *,
        repo_dir: str | Path,
        default_checkpoint_dir: str | Path | None = None,
        device_id: int = 0,
        default_t5_cpu: bool = True,
        default_convert_model_dtype: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.default_checkpoint_dir = (
            None if default_checkpoint_dir is None else Path(default_checkpoint_dir)
        )
        self.device_id = device_id
        self.default_t5_cpu = default_t5_cpu
        self.default_convert_model_dtype = default_convert_model_dtype
        self._modules_loaded = False
        self._torch = None
        self._wan = None
        self._wan_configs = None
        self._size_configs = None
        self._max_area_configs = None
        self._pil_image = None
        self._tv_tf = None
        self._np = None
        self._pipelines: dict[str, Any] = {}
        self._prompt_cache: dict[str, tuple[list[Any], list[Any]]] = {}
        self._conditioning_cache: dict[str, dict[str, Any]] = {}

    async def load(self) -> None:
        self._load_modules()

    def describe(self) -> dict[str, Any]:
        return {
            "name": "wan-official-model",
            "family": self.model_family,
            "repo_dir": str(self.repo_dir),
            "default_checkpoint_dir": None
            if self.default_checkpoint_dir is None
            else str(self.default_checkpoint_dir),
            "device": f"cuda:{self.device_id}",
        }

    # -- module loading ------------------------------------------------

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Wan repo_dir does not exist: {self.repo_dir}")
        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        import wan
        import wan.modules.attention as wan_attention
        import wan.modules.model as wan_model
        from PIL import Image
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

        if not wan_attention.FLASH_ATTN_2_AVAILABLE and not wan_attention.FLASH_ATTN_3_AVAILABLE:
            wan_model.flash_attention = wan_attention.attention

        self._np = np
        self._torch = torch
        self._tv_tf = TF
        self._wan = wan
        self._wan_configs = WAN_CONFIGS
        self._size_configs = SIZE_CONFIGS
        self._max_area_configs = MAX_AREA_CONFIGS
        self._pil_image = Image
        self._modules_loaded = True

    # -- pipeline lookup -----------------------------------------------

    def _task_key(self, task_type: str, model_size: str) -> str:
        if task_type == "text_to_video":
            return f"t2v-{model_size}"
        if task_type == "image_to_video":
            return f"i2v-{model_size}"
        raise NotImplementedError(f"Official Wan model does not support {task_type}")

    def _size_key(self, width: int, height: int) -> str:
        size_key = f"{width}*{height}"
        if size_key not in self._size_configs:
            raise ValueError(
                f"Wan official model does not support size {size_key}; "
                f"supported sizes are {sorted(self._size_configs.keys())}"
            )
        return size_key

    def _resolve_checkpoint_dir(self, ckpt_dir: str | None, task_key: str) -> Path:
        if ckpt_dir:
            return Path(ckpt_dir)
        if self.default_checkpoint_dir is not None:
            return self.default_checkpoint_dir
        fallback_map = {
            "t2v-A14B": self.repo_dir / "Wan2.2-T2V-A14B",
            "i2v-A14B": self.repo_dir / "Wan2.2-I2V-A14B",
        }
        ckpt_path = fallback_map.get(task_key)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No Wan checkpoint directory configured for {task_key}. "
                "Set ckpt_dir or the model default checkpoint path."
            )
        return ckpt_path

    def _validate_checkpoint_layout(
        self, task_key: str, checkpoint_dir: Path, config: Any
    ) -> None:
        required_paths: list[tuple[Path, str]] = [
            (checkpoint_dir / config.t5_checkpoint, "T5 checkpoint"),
            (checkpoint_dir / config.t5_tokenizer, "T5 tokenizer"),
            (checkpoint_dir / config.vae_checkpoint, "VAE checkpoint"),
        ]
        if task_key.startswith("i2v-"):
            required_paths.extend(
                [
                    (
                        checkpoint_dir / config.low_noise_checkpoint,
                        "low-noise checkpoint directory",
                    ),
                    (
                        checkpoint_dir / config.high_noise_checkpoint,
                        "high-noise checkpoint directory",
                    ),
                ]
            )
        for path, label in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Incomplete Wan checkpoint layout for {task_key}: missing {label} at {path}"
                )
        for subfolder_attr, label in (
            ("low_noise_checkpoint", "low-noise"),
            ("high_noise_checkpoint", "high-noise"),
        ):
            subfolder = getattr(config, subfolder_attr, None)
            if subfolder is None:
                continue
            shard_dir = checkpoint_dir / subfolder
            has_weight_file = any(shard_dir.glob("*.safetensors")) or any(shard_dir.glob("*.bin"))
            has_index = any(shard_dir.glob("*.index.json"))
            if not has_weight_file and not has_index:
                raise FileNotFoundError(
                    f"Incomplete Wan checkpoint layout for {task_key}: {label} model weights are missing under {shard_dir}"
                )

    def _pipeline_cache_key(
        self,
        task_key: str,
        checkpoint_dir: Path,
        t5_cpu: bool,
        convert_model_dtype: bool,
    ) -> str:
        return "|".join(
            [
                task_key,
                str(checkpoint_dir),
                str(self.device_id),
                str(t5_cpu),
                str(convert_model_dtype),
            ]
        )

    def _get_pipeline(self, request: VideoGenerationRequest) -> tuple[Any, str, Path]:
        self._load_modules()
        task_key = self._task_key(request.task_type, request.model_size)
        checkpoint_dir = self._resolve_checkpoint_dir(request.ckpt_dir, task_key)
        t5_cpu = request.t5_cpu if request.t5_cpu is not None else self.default_t5_cpu
        convert_dtype = (
            request.convert_model_dtype
            if request.convert_model_dtype is not None
            else self.default_convert_model_dtype
        )
        cache_key = self._pipeline_cache_key(task_key, checkpoint_dir, t5_cpu, convert_dtype)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is not None:
            return pipeline, task_key, checkpoint_dir

        config = self._wan_configs[task_key]
        self._validate_checkpoint_layout(task_key, checkpoint_dir, config)
        common_kwargs = {
            "config": config,
            "checkpoint_dir": str(checkpoint_dir),
            "device_id": self.device_id,
            "rank": 0,
            "t5_fsdp": False,
            "dit_fsdp": False,
            "use_sp": False,
            "t5_cpu": t5_cpu,
            "convert_model_dtype": convert_dtype,
        }
        if task_key.startswith("t2v-"):
            pipeline = self._wan.WanT2V(**common_kwargs)
        elif task_key.startswith("i2v-"):
            pipeline = self._wan.WanI2V(**common_kwargs)
        else:
            raise NotImplementedError(f"Official Wan model does not support task key {task_key}")
        self._pipelines[cache_key] = pipeline
        return pipeline, task_key, checkpoint_dir

    # -- stages --------------------------------------------------------

    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        pipeline, task_key, checkpoint_dir = self._get_pipeline(request)
        torch = self._torch
        n_prompt = request.negative_prompt or pipeline.sample_neg_prompt
        prompt_hash = _stable_hash(
            f"{request.model_name}|{request.prompt}|{request.negative_prompt}"
        )
        cache_key = f"{task_key}|{checkpoint_dir}|{prompt_hash}"
        cache_hit = cache_key in self._prompt_cache

        if cache_hit:
            cached_context, cached_context_null = self._prompt_cache[cache_key]
        else:
            if not pipeline.t5_cpu:
                pipeline.text_encoder.model.to(pipeline.device)
                prompt_context = pipeline.text_encoder([request.prompt], pipeline.device)
                prompt_context_null = pipeline.text_encoder([n_prompt], pipeline.device)
                if request.offload_model:
                    pipeline.text_encoder.model.cpu()
            else:
                prompt_context = pipeline.text_encoder([request.prompt], torch.device("cpu"))
                prompt_context_null = pipeline.text_encoder([n_prompt], torch.device("cpu"))
            cached_context = _clone_tensor_list_to_cpu(prompt_context)
            cached_context_null = _clone_tensor_list_to_cpu(prompt_context_null)
            self._prompt_cache[cache_key] = (cached_context, cached_context_null)

        device_context = _move_tensor_list_to_device(cached_context, pipeline.device)
        device_context_null = _move_tensor_list_to_device(cached_context_null, pipeline.device)
        return StageResult(
            state_updates={
                "pipeline": pipeline,
                "task_key": task_key,
                "checkpoint_dir": str(checkpoint_dir),
                "text_context": device_context,
                "text_context_null": device_context_null,
            },
            runtime_state_updates={
                "task_key": task_key,
                "checkpoint_dir": str(checkpoint_dir),
                "prompt_cache_key": cache_key,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(request.prompt.split())),
                "negative_prompt_used": bool(n_prompt),
            },
            notes=[f"Prompt encoding executed against {task_key} from {checkpoint_dir}."],
            cache_hit=cache_hit,
        )

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        pipeline = state["pipeline"]
        if request.task_type != "image_to_video":
            return StageResult(
                notes=["Conditioning stage skipped because the request is text-to-video."]
            )
        reference_path = resolve_wan_reference_path(request.references[0])
        conditioning_hash = _stable_hash(
            f"{request.task_type}|"
            f"{'|'.join(request.references)}|"
            f"{request.width}x{request.height}|"
            f"frames={request.frame_count}"
        )
        cache_key = f"{state['task_key']}|{reference_path}|{conditioning_hash}"
        cache_hit = cache_key in self._conditioning_cache

        if cache_hit:
            cached_conditioning = self._conditioning_cache[cache_key]
            conditioning = {
                "conditioning_tensor": _move_tensor_to_device(
                    cached_conditioning["conditioning_tensor"], pipeline.device
                ),
                "conditioning_shape": list(cached_conditioning["conditioning_shape"]),
                "decoded_size": list(cached_conditioning["decoded_size"]),
                "reference_path": cached_conditioning["reference_path"],
            }
        else:
            image = self._pil_image.open(reference_path).convert("RGB")
            img_tensor = self._tv_tf.to_tensor(image).sub_(0.5).div_(0.5).to(pipeline.device)
            frame_num = request.frame_count
            h, w = img_tensor.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                math.sqrt(request.width * request.height * aspect_ratio)
                // pipeline.vae_stride[1]
                // pipeline.patch_size[1]
                * pipeline.patch_size[1]
            )
            lat_w = round(
                math.sqrt(request.width * request.height / aspect_ratio)
                // pipeline.vae_stride[2]
                // pipeline.patch_size[2]
                * pipeline.patch_size[2]
            )
            height = lat_h * pipeline.vae_stride[1]
            width = lat_w * pipeline.vae_stride[2]
            torch = self._torch
            mask = torch.ones(1, frame_num, lat_h, lat_w, device=pipeline.device)
            mask[:, 1:] = 0
            mask = torch.concat(
                [torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]],
                dim=1,
            )
            mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
            mask = mask.transpose(1, 2)[0]
            y = pipeline.vae.encode(
                [
                    torch.concat(
                        [
                            torch.nn.functional.interpolate(
                                img_tensor[None].cpu(),
                                size=(height, width),
                                mode="bicubic",
                            ).transpose(0, 1),
                            torch.zeros(3, frame_num - 1, height, width),
                        ],
                        dim=1,
                    ).to(pipeline.device)
                ]
            )[0]
            conditioning = {
                "conditioning_tensor": torch.concat([mask, y]),
                "conditioning_shape": [int(lat_h), int(lat_w)],
                "decoded_size": [int(width), int(height)],
                "reference_path": reference_path,
            }
            self._conditioning_cache[cache_key] = {
                "conditioning_tensor": _clone_tensor_to_cpu(conditioning["conditioning_tensor"]),
                "conditioning_shape": list(conditioning["conditioning_shape"]),
                "decoded_size": list(conditioning["decoded_size"]),
                "reference_path": reference_path,
            }

        return StageResult(
            state_updates={"conditioning": conditioning},
            runtime_state_updates={
                "conditioning_shape": conditioning["conditioning_shape"],
                "conditioning_decoded_size": conditioning["decoded_size"],
                "reference_path": conditioning["reference_path"],
            },
            outputs={"reference_count": len(request.references)},
            notes=["Image conditioning tensor and mask were encoded for Wan I2V."],
            cache_hit=cache_hit,
        )

    async def denoise(self, request: VideoGenerationRequest, state: dict[str, Any]) -> StageResult:
        pipeline = state["pipeline"]
        torch = self._torch
        size_key = self._size_key(request.width, request.height)
        if request.seed is not None:
            seed = request.seed
            seed_policy = "explicit"
        else:
            seed = random.randint(0, sys.maxsize)
            seed_policy = "randomized"
        seed_g = torch.Generator(device=pipeline.device)
        seed_g.manual_seed(seed)
        low_noise_guidance_scale = float(request.guidance_scale)
        high_noise_guidance_scale = (
            low_noise_guidance_scale
            if request.high_noise_guidance_scale is None
            else float(request.high_noise_guidance_scale)
        )
        sample_solver = request.sample_solver

        with (
            torch.amp.autocast("cuda", dtype=pipeline.param_dtype),
            torch.no_grad(),
        ):
            if state["task_key"].startswith("t2v-"):
                width, height = self._size_configs[size_key]
                target_shape = (
                    pipeline.vae.model.z_dim,
                    (request.frame_count - 1) // pipeline.vae_stride[0] + 1,
                    height // pipeline.vae_stride[1],
                    width // pipeline.vae_stride[2],
                )
                seq_len = (
                    math.ceil(
                        (target_shape[2] * target_shape[3])
                        / (pipeline.patch_size[1] * pipeline.patch_size[2])
                        * target_shape[1]
                        / pipeline.sp_size
                    )
                    * pipeline.sp_size
                )
                latents = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=pipeline.device,
                        generator=seed_g,
                    )
                ]
                arg_c = {"context": state["text_context"], "seq_len": seq_len}
                arg_null = {"context": state["text_context_null"], "seq_len": seq_len}
            else:
                max_area = self._max_area_configs[size_key]
                frame_num = request.frame_count
                lat_h, lat_w = state["conditioning"]["conditioning_shape"]
                max_seq_len = (
                    ((frame_num - 1) // pipeline.vae_stride[0] + 1)
                    * lat_h
                    * lat_w
                    // (pipeline.patch_size[1] * pipeline.patch_size[2])
                )
                max_seq_len = math.ceil(max_seq_len / pipeline.sp_size) * pipeline.sp_size
                latents = torch.randn(
                    16,
                    (frame_num - 1) // pipeline.vae_stride[0] + 1,
                    lat_h,
                    lat_w,
                    dtype=torch.float32,
                    generator=seed_g,
                    device=pipeline.device,
                )
                arg_c = {
                    "context": [state["text_context"][0]],
                    "seq_len": max_seq_len,
                    "y": [state["conditioning"]["conditioning_tensor"]],
                }
                arg_null = {
                    "context": state["text_context_null"],
                    "seq_len": max_seq_len,
                    "y": [state["conditioning"]["conditioning_tensor"]],
                }

            boundary = pipeline.boundary * pipeline.num_train_timesteps
            solver_name = sample_solver
            if solver_name == "unipc":
                scheduler = importlib.import_module(
                    "wan.utils.fm_solvers_unipc"
                ).FlowUniPCMultistepScheduler(
                    num_train_timesteps=pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                scheduler.set_timesteps(
                    request.num_steps, device=pipeline.device, shift=request.shift
                )
                timesteps = scheduler.timesteps
            else:
                fm_solvers = importlib.import_module("wan.utils.fm_solvers")
                scheduler = fm_solvers.FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = fm_solvers.get_sampling_sigmas(request.num_steps, request.shift)
                timesteps, _ = fm_solvers.retrieve_timesteps(
                    scheduler,
                    device=pipeline.device,
                    sigmas=sampling_sigmas,
                )

            for t in timesteps:
                timestep = torch.stack([t]).to(pipeline.device)
                model = pipeline._prepare_model_for_timestep(t, boundary, request.offload_model)
                sample_guide_scale = (
                    high_noise_guidance_scale if t.item() >= boundary else low_noise_guidance_scale
                )

                if state["task_key"].startswith("t2v-"):
                    noise_pred_cond = model(latents, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model(latents, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents = [temp_x0.squeeze(0)]
                else:
                    latent_model_input = [latents.to(pipeline.device)]
                    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                    if request.offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                    if request.offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents = temp_x0.squeeze(0)

            final_latents = latents if isinstance(latents, list) else [latents]
            if request.offload_model:
                pipeline.low_noise_model.cpu()
                pipeline.high_noise_model.cpu()
                torch.cuda.empty_cache()

        latent_shape = list(final_latents[0].shape)
        return StageResult(
            state_updates={"latents": final_latents, "fps": pipeline.config.sample_fps},
            runtime_state_updates={
                "latent_shape": latent_shape,
                "seed": seed,
                "seed_policy": seed_policy,
                "sample_solver": solver_name,
                "guide_scale_pair": [low_noise_guidance_scale, high_noise_guidance_scale],
                "boundary_timestep": boundary,
                "max_area": None if state["task_key"].startswith("t2v-") else max_area,
            },
            outputs={
                "solver": solver_name,
                "num_steps": request.num_steps,
                "guidance_scale": [low_noise_guidance_scale, high_noise_guidance_scale],
            },
            notes=[
                f"Diffusion completed on {len(timesteps)} timesteps for {state['task_key']}.",
            ],
        )

    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        pipeline = state["pipeline"]
        video = pipeline.vae.decode(state["latents"])[0]
        if request.offload_model:
            gc.collect()
            self._torch.cuda.synchronize()
        return StageResult(
            state_updates={"video_tensor": video},
            runtime_state_updates={
                "decoded_frame_count": int(video.shape[1]),
                "decoded_spatial_size": [int(video.shape[3]), int(video.shape[2])],
            },
            outputs={"fps": state["fps"]},
            notes=["VAE decode completed and produced frame tensors."],
        )

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        video = state["video_tensor"]
        return StageResult(
            state_updates={
                "output_fps": int(state["fps"]),
                "_pipeline_output": video,
            },
            runtime_state_updates={"output_fps": int(state["fps"])},
            outputs={
                "frame_count": int(video.shape[1]),
                "channels": int(video.shape[0]),
            },
            notes=["Postprocess kept the official Wan tensor layout for persistence."],
        )
