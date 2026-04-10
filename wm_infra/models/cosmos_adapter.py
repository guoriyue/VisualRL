"""Cosmos video generation model adapter."""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from typing import Any

from wm_infra.models.video_generation import (
    StageResult,
    VideoGenerationModel,
    VideoGenerationRequest,
)


class CosmosLocalExecutor(ABC):
    """Executor interface for local/in-process Cosmos generation."""

    execution_mode: str = "in_process"

    async def load(self) -> None:  # noqa: B027
        """Resolve local resources needed for inference (optional override)."""

    @abstractmethod
    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Produce prompt-side conditioning."""

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Produce reference-side conditioning."""
        return StageResult(outputs={"reference_count": len(request.references)})

    @abstractmethod
    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Run the local generation loop and return latent/video state."""

    @abstractmethod
    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Decode latent state into video frames."""

    @abstractmethod
    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Standardise decoded frames for backend persistence."""

    def describe(self) -> dict[str, Any]:
        return {"execution_mode": self.execution_mode, "executor": self.__class__.__name__}


class StubCosmosLocalExecutor(CosmosLocalExecutor):
    """Deterministic in-memory stub for Cosmos local execution."""

    execution_mode = "stub"

    def _effective_seed(self, request: VideoGenerationRequest) -> int:
        if request.seed is not None:
            return int(request.seed)
        digest = hashlib.sha256(
            "|".join(
                [
                    request.prompt,
                    request.negative_prompt,
                    request.task_type,
                    request.model_name,
                    str(request.width),
                    str(request.height),
                    str(request.frame_count),
                ]
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _build_stub_latents(self, request: VideoGenerationRequest) -> Any:
        import numpy as np

        frame_count = max(1, int(request.frame_count or 16))
        height = max(16, int(request.height or 640))
        width = max(16, int(request.width or 1024))
        seed = self._effective_seed(request)
        base_phase = (seed % 360) * math.pi / 180.0

        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        latents = np.empty((frame_count, height, width, 3), dtype=np.float32)
        for frame_idx in range(frame_count):
            phase = base_phase + frame_idx * 0.23
            latents[frame_idx, :, :, 0] = np.mod(x + phase, 1.0)
            latents[frame_idx, :, :, 1] = np.mod(y + phase * 0.5, 1.0)
            latents[frame_idx, :, :, 2] = np.mod((x * 0.5 + y * 0.5) + phase * 0.25, 1.0)
        return latents

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        prompt_tokens_estimate = max(1, len((request.prompt or "").split()))
        return StageResult(
            state_updates={
                "prompt_text": request.prompt,
                "negative_prompt_text": request.negative_prompt,
            },
            outputs={"prompt_tokens_estimate": prompt_tokens_estimate},
            notes=["Cosmos stub prepared prompt-side conditioning metadata."],
        )

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return StageResult(
            state_updates={
                "has_reference": bool(request.references),
                "reference_count": len(request.references),
            },
            outputs={"reference_count": len(request.references)},
            notes=["Cosmos stub prepared reference-side conditioning metadata."],
        )

    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        started = time.perf_counter()
        seed = self._effective_seed(request)
        latents = self._build_stub_latents(request)
        elapsed_s = round(time.perf_counter() - started, 6)
        return StageResult(
            state_updates={
                "latent_frames": latents,
                "seed": seed,
                "runner_mode": self.execution_mode,
                "elapsed_s": elapsed_s,
            },
            runtime_state_updates={
                "latent_shape": list(latents.shape),
                "seed": seed,
            },
            outputs={
                "runner_mode": self.execution_mode,
                "latent_shape": list(latents.shape),
                "elapsed_s": elapsed_s,
            },
            notes=["Cosmos stub generated deterministic latent frames in memory."],
        )

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        import numpy as np

        latent_frames = np.asarray(state["latent_frames"], dtype=np.float32)
        decoded_frames = np.clip(latent_frames, 0.0, 1.0)
        return StageResult(
            state_updates={"video_frames": decoded_frames},
            runtime_state_updates={
                "decoded_frame_count": int(decoded_frames.shape[0]),
                "decoded_spatial_size": [
                    int(decoded_frames.shape[2]),
                    int(decoded_frames.shape[1]),
                ],
            },
            outputs={"fps": request.fps or 16},
            notes=["Cosmos stub decode_vae mapped latent frames into decoded video frames."],
        )

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        import numpy as np

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
            notes=["Cosmos stub postprocess standardised frames for backend persistence."],
        )


class CosmosGenerationModel(VideoGenerationModel):
    """Cosmos video generation model backed by an in-process local executor."""

    model_family = "cosmos"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        shell_runner: str | None = None,
        timeout_s: int = 600,
        executor: CosmosLocalExecutor | None = None,
        variant: str | None = None,
        model_size: str = "7B",
        model_id_or_path: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
    ) -> None:
        if base_url is not None or api_key is not None or shell_runner is not None:
            raise ValueError(
                "CosmosGenerationModel only supports in-process execution; "
                "base_url, api_key, and shell_runner are no longer supported"
            )
        self.model_name = model_name or "cosmos-predict1-7b-video2world"
        self.timeout_s = timeout_s

        if executor is not None:
            self._executor = executor
        elif variant is not None:
            from wm_infra.controlplane.schemas import CosmosVariant
            from wm_infra.models.cosmos_predict1 import DiffusersCosmosPredict1Executor

            self._executor = DiffusersCosmosPredict1Executor(
                variant=CosmosVariant(variant),
                model_size=model_size,
                model_id_or_path=model_id_or_path,
                device_id=device_id,
                dtype=dtype,
                enable_cpu_offload=enable_cpu_offload,
            )
        else:
            self._executor = StubCosmosLocalExecutor()

    @property
    def mode(self) -> str:
        return self._executor.execution_mode

    async def load(self) -> None:
        await self._executor.load()

    def describe(self) -> dict[str, Any]:
        return {
            "name": "cosmos-generation-model",
            "family": self.model_family,
            "model_name": self.model_name,
            "mode": self.mode,
            "executor": self._executor.describe(),
        }

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.encode_text(request, state)

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.encode_conditioning(request, state)

    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.denoise(request, state)

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.decode_vae(request, state)

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.postprocess(request, state)
