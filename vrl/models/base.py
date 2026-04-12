"""Model contract: video generation model base + execution DTOs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Model execution contract types (moved from vrl/schemas/video_generation.py)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VideoGenerationRequest:
    """Model execution parameters. Backend-agnostic scalars; extras in ``extra``."""

    prompt: str = ""
    negative_prompt: str = ""
    references: list[str] = field(default_factory=list)
    task_type: str = "text_to_video"
    width: int = 1024
    height: int = 640
    frame_count: int = 16
    num_steps: int = 35
    guidance_scale: float = 5.0
    high_noise_guidance_scale: float | None = None
    seed: int | None = None
    model_name: str = ""
    model_size: str = "A14B"
    ckpt_dir: str | None = None
    fps: int = 16
    sample_solver: str = "dpmpp"
    shift: float = 1.0
    t5_cpu: bool = True
    convert_model_dtype: bool = True
    offload_model: bool = False
    # Action-conditioning
    action_sequence: list[list[float]] | None = None
    action_dim: int | None = None
    action_conditioning_mode: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelResult:
    """Normalized output from a model method (encode, generate, decode, etc.)."""

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"


class VideoGenerationModel(ABC):
    """Five-stage generation: encode_text → encode_conditioning → generate → decode_vae → postprocess."""

    model_family: str = "video_generation"

    @abstractmethod
    async def load(self) -> None:
        """Load model resources."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable model metadata."""

    @abstractmethod
    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> ModelResult:
        """Produce text embeddings."""

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> ModelResult:
        """Produce conditioning tensors from references, actions, or metadata."""
        return ModelResult(notes=["No conditioning inputs were provided."])

    @abstractmethod
    async def generate(self, request: VideoGenerationRequest, state: dict[str, Any]) -> ModelResult:
        """Run generation (diffusion sampling, AR decode, etc.)."""

    async def denoise_init(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> Any:
        """Set up per-step denoising state. Returns a DenoiseLoopState."""
        raise NotImplementedError

    async def denoise_step(
        self, request: VideoGenerationRequest, state: dict[str, Any], denoise_state: Any
    ) -> ModelResult:
        """One denoising step; mutates denoise_state in-place."""
        raise NotImplementedError

    async def denoise_finalize(
        self, request: VideoGenerationRequest, state: dict[str, Any], denoise_state: Any
    ) -> ModelResult:
        """Package final latents after all steps."""
        raise NotImplementedError

    @abstractmethod
    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> ModelResult:
        """Decode latents into frames."""

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> ModelResult:
        """Assemble final output artifacts."""
        return ModelResult(notes=["Postprocess stage is a passthrough."])

    # -- batch methods (default: sequential fallback) ---

    async def batch_encode_text(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[ModelResult]:
        return [await self.encode_text(r, s) for r, s in zip(requests, states)]

    async def batch_encode_conditioning(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[ModelResult]:
        return [await self.encode_conditioning(r, s) for r, s in zip(requests, states)]

    async def batch_generate(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[ModelResult]:
        return [await self.generate(r, s) for r, s in zip(requests, states)]

    async def batch_denoise_init(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[Any]:
        return [await self.denoise_init(r, s) for r, s in zip(requests, states)]

    async def batch_denoise_step(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
        denoise_states: list[Any],
    ) -> list[ModelResult]:
        return [
            await self.denoise_step(r, s, d)
            for r, s, d in zip(requests, states, denoise_states)
        ]

    async def batch_denoise_finalize(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
        denoise_states: list[Any],
    ) -> list[ModelResult]:
        return [
            await self.denoise_finalize(r, s, d)
            for r, s, d in zip(requests, states, denoise_states)
        ]

    async def batch_decode_vae(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[ModelResult]:
        return [await self.decode_vae(r, s) for r, s in zip(requests, states)]

    async def batch_postprocess(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[ModelResult]:
        return [await self.postprocess(r, s) for r, s in zip(requests, states)]
