"""Unified model contract for staged temporal generation.

All models — Wan, Cosmos, action-conditioned interactive video generators
(Matrix-Game-3), and future temporal models — implement the five-stage
contract: encode_text -> encode_conditioning -> denoise -> decode_vae -> postprocess.

Backend and serving concerns remain separate from this layer (no controlplane
schemas, no WanExecutionContext, no workload/compute_profile lifecycle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class VideoGenerationModel(ABC):
    """Unified abstract base class for all temporal generation models.

    This is THE model contract for wm-infra. The five-stage interface —
    ``encode_text`` / ``encode_conditioning`` / ``denoise`` / ``decode_vae`` /
    ``postprocess`` — covers Wan, Cosmos, and action-conditioned interactive
    video generators such as Matrix-Game-3.

    Action-conditioned models override ``encode_conditioning`` to incorporate
    per-frame action inputs via the ``action_sequence`` /
    ``action_conditioning_mode`` fields on ``VideoGenerationRequest``.

    The model owns the forward logic but **not** serving infra (workload
    lifecycle, CUDA graph capture, compute profiles, queue scheduling).
    """

    model_family: str = "video_generation"

    @abstractmethod
    async def load(self) -> None:
        """Load / resolve model resources (checkpoints, modules, etc.)."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable model metadata dict."""

    @abstractmethod
    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Produce prompt or text embeddings for the next stage."""

    async def encode_conditioning(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Produce conditioning tensors from references, actions, or metadata.

        When ``request.action_sequence`` is set and
        ``request.action_conditioning_mode != "none"``, subclasses should
        encode the action inputs into conditioning tensors appropriate for
        their architecture (concatenation, cross-attention, adaptive layer
        norm, etc.).
        """
        return StageResult(notes=["No conditioning inputs were provided."])

    @abstractmethod
    async def denoise(self, request: VideoGenerationRequest, state: dict[str, Any]) -> StageResult:
        """Run the main generation step, usually diffusion or a related sampler."""

    async def denoise_init(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> Any:
        """Set up per-step denoising state.  Returns a ``DenoiseLoopState``.

        Models that support per-step denoise control (e.g. WanOfficial)
        implement this plus ``denoise_step`` and ``denoise_finalize``.
        The default raises ``NotImplementedError`` so the runner falls
        back to calling ``denoise()`` as a single black-box pass.
        """
        raise NotImplementedError

    async def denoise_step(
        self, request: VideoGenerationRequest, state: dict[str, Any], denoise_state: Any
    ) -> StageResult:
        """Execute one denoising step and mutate *denoise_state* in-place."""
        raise NotImplementedError

    async def denoise_finalize(
        self, request: VideoGenerationRequest, state: dict[str, Any], denoise_state: Any
    ) -> StageResult:
        """Package final latents after all denoise steps are complete."""
        raise NotImplementedError

    @abstractmethod
    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Decode latent samples into frames or other media outputs."""

    async def postprocess(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Assemble decoded outputs into delivery-ready artifacts."""
        return StageResult(notes=["Postprocess stage is a passthrough."])

    # ------------------------------------------------------------------
    # Batch methods — default sequential fallback
    # ------------------------------------------------------------------
    # Concrete models can override specific batch_* methods for true GPU
    # batching.  The defaults loop over the single-request methods.

    async def batch_encode_text(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[StageResult]:
        return [await self.encode_text(r, s) for r, s in zip(requests, states)]

    async def batch_encode_conditioning(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[StageResult]:
        return [await self.encode_conditioning(r, s) for r, s in zip(requests, states)]

    async def batch_denoise(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[StageResult]:
        return [await self.denoise(r, s) for r, s in zip(requests, states)]

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
    ) -> list[StageResult]:
        return [
            await self.denoise_step(r, s, d)
            for r, s, d in zip(requests, states, denoise_states)
        ]

    async def batch_denoise_finalize(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
        denoise_states: list[Any],
    ) -> list[StageResult]:
        return [
            await self.denoise_finalize(r, s, d)
            for r, s, d in zip(requests, states, denoise_states)
        ]

    async def batch_decode_vae(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[StageResult]:
        return [await self.decode_vae(r, s) for r, s in zip(requests, states)]

    async def batch_postprocess(
        self,
        requests: list[VideoGenerationRequest],
        states: list[dict[str, Any]],
    ) -> list[StageResult]:
        return [await self.postprocess(r, s) for r, s in zip(requests, states)]
