"""Unified model contract for staged temporal generation.

All models — Wan, Cosmos, action-conditioned interactive video generators
(Matrix-Game-3), and future temporal models — implement the five-stage
contract: encode_text -> encode_conditioning -> denoise -> decode_vae -> postprocess.

Backend and serving concerns remain separate from this layer (no controlplane
schemas, no WanExecutionContext, no workload/compute_profile lifecycle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VideoGenerationRequest:
    """Model-level request for staged generation.

    The request stays backend-agnostic and uses scalars only. Model-specific
    inputs can be carried in ``extra`` when a generator needs action traces,
    trajectory hints, camera metadata, or other family-specific fields.

    Action-conditioning fields (``action_sequence``, ``action_dim``,
    ``action_conditioning_mode``) support interactive video generators
    such as Matrix-Game-3 that accept per-frame action inputs.
    """

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
    # Action-conditioning for interactive video generators
    action_sequence: list[list[float]] | None = None
    action_dim: int | None = None
    action_conditioning_mode: str = "none"  # "none" | "concat" | "cross_attn" | "adaptive"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageResult:
    """Normalized stage output used by the shared staged-generation contract.

    The same result shape works for Wan, Cosmos, and action-conditioned
    interactive video generators, even when their execution substrates differ.
    ``WanStageUpdate`` in ``backends.wan.engine`` is now an alias for this type.
    """

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"


@dataclass(slots=True)
class VideoGenerationOutput:
    """Final result container for a full generation run."""

    stage_results: list[StageResult] = field(default_factory=list)
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    async def denoise(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        """Run the main generation step, usually diffusion or a related sampler."""

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
