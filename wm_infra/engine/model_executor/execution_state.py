"""Mutable per-request execution state for iterative video generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wm_infra.engine.types import VideoExecutionPhase
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


@dataclass(frozen=True)
class PhaseGroupKey:
    """Key for grouping requests that can be batched together.

    Requests with the same (phase, height, width, frame_count) can be
    dispatched as a single batched model call.
    """

    phase: VideoExecutionPhase
    height: int
    width: int
    frame_count: int


@dataclass
class DenoiseLoopState:
    """Per-step denoising state for models that support iterative denoise.

    Populated by ``denoise_init()`` and mutated by each ``denoise_step()``
    call.  Models that run denoise as a black box (Cosmos, WanDiffusers)
    leave this as ``None``.
    """

    latents: Any = None
    timesteps: Any = None
    current_step: int = 0
    total_steps: int = 0
    seed_generator: Any = None
    arg_c: dict = field(default_factory=dict)
    arg_null: dict = field(default_factory=dict)
    boundary: float = 0.0
    high_noise_guidance_scale: float = 5.0
    low_noise_guidance_scale: float = 5.0
    scheduler: Any = None
    seed: int = 0
    seed_policy: str = "randomized"
    solver_name: str = "dpmpp"
    pipeline: Any = None
    task_key: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class VideoExecutionState:
    """Mutable per-request state stored as ``SchedulerRequest.data``.

    The ``VideoIterationRunner`` wraps the original
    ``VideoGenerationRequest`` in this state on first execution.  Each
    ``advance()`` call reads/writes ``phase`` and ``pipeline_state``.
    When the request reaches ``DONE``, the ``VideoDiffusionIterationController``
    replaces ``request.data`` with ``stage_results`` for caller compatibility.
    """

    request: VideoGenerationRequest
    phase: VideoExecutionPhase = VideoExecutionPhase.ENCODE_TEXT
    pipeline_state: dict = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    denoise_state: DenoiseLoopState | None = None
    supports_per_step_denoise: bool = False
