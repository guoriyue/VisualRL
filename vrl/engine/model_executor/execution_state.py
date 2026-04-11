"""Execution state types for the model executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadSignature:
    """Batch grouping key: requests with the same signature can be batched."""

    model_name: str
    task_type: str
    height: int
    width: int
    frame_count: int
    num_steps: int


@dataclass
class DenoiseLoopState:
    """Per-step denoising progress. Model-agnostic.

    The engine only reads ``current_step`` and ``total_steps`` to decide
    when denoising is complete.

    All model-specific state (latents, scheduler, guidance args, etc.)
    lives in ``model_state`` — an opaque object created by
    ``denoise_init()`` and consumed by ``denoise_step()`` /
    ``denoise_finalize()``. The engine never inspects it.
    """

    current_step: int = 0
    total_steps: int = 0
    model_state: Any = None
