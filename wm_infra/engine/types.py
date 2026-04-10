"""Model-agnostic engine types following sglang-omni's request-level pattern.

Replaces the previous RL-specific types (Phase, EntityRequest, StepResult,
etc.) with generic request-level types that work for any model backend:
diffusion, world-model rollouts, autoregressive generation, etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------


class SchedulerStatus(Enum):
    """Lifecycle state of a request inside the scheduler."""

    WAITING = "waiting"
    RUNNING = "running"
    WAITING_FEEDBACK = "waiting_feedback"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(slots=True)
class SchedulerRequest:
    """One request tracked by the scheduler.

    ``data`` is opaque and model-specific (e.g. ``VideoGenerationRequest``).
    The scheduler never inspects it — only the ``BatchPlanner`` and
    ``IterationController`` do.
    """

    request_id: str
    data: Any
    status: SchedulerStatus = SchedulerStatus.WAITING
    arrival_time: float = field(default_factory=time.monotonic)
    finish_time: float | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerOutput:
    """Batch of requests selected for execution on this iteration."""

    requests: list[SchedulerRequest] = field(default_factory=list)
    batch_data: Any = None
    step_id: int = 0


@dataclass(slots=True)
class RequestOutput:
    """Result for a single request after one execution pass."""

    request_id: str
    data: Any = None
    finished: bool = False
    finish_reason: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelRunnerOutput:
    """Aggregated results from one model execution pass."""

    outputs: dict[str, RequestOutput] = field(default_factory=dict)
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
