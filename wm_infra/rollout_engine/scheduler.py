"""Rollout scheduler: batches and prioritizes world model prediction requests.

Unlike LLM scheduling (one token at a time, variable prompt lengths),
world model scheduling deals with:
- Variable-length rollouts (different number of future steps)
- Action-conditioned steps (each step needs an action input)
- Batching across rollout steps (different rollouts at different steps)
- Optional branching (fork a rollout to explore alternatives)
- Video-heavy jobs where frame count and resolution can dominate memory pressure
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from wm_infra.config import SchedulerConfig, SchedulerPolicy
from wm_infra.controlplane.schemas import RolloutTaskConfig, VideoMemoryProfile


LOW_VRAM_MEMORY_MULTIPLIER = 0.65
HIGH_QUALITY_MEMORY_MULTIPLIER = 1.25
DEFAULT_FRAME_COUNT = 1
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_RESOURCE_UNITS_PER_GB = 3.0


@dataclass(slots=True)
class RolloutRequest:
    """A pending rollout request."""

    request_id: str
    num_steps: int
    priority: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    deadline: Optional[float] = None
    frame_count: int = DEFAULT_FRAME_COUNT
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    memory_profile: Optional[str] = None
    estimated_resource_units: Optional[float] = None

    def estimate_resource_units(self) -> float:
        if self.estimated_resource_units is not None:
            return self.estimated_resource_units
        megapixels = (self.width * self.height) / 1_000_000
        frame_pressure = max(self.frame_count, 1) * max(self.num_steps, 1)
        multiplier = 1.0
        if self.memory_profile == VideoMemoryProfile.LOW_VRAM.value:
            multiplier = LOW_VRAM_MEMORY_MULTIPLIER
        elif self.memory_profile == VideoMemoryProfile.HIGH_QUALITY.value:
            multiplier = HIGH_QUALITY_MEMORY_MULTIPLIER
        return max(frame_pressure * max(megapixels, 0.1) * multiplier, 0.1)

    @classmethod
    def from_task_config(cls, request_id: str, task_config: Optional[RolloutTaskConfig], *, priority: float = 0.0, deadline: Optional[float] = None) -> "RolloutRequest":
        task_config = task_config or RolloutTaskConfig()
        memory_profile = task_config.memory_profile.value if task_config.memory_profile else None
        return cls(
            request_id=request_id,
            num_steps=task_config.num_steps,
            priority=priority,
            deadline=deadline,
            frame_count=task_config.frame_count or DEFAULT_FRAME_COUNT,
            width=task_config.width or DEFAULT_WIDTH,
            height=task_config.height or DEFAULT_HEIGHT,
            memory_profile=memory_profile,
        )


@dataclass(slots=True)
class ScheduledBatch:
    request_ids: list[str]
    step_indices: list[int]
    actions: list

    @property
    def size(self) -> int:
        return len(self.request_ids)


class RolloutScheduler:
    """Schedules world model rollout steps across concurrent requests."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._pending: Deque[RolloutRequest] = deque()
        self._active: dict[str, RolloutRequest] = {}
        self._step_counts: dict[str, int] = {}
        self._waiting_since: dict[str, float] = {}

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_active(self) -> int:
        return len(self._active)

    def submit(self, request: RolloutRequest) -> None:
        self._pending.append(request)
        self._waiting_since[request.request_id] = time.monotonic()

    def admit(self) -> list[str]:
        admitted = []
        while self._pending and self.num_active < self.config.max_concurrent_rollouts:
            req = self._pending.popleft()
            self._active[req.request_id] = req
            self._step_counts[req.request_id] = 0
            admitted.append(req.request_id)
        return admitted

    def schedule_batch(self) -> ScheduledBatch:
        self.admit()
        if not self._active:
            return ScheduledBatch(request_ids=[], step_indices=[], actions=[])

        candidates = list(self._active.values())
        if self.config.policy == SchedulerPolicy.SJF:
            candidates.sort(key=lambda r: r.num_steps - self._step_counts.get(r.request_id, 0))
        elif self.config.policy == SchedulerPolicy.DEADLINE:
            candidates.sort(key=lambda r: r.deadline or float("inf"))
        elif self.config.policy == SchedulerPolicy.MEMORY_AWARE:
            candidates.sort(key=lambda r: (r.estimate_resource_units(), -r.priority))

        now = time.monotonic()
        urgent = [r for r in candidates if (now - self._waiting_since.get(r.request_id, now)) * 1000 > self.config.max_waiting_time_ms]
        if urgent:
            candidates = urgent + [c for c in candidates if c not in urgent]

        selected = []
        consumed_units = 0.0
        for candidate in candidates:
            if len(selected) >= self.config.max_batch_size:
                break
            units = candidate.estimate_resource_units()
            if self.config.max_batch_resource_units is not None and selected and consumed_units + units > self.config.max_batch_resource_units:
                continue
            selected.append(candidate)
            consumed_units += units

        if not selected:
            selected = candidates[:1]

        return ScheduledBatch(
            request_ids=[r.request_id for r in selected],
            step_indices=[self._step_counts.get(r.request_id, 0) for r in selected],
            actions=[],
        )

    def step_completed(self, request_id: str) -> bool:
        self._step_counts[request_id] = self._step_counts.get(request_id, 0) + 1
        req = self._active.get(request_id)
        return bool(req and self._step_counts[request_id] >= req.num_steps)

    def complete(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)
        self._pending = deque(r for r in self._pending if r.request_id != request_id)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)
