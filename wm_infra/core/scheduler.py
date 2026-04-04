"""Rollout scheduler: batches and prioritizes world model prediction requests.

Unlike LLM scheduling (one token at a time, variable prompt lengths),
world model scheduling deals with:
- Variable-length rollouts (different number of future steps)
- Action-conditioned steps (each step needs an action input)
- Batching across rollout steps (different rollouts at different steps)
- Optional branching (fork a rollout to explore alternatives)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from wm_infra.config import SchedulerConfig, SchedulerPolicy


@dataclass(slots=True)
class RolloutRequest:
    """A pending rollout request."""

    request_id: str
    num_steps: int
    priority: float = 0.0  # higher = more urgent
    created_at: float = field(default_factory=time.monotonic)
    deadline: Optional[float] = None  # optional absolute deadline


@dataclass(slots=True)
class ScheduledBatch:
    """A batch of rollout steps ready to execute."""

    request_ids: list[str]
    step_indices: list[int]  # which step each request is at
    actions: list  # action tensors for each request

    @property
    def size(self) -> int:
        return len(self.request_ids)


class RolloutScheduler:
    """Schedules world model rollout steps across concurrent requests.

    Supports three policies:
    - FCFS: first-come first-serve
    - SJF: shortest-job-first (fewest remaining steps)
    - DEADLINE: earliest-deadline-first
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._pending: Deque[RolloutRequest] = deque()
        self._active: dict[str, RolloutRequest] = {}
        self._step_counts: dict[str, int] = {}  # request_id -> current step
        self._waiting_since: dict[str, float] = {}

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_active(self) -> int:
        return len(self._active)

    def submit(self, request: RolloutRequest) -> None:
        """Submit a new rollout request."""
        self._pending.append(request)
        self._waiting_since[request.request_id] = time.monotonic()

    def admit(self) -> list[str]:
        """Move pending requests to active set. Returns admitted IDs."""
        admitted = []
        while (
            self._pending
            and self.num_active < self.config.max_concurrent_rollouts
        ):
            req = self._pending.popleft()
            self._active[req.request_id] = req
            self._step_counts[req.request_id] = 0
            admitted.append(req.request_id)
        return admitted

    def schedule_batch(self) -> ScheduledBatch:
        """Select the next batch of rollout steps to execute.

        Automatically admits pending requests each time it's called,
        enabling continuous batching — new requests join mid-rollout.

        Returns:
            ScheduledBatch with request IDs and their current step indices
        """
        # Admit pending requests every scheduling round (continuous batching)
        self.admit()

        if not self._active:
            return ScheduledBatch(request_ids=[], step_indices=[], actions=[])

        # Sort active requests by policy
        candidates = list(self._active.values())
        if self.config.policy == SchedulerPolicy.SJF:
            candidates.sort(
                key=lambda r: r.num_steps - self._step_counts.get(r.request_id, 0)
            )
        elif self.config.policy == SchedulerPolicy.DEADLINE:
            candidates.sort(
                key=lambda r: r.deadline or float("inf")
            )
        # FCFS: keep insertion order (deque order)

        # Check waiting time — force-admit if waiting too long
        now = time.monotonic()
        urgent = [
            r for r in candidates
            if (now - self._waiting_since.get(r.request_id, now)) * 1000 > self.config.max_waiting_time_ms
        ]
        if urgent:
            candidates = urgent + [c for c in candidates if c not in urgent]

        # Build batch
        batch_size = min(len(candidates), self.config.max_batch_size)
        selected = candidates[:batch_size]

        return ScheduledBatch(
            request_ids=[r.request_id for r in selected],
            step_indices=[self._step_counts.get(r.request_id, 0) for r in selected],
            actions=[],  # filled by engine
        )

    def step_completed(self, request_id: str) -> bool:
        """Mark a step as completed for a request.

        Returns:
            True if the rollout is now complete
        """
        self._step_counts[request_id] = self._step_counts.get(request_id, 0) + 1
        req = self._active.get(request_id)
        if req and self._step_counts[request_id] >= req.num_steps:
            return True
        return False

    def complete(self, request_id: str) -> None:
        """Remove a completed request from the scheduler."""
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        """Cancel an active or pending request."""
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)
        self._pending = deque(r for r in self._pending if r.request_id != request_id)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)
