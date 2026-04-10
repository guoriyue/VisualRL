"""Pluggable component protocols for the model-agnostic engine.

Following sglang-omni's pattern, the scheduler delegates policy decisions
to three pluggable interfaces: BatchPlanner, ResourceManager, and
IterationController.  Default implementations are provided for the common
single-pass diffusion use case.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from wm_infra.engine.types import RequestOutput, SchedulerRequest

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BatchPlanner(Protocol):
    """Selects which requests to run and builds a batch payload."""

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        """Choose requests from *waiting* to admit this iteration."""
        ...

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        """Build an opaque batch payload consumed by ``ModelRunner``."""
        ...


@runtime_checkable
class ResourceManager(Protocol):
    """Tracks whether the system can accept more work."""

    def can_allocate(self, request: SchedulerRequest) -> bool: ...
    def allocate(self, request: SchedulerRequest) -> None: ...
    def free(self, request: SchedulerRequest) -> None: ...


@runtime_checkable
class IterationController(Protocol):
    """Decides per-request completion after each execution pass."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool: ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class FIFOBatchPlanner:
    """Admit up to *max_batch_size* requests in FIFO order."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        budget = self.max_batch_size - len(running)
        selected: list[SchedulerRequest] = []
        for req in waiting:
            if len(selected) >= budget:
                break
            if resource_manager.can_allocate(req):
                selected.append(req)
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return [r.data for r in requests]


class SimpleResourceManager:
    """Count-based resource manager: limits max concurrent requests."""

    def __init__(self, max_concurrent: int = 64) -> None:
        self.max_concurrent = max_concurrent
        self._allocated: int = 0

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return self._allocated < self.max_concurrent

    def allocate(self, request: SchedulerRequest) -> None:
        self._allocated += 1

    def free(self, request: SchedulerRequest) -> None:
        self._allocated = max(0, self._allocated - 1)


class SinglePassIterationController:
    """One execution pass = finished.  Suitable for diffusion pipelines."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        pass

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        return True
