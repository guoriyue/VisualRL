"""Pluggable protocols: BatchPlanner, ResourceManager, CacheManager."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.engine.types import RequestOutput, SchedulerRequest

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ResourceManager(Protocol):
    """Gate new-request admission based on available resources."""

    def can_allocate(self, request: SchedulerRequest) -> bool: ...
    def allocate(self, request: SchedulerRequest) -> None: ...
    def free(self, request: SchedulerRequest) -> None: ...


@runtime_checkable
class BatchPlanner(Protocol):
    """Select requests and build batch payload."""

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]: ...

    def build_batch(self, requests: list[SchedulerRequest]) -> Any: ...


@runtime_checkable
class CacheManager(Protocol):
    """Output cache for redundant execution skipping."""

    def get(self, request: SchedulerRequest) -> RequestOutput | None: ...
    def put(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class SimpleResourceManager:
    """Count-based resource manager: admits up to max_count concurrent requests."""

    def __init__(self, max_count: int = 32) -> None:
        self.max_count = max_count
        self._count = 0

    def can_allocate(self, request: SchedulerRequest) -> bool:
        return self._count < self.max_count

    def allocate(self, request: SchedulerRequest) -> None:
        self._count += 1

    def free(self, request: SchedulerRequest) -> None:
        self._count = max(0, self._count - 1)


class ContinuousBatchPlanner:
    """FIFO batch planner: running requests first, then waiting up to budget."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        selected: list[SchedulerRequest] = list(running)
        budget = self.max_batch_size - len(selected)
        for req in waiting:
            if budget <= 0:
                break
            if not resource_manager.can_allocate(req):
                break
            resource_manager.allocate(req)
            selected.append(req)
            budget -= 1
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return [r.data for r in requests]
