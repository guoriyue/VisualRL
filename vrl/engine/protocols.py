"""Engine protocols: BatchPlanner, ResourceManager, CacheManager."""

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


