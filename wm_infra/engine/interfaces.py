"""Pluggable component protocols for the model-agnostic engine.

The scheduler delegates policy decisions to two pluggable interfaces:
BatchPlanner and IterationController.
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
    ) -> list[SchedulerRequest]:
        """Choose requests to include in this iteration's batch."""
        ...

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        """Build an opaque batch payload consumed by the model runner."""
        ...


@runtime_checkable
class IterationController(Protocol):
    """Decides per-request completion after each execution pass."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool: ...


@runtime_checkable
class CacheManager(Protocol):
    """Output cache for skipping redundant model execution."""

    def get(self, request: SchedulerRequest) -> RequestOutput | None: ...
    def put(self, request: SchedulerRequest, output: RequestOutput) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class FeedbackMailbox(Protocol):
    """Non-blocking feedback mailbox keyed by request id."""

    def has(self, request_id: str) -> bool: ...
    def pop(self, request_id: str) -> Any | None: ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------


class ContinuousBatchPlanner:
    """Re-includes running requests every iteration, admits new ones up to budget.

    Running requests are included first so the ``VideoIterationRunner``
    can advance them to their next phase.  Remaining capacity is filled
    from waiting requests in FIFO order.
    """

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
    ) -> list[SchedulerRequest]:
        selected: list[SchedulerRequest] = list(running)
        budget = self.max_batch_size - len(selected)
        for req in waiting:
            if budget <= 0:
                break
            selected.append(req)
            budget -= 1
        return selected

    def build_batch(self, requests: list[SchedulerRequest]) -> Any:
        return [r.data for r in requests]


class VideoDiffusionIterationController:
    """Multi-step iteration controller for ``VideoIterationRunner``.

    The runner already advances the execution phase inside ``execute()``.
    This controller only handles error propagation and terminal detection.
    When a request finishes (``output.finished == True``), it replaces
    ``request.data`` with the accumulated ``stage_results`` list so
    callers see the same ``list[StageResult]`` format.
    """

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        if output.finish_reason == "error" and request.error is None:
            request.error = RuntimeError("Model execution failed")

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        if output.finished:
            from wm_infra.engine.model_executor.execution_state import VideoExecutionState

            state = request.data
            if isinstance(state, VideoExecutionState):
                request.data = state.stage_results
            return True
        return False
