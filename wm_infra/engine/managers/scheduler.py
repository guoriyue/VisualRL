"""Generic Scheduler — model-agnostic request lifecycle management.

Aligned with sglang-omni's Scheduler contract: overlap-safe update,
WAITING_FEEDBACK / resume, real streaming, completed retention.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from wm_infra.engine.interfaces import (
        BatchPlanner,
        IterationController,
    )

logger = logging.getLogger(__name__)


class Scheduler:
    """Generic request scheduler.

    Responsibilities:
    - Manage request lifecycle (WAITING -> RUNNING -> FINISHED/ABORTED)
    - Delegate batch selection to BatchPlanner
    - Delegate per-request updates to IterationController
    - Produce SchedulerOutput for the model runner
    """

    _COMPLETED_RETENTION_SOFT_LIMIT = 10000
    _COMPLETED_RETENTION_HARD_LIMIT = 5000

    def __init__(
        self,
        batch_planner: BatchPlanner,
        iteration_controller: IterationController,
        stream_adapter: Callable[[SchedulerRequest, RequestOutput], Any] | None = None,
    ):
        self.batch_planner = batch_planner
        self.iteration_controller = iteration_controller
        self._stream_adapter = stream_adapter

        # All active requests (id -> request).
        self.requests: dict[str, SchedulerRequest] = {}
        # Queue membership tracked by id only.
        self.waiting: deque[str] = deque()
        self.running: list[str] = []

        # Result futures (created lazily in get_result).
        self._futures: dict[str, asyncio.Future[SchedulerRequest]] = {}
        self._step_id = 0

        # Bounded retention of terminal requests so late callers of
        # get_result() still resolve immediately.
        self._completed_requests: dict[str, SchedulerRequest] = {}
        self._completed_order: deque[str] = deque()

        # Persistent abort tracking for overlap-safe update().
        self._aborted_ids: set[str] = set()

        # Streaming state.
        self._stream_queues: dict[str, asyncio.Queue[Any]] = {}
        self._completed_stream_queues: dict[str, asyncio.Queue[Any]] = {}
        self._completed_stream_order: deque[str] = deque()
        self._stream_done = object()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_request(self, request_id: str, data: Any) -> None:
        """Add a new request with model-specific data."""
        request = SchedulerRequest(
            request_id=request_id,
            data=data,
            arrival_time=time.monotonic(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)

    def abort_request(self, request_id: str) -> None:
        """Abort a request."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_ids.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED)

    def fail_request(self, request_id: str, error: Exception) -> None:
        """Fail a request with an error, propagating it to any waiting caller."""
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_ids.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED, error=error)

    def has_requests(self) -> bool:
        """Check if there are any requests to process."""
        return len(self.waiting) > 0 or len(self.running) > 0

    def resume_request(self, request_id: str) -> None:
        """Resume a WAITING_FEEDBACK request back to RUNNING."""
        request = self.requests.get(request_id)
        if request is None:
            return
        if request.status == SchedulerStatus.WAITING_FEEDBACK:
            request.status = SchedulerStatus.RUNNING

    async def get_result(self, request_id: str) -> SchedulerRequest:
        """Wait for a request to complete and return the finished request."""
        request = self._get_request(request_id)
        if request is None:
            raise KeyError(f"Unknown request: {request_id}")

        while True:
            request = self._get_request(request_id)
            if request is None:
                raise KeyError(f"Unknown request: {request_id}")
            if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
                self._futures.pop(request_id, None)
                if request.error is not None:
                    raise request.error
                return request

            # Lazy future creation, recover from stale/cancelled futures.
            loop = asyncio.get_running_loop()
            future = self._futures.get(request_id)
            if future is None or future.cancelled():
                future = loop.create_future()
                self._futures[request_id] = future

            await asyncio.shield(future)

    async def stream(self, request_id: str) -> AsyncIterator[Any]:
        """Yield per-step stream data for a request."""
        queue = self._subscribe_stream(request_id)
        try:
            while True:
                item = await queue.get()
                if item is self._stream_done:
                    return
                yield item
        finally:
            self._stream_queues.pop(request_id, None)
            self._completed_stream_queues.pop(request_id, None)

    def prepare_stream(self, request_id: str) -> None:
        """Pre-register a stream queue before request submission."""
        self._subscribe_stream(request_id)

    def discard_stream(self, request_id: str) -> None:
        """Drop a pre-registered stream queue for a failed submission."""
        self._stream_queues.pop(request_id, None)
        if request_id in self._completed_stream_queues:
            self._completed_stream_queues.pop(request_id, None)
            self._remove_completed_stream_order(request_id)

    # -----------------------------------------------------------------
    # Core scheduling
    # -----------------------------------------------------------------

    def schedule(self) -> SchedulerOutput | None:
        """Schedule next batch. Returns None if no work."""
        if not self.waiting and not self.running:
            return None

        self._step_id += 1

        waiting_reqs = [self.requests[rid] for rid in self.waiting]
        running_reqs = [
            self.requests[rid]
            for rid in self.running
            if self.requests[rid].status != SchedulerStatus.WAITING_FEEDBACK
        ]

        selected = self.batch_planner.select_requests(waiting_reqs, running_reqs)

        if not selected:
            return None

        for request in selected:
            if request.request_id in self.waiting:
                self.waiting.remove(request.request_id)
                self.running.append(request.request_id)
                request.status = SchedulerStatus.RUNNING

        batch_data = self.batch_planner.build_batch(selected)

        return SchedulerOutput(
            requests=selected,
            batch_data=batch_data,
            step_id=self._step_id,
        )

    def update(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[SchedulerRequest]:
        """Update state from model output. Returns finished requests.

        Overlap-safe: iterates over scheduler_output.requests (the batch
        that was scheduled), not model_output keys. Guards against
        requests that were aborted or finished between schedule() and
        this update() call.
        """
        finished: list[SchedulerRequest] = []

        for request in scheduler_output.requests:
            if request.request_id in self._aborted_ids:
                continue
            if request.status != SchedulerStatus.RUNNING:
                continue

            output = model_output.outputs.get(request.request_id)
            if output is None:
                logger.warning("Missing output for request_id=%s", request.request_id)
                continue

            self.iteration_controller.update_request(request, output)
            self._emit_stream(request, output)

            if self.iteration_controller.is_finished(request, output):
                if request.error is not None:
                    self._finish_request(
                        request, status=SchedulerStatus.ABORTED, error=request.error
                    )
                else:
                    self._finish_request(request)
                finished.append(request)

        return finished

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def num_waiting(self) -> int:
        return len(self.waiting)

    def num_running(self) -> int:
        return len(self.running)

    def get_request(self, request_id: str) -> SchedulerRequest | None:
        """Look up a request across active and completed."""
        return self._get_request(request_id)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _emit_stream(self, request: SchedulerRequest, output: RequestOutput) -> None:
        if self._stream_adapter is None:
            return
        queue = self._stream_queues.get(request.request_id)
        if queue is None:
            return
        item = self._stream_adapter(request, output)
        if item is None:
            return
        queue.put_nowait(item)

    def _finish_request(
        self,
        request: SchedulerRequest,
        status: SchedulerStatus = SchedulerStatus.FINISHED,
        error: Exception | None = None,
    ) -> None:
        request.status = status
        if error is not None:
            request.error = error
        request.finish_time = time.monotonic()

        # Remove from queues.
        if request.request_id in self.running:
            self.running.remove(request.request_id)
        if request.request_id in self.waiting:
            self.waiting.remove(request.request_id)

        self._aborted_ids.discard(request.request_id)
        self.requests.pop(request.request_id, None)
        self._remember_completed(request)

        # Resolve future.
        future = self._futures.pop(request.request_id, None)
        if future is not None and not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(request)

        # Close stream queue.
        queue = self._stream_queues.pop(request.request_id, None)
        if queue is not None:
            queue.put_nowait(self._stream_done)
            self._remember_completed_stream(request.request_id, queue)

    def _get_request(self, request_id: str) -> SchedulerRequest | None:
        request = self.requests.get(request_id)
        if request is not None:
            return request
        return self._completed_requests.get(request_id)

    def _subscribe_stream(self, request_id: str) -> asyncio.Queue[Any]:
        queue = self._stream_queues.get(request_id)
        if queue is None:
            queue = self._completed_stream_queues.pop(request_id, None)
            if queue is not None:
                self._remove_completed_stream_order(request_id)
        if queue is None:
            queue = asyncio.Queue()
        self._stream_queues[request_id] = queue
        # If already terminal, send done immediately.
        request = self._get_request(request_id)
        if request is not None and request.status in (
            SchedulerStatus.FINISHED,
            SchedulerStatus.ABORTED,
        ):
            queue.put_nowait(self._stream_done)
        return queue

    def _remember_completed(self, request: SchedulerRequest) -> None:
        rid = request.request_id
        if rid not in self._completed_requests:
            self._completed_order.append(rid)
        self._completed_requests[rid] = request
        if len(self._completed_order) <= self._COMPLETED_RETENTION_SOFT_LIMIT:
            return
        while len(self._completed_order) > self._COMPLETED_RETENTION_HARD_LIMIT:
            stale = self._completed_order.popleft()
            self._completed_requests.pop(stale, None)

    def _remember_completed_stream(
        self, request_id: str, queue: asyncio.Queue[Any]
    ) -> None:
        if request_id not in self._completed_stream_queues:
            self._completed_stream_order.append(request_id)
        self._completed_stream_queues[request_id] = queue
        while len(self._completed_stream_order) > self._COMPLETED_RETENTION_HARD_LIMIT:
            stale = self._completed_stream_order.popleft()
            self._completed_stream_queues.pop(stale, None)

    def _remove_completed_stream_order(self, request_id: str) -> None:
        try:
            self._completed_stream_order.remove(request_id)
        except ValueError:
            return
