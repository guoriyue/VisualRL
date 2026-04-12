"""Request scheduler: lifecycle, batching, streaming, overlap-safe update."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from vrl.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from vrl.engine.protocols import (
        BatchPlanner,
        ResourceManager,
    )

logger = logging.getLogger(__name__)


class Scheduler:
    """Request scheduler: WAITING → RUNNING → FINISHED/ABORTED."""

    _COMPLETED_RETENTION_SOFT_LIMIT = 10000
    _COMPLETED_RETENTION_HARD_LIMIT = 5000

    def __init__(
        self,
        batch_planner: BatchPlanner,
        resource_manager: ResourceManager,
        stream_adapter: Callable[[SchedulerRequest, RequestOutput], Any] | None = None,
    ):
        self.batch_planner = batch_planner
        self.resource_manager = resource_manager
        self._stream_adapter = stream_adapter

        self.requests: dict[str, SchedulerRequest] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []
        self._futures: dict[str, asyncio.Future[SchedulerRequest]] = {}
        self._step_id = 0
        self._completed_requests: dict[str, SchedulerRequest] = {}
        self._completed_order: deque[str] = deque()
        self._aborted_ids: set[str] = set()
        self._stream_queues: dict[str, asyncio.Queue[Any]] = {}
        self._completed_stream_queues: dict[str, asyncio.Queue[Any]] = {}
        self._completed_stream_order: deque[str] = deque()
        self._stream_done = object()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def add_request(self, request_id: str, data: Any) -> None:
        request = SchedulerRequest(
            request_id=request_id,
            data=data,
            arrival_time=time.monotonic(),
        )
        self.requests[request_id] = request
        self.waiting.append(request_id)

    def abort_request(self, request_id: str) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_ids.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED)

    def fail_request(self, request_id: str, error: Exception) -> None:
        request = self.requests.get(request_id)
        if request is None:
            return
        self._aborted_ids.add(request_id)
        self._finish_request(request, status=SchedulerStatus.ABORTED, error=error)

    def has_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    async def get_result(self, request_id: str) -> SchedulerRequest:
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
        self._subscribe_stream(request_id)

    def discard_stream(self, request_id: str) -> None:
        self._stream_queues.pop(request_id, None)
        if request_id in self._completed_stream_queues:
            self._completed_stream_queues.pop(request_id, None)
            self._remove_completed_stream_order(request_id)

    # -----------------------------------------------------------------
    # Core scheduling
    # -----------------------------------------------------------------

    def schedule(self) -> SchedulerOutput | None:
        if not self.waiting and not self.running:
            return None

        self._step_id += 1

        waiting_reqs = [self.requests[rid] for rid in self.waiting]
        running_reqs = [self.requests[rid] for rid in self.running]

        selected = self.batch_planner.select_requests(
            waiting_reqs, running_reqs, self.resource_manager
        )

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
        """Update state from model output. Returns finished requests. Overlap-safe."""
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

            self._emit_stream(request, output)

            if output.finished:
                if output.finish_reason == "error":
                    error = request.error or RuntimeError("Model execution failed")
                    self._finish_request(
                        request, status=SchedulerStatus.ABORTED, error=error
                    )
                else:
                    request.data = output.data
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

        # Free resource allocation.
        self.resource_manager.free(request)

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
