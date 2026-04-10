"""Model-agnostic request scheduler following sglang-omni's Scheduler pattern.

Manages request lifecycle: WAITING -> RUNNING -> FINISHED / ABORTED.
Delegates policy decisions to pluggable BatchPlanner, ResourceManager,
and IterationController interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

from wm_infra.engine.interfaces import (
    BatchPlanner,
    FIFOBatchPlanner,
    IterationController,
    ResourceManager,
    SimpleResourceManager,
    SinglePassIterationController,
)
from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    SchedulerStatus,
)

logger = logging.getLogger(__name__)


class Scheduler:
    """Model-agnostic request-level scheduler.

    Parameters
    ----------
    batch_planner : BatchPlanner | None
        Controls which requests to admit and how to build batches.
        Defaults to ``FIFOBatchPlanner()``.
    resource_manager : ResourceManager | None
        Tracks system capacity.  Defaults to ``SimpleResourceManager()``.
    iteration_controller : IterationController | None
        Decides per-request completion.  Defaults to ``SinglePassIterationController()``.
    """

    def __init__(
        self,
        batch_planner: BatchPlanner | None = None,
        resource_manager: ResourceManager | None = None,
        iteration_controller: IterationController | None = None,
    ) -> None:
        self.batch_planner = batch_planner or FIFOBatchPlanner()
        self.resource_manager = resource_manager or SimpleResourceManager()
        self.iteration_controller = iteration_controller or SinglePassIterationController()

        self._waiting: OrderedDict[str, SchedulerRequest] = OrderedDict()
        self._running: OrderedDict[str, SchedulerRequest] = OrderedDict()
        self._finished: OrderedDict[str, SchedulerRequest] = OrderedDict()

        self._results: dict[str, RequestOutput] = {}
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    def add_request(self, request_id: str, data: Any) -> SchedulerRequest:
        """Enqueue a new request for scheduling."""
        req = SchedulerRequest(
            request_id=request_id,
            data=data,
            status=SchedulerStatus.WAITING,
            arrival_time=time.monotonic(),
        )
        self._waiting[request_id] = req
        return req

    def abort_request(self, request_id: str) -> bool:
        """Abort a request.  Returns True if the request was found."""
        for queue in (self._waiting, self._running, self._finished):
            if request_id in queue:
                req = queue.pop(request_id)
                if req.status == SchedulerStatus.RUNNING:
                    self.resource_manager.free(req)
                req.status = SchedulerStatus.ABORTED
                req.finish_time = time.monotonic()
                self._finished[request_id] = req
                self._results[request_id] = RequestOutput(
                    request_id=request_id,
                    finished=True,
                    finish_reason="aborted",
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Core scheduling loop
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling iteration.

        Selects requests from the waiting queue, transitions them to
        RUNNING, and returns a ``SchedulerOutput`` with the batch.
        """
        waiting_list = list(self._waiting.values())
        running_list = list(self._running.values())

        selected = self.batch_planner.select_requests(
            waiting_list, running_list, self.resource_manager
        )

        for req in selected:
            self._waiting.pop(req.request_id, None)
            req.status = SchedulerStatus.RUNNING
            self.resource_manager.allocate(req)
            self._running[req.request_id] = req

        all_running = list(self._running.values())
        batch_data = self.batch_planner.build_batch(all_running) if all_running else None

        self._step_counter += 1
        return SchedulerOutput(
            requests=list(all_running),
            batch_data=batch_data,
            step_id=self._step_counter,
        )

    def update(self, runner_output: ModelRunnerOutput) -> list[RequestOutput]:
        """Update request states after model execution.

        Returns a list of ``RequestOutput`` for requests that finished.
        """
        completed: list[RequestOutput] = []

        for request_id, output in runner_output.outputs.items():
            req = self._running.get(request_id)
            if req is None:
                continue

            self.iteration_controller.update_request(req, output)

            if self.iteration_controller.is_finished(req, output):
                self._running.pop(request_id)
                self.resource_manager.free(req)
                req.status = SchedulerStatus.FINISHED
                req.finish_time = time.monotonic()
                self._finished[request_id] = req
                self._results[request_id] = output
                completed.append(output)

        return completed

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def get_result(self, request_id: str) -> RequestOutput | None:
        """Retrieve the result for a finished request (non-destructive)."""
        return self._results.get(request_id)

    def pop_result(self, request_id: str) -> RequestOutput | None:
        """Retrieve and remove the result for a finished request."""
        self._finished.pop(request_id, None)
        return self._results.pop(request_id, None)

    def stream(self, request_id: str) -> asyncio.Queue[RequestOutput]:
        """Return an asyncio.Queue that receives incremental outputs.

        For multi-step iteration controllers, each step's output is
        pushed to the queue.  The final output has ``finished=True``.

        NOTE: Streaming support is a placeholder for future multi-step
        iteration controllers.  Single-pass controllers produce one item.
        """
        q: asyncio.Queue[RequestOutput] = asyncio.Queue()
        # For single-pass, the result will be pushed by the engine loop
        # after update() returns completed outputs.
        return q

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)

    def num_finished(self) -> int:
        return len(self._finished)

    def has_work(self) -> bool:
        """Return True if there are waiting or running requests."""
        return bool(self._waiting or self._running)

    def get_request(self, request_id: str) -> SchedulerRequest | None:
        """Look up a request across all queues."""
        for queue in (self._waiting, self._running, self._finished):
            if request_id in queue:
                return queue[request_id]
        return None
