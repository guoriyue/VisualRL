"""Generic async engine loop following sglang-omni's OmniEngine pattern.

The ``EngineLoop`` is the top-level orchestrator.  It owns a ``Scheduler``
and a ``ModelRunner``, accepts requests via ``add_request()``, and runs a
persistent ``schedule() -> execute() -> update()`` loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import Executor
from typing import Any

from wm_infra.engine.interfaces import (
    BatchPlanner,
    IterationController,
    ResourceManager,
)
from wm_infra.engine.managers.scheduler import Scheduler
from wm_infra.engine.model_executor.model_runner import ModelFn, ModelRunner
from wm_infra.engine.types import RequestOutput

logger = logging.getLogger(__name__)


class EngineLoop:
    """Persistent async engine loop: schedule -> execute -> update.

    Parameters
    ----------
    model_fn : ModelFn
        Async callable that processes a single ``SchedulerRequest``.
    batch_planner : BatchPlanner | None
        Optional custom batch planner for the scheduler.
    resource_manager : ResourceManager | None
        Optional custom resource manager for the scheduler.
    iteration_controller : IterationController | None
        Optional custom iteration controller for the scheduler.
    executor : Executor | None
        Optional thread pool executor for GPU-bound work.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        *,
        batch_planner: BatchPlanner | None = None,
        resource_manager: ResourceManager | None = None,
        iteration_controller: IterationController | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.scheduler = Scheduler(
            batch_planner=batch_planner,
            resource_manager=resource_manager,
            iteration_controller=iteration_controller,
        )
        self.model_runner = ModelRunner(model_fn)
        self._executor = executor

        self._futures: dict[str, asyncio.Future[RequestOutput]] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._has_work = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the persistent engine loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the engine loop gracefully."""
        self._running = False
        self._has_work.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for future in self._futures.values():
            if not future.done():
                future.cancel()
        self._futures.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Request API
    # ------------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> RequestOutput:
        """Submit a request and await its completion."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RequestOutput] = loop.create_future()
        self._futures[request_id] = future
        self.scheduler.add_request(request_id, data)
        self._has_work.set()
        return await future

    def add_request_nowait(self, request_id: str, data: Any) -> asyncio.Future[RequestOutput]:
        """Submit a request without awaiting. Returns the future."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[RequestOutput] = loop.create_future()
        self._futures[request_id] = future
        self.scheduler.add_request(request_id, data)
        self._has_work.set()
        return future

    async def get_result(self, request_id: str) -> RequestOutput | None:
        """Retrieve the result for a completed request."""
        return self.scheduler.get_result(request_id)

    def abort_request(self, request_id: str) -> bool:
        """Abort a request and resolve its future."""
        success = self.scheduler.abort_request(request_id)
        if success:
            result = self.scheduler.pop_result(request_id)
            future = self._futures.pop(request_id, None)
            if future is not None and not future.done() and result is not None:
                future.set_result(result)
        return success

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Persistent loop: schedule -> execute -> update."""
        while self._running:
            try:
                has_active = await self._iteration()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Engine loop iteration failed")
                has_active = False

            if has_active:
                await asyncio.sleep(0)
            else:
                self._has_work.clear()
                await self._has_work.wait()

    async def _iteration(self) -> bool:
        """One iteration of the engine loop."""
        # (1) Schedule: select requests for execution
        sched_output = self.scheduler.schedule()

        if not sched_output.requests:
            return self.scheduler.has_work()

        # (2) Execute: run model on the batch
        runner_output = await self.model_runner.execute(sched_output)

        # (3) Update: transition completed requests and resolve futures
        completed = self.scheduler.update(runner_output)
        for output in completed:
            future = self._futures.pop(output.request_id, None)
            if future is not None and not future.done():
                future.set_result(output)

        return self.scheduler.has_work()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def num_waiting(self) -> int:
        return self.scheduler.num_waiting()

    def num_running(self) -> int:
        return self.scheduler.num_running()

    def num_pending(self) -> int:
        return self.scheduler.num_waiting()
