"""EngineLoop: schedule → execute → update. GPU work runs in a worker thread."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from vrl.engine.managers.scheduler import Scheduler
from vrl.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from vrl.engine.protocols import CacheManager

logger = logging.getLogger(__name__)


class EngineLoop:
    """Async engine loop: schedule → execute → update."""

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: Any,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.cache_manager = cache_manager

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

    # -----------------------------------------------------------------
    # Engine public API
    # -----------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> None:
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def stream(self, request_id: str):
        async for item in self.scheduler.stream(request_id):
            yield item

    def prepare_stream(self, request_id: str) -> None:
        self.scheduler.prepare_stream(request_id)

    def discard_stream(self, request_id: str) -> None:
        self.scheduler.discard_stream(request_id)

    async def abort(self, request_id: str) -> None:
        self.scheduler.abort_request(request_id)

    def add_request_nowait(
        self, request_id: str, data: Any
    ) -> asyncio.Task[RequestOutput]:
        self.scheduler.add_request(request_id, data)
        return asyncio.create_task(self._get_result_as_output(request_id))

    async def _get_result_as_output(self, request_id: str) -> RequestOutput:
        request = await self.scheduler.get_result(request_id)
        return RequestOutput(
            request_id=request_id,
            data=request.data,
            finished=True,
            finish_reason=(
                "completed"
                if request.status == SchedulerStatus.FINISHED
                else "aborted"
            ),
        )

    def abort_request(self, request_id: str) -> bool:
        req = self.scheduler.get_request(request_id)
        if req is None:
            return False
        self.scheduler.abort_request(request_id)
        return True

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("EngineLoop started")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        logger.info("EngineLoop stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -----------------------------------------------------------------
    # Processing loop
    # -----------------------------------------------------------------

    async def _run_loop(self) -> None:
        while self._running:
            await self._step()
            await asyncio.sleep(0)

    async def _step(self) -> bool:
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            await asyncio.sleep(0.001)
            return False

        try:
            if self.cache_manager is not None:
                scheduler_output = await self._filter_cached(scheduler_output)
                if scheduler_output is None:
                    return True

            model_output = await self._execute_async(scheduler_output)

            if self.cache_manager is not None:
                self._update_cache_sync(scheduler_output, model_output)

            finished = self.scheduler.update(scheduler_output, model_output)
            if finished:
                for req in finished:
                    logger.debug("Request %s finished", req.request_id)

        except Exception as e:
            logger.exception(
                "EngineLoop step failed, failing %d request(s)",
                len(scheduler_output.requests),
            )
            self._fail_requests(scheduler_output, e)
            return False

        return True

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _should_execute_in_thread(self) -> bool:
        flag = getattr(self.model_runner, "execute_in_thread", None)
        if flag is not None:
            return flag
        device = getattr(self.model_runner, "device", None)
        device_type = getattr(device, "type", str(device) if device is not None else "")
        return str(device_type) != "cpu"

    async def _execute_async(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        if self._should_execute_in_thread():
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.model_runner.execute, scheduler_output
            )
        else:
            return self.model_runner.execute(scheduler_output)

    def _update_cache_sync(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> None:
        assert self.cache_manager is not None
        for request in scheduler_output.requests:
            output = model_output.outputs.get(request.request_id)
            if output is not None:
                self.cache_manager.put(request, output)

    def _fail_requests(
        self, scheduler_output: SchedulerOutput, error: Exception
    ) -> None:
        for request in scheduler_output.requests:
            with suppress(Exception):
                self.scheduler.fail_request(request.request_id, error)

    async def _filter_cached(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput | None:
        assert self.cache_manager is not None
        cached_outputs: dict[str, Any] = {}
        uncached_requests = []

        for request in scheduler_output.requests:
            cached = self.cache_manager.get(request)
            if cached is not None:
                cached_outputs[request.request_id] = cached
            else:
                uncached_requests.append(request)

        if not uncached_requests:
            req_ids = [r.request_id for r in scheduler_output.requests]
            req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}
            model_output = ModelRunnerOutput(
                outputs=cached_outputs,
                req_ids=req_ids,
                req_id_to_index=req_id_to_index,
            )
            self.scheduler.update(scheduler_output, model_output)
            return None

        batch_data = self.scheduler.batch_planner.build_batch(uncached_requests)
        return SchedulerOutput(
            requests=uncached_requests,
            batch_data=batch_data,
            step_id=scheduler_output.step_id,
        )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def num_waiting(self) -> int:
        return self.scheduler.num_waiting()

    def num_running(self) -> int:
        return self.scheduler.num_running()

    def num_pending(self) -> int:
        return self.scheduler.num_waiting()
