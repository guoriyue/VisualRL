"""EngineLoop — unified engine combining Scheduler and model runner.

Execution model:
    schedule(N) -> execute(N) -> update(N) -> schedule(N+1) -> ...

GPU work runs in a worker thread via ``run_in_executor`` so the
asyncio event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from wm_infra.engine.managers.scheduler import Scheduler
from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerStatus,
)

if TYPE_CHECKING:
    from wm_infra.engine.interfaces import CacheManager, FeedbackMailbox

logger = logging.getLogger(__name__)


class EngineLoop:
    """Persistent async engine loop: schedule -> execute -> update.

    Parameters
    ----------
    scheduler : Scheduler
        Owns request lifecycle.
    model_runner
        Stateless model executor (e.g. ``VideoIterationRunner``).
    cache_manager : CacheManager | None
        Optional output cache.
    feedback_mailbox
        Optional queue for external feedback (e.g. pause/resume).
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: Any,
        cache_manager: CacheManager | None = None,
        feedback_mailbox: FeedbackMailbox | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.cache_manager = cache_manager
        self._feedback_mailbox = feedback_mailbox

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

    # -----------------------------------------------------------------
    # Engine public API
    # -----------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        """Wait for a request to complete and return its data."""
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def stream(self, request_id: str):
        """Stream per-step outputs for a request."""
        async for item in self.scheduler.stream(request_id):
            yield item

    def prepare_stream(self, request_id: str) -> None:
        """Pre-register stream delivery before request execution starts."""
        self.scheduler.prepare_stream(request_id)

    def discard_stream(self, request_id: str) -> None:
        """Discard a pre-registered stream queue for failed submissions."""
        self.scheduler.discard_stream(request_id)

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_request(request_id)

    def add_request_nowait(
        self, request_id: str, data: Any
    ) -> asyncio.Task[RequestOutput]:
        """Submit a request, return a Task that resolves to RequestOutput.

        Used by IPC server for non-blocking submission with callbacks.
        """
        self.scheduler.add_request(request_id, data)
        return asyncio.create_task(self._get_result_as_output(request_id))

    async def _get_result_as_output(self, request_id: str) -> RequestOutput:
        """Await completion and wrap as RequestOutput for IPC compat."""
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
        """Abort a request. Returns True if found. IPC compat."""
        req = self.scheduler.get_request(request_id)
        if req is None:
            return False
        self.scheduler.abort_request(request_id)
        return True

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def start(self) -> None:
        """Start the persistent engine loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("EngineLoop started")

    async def stop(self) -> None:
        """Stop the engine loop gracefully."""
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
            if self._feedback_mailbox is not None:
                self._check_feedback()
            await asyncio.sleep(0.001)
            return False

        try:
            # Cache check.
            if self.cache_manager is not None:
                scheduler_output = await self._filter_cached(scheduler_output)
                if scheduler_output is None:
                    return True

            # Execute.
            model_output = await self._execute_async(scheduler_output)

            # Cache update.
            if self.cache_manager is not None:
                self._update_cache_sync(scheduler_output, model_output)

            # Update scheduler state.
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

        # Feedback check.
        self._check_feedback_for_output(scheduler_output, model_output)

        if self._feedback_mailbox is not None:
            self._check_feedback()

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
        """Execute model forward pass, choosing sync+thread or direct."""
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

    def _check_feedback_for_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> None:
        """Set WAITING_FEEDBACK for requests that need it."""
        iter_ctrl = self.scheduler.iteration_controller
        if not hasattr(iter_ctrl, "needs_feedback"):
            return
        for request in scheduler_output.requests:
            if request.status in (SchedulerStatus.FINISHED, SchedulerStatus.ABORTED):
                continue
            output = model_output.outputs.get(request.request_id)
            if output is not None and iter_ctrl.needs_feedback(request, output):
                request.status = SchedulerStatus.WAITING_FEEDBACK

    def _check_feedback(self) -> None:
        """Check feedback mailbox and resume WAITING_FEEDBACK requests."""
        if self._feedback_mailbox is None:
            return
        mailbox = self._feedback_mailbox
        for req_id, request in list(self.scheduler.requests.items()):
            if request.status != SchedulerStatus.WAITING_FEEDBACK:
                continue
            if not mailbox.has(req_id):
                continue
            item = mailbox.pop(req_id)
            if item is None:
                continue
            try:
                if isinstance(item, BaseException):
                    err = item if isinstance(item, Exception) else RuntimeError(str(item))
                    self.scheduler.fail_request(req_id, err)
                    continue
                # Apply feedback via iteration controller if supported.
                iter_ctrl = self.scheduler.iteration_controller
                if hasattr(iter_ctrl, "apply_feedback"):
                    data = getattr(item, "data", item)
                    iter_ctrl.apply_feedback(request, data)
                self.scheduler.resume_request(req_id)
            except Exception as e:
                logger.error("Feedback handling failed for %s: %s", req_id, e)
                with suppress(Exception):
                    self.scheduler.fail_request(
                        req_id,
                        e if isinstance(e, Exception) else RuntimeError(str(e)),
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
