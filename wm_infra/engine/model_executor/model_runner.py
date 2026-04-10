"""Stateless model runner following sglang-omni's ModelRunner pattern.

The ``ModelRunner`` wraps an arbitrary callable (e.g. ``ComposedPipeline.run``)
and executes it under ``torch.inference_mode()`` for the batch of requests
selected by the scheduler.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

logger = logging.getLogger(__name__)

ModelFn = Callable[[SchedulerRequest], Awaitable[Any]]


class ModelRunner:
    """Stateless executor that runs a model function on scheduled requests.

    Parameters
    ----------
    model_fn : ModelFn
        Async callable that takes a ``SchedulerRequest`` and returns
        an opaque result.  For pipeline-based models this is typically
        ``ComposedPipeline.run(request.data, state)``.
    """

    def __init__(self, model_fn: ModelFn) -> None:
        self.model_fn = model_fn

    async def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Execute the model function for every request in the batch.

        Uses ``torch.inference_mode()`` when torch is available.
        """
        outputs: dict[str, RequestOutput] = {}

        for request in scheduler_output.requests:
            try:
                result = await self._run_one(request)
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=result,
                    finished=True,
                    finish_reason="completed",
                )
            except Exception as exc:
                logger.error("Request %s failed: %s", request.request_id, exc)
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=None,
                    finished=True,
                    finish_reason="error",
                )
                request.error = exc

        return ModelRunnerOutput(outputs=outputs)

    async def _run_one(self, request: SchedulerRequest) -> Any:
        """Run the model function for a single request."""
        try:
            import torch

            with torch.inference_mode():
                return await self.model_fn(request)
        except ImportError:
            return await self.model_fn(request)
