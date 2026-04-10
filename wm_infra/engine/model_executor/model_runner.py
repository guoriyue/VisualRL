"""ModelRunner — stateless model executor.

Aligned with sglang-omni: batched sync execution via
InputPreparer -> model forward -> OutputProcessor.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    from wm_infra.engine.interfaces import InputPreparer, OutputProcessor

logger = logging.getLogger(__name__)


class ModelRunner:
    """Batched stateless model executor.

    Uses InputPreparer to convert SchedulerOutput into model input
    tensors, runs a single batched forward pass, then uses
    OutputProcessor to split results back into per-request outputs.

    ``execute()`` is synchronous — the engine calls it via
    ``run_in_executor()`` so the event loop stays responsive.
    """

    def __init__(
        self,
        model: Any,
        input_preparer: InputPreparer,
        output_processor: OutputProcessor,
        *,
        device: Any = "cuda",
    ) -> None:
        import torch

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.input_preparer = input_preparer
        self.output_processor = output_processor
        self.model = model.to(device)
        self.model.eval()

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run one batched forward pass (sync)."""
        import torch

        if not scheduler_output.requests:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_inputs = self.input_preparer.prepare(scheduler_output, self.device)

        if isinstance(model_inputs, dict) and model_inputs.get("_skip_all"):
            model_output = {}
        else:
            with torch.inference_mode():
                model_output = self.model(**model_inputs)

        outputs: dict[str, RequestOutput] = self.output_processor.process(
            model_output, scheduler_output
        )

        req_ids = [r.request_id for r in scheduler_output.requests]
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )


# ---------------------------------------------------------------------------
# Compat layer: async callable model runner for pipeline-style execution
# ---------------------------------------------------------------------------

ModelFn = Callable[[SchedulerRequest], Awaitable[Any]]


class CallableModelRunner:
    """Wraps an async per-request callable into the ModelRunner interface.

    Used when the model is a ``ComposedPipeline.run`` or similar async
    callable that processes one request at a time.  The engine calls
    ``execute_async()`` directly (no ``run_in_executor``).
    """

    # Tells the engine to call execute_async() instead of run_in_executor.
    execute_in_thread: bool = False

    def __init__(self, model_fn: ModelFn) -> None:
        self.model_fn = model_fn

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Sync execute — not supported, use execute_async()."""
        raise NotImplementedError(
            "CallableModelRunner is async-only. "
            "Use execute_async() or set execute_in_thread=False."
        )

    async def execute_async(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run the callable for each request in the batch."""
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

        req_ids = [r.request_id for r in scheduler_output.requests]
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    async def _run_one(self, request: SchedulerRequest) -> Any:
        """Run the model function for a single request."""
        try:
            import torch

            with torch.inference_mode():
                return await self.model_fn(request)
        except ImportError:
            return await self.model_fn(request)
