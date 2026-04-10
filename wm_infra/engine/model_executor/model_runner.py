"""ModelRunner — stateless model executor.

Aligned with sglang-omni: batched sync execution via
InputPreparer -> model forward -> OutputProcessor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    from wm_infra.engine.interfaces import InputPreparer, OutputProcessor
    from wm_infra.engine.model_executor.pipeline import ComposedPipeline

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
# Pipeline model runner: sync wrapper around async ComposedPipeline
# ---------------------------------------------------------------------------


class PipelineModelRunner:
    """Sync model runner bridging async ComposedPipeline.

    ``execute()`` is synchronous and runs in a worker thread via
    ``run_in_executor``.  Inside the thread it spins up a private
    event loop (``asyncio.run``) to drive the async pipeline stages.
    """

    execute_in_thread: bool = True  # engine dispatches via run_in_executor

    def __init__(self, pipeline: ComposedPipeline) -> None:
        self.pipeline = pipeline

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs: dict[str, RequestOutput] = {}

        for request in scheduler_output.requests:
            try:
                result = self._run_one(request)
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

    def _run_one(self, request: SchedulerRequest) -> Any:
        """Run pipeline for a single request, bridging async->sync."""
        import asyncio

        try:
            import torch

            with torch.inference_mode():
                return asyncio.run(self.pipeline.run(request.data, {}))
        except ImportError:
            return asyncio.run(self.pipeline.run(request.data, {}))
