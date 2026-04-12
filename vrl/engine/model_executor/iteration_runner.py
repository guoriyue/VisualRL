"""PipelineRunner: full pipeline per execute(), grouped by workload signature."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from vrl.engine.model_executor.execution_state import WorkloadSignature
from vrl.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from vrl.models.base import VideoGenerationModel
from vrl.schemas.video_generation import StageResult, VideoGenerationRequest

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Whole-request pipeline runner: runs all stages in one execute() call."""

    execute_in_thread: bool = True

    def __init__(self, model: VideoGenerationModel) -> None:
        self.model = model

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs: dict[str, RequestOutput] = {}

        # Validate requests; isolate failures early
        ready: list[tuple[SchedulerRequest, VideoGenerationRequest]] = []
        for request in scheduler_output.requests:
            try:
                vgr = self._extract_request(request)
                ready.append((request, vgr))
            except Exception as exc:
                logger.error(
                    "Request %s failed during init: %s", request.request_id, exc
                )
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=None,
                    finished=True,
                    finish_reason="error",
                )
                request.error = exc

        # Group by workload signature
        groups = self._group_by_workload(ready)

        # Run full pipeline per group
        for _sig, members in groups.items():
            try:
                group_outputs = self._run_pipeline(members)
                for (sched_req, _vgr), output in zip(members, group_outputs):
                    outputs[sched_req.request_id] = output
            except Exception as exc:
                logger.error("Pipeline group failed: %s", exc)
                for sched_req, _vgr in members:
                    outputs[sched_req.request_id] = RequestOutput(
                        request_id=sched_req.request_id,
                        data=None,
                        finished=True,
                        finish_reason="error",
                    )
                    sched_req.error = exc

        req_ids = [r.request_id for r in scheduler_output.requests]
        req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    def _extract_request(self, request: SchedulerRequest) -> VideoGenerationRequest:
        if isinstance(request.data, VideoGenerationRequest):
            return request.data
        raise TypeError(
            f"Expected VideoGenerationRequest, got {type(request.data).__name__}"
        )

    @staticmethod
    def _group_by_workload(
        ready: list[tuple[SchedulerRequest, VideoGenerationRequest]],
    ) -> dict[WorkloadSignature, list[tuple[SchedulerRequest, VideoGenerationRequest]]]:
        groups: dict[
            WorkloadSignature,
            list[tuple[SchedulerRequest, VideoGenerationRequest]],
        ] = defaultdict(list)
        for sched_req, vgr in ready:
            sig = WorkloadSignature(
                model_name=vgr.model_name,
                task_type=vgr.task_type,
                height=vgr.height,
                width=vgr.width,
                frame_count=vgr.frame_count,
                num_steps=vgr.num_steps,
            )
            groups[sig].append((sched_req, vgr))
        return dict(groups)

    def _run_pipeline(
        self,
        members: list[tuple[SchedulerRequest, VideoGenerationRequest]],
    ) -> list[RequestOutput]:
        requests = [vgr for _, vgr in members]
        states: list[dict[str, Any]] = [{} for _ in members]
        all_stage_results: list[list[StageResult]] = [[] for _ in members]

        # 1. encode_text
        results = asyncio.run(self.model.batch_encode_text(requests, states))
        for i, result in enumerate(results):
            states[i].update(result.state_updates)
            all_stage_results[i].append(result)

        # 2. encode_conditioning
        results = asyncio.run(self.model.batch_encode_conditioning(requests, states))
        for i, result in enumerate(results):
            states[i].update(result.state_updates)
            all_stage_results[i].append(result)

        # 3. generate
        results = asyncio.run(self.model.batch_generate(requests, states))
        for i, result in enumerate(results):
            states[i].update(result.state_updates)
            all_stage_results[i].append(result)

        # 4. decode_vae
        results = asyncio.run(self.model.batch_decode_vae(requests, states))
        for i, result in enumerate(results):
            states[i].update(result.state_updates)
            all_stage_results[i].append(result)

        # 5. postprocess
        results = asyncio.run(self.model.batch_postprocess(requests, states))
        for i, result in enumerate(results):
            states[i].update(result.state_updates)
            all_stage_results[i].append(result)

        # Package outputs — always finished
        pipeline_outputs: list[RequestOutput] = []
        for (sched_req, _vgr), stage_results in zip(members, all_stage_results):
            pipeline_outputs.append(
                RequestOutput(
                    request_id=sched_req.request_id,
                    data=stage_results,
                    finished=True,
                    finish_reason="completed",
                )
            )
        return pipeline_outputs
