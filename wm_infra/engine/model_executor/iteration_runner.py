"""VideoIterationRunner — per-step model executor for iterative video generation.

``VideoIterationRunner`` advances ONE execution phase per ``execute()``
call.  The engine loops ``schedule() -> execute() -> update()`` many
times per request, giving it per-step visibility for batching,
pause/resume, branching, and streaming.

Requests at the same phase with the same resolution are grouped and
dispatched as a single batched model call via ``model.batch_*`` methods.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from wm_infra.engine.model_executor.execution_state import (
    PhaseGroupKey,
    VideoExecutionState,
)
from wm_infra.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
    VideoExecutionPhase,
)
from wm_infra.models.base import VideoGenerationModel
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest

logger = logging.getLogger(__name__)


class VideoIterationRunner:
    """Per-step model runner that advances one phase per execute() call.

    ``execute()`` is synchronous and runs in a worker thread via
    ``run_in_executor``.  Inside the thread it uses ``asyncio.run()``
    to bridge async model methods.

    Requests at the same ``(phase, height, width, frame_count)`` are
    grouped and dispatched as a single batch call.
    """

    execute_in_thread: bool = True

    def __init__(self, model: VideoGenerationModel) -> None:
        self.model = model
        self._supports_per_step: bool | None = None

    def _check_per_step_support(self) -> bool:
        if self._supports_per_step is None:
            try:
                self.model.denoise_init
                self.model.denoise_step
                self.model.denoise_finalize
                # Check that they aren't the default NotImplementedError stubs
                # by verifying the method is overridden on the concrete class
                base_init = VideoGenerationModel.denoise_init
                self._supports_per_step = type(self.model).denoise_init is not base_init
            except AttributeError:
                self._supports_per_step = False
        return self._supports_per_step

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs: dict[str, RequestOutput] = {}

        # Phase 1: ensure state for all requests; isolate failures
        ready: list[tuple[SchedulerRequest, VideoExecutionState]] = []
        for request in scheduler_output.requests:
            try:
                state = self._ensure_state(request)
                ready.append((request, state))
            except Exception as exc:
                logger.error("Request %s failed during init: %s", request.request_id, exc)
                outputs[request.request_id] = RequestOutput(
                    request_id=request.request_id,
                    data=None,
                    finished=True,
                    finish_reason="error",
                )
                request.error = exc

        # Phase 2: group by (phase, height, width, frame_count)
        groups = self._group_by_phase(ready)

        # Phase 3: dispatch each group as a batch
        for key, members in groups.items():
            try:
                results = self._advance_batch(key, members)
                for (request, state), result in zip(members, results):
                    finished = state.phase == VideoExecutionPhase.DONE
                    outputs[request.request_id] = RequestOutput(
                        request_id=request.request_id,
                        data=result,
                        finished=finished,
                        finish_reason="completed" if finished else None,
                    )
            except Exception as exc:
                # Entire group fails atomically
                logger.error("Batch %s failed: %s", key, exc)
                for request, _state in members:
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

    def _ensure_state(self, request: SchedulerRequest) -> VideoExecutionState:
        """Wrap raw VideoGenerationRequest in VideoExecutionState on first call."""
        if isinstance(request.data, VideoExecutionState):
            return request.data
        if not isinstance(request.data, VideoGenerationRequest):
            raise TypeError(
                f"Expected VideoGenerationRequest or VideoExecutionState, "
                f"got {type(request.data).__name__}"
            )
        state = VideoExecutionState(
            request=request.data,
            supports_per_step_denoise=self._check_per_step_support(),
        )
        request.data = state
        return state

    @staticmethod
    def _group_by_phase(
        ready: list[tuple[SchedulerRequest, VideoExecutionState]],
    ) -> dict[PhaseGroupKey, list[tuple[SchedulerRequest, VideoExecutionState]]]:
        """Group requests by (phase, height, width, frame_count)."""
        groups: dict[PhaseGroupKey, list[tuple[SchedulerRequest, VideoExecutionState]]] = defaultdict(list)
        for request, state in ready:
            req: VideoGenerationRequest = state.request
            key = PhaseGroupKey(
                phase=state.phase,
                height=req.height,
                width=req.width,
                frame_count=req.frame_count,
            )
            groups[key].append((request, state))
        return dict(groups)

    def _advance_batch(
        self,
        key: PhaseGroupKey,
        members: list[tuple[SchedulerRequest, VideoExecutionState]],
    ) -> list[StageResult]:
        """Dispatch one batch call for all members and apply state updates."""
        phase = key.phase
        requests = [s.request for _, s in members]
        states_list = [s.pipeline_state for _, s in members]
        exec_states = [s for _, s in members]

        if phase == VideoExecutionPhase.ENCODE_TEXT:
            results = asyncio.run(self.model.batch_encode_text(requests, states_list))
            for state, result in zip(exec_states, results):
                state.pipeline_state.update(result.state_updates)
                state.stage_results.append(result)
                state.phase = VideoExecutionPhase.ENCODE_CONDITIONING

        elif phase == VideoExecutionPhase.ENCODE_CONDITIONING:
            results = asyncio.run(self.model.batch_encode_conditioning(requests, states_list))
            for state, result in zip(exec_states, results):
                state.pipeline_state.update(result.state_updates)
                state.stage_results.append(result)
                if state.supports_per_step_denoise:
                    state.phase = VideoExecutionPhase.DENOISE_INIT
                else:
                    state.phase = VideoExecutionPhase.DENOISE_INIT

        elif phase == VideoExecutionPhase.DENOISE_INIT:
            per_step = exec_states[0].supports_per_step_denoise
            if per_step:
                denoise_states = asyncio.run(self.model.batch_denoise_init(requests, states_list))
                results = []
                for state, ds in zip(exec_states, denoise_states):
                    state.denoise_state = ds
                    state.phase = VideoExecutionPhase.DENOISE_STEP
                    results.append(StageResult(notes=["Denoise loop initialized."]))
            else:
                # Black-box: run full denoise() in one shot
                results = asyncio.run(self.model.batch_denoise(requests, states_list))
                for state, result in zip(exec_states, results):
                    state.pipeline_state.update(result.state_updates)
                    state.stage_results.append(result)
                    state.phase = VideoExecutionPhase.DECODE_VAE

        elif phase == VideoExecutionPhase.DENOISE_STEP:
            denoise_states = [s.denoise_state for s in exec_states]
            results = asyncio.run(
                self.model.batch_denoise_step(requests, states_list, denoise_states)
            )
            for state in exec_states:
                ds = state.denoise_state
                if ds.current_step >= ds.total_steps:
                    state.phase = VideoExecutionPhase.DENOISE_FINALIZE

        elif phase == VideoExecutionPhase.DENOISE_FINALIZE:
            denoise_states = [s.denoise_state for s in exec_states]
            results = asyncio.run(
                self.model.batch_denoise_finalize(requests, states_list, denoise_states)
            )
            for state, result in zip(exec_states, results):
                state.pipeline_state.update(result.state_updates)
                state.stage_results.append(result)
                state.denoise_state = None
                state.phase = VideoExecutionPhase.DECODE_VAE

        elif phase == VideoExecutionPhase.DECODE_VAE:
            results = asyncio.run(self.model.batch_decode_vae(requests, states_list))
            for state, result in zip(exec_states, results):
                state.pipeline_state.update(result.state_updates)
                state.stage_results.append(result)
                state.phase = VideoExecutionPhase.POSTPROCESS

        elif phase == VideoExecutionPhase.POSTPROCESS:
            results = asyncio.run(self.model.batch_postprocess(requests, states_list))
            for state, result in zip(exec_states, results):
                state.pipeline_state.update(result.state_updates)
                state.stage_results.append(result)
                state.phase = VideoExecutionPhase.DONE

        else:
            raise RuntimeError(f"Unexpected phase: {phase}")

        return results
