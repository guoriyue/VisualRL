"""Tests for PipelineRunner: full-pipeline execution in one call."""

from __future__ import annotations

from typing import Any

import pytest

from vrl.engine.model_executor.execution_state import (
    DenoiseLoopState,
    WorkloadSignature,
)
from vrl.engine.model_executor.iteration_runner import PipelineRunner
from vrl.engine.types import (
    SchedulerOutput,
    SchedulerRequest,
)
from vrl.models.base import VideoGenerationModel
from vrl.schemas.video_generation import StageResult, VideoGenerationRequest


# ---------------------------------------------------------------------------
# Mock model that records batch calls
# ---------------------------------------------------------------------------


class MockVideoModel(VideoGenerationModel):
    """Minimal mock that records which batch methods are called and with how many items."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []  # (method_name, batch_size)

    async def load(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"name": "mock"}

    async def encode_text(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return StageResult(notes=["encoded text"])

    async def denoise(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return StageResult(notes=["denoised"])

    async def decode_vae(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return StageResult(notes=["decoded"])

    # Override batch methods to record calls (still delegate to sequential defaults)

    async def batch_encode_text(self, requests, states):
        self.calls.append(("batch_encode_text", len(requests)))
        return await super().batch_encode_text(requests, states)

    async def batch_encode_conditioning(self, requests, states):
        self.calls.append(("batch_encode_conditioning", len(requests)))
        return await super().batch_encode_conditioning(requests, states)

    async def batch_denoise(self, requests, states):
        self.calls.append(("batch_denoise", len(requests)))
        return await super().batch_denoise(requests, states)

    async def batch_decode_vae(self, requests, states):
        self.calls.append(("batch_decode_vae", len(requests)))
        return await super().batch_decode_vae(requests, states)

    async def batch_postprocess(self, requests, states):
        self.calls.append(("batch_postprocess", len(requests)))
        return await super().batch_postprocess(requests, states)


class MockVideoModelWithPerStepDenoise(MockVideoModel):
    """Mock that supports per-step denoise (denoise_init/step/finalize)."""

    async def denoise_init(self, request, state) -> DenoiseLoopState:
        return DenoiseLoopState(total_steps=3, current_step=0)

    async def denoise_step(self, request, state, denoise_state) -> StageResult:
        denoise_state.current_step += 1
        return StageResult(notes=[f"step {denoise_state.current_step}"])

    async def denoise_finalize(self, request, state, denoise_state) -> StageResult:
        return StageResult(notes=["finalized"])

    async def batch_denoise_init(self, requests, states):
        self.calls.append(("batch_denoise_init", len(requests)))
        return await super().batch_denoise_init(requests, states)

    async def batch_denoise_step(self, requests, states, denoise_states):
        self.calls.append(("batch_denoise_step", len(requests)))
        return await super().batch_denoise_step(requests, states, denoise_states)

    async def batch_denoise_finalize(self, requests, states, denoise_states):
        self.calls.append(("batch_denoise_finalize", len(requests)))
        return await super().batch_denoise_finalize(requests, states, denoise_states)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req-1",
    height: int = 640,
    width: int = 1024,
    frame_count: int = 16,
    num_steps: int = 35,
) -> SchedulerRequest:
    vgr = VideoGenerationRequest(
        prompt="test",
        height=height,
        width=width,
        frame_count=frame_count,
        num_steps=num_steps,
    )
    return SchedulerRequest(request_id=request_id, data=vgr)


def _make_scheduler_output(*requests: SchedulerRequest) -> SchedulerOutput:
    return SchedulerOutput(requests=list(requests))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test: single request runs all 5 stages and returns finished."""

    def test_single_request_full_pipeline(self):
        model = MockVideoModel()
        runner = PipelineRunner(model)

        r1 = _make_request("r1")
        output = runner.execute(_make_scheduler_output(r1))

        assert output.outputs["r1"].finished is True
        assert output.outputs["r1"].finish_reason == "completed"
        assert output.outputs["r1"].data is not None
        # Should have 5 stage results
        assert len(output.outputs["r1"].data) == 5
        assert model.calls == [
            ("batch_encode_text", 1),
            ("batch_encode_conditioning", 1),
            ("batch_denoise", 1),
            ("batch_decode_vae", 1),
            ("batch_postprocess", 1),
        ]


class TestBatchedPipeline:
    """Test: 3 requests with same signature → batched through all stages."""

    def test_three_requests_batched(self):
        model = MockVideoModel()
        runner = PipelineRunner(model)

        reqs = [_make_request(f"r{i}") for i in range(3)]
        output = runner.execute(_make_scheduler_output(*reqs))

        # All should be finished
        for i in range(3):
            assert output.outputs[f"r{i}"].finished is True
            assert output.outputs[f"r{i}"].finish_reason == "completed"

        # All 5 stages should be called with batch_size=3
        assert model.calls == [
            ("batch_encode_text", 3),
            ("batch_encode_conditioning", 3),
            ("batch_denoise", 3),
            ("batch_decode_vae", 3),
            ("batch_postprocess", 3),
        ]


class TestWorkloadSignatureGrouping:
    """Test: different resolutions → different groups, different batch sizes."""

    def test_different_resolutions_separate_groups(self):
        model = MockVideoModel()
        runner = PipelineRunner(model)

        r1 = _make_request("r1", height=640, width=1024)
        r2 = _make_request("r2", height=640, width=1024)
        r3 = _make_request("r3", height=320, width=512)

        output = runner.execute(_make_scheduler_output(r1, r2, r3))

        # All should be finished
        for rid in ("r1", "r2", "r3"):
            assert output.outputs[rid].finished is True

        # Should have 10 batch calls: 5 stages × 2 groups
        assert len(model.calls) == 10

        # Each stage should appear twice (once for each group)
        stage_counts: dict[str, list[int]] = {}
        for name, size in model.calls:
            stage_counts.setdefault(name, []).append(size)

        for stage_name in (
            "batch_encode_text",
            "batch_encode_conditioning",
            "batch_denoise",
            "batch_decode_vae",
            "batch_postprocess",
        ):
            sizes = sorted(stage_counts[stage_name])
            assert sizes == [1, 2]


class TestErrorIsolation:
    """Test: bad request doesn't block other requests."""

    def test_bad_request_doesnt_block_others(self):
        model = MockVideoModel()
        runner = PipelineRunner(model)

        r1 = _make_request("r1")
        r2 = SchedulerRequest(request_id="r2", data=None)  # broken data

        output = runner.execute(_make_scheduler_output(r1, r2))

        # r2 should error
        assert output.outputs["r2"].finished is True
        assert output.outputs["r2"].finish_reason == "error"
        # r1 should succeed
        assert output.outputs["r1"].finished is True
        assert output.outputs["r1"].finish_reason == "completed"


class TestPerStepDenoise:
    """Test: per-step denoise model runs init → step × N → finalize."""

    def test_per_step_denoise_pipeline(self):
        model = MockVideoModelWithPerStepDenoise()
        runner = PipelineRunner(model)

        r1 = _make_request("r1")
        output = runner.execute(_make_scheduler_output(r1))

        assert output.outputs["r1"].finished is True
        assert output.outputs["r1"].finish_reason == "completed"

        # Should have: encode_text, encode_conditioning, denoise_init, 3× denoise_step, denoise_finalize, decode_vae, postprocess
        method_names = [name for name, _ in model.calls]
        assert method_names == [
            "batch_encode_text",
            "batch_encode_conditioning",
            "batch_denoise_init",
            "batch_denoise_step",
            "batch_denoise_step",
            "batch_denoise_step",
            "batch_denoise_finalize",
            "batch_decode_vae",
            "batch_postprocess",
        ]


class TestWorkloadSignatureFields:
    """Test: WorkloadSignature captures the right fields."""

    def test_signature_grouping_by_num_steps(self):
        model = MockVideoModel()
        runner = PipelineRunner(model)

        r1 = _make_request("r1", num_steps=35)
        r2 = _make_request("r2", num_steps=50)

        output = runner.execute(_make_scheduler_output(r1, r2))

        # Different num_steps → 2 groups → 10 batch calls
        assert len(model.calls) == 10
        for rid in ("r1", "r2"):
            assert output.outputs[rid].finished is True


class TestGroupByWorkloadStatic:
    """Test the static _group_by_workload method directly."""

    def test_grouping(self):
        vgr1 = VideoGenerationRequest(prompt="a", height=640, width=1024, frame_count=16, num_steps=35)
        vgr2 = VideoGenerationRequest(prompt="b", height=640, width=1024, frame_count=16, num_steps=35)
        vgr3 = VideoGenerationRequest(prompt="c", height=320, width=512, frame_count=16, num_steps=35)

        r1 = SchedulerRequest(request_id="r1", data=vgr1)
        r2 = SchedulerRequest(request_id="r2", data=vgr2)
        r3 = SchedulerRequest(request_id="r3", data=vgr3)

        groups = PipelineRunner._group_by_workload([(r1, vgr1), (r2, vgr2), (r3, vgr3)])

        assert len(groups) == 2
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]
