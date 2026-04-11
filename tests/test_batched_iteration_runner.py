"""Tests for per-phase batching in VideoIterationRunner."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from wm_infra.engine.model_executor.execution_state import (
    DenoiseLoopState,
    PhaseGroupKey,
    VideoExecutionState,
)
from wm_infra.engine.model_executor.iteration_runner import VideoIterationRunner
from wm_infra.engine.types import (
    SchedulerOutput,
    SchedulerRequest,
    VideoExecutionPhase,
)
from wm_infra.models.base import VideoGenerationModel
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


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
    phase: VideoExecutionPhase | None = None,
) -> SchedulerRequest:
    vgr = VideoGenerationRequest(
        prompt="test",
        height=height,
        width=width,
        frame_count=frame_count,
    )
    if phase is not None:
        state = VideoExecutionState(request=vgr)
        state.phase = phase
        data = state
    else:
        data = vgr
    return SchedulerRequest(request_id=request_id, data=data)


def _make_scheduler_output(*requests: SchedulerRequest) -> SchedulerOutput:
    return SchedulerOutput(requests=list(requests))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGroupingLogic:
    """Test 1: _group_by_phase correctly groups by (phase, h, w, frames)."""

    def test_mixed_phases_and_resolutions(self):
        r1 = _make_request("r1", phase=VideoExecutionPhase.ENCODE_TEXT)
        r2 = _make_request("r2", phase=VideoExecutionPhase.DECODE_VAE)
        r3 = _make_request("r3", phase=VideoExecutionPhase.ENCODE_TEXT)
        r4 = _make_request("r4", height=320, width=512, phase=VideoExecutionPhase.ENCODE_TEXT)

        ready = [(r, r.data) for r in [r1, r2, r3, r4]]
        groups = VideoIterationRunner._group_by_phase(ready)

        assert len(groups) == 3
        # r1 and r3 share (ENCODE_TEXT, 640, 1024, 16)
        encode_key = PhaseGroupKey(VideoExecutionPhase.ENCODE_TEXT, 640, 1024, 16)
        assert len(groups[encode_key]) == 2
        assert groups[encode_key][0][0].request_id == "r1"
        assert groups[encode_key][1][0].request_id == "r3"


class TestSingleRequestFallback:
    """Test 2: A batch of 1 works identically to the old serial path."""

    def test_single_request_advances(self):
        model = MockVideoModel()
        runner = VideoIterationRunner(model)

        r1 = _make_request("r1")
        output = runner.execute(_make_scheduler_output(r1))

        assert output.outputs["r1"].finished is False
        assert output.outputs["r1"].finish_reason is None
        state = r1.data
        assert isinstance(state, VideoExecutionState)
        assert state.phase == VideoExecutionPhase.ENCODE_CONDITIONING
        assert model.calls == [("batch_encode_text", 1)]


class TestMultiRequestSamePhase:
    """Test 3: 3 requests at ENCODE_TEXT → 1 batch call with batch_size=3."""

    def test_three_requests_batched(self):
        model = MockVideoModel()
        runner = VideoIterationRunner(model)

        reqs = [_make_request(f"r{i}") for i in range(3)]
        output = runner.execute(_make_scheduler_output(*reqs))

        assert model.calls == [("batch_encode_text", 3)]
        for r in reqs:
            assert r.data.phase == VideoExecutionPhase.ENCODE_CONDITIONING


class TestMultiPhaseSplitting:
    """Test 4: 2 at ENCODE_TEXT + 2 at DECODE_VAE → 2 batch calls."""

    def test_two_groups(self):
        model = MockVideoModel()
        runner = VideoIterationRunner(model)

        r1 = _make_request("r1", phase=VideoExecutionPhase.ENCODE_TEXT)
        r2 = _make_request("r2", phase=VideoExecutionPhase.ENCODE_TEXT)
        r3 = _make_request("r3", phase=VideoExecutionPhase.DECODE_VAE)
        r4 = _make_request("r4", phase=VideoExecutionPhase.DECODE_VAE)

        output = runner.execute(_make_scheduler_output(r1, r2, r3, r4))

        method_names = {name for name, _ in model.calls}
        assert "batch_encode_text" in method_names
        assert "batch_decode_vae" in method_names
        assert len(model.calls) == 2

        # Check correct batch sizes
        for name, size in model.calls:
            assert size == 2


class TestResolutionSplitting:
    """Test 5: 2x1024x640 + 1x512x320 → 2 groups."""

    def test_different_resolutions(self):
        model = MockVideoModel()
        runner = VideoIterationRunner(model)

        r1 = _make_request("r1", height=640, width=1024)
        r2 = _make_request("r2", height=640, width=1024)
        r3 = _make_request("r3", height=320, width=512)

        output = runner.execute(_make_scheduler_output(r1, r2, r3))

        assert len(model.calls) == 2
        sizes = sorted(size for _, size in model.calls)
        assert sizes == [1, 2]


class TestErrorIsolation:
    """Test 6: _ensure_state failure doesn't kill other requests."""

    def test_bad_request_doesnt_block_others(self):
        model = MockVideoModel()
        runner = VideoIterationRunner(model)

        # r1 is normal
        r1 = _make_request("r1")
        # r2 has broken data that will cause _ensure_state to fail
        r2 = SchedulerRequest(request_id="r2", data=None)

        output = runner.execute(_make_scheduler_output(r1, r2))

        # r2 should error
        assert output.outputs["r2"].finished is True
        assert output.outputs["r2"].finish_reason == "error"
        # r1 should succeed normally
        assert output.outputs["r1"].finished is False
        assert r1.data.phase == VideoExecutionPhase.ENCODE_CONDITIONING


class TestDenoiseStepBatching:
    """Test 7: Different current_step, same shape → one batch group."""

    def test_denoise_step_same_shape_batched(self):
        model = MockVideoModelWithPerStepDenoise()
        runner = VideoIterationRunner(model)

        # Create two requests already at DENOISE_STEP with different current_step
        r1 = _make_request("r1", phase=VideoExecutionPhase.DENOISE_STEP)
        r1.data.supports_per_step_denoise = True
        r1.data.denoise_state = DenoiseLoopState(total_steps=5, current_step=1)

        r2 = _make_request("r2", phase=VideoExecutionPhase.DENOISE_STEP)
        r2.data.supports_per_step_denoise = True
        r2.data.denoise_state = DenoiseLoopState(total_steps=5, current_step=3)

        output = runner.execute(_make_scheduler_output(r1, r2))

        # Should be one batch call
        assert model.calls == [("batch_denoise_step", 2)]
        # current_step should have advanced
        assert r1.data.denoise_state.current_step == 2
        assert r2.data.denoise_state.current_step == 4
        # Neither is done yet
        assert r1.data.phase == VideoExecutionPhase.DENOISE_STEP
        assert r2.data.phase == VideoExecutionPhase.DENOISE_STEP
