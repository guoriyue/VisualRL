"""Tests for EngineLoop with SimpleResourceManager."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vrl.engine import (
    ContinuousBatchPlanner,
    EngineLoop,
    Scheduler,
    SimpleResourceManager,
)
from vrl.engine.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
)
from vrl.models.base import VideoGenerationRequest


class _EchoRunner:
    """Trivial runner that echoes request data as finished output."""

    execute_in_thread = False

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        outputs = {}
        for req in scheduler_output.requests:
            outputs[req.request_id] = RequestOutput(
                request_id=req.request_id,
                data=req.data,
                finished=True,
                finish_reason="completed",
            )
        req_ids = [r.request_id for r in scheduler_output.requests]
        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )


def _build_engine(max_count: int = 32, max_batch_size: int = 32) -> EngineLoop:
    return EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=max_batch_size),
            resource_manager=SimpleResourceManager(max_count=max_count),
        ),
        model_runner=_EchoRunner(),
    )


@pytest.mark.asyncio
async def test_engine_loop_basic():
    """Single request flows through EngineLoop to completion."""
    engine = _build_engine()
    await engine.start()
    try:
        await engine.add_request("req-1", {"hello": "world"})
        result = await asyncio.wait_for(engine.get_result("req-1"), timeout=2.0)
        assert result == {"hello": "world"}
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_resource_manager_limits_admission():
    """SimpleResourceManager with max_count=1 only admits one request at a time."""
    resource_manager = SimpleResourceManager(max_count=1)
    engine = EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=10),
            resource_manager=resource_manager,
        ),
        model_runner=_EchoRunner(),
    )
    await engine.start()
    try:
        # Submit two requests
        await engine.add_request("req-1", "data-1")
        await engine.add_request("req-2", "data-2")

        # Both should eventually complete since the echo runner finishes immediately
        r1 = await asyncio.wait_for(engine.get_result("req-1"), timeout=2.0)
        r2 = await asyncio.wait_for(engine.get_result("req-2"), timeout=2.0)

        assert r1 == "data-1"
        assert r2 == "data-2"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_loop_abort():
    """Abort cancels a request."""
    engine = _build_engine()
    await engine.start()
    try:
        await engine.add_request("req-1", "data")
        engine.abort_request("req-1")
        # get_result should raise since the request was aborted with an error
        # (abort sets status=ABORTED but no error, so it returns the request)
        request = await asyncio.wait_for(
            engine.scheduler.get_result("req-1"), timeout=2.0
        )
        from vrl.engine.types import SchedulerStatus

        assert request.status == SchedulerStatus.ABORTED
    finally:
        await engine.stop()
