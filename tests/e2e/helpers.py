"""Helpers for real-model staged-generation tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from wm_infra.engine.interfaces import (
    FIFOBatchPlanner,
    SimpleResourceManager,
    SinglePassIterationController,
)
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.managers.scheduler import Scheduler
from wm_infra.engine.model_executor.model_runner import PipelineModelRunner
from wm_infra.engine.model_executor.pipeline import ComposedPipeline
from wm_infra.engine.model_executor.stages.base import PipelineStage
from wm_infra.models.base import VideoGenerationModel
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class ModelMethodStage(PipelineStage):
    """Wrap a single ``VideoGenerationModel`` stage method as a pipeline stage."""

    def __init__(self, name: str, method) -> None:
        self.name = name
        self._method = method

    async def forward(
        self, request: VideoGenerationRequest, state: dict[str, Any]
    ) -> StageResult:
        return await self._method(request, state)


def model_to_pipeline(model: VideoGenerationModel) -> ComposedPipeline:
    """Build the canonical five-stage pipeline for a staged-generation model."""
    return ComposedPipeline(
        stages=[
            ModelMethodStage("encode_text", model.encode_text),
            ModelMethodStage("encode_conditioning", model.encode_conditioning),
            ModelMethodStage("denoise", model.denoise),
            ModelMethodStage("decode_vae", model.decode_vae),
            ModelMethodStage("postprocess", model.postprocess),
        ]
    )


def require_real_model_opt_in() -> None:
    """Skip unless real-model tests were explicitly enabled."""
    if os.environ.get("WM_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set WM_RUN_REAL_MODEL_TESTS=1 to run real-model E2E tests.")


def require_cuda() -> None:
    """Skip unless CUDA is available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-model E2E tests.")


def resolve_hf_snapshot(cache_name: str) -> Path | None:
    """Resolve a local Hugging Face cache snapshot directory by cache name."""
    root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    ref = root / "refs" / "main"
    if not ref.exists():
        return None
    revision = ref.read_text().strip()
    snapshot = root / "snapshots" / revision
    if not snapshot.exists():
        return None
    return snapshot


def build_engine(model) -> EngineLoop:
    """Build a single-request engine around a staged-generation model."""
    pipeline = model_to_pipeline(model)

    return EngineLoop(
        scheduler=Scheduler(
            batch_planner=FIFOBatchPlanner(max_batch_size=1),
            resource_manager=SimpleResourceManager(max_concurrent=1),
            iteration_controller=SinglePassIterationController(),
        ),
        model_runner=PipelineModelRunner(pipeline),
    )
