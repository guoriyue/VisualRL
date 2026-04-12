"""Helpers for real-model staged-generation tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vrl.engine.managers.batch_planner import ContinuousBatchPlanner
from vrl.engine.managers.resource_manager import SimpleResourceManager
from vrl.engine.managers.engine_loop import EngineLoop
from vrl.engine.managers.scheduler import Scheduler
from vrl.engine.model_executor.iteration_runner import PipelineRunner
from vrl.models.base import VideoGenerationModel


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
    return EngineLoop(
        scheduler=Scheduler(
            batch_planner=ContinuousBatchPlanner(max_batch_size=1),
            resource_manager=SimpleResourceManager(max_count=1),
        ),
        model_runner=PipelineRunner(model),
    )
