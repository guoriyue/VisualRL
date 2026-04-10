"""ComposedPipeline — stage assembly and sequential execution."""

from __future__ import annotations

import time
from typing import Any

from wm_infra.engine.model_executor.config import PipelineConfig
from wm_infra.engine.model_executor.stages.base import PipelineStage
from wm_infra.models.video_generation import StageResult, VideoGenerationRequest


class ComposedPipeline:
    """Assembles and executes a sequence of ``PipelineStage`` objects.

    Usage::

        pipeline = ComposedPipeline(
            stages=[
                MyTextEncodingStage(...),
                MyConditioningStage(...),
                DiffusersDenoisingStage(...),
                PassthroughDecodeStage(),
                Uint8PostprocessStage(),
            ],
            config=PipelineConfig(device_id=0),
        )
        results = await pipeline.run(request, state)

    The pipeline threads ``state`` through each stage, applying
    ``result.state_updates`` after every stage so that downstream
    stages see upstream outputs.
    """

    def __init__(
        self,
        stages: list[PipelineStage] | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.stages: list[PipelineStage] = list(stages) if stages else []
        self.config = config or PipelineConfig()

    def add_stage(self, stage: PipelineStage) -> None:
        """Append a stage to the pipeline."""
        self.stages.append(stage)

    async def run(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any] | None = None,
    ) -> list[StageResult]:
        """Execute all stages in order, threading state through each."""
        if state is None:
            state = {}
        state.setdefault("_pipeline_config", self.config)

        results: list[StageResult] = []
        for stage in self.stages:
            await stage.verify_input(request, state)
            started = time.perf_counter()
            result = await stage.forward(request, state)
            elapsed = time.perf_counter() - started

            # Apply state updates so downstream stages see them
            state.update(result.state_updates)

            # Record timing in runtime metadata
            result.runtime_state_updates.setdefault(
                "_stage_elapsed_s", round(elapsed, 6)
            )
            result.runtime_state_updates.setdefault("_stage_name", stage.name)

            await stage.verify_output(request, state, result)
            results.append(result)

        return results

    async def run_stage(
        self,
        stage_name: str,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Execute a single named stage."""
        for stage in self.stages:
            if stage.name == stage_name:
                await stage.verify_input(request, state)
                result = await stage.forward(request, state)
                state.update(result.state_updates)
                await stage.verify_output(request, state, result)
                return result
        raise KeyError(f"No stage named {stage_name!r} in pipeline")

    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable pipeline metadata."""
        return {
            "pipeline_class": self.__class__.__name__,
            "config": {
                "device_id": self.config.device_id,
                "dtype": self.config.dtype,
                "enable_cpu_offload": self.config.enable_cpu_offload,
            },
            "stages": [s.describe() for s in self.stages],
        }

    def __repr__(self) -> str:
        stage_names = [s.name for s in self.stages]
        return f"<ComposedPipeline stages={stage_names}>"
