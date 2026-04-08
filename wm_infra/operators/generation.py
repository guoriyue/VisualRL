"""Generation-model invocation operators."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wm_infra.runtime import (
    CallableGenerationStage,
    ComposedGenerationPipeline,
    GenerationPipelineRun,
    GenerationPipelineStageSpec,
    GenerationRuntimeBackend,
    GenerationRuntimeConfig,
    GenerationStageUpdate,
)
from wm_infra.operators.base import ModelOperator, OperatorFamily


@dataclass(slots=True)
class CosmosGenerationContext:
    """Execution payload for one Cosmos generation pipeline run."""

    output_dir: str | Path
    request: Any
    task_config: Any
    cosmos_config: Any


class CosmosGenerationOperator(ModelOperator):
    """Operator wrapper around Cosmos generation runners."""

    operator_name = "cosmos-runner"
    family = OperatorFamily.GENERATION

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.runtime_config = GenerationRuntimeConfig(
            model_path=getattr(runner, "model_name", None),
            local_scheduler=False,
            backend=GenerationRuntimeBackend.EXTERNAL,
        )

    @property
    def mode(self) -> str:
        return self.runner.mode

    def runtime_descriptor(self) -> dict[str, Any]:
        self.load()
        return {
            "runtime": self.runtime_config.describe(),
            "runner_mode": self.mode,
        }

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            **self.runtime_descriptor(),
        }

    def load(self) -> str:
        mode = self.runner.load()
        self.runtime_config = GenerationRuntimeConfig(
            model_path=getattr(self.runner, "model_name", None),
            local_scheduler=(mode != "nim"),
            host=None if mode != "nim" else getattr(self.runner, "base_url", None),
            output_path=None,
            backend=GenerationRuntimeBackend.EXTERNAL if mode in {"nim", "shell"} else GenerationRuntimeBackend.NATIVE,
        )
        return mode

    def generate(
        self,
        *,
        output_dir: str | Path,
        request: Any,
        task_config: Any,
        cosmos_config: Any,
    ) -> Any:
        return self.runner.run(
            output_dir=output_dir,
            request=request,
            task_config=task_config,
            cosmos_config=cosmos_config,
        )

    async def generate_pipeline(
        self,
        *,
        output_dir: str | Path,
        request: Any,
        task_config: Any,
        cosmos_config: Any,
    ) -> GenerationPipelineRun:
        context = CosmosGenerationContext(
            output_dir=output_dir,
            request=request,
            task_config=task_config,
            cosmos_config=cosmos_config,
        )
        self.load()

        async def _run_infer(ctx: CosmosGenerationContext, _state: dict[str, Any]) -> GenerationStageUpdate:
            result = await asyncio.to_thread(
                self.runner.run,
                output_dir=ctx.output_dir,
                request=ctx.request,
                task_config=ctx.task_config,
                cosmos_config=ctx.cosmos_config,
            )
            return GenerationStageUpdate(
                state_updates={"_pipeline_output": result},
                runtime_state_updates={"output_path": result.output_path},
                outputs={
                    "runner_mode": result.mode,
                    "output_path": result.output_path,
                    "elapsed_s": result.elapsed_s,
                },
                notes=[
                    "Cosmos generation ran through a composed generation pipeline.",
                ],
            )

        pipeline = ComposedGenerationPipeline(
            pipeline_name="cosmos-generation",
            execution_backend="single_stage_generation_runner",
            stages=[
                CallableGenerationStage(
                    GenerationPipelineStageSpec(
                        name="infer",
                        component="cosmos_runner",
                        device="external",
                        worker="runner",
                    ),
                    _run_infer,
                )
            ],
            runtime_config=self.runtime_config,
            initial_log_lines=[
                "Cosmos generation pipeline started.",
                f"Runner mode={self.mode}.",
            ],
            build_metadata=lambda stage_records, _runtime_state: {
                "runner_mode": self.mode,
                "supports_cross_request_batching": False,
                "stage_family": "generation",
            },
        )
        return await pipeline.run(context)


class WanInProcessGenerationOperator(ModelOperator):
    """Operator wrapper for in-process stage-oriented Wan execution."""

    operator_name = "wan-in-process"
    family = OperatorFamily.GENERATION

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.runtime_config = GenerationRuntimeConfig(
            local_scheduler=True,
            backend=GenerationRuntimeBackend.NATIVE,
        )

    def runtime_descriptor(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_config.describe(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            **self.runtime_descriptor(),
        }

    async def generate(self, context: Any) -> Any:
        return await self.scheduler.run(context)
