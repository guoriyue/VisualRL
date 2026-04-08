"""Concrete composed generation pipeline built on the runtime substrate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from wm_infra.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from wm_infra.runtime.pipelines_core.executors.sync_executor import SyncPipelineExecutor
from wm_infra.runtime.pipelines_core.stages.base import PipelineStage, StageParallelismType
from wm_infra.runtime.server_args import GenerationServerArgs


GenerationRuntimeConfig = GenerationServerArgs


@dataclass(frozen=True, slots=True)
class GenerationPipelineStageSpec:
    """One named stage in a composed generation pipeline."""

    name: str
    component: str
    device: str
    worker: str
    optional: bool = False


@dataclass(slots=True)
class GenerationStageUpdate:
    """Normalized payload returned by one stage."""

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"


@dataclass(slots=True)
class GenerationPipelineRun:
    """Structured result from a composed pipeline execution."""

    pipeline_metadata: dict[str, Any]
    stage_records: list[dict[str, Any]]
    stage_state: dict[str, Any]
    log_text: str
    output: Any | None = None


class CallableGenerationStage(PipelineStage):
    """Adapter that turns an async callable into a pipeline stage."""

    parallelism = StageParallelismType.REPLICATED

    def __init__(
        self,
        spec: GenerationPipelineStageSpec,
        run_fn: Callable[[Any, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.spec = spec
        self._run_fn = run_fn

    async def forward(self, context: Any, state: dict[str, Any]) -> Any:
        return await self._run_fn(context, state)


BeforeStageHook = Callable[[GenerationPipelineStageSpec, Any, dict[str, Any]], dict[str, Any] | None]
AfterStageHook = Callable[
    [GenerationPipelineStageSpec, Any, dict[str, Any], dict[str, Any], Any],
    dict[str, Any] | None,
]
BuildMetadataHook = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any] | None]


class ComposedGenerationPipeline(ComposedPipelineBase):
    """Stateful generation pipeline with stage records and runtime metadata."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        execution_backend: str,
        stages: list[CallableGenerationStage],
        runtime_config: GenerationRuntimeConfig | None = None,
        before_stage: BeforeStageHook | None = None,
        after_stage: AfterStageHook | None = None,
        build_metadata: BuildMetadataHook | None = None,
        initial_log_lines: list[str] | None = None,
    ) -> None:
        super().__init__(
            pipeline_name=pipeline_name,
            server_args=runtime_config or GenerationRuntimeConfig(),
            executor=SyncPipelineExecutor(),
        )
        self.execution_backend = execution_backend
        self._before_stage = before_stage
        self._after_stage = after_stage
        self._build_metadata = build_metadata
        self._initial_log_lines = list(initial_log_lines or [])
        for stage in stages:
            self.add_stage(stage)

    async def run(self, context: Any) -> GenerationPipelineRun:
        import time

        execution_state: dict[str, Any] = {}
        runtime_state: dict[str, Any] = {}
        stage_records: list[dict[str, Any]] = []
        log_lines = list(self._initial_log_lines)

        for stage in self.stages:
            spec = stage.spec
            stage_record: dict[str, Any] = {}
            if self._before_stage is not None:
                stage_record.update(self._before_stage(spec, context, execution_state) or {})

            update = await stage(context, execution_state)
            profile = getattr(update, "_stage_profile", None)
            if profile is not None:
                elapsed_ms = profile.elapsed_ms
            else:
                outputs = getattr(update, "outputs", {})
                stage_profile = outputs.get("stage_profile", {}) if isinstance(outputs, dict) else {}
                elapsed_ms = float(stage_profile.get("elapsed_ms", 0.0))
            now = time.time()
            stage_record.update(
                {
                    "name": spec.name,
                    "component": spec.component,
                    "device": spec.device,
                    "worker": spec.worker,
                    "optional": spec.optional,
                    "status": getattr(update, "status", "succeeded"),
                    "cache_hit": getattr(update, "cache_hit", None),
                    "started_at": now - (elapsed_ms / 1000.0),
                    "completed_at": now,
                    "elapsed_ms": elapsed_ms,
                    "outputs": getattr(update, "outputs", {}),
                    "notes": list(getattr(update, "notes", [])),
                }
            )
            execution_state.update(getattr(update, "state_updates", {}))
            runtime_state.update(getattr(update, "runtime_state_updates", {}))
            if self._after_stage is not None:
                stage_record.update(
                    self._after_stage(spec, context, execution_state, runtime_state, update) or {}
                )
            stage_records.append(stage_record)
            log_lines.append(
                f"[{spec.name}] device={spec.device} worker={spec.worker} "
                f"status={stage_record['status']} elapsed_ms={elapsed_ms}"
            )
            for note in stage_record["notes"]:
                log_lines.append(f"[{spec.name}] {note}")

        metadata = {
            "pipeline_name": self.pipeline_name,
            "execution_backend": self.execution_backend,
            "stage_count": len(stage_records),
            "stage_sequence": [stage["name"] for stage in stage_records],
            "stage_devices": {stage["name"]: stage["device"] for stage in stage_records},
            "runtime_config": self.server_args.describe(),
        }
        if self._build_metadata is not None:
            metadata.update(self._build_metadata(stage_records, runtime_state) or {})

        return GenerationPipelineRun(
            pipeline_metadata=metadata,
            stage_records=stage_records,
            stage_state=runtime_state,
            log_text="\n".join(log_lines) + "\n",
            output=execution_state.get("_pipeline_output"),
        )
