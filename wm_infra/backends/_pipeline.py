"""Pipeline framework for composed generation backends.

Extracted from the former ``wm_infra.runtime`` package so that Wan and Cosmos
backends can share the stage/pipeline primitives without a top-level module.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from wm_infra.models.video_generation import VideoGenerationModel


# ---------------------------------------------------------------------------
# Server args / runtime config
# ---------------------------------------------------------------------------

class GenerationRuntimeBackend(str, Enum):
    """Execution backend family for one runtime."""

    NATIVE = "native"
    EXTERNAL = "external"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class GenerationServerArgs:
    """Runtime shape for a generation server/process topology."""

    model_path: str | None = None
    host: str | None = None
    port: int | None = None
    scheduler_port: int | None = None
    num_gpus: int = 1
    tp_size: int = 1
    sp_degree: int = 1
    enable_cfg_parallel: bool = False
    warmup: bool = False
    local_scheduler: bool = True
    attention_backend: str | None = None
    output_path: str | None = None
    backend: GenerationRuntimeBackend = GenerationRuntimeBackend.NATIVE

    @property
    def local_mode(self) -> bool:
        return self.host is None or self.port is None

    def describe(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "scheduler_port": self.scheduler_port,
            "num_gpus": self.num_gpus,
            "tp_size": self.tp_size,
            "sp_degree": self.sp_degree,
            "enable_cfg_parallel": self.enable_cfg_parallel,
            "warmup": self.warmup,
            "local_scheduler": self.local_scheduler,
            "local_mode": self.local_mode,
            "attention_backend": self.attention_backend,
            "output_path": self.output_path,
            "backend": self.backend.value,
        }


GenerationRuntimeConfig = GenerationServerArgs


# ---------------------------------------------------------------------------
# Pipeline stage primitives
# ---------------------------------------------------------------------------

class StageParallelismType(str, Enum):
    """Execution semantics borrowed from SGLang Diffusion."""

    REPLICATED = "replicated"
    MAIN_RANK_ONLY = "main_rank_only"
    CFG_PARALLEL = "cfg_parallel"


@dataclass(frozen=True, slots=True)
class StageProfile:
    """Lightweight per-stage timing profile."""

    elapsed_ms: float


class PipelineStage(ABC):
    """Base class for one pipeline stage."""

    parallelism: StageParallelismType = StageParallelismType.REPLICATED

    def verify_input(self, _context: Any, _state: dict[str, Any]) -> None:
        """Stage-specific input validation hook."""

    def verify_output(self, _output: Any) -> None:
        """Stage-specific output validation hook."""

    @abstractmethod
    async def forward(self, context: Any, state: dict[str, Any]) -> Any:
        """Run the stage and return a stage update."""

    async def __call__(self, context: Any, state: dict[str, Any]) -> Any:
        self.verify_input(context, state)
        started = time.perf_counter()
        output = await self.forward(context, state)
        self.verify_output(output)
        profile = StageProfile(elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2))
        outputs = getattr(output, "outputs", None)
        if isinstance(outputs, dict):
            outputs.setdefault("stage_profile", {"elapsed_ms": profile.elapsed_ms})
        else:
            try:
                setattr(output, "_stage_profile", profile)
            except AttributeError:
                pass
        return output


# ---------------------------------------------------------------------------
# Composed pipeline base
# ---------------------------------------------------------------------------

class ComposedPipelineBase:
    """Own the ordered stage list and delegate execution."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        server_args: GenerationServerArgs | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.server_args = server_args or GenerationServerArgs()
        self._stages: list[Any] = []

    @property
    def stages(self) -> list[Any]:
        return list(self._stages)

    def add_stage(self, stage: Any) -> None:
        self._stages.append(stage)


# ---------------------------------------------------------------------------
# Generation pipeline types
# ---------------------------------------------------------------------------

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
    execution_state: dict[str, Any] = field(default_factory=dict)
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


def build_video_generation_stages(
    *,
    model: VideoGenerationModel,
    request_key: str = "vg_request",
    component_prefix: str = "model",
    device_map: dict[str, str] | None = None,
    worker_map: dict[str, str] | None = None,
) -> list[CallableGenerationStage]:
    """Build canonical generation stages for a ``VideoGenerationModel``.

    This keeps model stage ordering in the shared generation substrate rather
    than re-encoding it in each backend adapter.
    """

    stage_defs = [
        ("encode_text", model.encode_text),
        ("encode_conditioning", model.encode_conditioning),
        ("denoise", model.denoise),
        ("decode_vae", model.decode_vae),
        ("postprocess", model.postprocess),
    ]
    resolved_device_map = device_map or {}
    resolved_worker_map = worker_map or {}
    stages: list[CallableGenerationStage] = []
    for stage_name, stage_fn in stage_defs:
        spec = GenerationPipelineStageSpec(
            name=stage_name,
            component=f"{component_prefix}-{stage_name.replace('_', '-')}",
            device=resolved_device_map.get(stage_name, "cpu"),
            worker=resolved_worker_map.get(stage_name, "model"),
        )

        async def _run_stage(
            context: Any,
            state: dict[str, Any],
            *,
            _stage_fn: Callable[[Any, dict[str, Any]], Awaitable[Any]] = stage_fn,
        ) -> Any:
            if isinstance(context, dict):
                request = context[request_key]
            else:
                request = getattr(context, request_key)
            return await _stage_fn(request, state)

        stages.append(CallableGenerationStage(spec, _run_stage))
    return stages


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
        )
        self.execution_backend = execution_backend
        self._before_stage = before_stage
        self._after_stage = after_stage
        self._build_metadata = build_metadata
        self._initial_log_lines = list(initial_log_lines or [])
        for stage in stages:
            self.add_stage(stage)

    async def run(self, context: Any) -> GenerationPipelineRun:
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
            execution_state=dict(execution_state),
            output=execution_state.get("_pipeline_output"),
        )


__all__ = [
    "CallableGenerationStage",
    "build_video_generation_stages",
    "ComposedGenerationPipeline",
    "ComposedPipelineBase",
    "GenerationPipelineRun",
    "GenerationPipelineStageSpec",
    "GenerationRuntimeBackend",
    "GenerationRuntimeConfig",
    "GenerationServerArgs",
    "GenerationStageUpdate",
    "PipelineStage",
    "StageParallelismType",
    "StageProfile",
]
