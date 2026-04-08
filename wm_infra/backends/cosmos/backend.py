"""Cosmos Predict backend for world-generation sample production."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.backends.cosmos.runner import CosmosRunResult, CosmosRunner
from wm_infra.backends.cosmos.runtime import (
    CosmosExecutionEntity,
    CosmosRuntimeTrace,
    CosmosStageRecord,
    CosmosInputCache,
    prompt_reference_key,
    queue_lane_for_request,
)
from wm_infra.operators import CosmosGenerationOperator
from wm_infra.backends.cosmos.scheduler import CosmosChunkScheduler
from wm_infra.controlplane import (
    ArtifactKind,
    ArtifactRecord,
    CosmosTaskConfig,
    ProduceSampleRequest,
    RolloutTaskConfig,
    SampleRecord,
    SampleStatus,
    TaskType,
    WorldModelKind,
    estimate_cosmos_request,
)


class CosmosPredictBackend(ProduceSampleBackend):
    """Queue-friendly Cosmos backend using NIM, shell, or stub execution."""

    world_model_kind = WorldModelKind.GENERATION
    capability_flags = frozenset({"continuation", "multistage_pipeline", "video_artifact"})

    def __init__(
        self,
        output_root: str | Path,
        *,
        backend_name: str = "cosmos-predict",
        runner: CosmosRunner | None = None,
        max_chunk_size: int = 4,
    ) -> None:
        self.backend_name = backend_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner or CosmosRunner()
        self._operator = CosmosGenerationOperator(self.runner)
        self._job_queue = None
        self._scheduler = CosmosChunkScheduler(max_chunk_size=max_chunk_size)
        self._input_cache = CosmosInputCache()

    @property
    def runner_mode(self) -> str:
        return self._operator.mode

    def _sample_dir(self, sample_id: str) -> Path:
        path = self.output_root / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _file_details(path: Path) -> tuple[int | None, str | None]:
        if not path.exists() or not path.is_file():
            return None, None
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()

    def _artifact_record(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        path: Path,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        bytes_size, sha256 = self._file_details(path)
        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            uri=f"file://{path}",
            mime_type=mime_type,
            bytes=bytes_size,
            sha256=sha256,
            metadata={"exists": path.exists(), **(metadata or {})},
        )

    def _effective_task_config(self, request: ProduceSampleRequest) -> RolloutTaskConfig:
        task_config = request.task_config.model_copy(deep=True) if request.task_config is not None else RolloutTaskConfig()
        if task_config.frame_count is None:
            task_config.frame_count = 16
        if task_config.num_steps == 1:
            task_config.num_steps = 35
        if task_config.width is None:
            task_config.width = request.sample_spec.width or 1024
        if task_config.height is None:
            task_config.height = request.sample_spec.height or 640
        return task_config

    def _effective_cosmos_config(self, request: ProduceSampleRequest, task_config: RolloutTaskConfig) -> CosmosTaskConfig:
        cosmos_config = request.cosmos_config.model_copy(deep=True) if request.cosmos_config is not None else CosmosTaskConfig()
        if request.task_type == TaskType.TEXT_TO_VIDEO:
            cosmos_config.variant = type(cosmos_config.variant).PREDICT1_TEXT2WORLD
        return cosmos_config

    def _validate_request(self, request: ProduceSampleRequest, cosmos_config: CosmosTaskConfig) -> None:
        if request.task_type not in {TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO}:
            raise ValueError(f"Backend {self.backend_name} only supports Cosmos video world-generation tasks")
        if cosmos_config.variant.value.endswith("video2world") and not request.sample_spec.references:
            raise ValueError("Cosmos video2world requests require at least one sample_spec.references item")
        if cosmos_config.variant.value.endswith("text2world") and not (request.sample_spec.prompt or "").strip():
            raise ValueError("Cosmos text2world requests require a non-empty sample_spec.prompt")

    def _execution_entity(
        self,
        *,
        sample_id: str,
        request: ProduceSampleRequest,
        task_config: RolloutTaskConfig,
        cosmos_config: CosmosTaskConfig,
    ) -> CosmosExecutionEntity:
        cache_key = prompt_reference_key(request.sample_spec.prompt or "", request.sample_spec.references)
        reuse_hit = self._input_cache.get(cache_key) is not None
        queue_lane = queue_lane_for_request(bool(request.sample_spec.references), reuse_hit)
        from wm_infra.backends.cosmos.runtime import CosmosBatchSignature

        batch_signature = CosmosBatchSignature(
            backend=self.backend_name,
            model=request.model,
            stage="infer",
            runner_mode=self._operator.load(),
            variant=cosmos_config.variant.value,
            width=task_config.width or 0,
            height=task_config.height or 0,
            frame_count=task_config.frame_count or 0,
            num_steps=task_config.num_steps,
            fps=cosmos_config.frames_per_second,
            has_reference=bool(request.sample_spec.references),
        )
        return CosmosExecutionEntity(
            entity_id=f"{sample_id}:infer",
            sample_id=sample_id,
            stage="infer",
            priority=request.priority,
            batch_signature=batch_signature,
            queue_lane=queue_lane,
            reference_key=cache_key,
        )

    async def execute_job(self, request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        self.validate_world_model_kind(request)
        sample_dir = self._sample_dir(sample_id)
        request_path = sample_dir / "request.json"
        runtime_path = sample_dir / "runtime.json"
        log_path = sample_dir / "runner.log"

        task_config = self._effective_task_config(request)
        cosmos_config = self._effective_cosmos_config(request, task_config)
        self._validate_request(request, cosmos_config)
        estimate = estimate_cosmos_request(task_config, cosmos_config)

        request_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "request": request.model_dump(mode="json"),
                    "effective_task_config": task_config.model_dump(mode="json"),
                    "effective_cosmos_config": cosmos_config.model_dump(mode="json"),
                    "resource_estimate": estimate.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )

        trace = CosmosRuntimeTrace()
        entity = self._execution_entity(
            sample_id=sample_id,
            request=request,
            task_config=task_config,
            cosmos_config=cosmos_config,
        )
        scheduler_decision = self._scheduler.schedule([entity], estimated_units=estimate.estimated_units)[0]
        trace.record(
            CosmosStageRecord(
                stage="admission",
                entity_id=entity.entity_id,
                queue_lane=entity.queue_lane.value,
                elapsed_ms=0.0,
                chunk_id=scheduler_decision.chunk.chunk_id,
                chunk_size=scheduler_decision.chunk.size,
                expected_occupancy=scheduler_decision.chunk.expected_occupancy,
                metadata=scheduler_decision.scheduler_inputs,
            )
        )

        start_materialize = time.perf_counter()
        cache_key = entity.reference_key
        reuse_hit = self._input_cache.get(cache_key) is not None
        trace.record(
            CosmosStageRecord(
                stage="input_materialize",
                entity_id=entity.entity_id,
                queue_lane=entity.queue_lane.value,
                elapsed_ms=round((time.perf_counter() - start_materialize) * 1000.0, 3),
                metadata={"reference_reuse_hit": reuse_hit},
            )
        )

        run_started = time.perf_counter()
        pipeline_run = await self._operator.generate_pipeline(
            output_dir=sample_dir,
            request=request,
            task_config=task_config,
            cosmos_config=cosmos_config,
        )
        result = pipeline_run.output
        if not isinstance(result, CosmosRunResult):
            raise RuntimeError("Cosmos generation operator did not return a CosmosRunResult")
        infer_stage = next((stage for stage in pipeline_run.stage_records if stage["name"] == "infer"), None)
        trace.record(
            CosmosStageRecord(
                stage="infer",
                entity_id=entity.entity_id,
                queue_lane=entity.queue_lane.value,
                elapsed_ms=round((time.perf_counter() - run_started) * 1000.0, 3),
                chunk_id=scheduler_decision.chunk.chunk_id,
                chunk_size=scheduler_decision.chunk.size,
                expected_occupancy=scheduler_decision.chunk.expected_occupancy,
                metadata={
                    "runner_mode": result.mode,
                    "pipeline_stage_elapsed_ms": None if infer_stage is None else infer_stage["elapsed_ms"],
                },
            )
        )

        self._input_cache.put(cache_key)

        log_payload = {
            "runner_mode": result.mode,
            "command": result.command,
            "error": result.error,
            "extra": result.extra,
            "response_payload": result.response_payload,
        }
        log_path.write_text(json.dumps(log_payload, indent=2, sort_keys=True))
        trace.record(
            CosmosStageRecord(
                stage="artifact_persist",
                entity_id=entity.entity_id,
                queue_lane=entity.queue_lane.value,
                elapsed_ms=0.0,
            )
        )

        runtime_payload = {
            "runner_mode": result.mode,
            "model_name": result.model_name,
            "scheduler": {
                **scheduler_decision.scheduler_inputs,
                "prompt_or_reference_hot": reuse_hit,
            },
            "chunk_summary": trace.chunk_summary(),
            "cache": self._input_cache.snapshot(),
            "reference_reuse_hit": reuse_hit,
            "stage_timings_ms": trace.stage_timings_ms(),
            "stage_graph": ["admission", "input_materialize", "infer", "artifact_persist", "controlplane_commit"],
            "pipeline": pipeline_run.pipeline_metadata,
            "stages": pipeline_run.stage_records,
            "output_path": result.output_path,
            "error": result.error,
            "operator": self._operator.describe(),
        }

        status = SampleStatus.SUCCEEDED if result.error is None else SampleStatus.FAILED
        runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True))
        record = SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            world_model_kind=self.world_model_kind,
            model_revision=request.model_revision,
            status=status,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            task_config=task_config,
            cosmos_config=cosmos_config,
            resource_estimate=estimate,
            runtime=runtime_payload,
            metadata={
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "runner_mode": result.mode,
                "stage_timings_ms": runtime_payload["stage_timings_ms"],
            },
            metrics={
                "elapsed_ms": round(result.elapsed_s * 1000.0, 3),
                "scheduler_expected_occupancy": round(scheduler_decision.chunk.expected_occupancy, 3),
            },
        )
        record.artifacts.extend(
            [
                self._artifact_record(
                    artifact_id=f"{sample_id}:video",
                    kind=ArtifactKind.VIDEO,
                    path=Path(result.output_path),
                    mime_type="video/mp4",
                    metadata={"runner_mode": result.mode, "variant": cosmos_config.variant.value},
                ),
                self._artifact_record(
                    artifact_id=f"{sample_id}:log",
                    kind=ArtifactKind.LOG,
                    path=log_path,
                    mime_type="application/json",
                ),
                self._artifact_record(
                    artifact_id=f"{sample_id}:metadata",
                    kind=ArtifactKind.METADATA,
                    path=runtime_path,
                    mime_type="application/json",
                ),
            ]
        )
        return record

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        sample_id = str(uuid.uuid4())
        return await self.execute_job(request, sample_id)

    def submit_async(self, request: ProduceSampleRequest) -> SampleRecord:
        if self._job_queue is None:
            raise RuntimeError("Cosmos backend async queue is not configured")

        self.validate_world_model_kind(request)
        task_config = self._effective_task_config(request)
        cosmos_config = self._effective_cosmos_config(request, task_config)
        self._validate_request(request, cosmos_config)
        estimate = estimate_cosmos_request(task_config, cosmos_config)

        sample_id = str(uuid.uuid4())
        entry = self._job_queue.submit(sample_id, request)
        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            world_model_kind=self.world_model_kind,
            model_revision=request.model_revision,
            status=SampleStatus.QUEUED,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            task_config=task_config,
            cosmos_config=cosmos_config,
            resource_estimate=estimate,
            runtime={
                "queue": {
                    "queue_name": "cosmos",
                    "pending": self._job_queue.pending_count,
                    "running": self._job_queue.running_count,
                    "position": self._job_queue.position(sample_id),
                    "submitted_at": entry.submitted_at,
                }
            },
            metadata={
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "runner_mode": self.runner.load(),
                "operator": self._operator.describe(),
            },
        )
