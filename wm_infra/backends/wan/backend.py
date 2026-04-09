"""Wan 2.2 video backend with in-process stage scheduling and queue shaping."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.engine.metrics import SERVING_COMPILED_PROFILE_EVENTS, SERVING_TRANSFER_BYTES
from wm_infra.controlplane.resource_estimator import estimate_wan_request
from wm_infra.controlplane.schemas import (
    ArtifactKind,
    ArtifactRecord,
    ProduceSampleRequest,
    SampleRecord,
    SampleStatus,
    TaskType,
    WanTaskConfig,
    WorldModelKind,
)
from wm_infra.backends.wan.runtime import (
    WarmedWanEnginePool,
    build_wan_execution_family,
    build_wan_residency_records,
    build_quality_cost_hints,
    build_wan_batch_key,
    build_wan_batch_signature,
    build_wan_scheduler_payload,
    build_wan_transfer_plan,
    default_wan_prewarm_signatures,
    wan_batch_compatibility_score,
)
from .engine import (
    DiffusersWanI2VAdapter,
    HybridWanInProcessAdapter,
    OfficialWanInProcessAdapter,
    StubWanEngineAdapter,
    WanEngineAdapter,
    WanExecutionContext,
    WanStageScheduler,
    load_wan_engine_adapter,
)
from wm_infra.backends._pipeline import GenerationRuntimeBackend, GenerationRuntimeConfig


def _autodetect_wan_i2v_diffusers_dir() -> str | None:
    snapshot_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers"
        / "snapshots"
    )
    if not snapshot_root.exists():
        return None
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir() and (path / "model_index.json").exists())
    if not snapshots:
        return None
    return str(snapshots[-1])


class WanVideoBackend(ProduceSampleBackend):
    """Wan 2.2 video generation backend that only serves in-process adapters."""

    world_model_kind = WorldModelKind.GENERATION
    capability_flags = frozenset({"multistage_pipeline", "video_artifact"})

    def __init__(
        self,
        output_root: str | Path,
        *,
        backend_name: str = "wan-video",
        wan_admission_max_units: float | None = None,
        wan_admission_max_vram_gb: float | None = 32.0,
        max_batch_size: int = 4,
        batch_wait_ms: float = 2.0,
        warm_pool_size: int = 16,
        prewarm_common_signatures: bool = False,
        engine_adapter: WanEngineAdapter | None = None,
        wan_engine_adapter: str | None = None,
        # Official runner config
        wan_repo_dir: str | None = None,
        wan_conda_env: str | None = None,
        wan_ckpt_dir: str | None = None,
        wan_i2v_diffusers_dir: str | None = None,
        conda_sh_path: str | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.wan_admission_max_units = wan_admission_max_units
        self.wan_admission_max_vram_gb = wan_admission_max_vram_gb
        self.max_batch_size = max(1, max_batch_size)
        self.batch_wait_ms = max(batch_wait_ms, 0.0)
        self.warm_pool_size = max(1, warm_pool_size)
        self.prewarm_common_signatures = prewarm_common_signatures

        # Official runner: local Wan2.2 repo invocation
        self.wan_repo_dir = wan_repo_dir or os.environ.get("WM_WAN_REPO_DIR")
        self.wan_conda_env = wan_conda_env or os.environ.get("WM_WAN_CONDA_ENV", "kosen")
        self.wan_ckpt_dir = wan_ckpt_dir or os.environ.get("WM_WAN_CKPT_DIR")
        self.wan_i2v_diffusers_dir = (
            wan_i2v_diffusers_dir
            or os.environ.get("WM_WAN_I2V_DIFFUSERS_DIR")
            or _autodetect_wan_i2v_diffusers_dir()
        )
        self.conda_sh_path = conda_sh_path or os.environ.get("WM_CONDA_SH_PATH")
        self.wan_engine_adapter_spec = wan_engine_adapter or os.environ.get("WM_WAN_ENGINE_ADAPTER")
        if self.wan_engine_adapter_spec == "disabled":
            raise ValueError("WanVideoBackend requires an in-process engine adapter; 'disabled' is no longer supported")
        resolved_engine_adapter = engine_adapter or load_wan_engine_adapter(
            self.wan_engine_adapter_spec,
            repo_dir=self.wan_repo_dir,
            default_checkpoint_dir=self.wan_ckpt_dir,
            i2v_diffusers_dir=self.wan_i2v_diffusers_dir,
        )
        if resolved_engine_adapter is None:
            official_adapter = None
            if self.wan_repo_dir is not None:
                official_adapter = OfficialWanInProcessAdapter(
                    repo_dir=self.wan_repo_dir,
                    default_checkpoint_dir=self.wan_ckpt_dir,
                )
            image_to_video_adapter = None
            if self.wan_i2v_diffusers_dir is not None:
                image_to_video_adapter = DiffusersWanI2VAdapter(default_model_dir=self.wan_i2v_diffusers_dir)
            if official_adapter is not None and image_to_video_adapter is not None:
                resolved_engine_adapter = HybridWanInProcessAdapter(
                    official_adapter=official_adapter,
                    image_to_video_adapter=image_to_video_adapter,
                )
            elif official_adapter is not None:
                resolved_engine_adapter = official_adapter
            elif image_to_video_adapter is not None:
                resolved_engine_adapter = image_to_video_adapter
        if resolved_engine_adapter is None:
            resolved_engine_adapter = StubWanEngineAdapter()
        self.engine_adapter = resolved_engine_adapter
        self._stage_scheduler = WanStageScheduler(self.engine_adapter)
        self._wan_runtime_config = GenerationRuntimeConfig(
            local_scheduler=True,
            backend=GenerationRuntimeBackend.NATIVE,
        )
        self._stage_scheduler.runtime_config = self._wan_runtime_config
        self._engine_pool = WarmedWanEnginePool(
            max_profiles=self.warm_pool_size,
            prewarmed_signatures=(
                default_wan_prewarm_signatures(backend_name, self.runner_mode)
                if self.prewarm_common_signatures
                else None
            ),
        )

    @property
    def runner_mode(self) -> str:
        return self.engine_adapter.mode

    @property
    def execution_backend(self) -> str:
        return self.engine_adapter.execution_backend

    def _sample_dir(self, sample_id: str) -> Path:
        path = self.output_root / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_wan_config(self, request: ProduceSampleRequest) -> WanTaskConfig:
        return request.wan_config or WanTaskConfig(
            width=request.sample_spec.width or 832,
            height=request.sample_spec.height or 480,
        )

    def _validate_request(self, request: ProduceSampleRequest) -> None:
        if request.task_type not in {TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO}:
            raise ValueError(f"Backend {self.backend_name} only supports Wan 2.2-style video tasks")

        references = request.sample_spec.references
        if request.task_type == TaskType.TEXT_TO_VIDEO and not (request.sample_spec.prompt or "").strip():
            raise ValueError("Wan 2.2 text_to_video requests require a non-empty sample_spec.prompt")
        if request.task_type in {TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO} and not references:
            raise ValueError(f"Wan 2.2 {request.task_type.value} requests require at least one sample_spec.references item")
        if self.runner_mode == "official" and request.task_type == TaskType.VIDEO_TO_VIDEO:
            raise ValueError("Official Wan 2.2 runner wiring currently supports text_to_video and image_to_video only")

    def _admission_result(self, request: ProduceSampleRequest, wan_config: WanTaskConfig) -> tuple[bool, dict[str, Any], Any]:
        estimate = estimate_wan_request(wan_config)
        reasons: list[str] = []
        if self.wan_admission_max_units is not None and estimate.estimated_units > self.wan_admission_max_units:
            reasons.append(
                f"estimated_units {estimate.estimated_units:.2f} exceeds limit {self.wan_admission_max_units:.2f}"
            )
        if (
            self.wan_admission_max_vram_gb is not None
            and estimate.estimated_vram_gb is not None
            and estimate.estimated_vram_gb > self.wan_admission_max_vram_gb
        ):
            reasons.append(
                f"estimated_vram_gb {estimate.estimated_vram_gb:.2f} exceeds limit {self.wan_admission_max_vram_gb:.2f}"
            )
        admitted = not reasons
        return admitted, {
            "admitted": admitted,
            "reasons": reasons,
            "max_units": self.wan_admission_max_units,
            "max_vram_gb": self.wan_admission_max_vram_gb,
            "quality_cost_hints": build_quality_cost_hints(
                wan_config,
                max_units=self.wan_admission_max_units,
                max_vram_gb=self.wan_admission_max_vram_gb,
            ),
        }, estimate

    def queue_batch_key(self, request: ProduceSampleRequest) -> tuple[Any, ...] | None:
        self._validate_request(request)
        wan_config = self._resolve_wan_config(request)
        return build_wan_batch_key(request, wan_config, runner_mode=self.runner_mode)

    def queue_batch_size_limit(self, configured_max_batch_size: int) -> int:
        return max(1, min(configured_max_batch_size, self.max_batch_size))

    def queue_batch_score(
        self,
        reference_request: ProduceSampleRequest,
        candidate_request: ProduceSampleRequest,
    ) -> float | None:
        """Rank near-shape Wan 2.2 requests when exact queue keys do not match."""

        try:
            self._validate_request(reference_request)
            self._validate_request(candidate_request)
        except ValueError:
            return None
        reference_config = self._resolve_wan_config(reference_request)
        candidate_config = self._resolve_wan_config(candidate_request)
        return wan_batch_compatibility_score(
            reference_request,
            reference_config,
            candidate_request,
            candidate_config,
            runner_mode=self.runner_mode,
        )

    def _build_request_payload(
        self,
        sample_id: str,
        request: ProduceSampleRequest,
        wan_config: WanTaskConfig,
        estimate: Any,
        plan_path: Path,
        log_path: Path,
        video_path: Path,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "task_type": request.task_type.value,
            "backend": request.backend,
            "model": request.model,
            "model_revision": request.model_revision,
            "sample_spec": request.sample_spec.model_dump(mode="json"),
            "wan_config": wan_config.model_dump(mode="json"),
            "resource_estimate": estimate.model_dump(mode="json"),
            "artifacts": {
                "request_path": str(plan_path),
                "log_path": str(log_path),
                "output_path": str(video_path),
            },
            "request_context": {
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "experiment": request.experiment.model_dump(mode="json") if request.experiment else None,
            },
            "execution_backend": self.execution_backend,
            "engine_adapter": None if self.engine_adapter is None else self.engine_adapter.describe(),
        }

    def _artifact_details(self, path: Path) -> tuple[int | None, str | None]:
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
        bytes_size, sha256 = self._artifact_details(path)
        payload = {"exists": path.exists()}
        if metadata:
            payload.update(metadata)
        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            uri=f"file://{path}",
            mime_type=mime_type,
            bytes=bytes_size,
            sha256=sha256,
            metadata=payload,
        )

    def _failure_payload(
        self,
        *,
        sample_id: str,
        status: SampleStatus,
        command: str | None,
        log_path: Path,
        video_path: Path,
        stdout: bytes | None,
        returncode: int | None,
        timed_out: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "status": status.value,
            "returncode": returncode,
            "timed_out": timed_out,
            "command": command,
            "log_path": str(log_path),
            "output_path": str(video_path),
            "output_exists": video_path.exists(),
            "tail": (stdout or b"")[-4000:].decode("utf-8", errors="replace"),
            "error": error,
        }

    async def _execute_job_impl(
        self,
        request: ProduceSampleRequest,
        sample_id: str,
        *,
        batch_size: int,
        batch_index: int,
        batch_sample_ids: list[str],
        engine_profile: dict[str, Any] | None = None,
    ) -> SampleRecord:
        """Execute one Wan 2.2 job with optional queue-batch metadata."""
        self._validate_request(request)

        wan_config = self._resolve_wan_config(request)
        signature = build_wan_batch_signature(request, wan_config, runner_mode=self.runner_mode)
        profile_claim = engine_profile or self._engine_pool.reserve(signature, batch_size=batch_size)
        compile_state = str(profile_claim.get("compile_state") or "unknown")
        SERVING_COMPILED_PROFILE_EVENTS.labels(backend=self.backend_name, event=compile_state).inc()
        execution_family = profile_claim.get("execution_family") or build_wan_execution_family(
            signature,
            batch_size=batch_size,
        ).as_dict()
        scheduler_payload = build_wan_scheduler_payload(
            signature,
            batch_size=batch_size,
            batch_index=batch_index,
            max_batch_size=self.max_batch_size,
            sample_ids=batch_sample_ids,
        )
        sample_dir = self._sample_dir(sample_id)
        plan_path = sample_dir / "request.json"
        log_path = sample_dir / "runner.log"
        video_path = sample_dir / "sample.mp4"
        runtime_path = sample_dir / "runtime.json"
        failure_path = sample_dir / "failure.json"

        estimate = estimate_wan_request(wan_config)
        request_payload = self._build_request_payload(sample_id, request, wan_config, estimate, plan_path, log_path, video_path)
        request_payload["scheduler"] = scheduler_payload
        request_payload["compiled_graph_pool"] = profile_claim
        request_payload["execution_family"] = execution_family
        plan_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True))

        started_at = time.time()
        mode = self.runner_mode
        transfer_plan = build_wan_transfer_plan(wan_config=wan_config, request=request)
        if transfer_plan.h2d_bytes:
            SERVING_TRANSFER_BYTES.labels(backend=self.backend_name, kind="h2d").observe(transfer_plan.h2d_bytes)
        runtime: dict[str, Any] = {
            "runner": mode,
            "request_path": str(plan_path),
            "output_path": str(video_path),
            "log_path": str(log_path),
            "runtime_path": str(runtime_path),
            "failure_path": str(failure_path),
            "status_history": [
                {"status": SampleStatus.QUEUED.value, "timestamp": started_at},
                {"status": SampleStatus.RUNNING.value, "timestamp": started_at},
            ],
            "started_at": started_at,
            "scheduler": scheduler_payload,
            "compiled_graph_pool": profile_claim,
            "execution_family": execution_family,
            "transfer_plan": transfer_plan.as_dict(),
            "residency": build_wan_residency_records(signature, batch_size=batch_size),
            "engine_pool_snapshot": self._engine_pool.snapshot(),
        }
        status = SampleStatus.ACCEPTED if mode == "stub" else SampleStatus.SUCCEEDED
        metadata: dict[str, Any] = {
            "evaluation_policy": request.evaluation_policy,
            "priority": request.priority,
            "labels": request.labels,
            "stubbed": mode == "stub",
            "runner_mode": mode,
            "queue_batched": batch_size > 1,
        }
        runtime["execution_backend"] = self.execution_backend
        runtime["engine"] = self.engine_adapter.describe()

        pipeline_context = WanExecutionContext(
            sample_id=sample_id,
            request=request,
            wan_config=wan_config,
            sample_dir=sample_dir,
            plan_path=plan_path,
            log_path=log_path,
            video_path=video_path,
            runtime_path=runtime_path,
            batch_size=batch_size,
            batch_index=batch_index,
            batch_sample_ids=batch_sample_ids,
            scheduler_payload=scheduler_payload,
            engine_profile=profile_claim,
        )
        pipeline_run = await self._stage_scheduler.run(pipeline_context)
        runtime["pipeline"] = pipeline_run.pipeline_metadata
        runtime["stages"] = pipeline_run.stage_records
        runtime["stage_state"] = pipeline_run.stage_state
        runtime["operator"] = {
            "name": "wan-in-process",
            "family": "generation",
            "runtime": self._wan_runtime_config.describe(),
        }
        metadata["engine_adapter"] = self.engine_adapter.adapter_name
        log_path.write_text(pipeline_run.log_text)
        if video_path.exists():
            status = SampleStatus.SUCCEEDED
        elif self.engine_adapter.supports_output_video:
            status = SampleStatus.FAILED
            metadata["runner_error"] = "in-process Wan adapter completed without persisting an output video"
            failure_path.write_text(
                json.dumps(
                    self._failure_payload(
                        sample_id=sample_id,
                        status=status,
                        command=None,
                        log_path=log_path,
                        video_path=video_path,
                        stdout=pipeline_run.log_text.encode("utf-8"),
                        returncode=None,
                        timed_out=False,
                        error=metadata["runner_error"],
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            status = SampleStatus.ACCEPTED

        completed_at = time.time()
        artifact_io_bytes = 0
        for path in (plan_path, log_path, runtime_path, video_path, failure_path):
            if path.exists():
                artifact_io_bytes += path.stat().st_size
        transfer_plan.add_artifact_io(artifact_io_bytes)
        if transfer_plan.artifact_io_bytes:
            SERVING_TRANSFER_BYTES.labels(backend=self.backend_name, kind="artifact_io").observe(
                transfer_plan.artifact_io_bytes
            )
        runtime["transfer_plan"] = transfer_plan.as_dict()
        runtime["residency"] = build_wan_residency_records(
            signature,
            batch_size=batch_size,
            artifact_io_bytes=artifact_io_bytes,
        )
        runtime["completed_at"] = completed_at
        runtime["elapsed_ms"] = round((completed_at - started_at) * 1000, 2)
        runtime["status_history"].append({"status": status.value, "timestamp": completed_at})
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True))

        artifacts = [
            self._artifact_record(
                artifact_id=f"{sample_id}:log",
                kind=ArtifactKind.LOG,
                path=log_path,
                mime_type="text/plain",
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:metadata",
                kind=ArtifactKind.METADATA,
                path=plan_path,
                mime_type="application/json",
                metadata={"role": "request"},
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:runtime",
                kind=ArtifactKind.METADATA,
                path=runtime_path,
                mime_type="application/json",
                metadata={"role": "runtime"},
            ),
        ]
        if video_path.exists():
            artifacts.insert(
                0,
                self._artifact_record(
                    artifact_id=f"{sample_id}:video",
                    kind=ArtifactKind.VIDEO,
                    path=video_path,
                    mime_type="video/mp4",
                    metadata={"stubbed": mode == "stub"},
                ),
            )
        if failure_path.exists():
            artifacts.append(
                self._artifact_record(
                    artifact_id=f"{sample_id}:failure",
                    kind=ArtifactKind.METADATA,
                    path=failure_path,
                    mime_type="application/json",
                    metadata={"role": "failure"},
                )
            )

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            world_model_kind=self.world_model_kind,
            model_revision=request.model_revision,
            status=status,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            wan_config=wan_config,
            resource_estimate=estimate,
            artifacts=artifacts,
            runtime=runtime,
            metadata=metadata,
        )

    async def execute_job(self, request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        """Execute a Wan 2.2 job synchronously (called by queue worker or produce_sample)."""

        return await self._execute_job_impl(
            request,
            sample_id,
            batch_size=1,
            batch_index=0,
            batch_sample_ids=[sample_id],
        )

    async def execute_job_batch(self, items: list[tuple[ProduceSampleRequest, str]]) -> list[SampleRecord]:
        """Execute a queue batch of compatible Wan 2.2 jobs under one warmed profile."""

        if not items:
            return []
        if len(items) == 1:
            request, sample_id = items[0]
            return [await self.execute_job(request, sample_id)]

        first_request, _first_sample_id = items[0]
        wan_config = self._resolve_wan_config(first_request)
        signature = build_wan_batch_signature(first_request, wan_config, runner_mode=self.runner_mode)
        expected_key = build_wan_batch_key(first_request, wan_config, runner_mode=self.runner_mode)
        for request, _sample_id in items[1:]:
            candidate_key = self.queue_batch_key(request)
            if candidate_key != expected_key and self.queue_batch_score(first_request, request) is None:
                raise ValueError("Wan 2.2 execute_job_batch received incompatible requests")
        shared_profile = self._engine_pool.reserve(signature, batch_size=len(items))
        sample_ids = [sample_id for _request, sample_id in items]
        records: list[SampleRecord] = []
        for index, (request, sample_id) in enumerate(items):
            records.append(
                await self._execute_job_impl(
                    request,
                    sample_id,
                    batch_size=len(items),
                    batch_index=index,
                    batch_sample_ids=sample_ids,
                    engine_profile=shared_profile,
                )
            )
        return records

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        self.validate_world_model_kind(request)
        self._validate_request(request)
        sample_id = str(uuid.uuid4())
        return await self.execute_job(request, sample_id)

    def submit_async(self, request: ProduceSampleRequest, *, queue) -> SampleRecord:
        self.validate_world_model_kind(request)
        self._validate_request(request)
        sample_id = str(uuid.uuid4())
        wan_config = self._resolve_wan_config(request)
        admitted, admission, estimate = self._admission_result(request, wan_config)
        queued_at = time.time()
        queue_position = None
        if admitted:
            queue.submit(sample_id, request)
            queue_position = queue.position(sample_id)

        batch_key = self.queue_batch_key(request)
        runtime = {
            "runner": self.runner_mode,
            "async": True,
            "execution_backend": self.execution_backend,
            "queued_at": queued_at,
            "admission": admission,
            "scheduler": {
                "queue_batch_key": list(batch_key) if isinstance(batch_key, tuple) else batch_key,
                "max_batch_size": self.max_batch_size,
                "batch_wait_ms": self.batch_wait_ms,
            },
            "status_history": [
                {"status": (SampleStatus.QUEUED if admitted else SampleStatus.REJECTED).value, "timestamp": queued_at},
            ],
        }
        signature = build_wan_batch_signature(request, wan_config, runner_mode=self.runner_mode)
        runtime["execution_family"] = build_wan_execution_family(signature, batch_size=1).as_dict()
        transfer_plan = build_wan_transfer_plan(wan_config=wan_config, request=request)
        if transfer_plan.h2d_bytes:
            SERVING_TRANSFER_BYTES.labels(backend=self.backend_name, kind="h2d").observe(transfer_plan.h2d_bytes)
        runtime["transfer_plan"] = transfer_plan.as_dict()
        runtime["residency"] = build_wan_residency_records(signature, batch_size=1)
        if self.engine_adapter is not None:
            runtime["engine"] = self.engine_adapter.describe()
            runtime["pipeline"] = {
                "execution_backend": self.execution_backend,
                "adapter": self.engine_adapter.describe(),
            }
        if queue_position is not None:
            runtime["queue_position"] = queue_position
            runtime["queue_snapshot"] = queue.snapshot()

        metadata = {
            "evaluation_policy": request.evaluation_policy,
            "priority": request.priority,
            "labels": request.labels,
            "stubbed": self.runner_mode == "stub",
            "async": True,
            "runner_mode": self.runner_mode,
        }
        if self.engine_adapter is not None:
            metadata["engine_adapter"] = self.engine_adapter.adapter_name
        if not admitted:
            metadata["runner_error"] = "; ".join(admission["reasons"])

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            world_model_kind=self.world_model_kind,
            model_revision=request.model_revision,
            status=SampleStatus.QUEUED if admitted else SampleStatus.REJECTED,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            wan_config=wan_config,
            resource_estimate=estimate,
            runtime=runtime,
            metadata=metadata,
        )
