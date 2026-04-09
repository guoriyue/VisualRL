"""Gateway routes for sample production and Gateway-managed discovery state."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from wm_infra.api.metrics import QUEUE_DEPTH, SAMPLE_DURATION, SAMPLE_TOTAL
from wm_infra.backends import BackendRegistry
from wm_infra.controlplane import ProduceSampleRequest, SampleManifestStore
from wm_infra.backends import CosmosPredictBackend, WanVideoBackend
from wm_infra.gateway.state import get_gateway_runtime


def _queue_for_backend(runtime, backend):
    if isinstance(backend, WanVideoBackend):
        return runtime.wan_job_queue
    if isinstance(backend, CosmosPredictBackend):
        return runtime.cosmos_job_queue
    return None


def register_sample_routes(app: FastAPI) -> None:
    """Register sample-production and backend-serving routes."""
    router = APIRouter()

    @router.get("/v1/health")
    async def health(request: Request):
        runtime = get_gateway_runtime(request)
        backends = runtime.backend_registry.names()
        return {"status": "ready", "backends": backends}

    @router.get("/v1/models")
    async def list_models():
        from wm_infra.models.registry import list_models as _list_models
        return {"models": _list_models()}

    @router.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.get("/v1/backends")
    async def list_backends(request: Request):
        runtime = get_gateway_runtime(request)
        registry: BackendRegistry = runtime.backend_registry
        backends = []
        for name in registry.names():
            backend = registry.get(name)
            info = {"name": name, "type": backend.__class__.__name__, **backend.backend_descriptor()}
            if isinstance(backend, WanVideoBackend):
                info.update(
                    {
                        "runner_mode": backend.runner_mode,
                        "execution_mode": "in_process",
                        "output_root": str(backend.output_root),
                        "async_queue": runtime.wan_job_queue is not None,
                        "execution_backend": backend.execution_backend,
                        "engine_adapter": None if backend.engine_adapter is None else backend.engine_adapter.describe(),
                        "wan_engine_adapter_spec": backend.wan_engine_adapter_spec,
                        "wan_ckpt_dir": backend.wan_ckpt_dir,
                        "wan_i2v_diffusers_dir": backend.wan_i2v_diffusers_dir,
                        "max_batch_size": backend.max_batch_size,
                        "batch_wait_ms": backend.batch_wait_ms,
                        "warm_pool_size": backend.warm_pool_size,
                        "prewarm_common_signatures": backend.prewarm_common_signatures,
                        "warm_pool": backend._engine_pool.snapshot(),
                        "admission_max_units": backend.wan_admission_max_units,
                        "admission_max_vram_gb": backend.wan_admission_max_vram_gb,
                        "operator": (
                            None if backend._wan_runtime_config is None else {
                                "name": "wan-in-process",
                                "family": "generation",
                                "runtime": backend._wan_runtime_config.describe(),
                            }
                        ),
                    }
                )
            if isinstance(backend, CosmosPredictBackend):
                info.update(
                    {
                        "runner_mode": backend.runner_mode,
                        "execution_mode": "in_process",
                        "model": backend._model.model_name,
                        "output_root": str(backend.output_root),
                        "async_queue": runtime.cosmos_job_queue is not None,
                        "operator": backend._operator_describe(),
                    }
                )
            backends.append(info)
        return {"backends": backends}

    @router.post("/v1/samples")
    async def produce_sample(request: ProduceSampleRequest, http_request: Request):
        import time as _time

        runtime = get_gateway_runtime(http_request)
        sample_t0 = _time.monotonic()
        sample_status = "error"
        store = runtime.sample_store

        backend = runtime.backend_registry.get(request.backend)
        if backend is None:
            raise HTTPException(status_code=404, detail=f"Unknown backend: {request.backend}")

        try:
            submit_async = getattr(backend, "submit_async", None)
            queue = _queue_for_backend(runtime, backend)
            if callable(submit_async) and queue is not None:
                try:
                    record = submit_async(request, queue=queue)
                except ValueError as exc:
                    SAMPLE_TOTAL.labels(backend=request.backend, status="error").inc()
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except RuntimeError as exc:
                    SAMPLE_TOTAL.labels(backend=request.backend, status="error").inc()
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                store.put(record)
                SAMPLE_TOTAL.labels(backend=request.backend, status=record.status.value).inc()
                sample_status = record.status.value
                return record.model_dump(mode="json")

            record = await backend.produce_sample(request)
            store.put(record)
            SAMPLE_TOTAL.labels(backend=request.backend, status=record.status.value).inc()
            sample_status = record.status.value
            return record.model_dump(mode="json")
        except ValueError as exc:
            SAMPLE_TOTAL.labels(backend=request.backend, status="error").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            SAMPLE_DURATION.labels(backend=request.backend, status=sample_status).observe(_time.monotonic() - sample_t0)

    @router.get("/v1/queue/status")
    async def queue_status(request: Request):
        runtime = get_gateway_runtime(request)
        wan_queue = runtime.wan_job_queue
        cosmos_queue = runtime.cosmos_job_queue
        if wan_queue is None and cosmos_queue is None:
            QUEUE_DEPTH.set(0)
            return {"queue_enabled": False}

        pending = (
            (wan_queue.pending_count if wan_queue else 0)
            + (cosmos_queue.pending_count if cosmos_queue else 0)
        )
        running = (
            (wan_queue.running_count if wan_queue else 0)
            + (cosmos_queue.running_count if cosmos_queue else 0)
        )
        QUEUE_DEPTH.set(pending)
        return {
            "queue_enabled": True,
            "pending": pending,
            "running": running,
            "total_tracked": (
                (wan_queue.total_count if wan_queue else 0)
                + (cosmos_queue.total_count if cosmos_queue else 0)
            ),
            "queues": {
                "wan": None if wan_queue is None else wan_queue.snapshot(),
                "cosmos": None if cosmos_queue is None else cosmos_queue.snapshot(),
            },
        }

    @router.get("/v1/samples")
    async def list_samples(request: Request, status: str | None = None, backend: str | None = None, experiment_id: str | None = None, limit: int = 50):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        records = store.list()
        if status is not None:
            records = [record for record in records if record.status.value == status]
        if backend is not None:
            records = [record for record in records if record.backend == backend]
        if experiment_id is not None:
            records = [record for record in records if record.experiment is not None and record.experiment.experiment_id == experiment_id]
        limit = max(1, min(limit, 200))
        records = records[:limit]
        return {"samples": [record.model_dump(mode="json") for record in records], "count": len(records)}

    @router.get("/v1/samples/{sample_id}")
    async def get_sample(sample_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return record.model_dump(mode="json")

    @router.get("/v1/samples/{sample_id}/artifacts")
    async def list_artifacts(sample_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return {"artifacts": [artifact.model_dump(mode="json") for artifact in record.artifacts], "count": len(record.artifacts)}

    @router.get("/v1/samples/{sample_id}/artifacts/{artifact_id}")
    async def get_artifact(sample_id: str, artifact_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        decoded_id = unquote(artifact_id)
        for artifact in record.artifacts:
            if artifact.artifact_id == decoded_id:
                return artifact.model_dump(mode="json")
        raise HTTPException(status_code=404, detail=f"Artifact not found: {decoded_id}")

    @router.get("/v1/samples/{sample_id}/artifacts/{artifact_id}/content")
    async def get_artifact_content(sample_id: str, artifact_id: str, request: Request):
        store: SampleManifestStore = get_gateway_runtime(request).sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        decoded_id = unquote(artifact_id)
        artifact = next((a for a in record.artifacts if a.artifact_id == decoded_id), None)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {decoded_id}")
        if not artifact.uri.startswith("file://"):
            raise HTTPException(status_code=400, detail="Only file:// artifacts can be served")

        file_path = Path(artifact.uri[7:])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Artifact file not found on disk")
        return FileResponse(path=str(file_path), media_type=artifact.mime_type or "application/octet-stream", filename=file_path.name)

    app.include_router(router)
