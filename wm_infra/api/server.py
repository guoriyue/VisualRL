"""FastAPI server for wm-infra temporal serving.

Provides two distinct surfaces:
- low-level rollout runtime endpoints for engine bring-up
- higher-level temporal sample-production endpoints for Wan/Genie-style backends

Endpoints include:
- POST /v1/rollout — submit a rollout prediction (supports stream=true for SSE)
- GET  /v1/rollout/{job_id} — get rollout result
- POST /v1/samples — produce a sample manifest (async for queue-backed backends)
- GET  /v1/samples — list samples
- GET  /v1/samples/{sample_id} — fetch a persisted sample manifest
- GET  /v1/samples/{sample_id}/artifacts/{artifact_id} — artifact metadata
- GET  /v1/samples/{sample_id}/artifacts/{artifact_id}/content — artifact bytes
- POST /v1/episodes, /v1/branches, /v1/state-handles, /v1/rollouts, /v1/checkpoints
- GET  corresponding temporal entity list/get endpoints
- GET  /v1/queue/status — async backend queue health
- GET  /v1/health — health check
- GET  /v1/models — list available models

Uses AsyncWorldModelEngine for non-blocking concurrent request handling.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import unquote

import numpy as np
import torch

from wm_infra.api.metrics import ACTIVE_ROLLOUTS, API_AUTH_FAILURES, QUEUE_DEPTH, REQUEST_DURATION, REQUEST_TOTAL, SAMPLE_DURATION, SAMPLE_TOTAL
from wm_infra.api.protocol import HealthResponse, RolloutRequest, RolloutResponse, SSE_DONE, StepResult
from wm_infra.backends import BackendRegistry, GenieJobQueue, GenieRolloutBackend, RolloutBackend, WanJobQueue, WanVideoBackend
from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.config import EngineConfig, load_config
from wm_infra.controlplane import (
    BranchCreate,
    CheckpointCreate,
    EpisodeCreate,
    ProduceSampleRequest,
    RolloutCreate,
    SampleManifestStore,
    StateHandleCreate,
    TemporalStore,
)
from wm_infra.core.engine import AsyncWorldModelEngine, RolloutJob
from wm_infra.models.dynamics import LatentDynamicsModel
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer

logger = logging.getLogger("wm_infra")
_engine: Optional[AsyncWorldModelEngine] = None


def _create_async_engine(config: EngineConfig, execution_mode: str = "chunked") -> AsyncWorldModelEngine:
    dynamics = LatentDynamicsModel(config.dynamics)
    tokenizer = VideoTokenizer(config.tokenizer)

    if config.model_path:
        state_dict = torch.load(config.model_path, map_location="cpu", weights_only=True)
        dynamics.load_state_dict(state_dict.get("dynamics", state_dict), strict=False)
        if "tokenizer" in state_dict:
            tokenizer.load_state_dict(state_dict["tokenizer"], strict=False)

    return AsyncWorldModelEngine(config, dynamics, tokenizer, execution_mode=execution_mode)


def _build_job(request: RolloutRequest, config: EngineConfig) -> RolloutJob:
    job = RolloutJob(
        job_id="",
        num_steps=request.num_steps,
        return_frames=request.return_frames,
        return_latents=request.return_latents,
        stream=request.stream,
    )

    if request.initial_latent is not None:
        job.initial_latent = torch.tensor(request.initial_latent, dtype=torch.float32)
    elif request.initial_observation_b64 is not None:
        img_bytes = base64.b64decode(request.initial_observation_b64)
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        job.initial_observation = torch.from_numpy(img_np).permute(2, 0, 1)
    else:
        n = config.state_cache.num_latent_tokens
        d = config.dynamics.latent_token_dim
        job.initial_latent = torch.randn(n, d)

    if request.actions is not None:
        job.actions = torch.tensor(request.actions, dtype=torch.float32)

    return job


def _build_default_store(config: EngineConfig) -> SampleManifestStore:
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return SampleManifestStore(root)


def _build_temporal_store(config: EngineConfig) -> TemporalStore:
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return TemporalStore(Path(root) / "temporal")


def create_app(
    config: Optional[EngineConfig] = None,
    sample_store: Optional[SampleManifestStore] = None,
    backend_registry: Optional[BackendRegistry] = None,
    temporal_store: Optional[TemporalStore] = None,
    execution_mode: str = "chunked",
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    if config is None:
        config = EngineConfig()
    if sample_store is None:
        sample_store = _build_default_store(config)
    if temporal_store is None:
        temporal_store = _build_temporal_store(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine
        logger.info("Initializing temporal runtime engine...")
        _engine = _create_async_engine(config, execution_mode=execution_mode)
        _engine.start()

        registry = backend_registry or BackendRegistry()
        if registry.get("rollout-engine") is None:
            registry.register(RolloutBackend(_engine))
        if registry.get("genie-rollout") is None:
            genie_output = config.controlplane.genie_output_root or str(
                Path(tempfile.gettempdir()) / "wm_infra_genie"
            )
            genie_runner = GenieRunner(
                model_name_or_path=config.controlplane.genie_model_name or "1x-technologies/GENIE_210M_v0",
                device=config.controlplane.genie_device or "cuda",
                num_prompt_frames=config.controlplane.genie_num_prompt_frames,
                maskgit_steps=config.controlplane.genie_maskgit_steps,
                temperature=config.controlplane.genie_temperature,
            )
            genie_backend = GenieRolloutBackend(
                temporal_store,
                output_root=genie_output,
                runner=genie_runner,
            )
            registry.register(genie_backend)
        if registry.get("wan-video") is None:
            wan_root = config.controlplane.wan_output_root or str(Path(tempfile.gettempdir()) / "wm_infra_wan")
            wan_backend = WanVideoBackend(
                wan_root,
                shell_runner=config.controlplane.wan_shell_runner,
                shell_runner_timeout_s=config.controlplane.wan_shell_runner_timeout_s,
                wan_admission_max_units=config.controlplane.wan_admission_max_units,
                wan_admission_max_vram_gb=config.controlplane.wan_admission_max_vram_gb,
                wan_repo_dir=config.controlplane.wan_repo_dir,
                wan_conda_env=config.controlplane.wan_conda_env,
                conda_sh_path=config.controlplane.conda_sh_path,
            )
            registry.register(wan_backend)
        else:
            wan_backend = registry.get("wan-video")

        wan_job_queue = None
        if isinstance(wan_backend, WanVideoBackend):
            wan_job_queue = WanJobQueue(
                execute_fn=wan_backend.execute_job,
                store=sample_store,
                queue_name="wan",
                max_queue_size=config.controlplane.wan_max_queue_size,
                max_concurrent=config.controlplane.wan_max_concurrent_jobs,
            )
            wan_backend._job_queue = wan_job_queue
            wan_job_queue.start()

        genie_job_queue = None
        genie_backend = registry.get("genie-rollout")
        if isinstance(genie_backend, GenieRolloutBackend):
            genie_job_queue = GenieJobQueue(
                execute_fn=genie_backend.execute_job,
                store=sample_store,
                queue_name="genie",
                max_queue_size=config.controlplane.genie_max_queue_size,
                max_concurrent=config.controlplane.genie_max_concurrent_jobs,
            )
            genie_backend._job_queue = genie_job_queue
            genie_job_queue.start()

        app.state.backend_registry = registry
        app.state.sample_store = sample_store
        app.state.temporal_store = temporal_store
        app.state.wan_job_queue = wan_job_queue
        app.state.genie_job_queue = genie_job_queue

        device_str = config.device.value if hasattr(config.device, "value") else str(config.device)
        logger.info(
            f"Engine ready: device={device_str}, dynamics={sum(p.numel() for p in _engine.engine.dynamics_model.parameters())} params"
        )
        yield
        logger.info("Shutting down engine")
        if wan_job_queue is not None:
            await wan_job_queue.stop()
        if genie_job_queue is not None:
            await genie_job_queue.stop()
        await _engine.stop()

    app = FastAPI(title="wm-infra", description="Temporal model serving and control-plane infrastructure", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def api_key_guard(request, call_next):
        api_key = config.server.api_key
        if api_key is None:
            return await call_next(request)

        path = request.url.path
        if (
            path in {"/v1/health", "/v1/models", "/metrics", "/openapi.json"}
            or path.startswith("/docs")
            or path.startswith("/redoc")
            or request.method in {"OPTIONS", "HEAD"}
        ):
            return await call_next(request)

        provided = request.headers.get("X-API-Key")
        if provided != api_key:
            API_AUTH_FAILURES.labels(endpoint=path).inc()
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        return await call_next(request)

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        ready = _engine is not None and _engine.is_running
        ACTIVE_ROLLOUTS.set(_engine.engine.state_manager.num_active if _engine else 0)
        return HealthResponse(
            status="ready" if ready else "not_ready",
            model_loaded=_engine is not None,
            engine_running=_engine.is_running if _engine else False,
            active_rollouts=_engine.engine.state_manager.num_active if _engine else 0,
            memory_used_gb=_engine.engine.state_manager.memory_used_gb if _engine else 0.0,
        )

    @app.get("/v1/models")
    async def list_models():
        from wm_infra.models.registry import list_models as _list_models
        return {"models": _list_models()}

    @app.get("/v1/backends")
    async def list_backends():
        registry: BackendRegistry = app.state.backend_registry
        backends = []
        for name in registry.names():
            backend = registry.get(name)
            info = {"name": name, "type": backend.__class__.__name__}
            if isinstance(backend, WanVideoBackend):
                info.update(
                    {
                        "runner_mode": backend.runner_mode,
                        "shell_runner_configured": backend.shell_runner is not None,
                        "shell_runner_timeout_s": backend.shell_runner_timeout_s,
                        "output_root": str(backend.output_root),
                        "async_queue": backend._job_queue is not None,
                        "admission_max_units": backend.wan_admission_max_units,
                        "admission_max_vram_gb": backend.wan_admission_max_vram_gb,
                    }
                )
            if isinstance(backend, GenieRolloutBackend):
                info.update({
                    "stateful": True,
                    "runner_mode": backend.runner_mode,
                    "model": backend.runner.model_name_or_path,
                    "output_root": str(backend.output_root) if backend.output_root else None,
                    "async_queue": backend._job_queue is not None,
                })
            backends.append(info)
        return {"backends": backends}

    @app.get("/metrics")
    async def metrics():
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/rollout")
    async def submit_rollout(request: RolloutRequest):
        import time as _time

        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        job = _build_job(request, config)

        if request.stream:
            REQUEST_TOTAL.labels(status="stream").inc()
            return StreamingResponse(
                _stream_rollout(job, request),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        t0 = _time.monotonic()
        try:
            result = await _engine.submit(job)
            REQUEST_TOTAL.labels(status="success").inc()
        except Exception:
            REQUEST_TOTAL.labels(status="error").inc()
            raise
        finally:
            REQUEST_DURATION.observe(_time.monotonic() - t0)

        response = RolloutResponse(job_id=result.job_id, model=request.model, steps_completed=result.steps_completed, elapsed_ms=result.elapsed_ms)
        if result.predicted_latents is not None:
            response.latents = result.predicted_latents.cpu().tolist()
        if result.predicted_frames is not None:
            frames_b64 = []
            for t in range(result.predicted_frames.shape[0]):
                frame = result.predicted_frames[t]
                frame_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
                try:
                    from PIL import Image
                    img = Image.fromarray(frame_np)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    frames_b64.append(base64.b64encode(buf.getvalue()).decode())
                except ImportError:
                    frames_b64.append("")
            response.frames_b64 = frames_b64
        return response

    async def _stream_rollout(job: RolloutJob, request: RolloutRequest) -> AsyncIterator[str]:
        async for step_idx, latent in _engine.submit_stream(job):
            step_result = StepResult(step=step_idx)
            if request.return_latents:
                step_result.latent = latent.cpu().tolist()
            yield step_result.to_sse()
        yield SSE_DONE

    @app.get("/v1/rollout/{job_id}")
    async def get_rollout(job_id: str):
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        result = _engine.engine.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": result.job_id, "steps_completed": result.steps_completed, "elapsed_ms": result.elapsed_ms}

    @app.post("/v1/samples")
    async def produce_sample(request: ProduceSampleRequest):
        import time as _time

        sample_t0 = _time.monotonic()
        sample_status = "error"
        registry: BackendRegistry = app.state.backend_registry
        store: SampleManifestStore = app.state.sample_store

        backend = registry.get(request.backend)
        if backend is None:
            raise HTTPException(status_code=404, detail=f"Unknown backend: {request.backend}")

        try:
            if (isinstance(backend, WanVideoBackend) and backend._job_queue is not None) or (
                isinstance(backend, GenieRolloutBackend) and backend._job_queue is not None
            ):
                try:
                    record = backend.submit_async(request)
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

    @app.get("/v1/samples")
    async def list_samples(status: str | None = None, backend: str | None = None, experiment_id: str | None = None, limit: int = 50):
        store: SampleManifestStore = app.state.sample_store
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

    @app.get("/v1/samples/{sample_id}")
    async def get_sample(sample_id: str):
        store: SampleManifestStore = app.state.sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return record.model_dump(mode="json")

    @app.get("/v1/samples/{sample_id}/artifacts")
    async def list_artifacts(sample_id: str):
        store: SampleManifestStore = app.state.sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return {"artifacts": [artifact.model_dump(mode="json") for artifact in record.artifacts], "count": len(record.artifacts)}

    @app.get("/v1/samples/{sample_id}/artifacts/{artifact_id}")
    async def get_artifact(sample_id: str, artifact_id: str):
        store: SampleManifestStore = app.state.sample_store
        record = store.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        decoded_id = unquote(artifact_id)
        for artifact in record.artifacts:
            if artifact.artifact_id == decoded_id:
                return artifact.model_dump(mode="json")
        raise HTTPException(status_code=404, detail=f"Artifact not found: {decoded_id}")

    @app.get("/v1/samples/{sample_id}/artifacts/{artifact_id}/content")
    async def get_artifact_content(sample_id: str, artifact_id: str):
        from fastapi.responses import FileResponse

        store: SampleManifestStore = app.state.sample_store
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

    @app.get("/v1/queue/status")
    async def queue_status():
        wan_queue = app.state.wan_job_queue
        genie_queue = app.state.genie_job_queue
        if wan_queue is None and genie_queue is None:
            QUEUE_DEPTH.set(0)
            return {"queue_enabled": False}
        pending = (wan_queue.pending_count if wan_queue else 0) + (genie_queue.pending_count if genie_queue else 0)
        running = (wan_queue.running_count if wan_queue else 0) + (genie_queue.running_count if genie_queue else 0)
        QUEUE_DEPTH.set(pending)
        return {
            "queue_enabled": True,
            "pending": pending,
            "running": running,
            "total_tracked": (wan_queue.total_count if wan_queue else 0) + (genie_queue.total_count if genie_queue else 0),
            "queues": {
                "wan": None if wan_queue is None else wan_queue.snapshot(),
                "genie": None if genie_queue is None else genie_queue.snapshot(),
            },
        }

    @app.post("/v1/episodes")
    async def create_episode(request: EpisodeCreate):
        store: TemporalStore = app.state.temporal_store
        return store.create_episode(request).model_dump(mode="json")

    @app.get("/v1/episodes")
    async def list_episodes():
        store: TemporalStore = app.state.temporal_store
        episodes = store.episodes.list()
        return {"episodes": [e.model_dump(mode="json") for e in episodes], "count": len(episodes)}

    @app.get("/v1/episodes/{episode_id}")
    async def get_episode(episode_id: str):
        store: TemporalStore = app.state.temporal_store
        episode = store.episodes.get(episode_id)
        if episode is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        return episode.model_dump(mode="json")

    @app.post("/v1/branches")
    async def create_branch(request: BranchCreate):
        store: TemporalStore = app.state.temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.parent_branch_id and store.branches.get(request.parent_branch_id) is None:
            raise HTTPException(status_code=404, detail="Parent branch not found")
        if request.forked_from_checkpoint_id and store.checkpoints.get(request.forked_from_checkpoint_id) is None:
            raise HTTPException(status_code=404, detail="Fork checkpoint not found")
        return store.create_branch(request).model_dump(mode="json")

    @app.get("/v1/branches")
    async def list_branches(episode_id: str | None = None):
        store: TemporalStore = app.state.temporal_store
        branches = store.branches.list()
        if episode_id is not None:
            branches = [b for b in branches if b.episode_id == episode_id]
        return {"branches": [b.model_dump(mode="json") for b in branches], "count": len(branches)}

    @app.get("/v1/branches/{branch_id}")
    async def get_branch(branch_id: str):
        store: TemporalStore = app.state.temporal_store
        branch = store.branches.get(branch_id)
        if branch is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        return branch.model_dump(mode="json")

    @app.post("/v1/state-handles")
    async def create_state_handle(request: StateHandleCreate):
        store: TemporalStore = app.state.temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.rollout_id and store.rollouts.get(request.rollout_id) is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        if request.checkpoint_id and store.checkpoints.get(request.checkpoint_id) is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return store.create_state_handle(request).model_dump(mode="json")

    @app.get("/v1/state-handles")
    async def list_state_handles(episode_id: str | None = None, branch_id: str | None = None):
        store: TemporalStore = app.state.temporal_store
        items = store.state_handles.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if branch_id is not None:
            items = [i for i in items if i.branch_id == branch_id]
        return {"state_handles": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @app.get("/v1/state-handles/{state_handle_id}")
    async def get_state_handle(state_handle_id: str):
        store: TemporalStore = app.state.temporal_store
        item = store.state_handles.get(state_handle_id)
        if item is None:
            raise HTTPException(status_code=404, detail="State handle not found")
        return item.model_dump(mode="json")

    @app.post("/v1/rollouts")
    async def create_temporal_rollout(request: RolloutCreate):
        store: TemporalStore = app.state.temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.input_state_handle_id and store.state_handles.get(request.input_state_handle_id) is None:
            raise HTTPException(status_code=404, detail="Input state handle not found")
        return store.create_rollout(request).model_dump(mode="json")

    @app.get("/v1/rollouts")
    async def list_temporal_rollouts(episode_id: str | None = None, branch_id: str | None = None):
        store: TemporalStore = app.state.temporal_store
        items = store.rollouts.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if branch_id is not None:
            items = [i for i in items if i.branch_id == branch_id]
        return {"rollouts": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @app.get("/v1/rollouts/{rollout_id}")
    async def get_temporal_rollout(rollout_id: str):
        store: TemporalStore = app.state.temporal_store
        item = store.rollouts.get(rollout_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        return item.model_dump(mode="json")

    @app.post("/v1/checkpoints")
    async def create_checkpoint(request: CheckpointCreate):
        store: TemporalStore = app.state.temporal_store
        if store.episodes.get(request.episode_id) is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        if request.rollout_id and store.rollouts.get(request.rollout_id) is None:
            raise HTTPException(status_code=404, detail="Rollout not found")
        if request.branch_id and store.branches.get(request.branch_id) is None:
            raise HTTPException(status_code=404, detail="Branch not found")
        if request.state_handle_id and store.state_handles.get(request.state_handle_id) is None:
            raise HTTPException(status_code=404, detail="State handle not found")
        checkpoint = store.create_checkpoint(request)
        if request.rollout_id:
            store.attach_checkpoint_to_rollout(request.rollout_id, checkpoint.checkpoint_id)
        return checkpoint.model_dump(mode="json")

    @app.get("/v1/checkpoints")
    async def list_checkpoints(episode_id: str | None = None, rollout_id: str | None = None):
        store: TemporalStore = app.state.temporal_store
        items = store.checkpoints.list()
        if episode_id is not None:
            items = [i for i in items if i.episode_id == episode_id]
        if rollout_id is not None:
            items = [i for i in items if i.rollout_id == rollout_id]
        return {"checkpoints": [i.model_dump(mode="json") for i in items], "count": len(items)}

    @app.get("/v1/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str):
        store: TemporalStore = app.state.temporal_store
        item = store.checkpoints.get(checkpoint_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        return item.model_dump(mode="json")

    return app


def main():
    import uvicorn

    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")


if __name__ == "__main__":
    main()
