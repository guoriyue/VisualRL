"""FastAPI server for world model inference.

Provides:
- POST /v1/rollout — submit a rollout prediction
- GET  /v1/rollout/{job_id} — get rollout result
- GET  /v1/health — health check
- GET  /v1/models — list available models

Uses AsyncWorldModelEngine for non-blocking concurrent request handling.
Multiple concurrent POST /v1/rollout requests are automatically batched
by the engine's background loop.
"""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np

from wm_infra.config import EngineConfig, DynamicsConfig, TokenizerConfig
from wm_infra.core.engine import AsyncWorldModelEngine, RolloutJob
from wm_infra.models.dynamics import LatentDynamicsModel
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer
from wm_infra.api.protocol import (
    RolloutRequest,
    RolloutResponse,
    HealthResponse,
    ModelInfo,
)

logger = logging.getLogger("wm_infra")

# Global engine instance
_engine: Optional[AsyncWorldModelEngine] = None


def _create_async_engine(config: EngineConfig) -> AsyncWorldModelEngine:
    """Create async engine with model and optional tokenizer."""
    dynamics = LatentDynamicsModel(config.dynamics)
    tokenizer = VideoTokenizer(config.tokenizer)

    if config.model_path:
        state_dict = torch.load(config.model_path, map_location="cpu", weights_only=True)
        dynamics.load_state_dict(state_dict.get("dynamics", state_dict), strict=False)
        if "tokenizer" in state_dict:
            tokenizer.load_state_dict(state_dict["tokenizer"], strict=False)

    return AsyncWorldModelEngine(config, dynamics, tokenizer)


def create_app(config: Optional[EngineConfig] = None):
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    if config is None:
        config = EngineConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine
        logger.info("Initializing world model engine...")
        _engine = _create_async_engine(config)
        _engine.start()
        logger.info(
            f"Engine ready: device={config.device.value}, "
            f"dynamics={sum(p.numel() for p in _engine.engine.dynamics_model.parameters())} params"
        )
        yield
        logger.info("Shutting down engine")
        await _engine.stop()

    app = FastAPI(
        title="wm-infra",
        description="World Model Inference Infrastructure",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_engine is not None,
            active_rollouts=_engine.engine.state_manager.num_active if _engine else 0,
            memory_used_gb=_engine.engine.state_manager.memory_used_gb if _engine else 0.0,
        )

    @app.get("/v1/models")
    async def list_models():
        from wm_infra.models.registry import list_models as _list_models
        return {"models": _list_models()}

    @app.post("/v1/rollout", response_model=RolloutResponse)
    async def submit_rollout(request: RolloutRequest):
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Build job
        job = RolloutJob(
            job_id="",
            num_steps=request.num_steps,
            return_frames=request.return_frames,
            return_latents=request.return_latents,
            stream=request.stream,
        )

        # Parse initial state
        if request.initial_latent is not None:
            job.initial_latent = torch.tensor(request.initial_latent, dtype=torch.float32)
        elif request.initial_observation_b64 is not None:
            img_bytes = base64.b64decode(request.initial_observation_b64)
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                job.initial_observation = torch.from_numpy(img_np).permute(2, 0, 1)
            except ImportError:
                raise HTTPException(status_code=400, detail="PIL required for image decoding")
        else:
            # Generate random initial state for demo
            N = config.state_cache.num_latent_tokens
            D = config.dynamics.latent_token_dim
            job.initial_latent = torch.randn(N, D)

        # Parse actions
        if request.actions is not None:
            job.actions = torch.tensor(request.actions, dtype=torch.float32)

        # Submit to async engine — awaits completion without blocking event loop
        result = await _engine.submit(job)

        response = RolloutResponse(
            job_id=result.job_id,
            model=request.model,
            steps_completed=result.steps_completed,
            elapsed_ms=result.elapsed_ms,
        )

        if result.predicted_latents is not None:
            response.latents = result.predicted_latents.cpu().tolist()

        if result.predicted_frames is not None:
            frames_b64 = []
            for t in range(result.predicted_frames.shape[0]):
                frame = result.predicted_frames[t]  # [C, H, W]
                frame_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                frame_np = frame_np.transpose(1, 2, 0)  # [H, W, C]
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

    @app.get("/v1/rollout/{job_id}")
    async def get_rollout(job_id: str):
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        result = _engine.engine.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": result.job_id,
            "steps_completed": result.steps_completed,
            "elapsed_ms": result.elapsed_ms,
        }

    return app


def main():
    """Entry point for `wm-serve` CLI command."""
    import uvicorn

    config = EngineConfig()
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
