"""Core gateway routes: health, models, rollout."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from wm_infra.gateway.state import get_gateway_runtime


class RolloutSubmitRequest(BaseModel):
    """Request body for POST /v1/rollout."""

    request_id: str | None = None
    num_steps: int
    priority: float = 0.0
    action_sequence: list[float | int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    prefix_hash: str | None = None


def register_routes(app: FastAPI) -> None:
    """Register core serving routes."""
    router = APIRouter()

    @router.get("/v1/health")
    async def health():
        return {"status": "ready"}

    @router.get("/v1/models")
    async def list_models():
        from wm_infra.models.registry import list_models as _list_models

        return {"models": _list_models()}

    # ------------------------------------------------------------------
    # Rollout routes (active only when engine_client is connected)
    # ------------------------------------------------------------------

    @router.post("/v1/rollout", status_code=202)
    async def submit_rollout(request: Request, body: RolloutSubmitRequest):
        """Submit a rollout request to the engine."""
        runtime = get_gateway_runtime(request)
        if runtime.engine_client is None:
            raise HTTPException(status_code=503, detail="Engine IPC not configured")

        request_id = body.request_id or uuid.uuid4().hex
        result = await runtime.engine_client.submit(
            request_id=request_id,
            num_steps=body.num_steps,
            priority=body.priority,
            action_sequence=body.action_sequence,
            metadata=body.metadata,
            prefix_hash=body.prefix_hash,
        )
        return JSONResponse(
            status_code=202,
            content={
                "request_id": request_id,
                "accepted": result.get("accepted", False),
                "queue_position": result.get("queue_position", 0),
                "error": result.get("error"),
            },
        )

    @router.get("/v1/rollout/{request_id}")
    async def get_rollout(request: Request, request_id: str):
        """Get rollout status and result."""
        runtime = get_gateway_runtime(request)
        if runtime.engine_client is None:
            raise HTTPException(status_code=503, detail="Engine IPC not configured")

        status = await runtime.engine_client.get_status(request_id)
        phase = status.get("phase", "unknown")

        if phase in {"done", "finished"}:
            result = await runtime.engine_client.get_result(request_id)
            return {
                "request_id": request_id,
                "phase": phase,
                "done": result.get("done", False),
                "step_index": status.get("step_index", 0),
                "num_steps": status.get("num_steps", 0),
                "artifact": result.get("artifact"),
                "meta": result.get("meta"),
            }

        return {
            "request_id": request_id,
            "phase": phase,
            "done": False,
            "step_index": status.get("step_index", 0),
            "num_steps": status.get("num_steps", 0),
        }

    @router.delete("/v1/rollout/{request_id}")
    async def cancel_rollout(request: Request, request_id: str):
        """Cancel a rollout request."""
        runtime = get_gateway_runtime(request)
        if runtime.engine_client is None:
            raise HTTPException(status_code=503, detail="Engine IPC not configured")

        result = await runtime.engine_client.cancel(request_id)
        return {
            "request_id": request_id,
            "cancelled": result.get("cancelled", False),
        }

    app.include_router(router)
