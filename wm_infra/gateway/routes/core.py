"""Core gateway routes: health, models, metrics."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


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

    @router.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)
