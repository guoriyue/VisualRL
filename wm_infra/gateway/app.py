"""Gateway app factory and entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from wm_infra.config import EngineConfig, load_config
from wm_infra.engine.metrics import API_AUTH_FAILURES
from wm_infra.gateway.bootstrap import build_gateway_lifespan, create_gateway_runtime
from wm_infra.gateway.routes import register_routes
from wm_infra.gateway.state import bind_gateway_runtime


def create_app(config: EngineConfig | None = None):
    """Create the Gateway application."""
    resolved_config = config or EngineConfig()
    runtime = create_gateway_runtime(resolved_config)
    app = FastAPI(
        title="wm-infra",
        description="Video generation model serving",
        version="0.1.0",
        lifespan=build_gateway_lifespan(runtime),
    )
    bind_gateway_runtime(app, runtime)

    @app.middleware("http")
    async def api_key_guard(request, call_next):
        api_key = runtime.config.server.api_key
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

    register_routes(app)
    return app


def main() -> None:
    """Run the Gateway app with uvicorn."""
    import uvicorn

    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
