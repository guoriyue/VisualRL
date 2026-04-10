"""Gateway bootstrap helpers for assembling runtime dependencies."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from wm_infra.config import EngineConfig
from wm_infra.gateway.state import GatewayRuntime

logger = logging.getLogger("wm_infra")


def create_gateway_runtime(config: EngineConfig) -> GatewayRuntime:
    """Assemble static Gateway dependencies."""
    return GatewayRuntime(config=config)


def build_gateway_lifespan(runtime: GatewayRuntime):
    """Create the FastAPI lifespan."""

    @asynccontextmanager
    async def lifespan(_app):
        device_str = (
            runtime.config.device.value
            if hasattr(runtime.config.device, "value")
            else str(runtime.config.device)
        )
        logger.info("Gateway ready: device=%s", device_str)
        yield
        logger.info("Shutting down gateway")

    return lifespan
