"""Gateway bootstrap helpers for assembling runtime dependencies."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from wm_infra.config import EngineConfig
from wm_infra.gateway.state import GatewayRuntime

logger = logging.getLogger("wm_infra")


def create_gateway_runtime(config: EngineConfig) -> GatewayRuntime:
    """Assemble static Gateway dependencies."""
    client = None
    if config.ipc.enabled:
        from wm_infra.engine.ipc.client import EngineIPCClient

        client = EngineIPCClient(ipc_path=config.ipc.socket_path)
    return GatewayRuntime(config=config, engine_client=client)


def build_gateway_lifespan(runtime: GatewayRuntime):
    """Create the FastAPI lifespan."""

    @asynccontextmanager
    async def lifespan(_app):
        device_str = (
            runtime.config.device.value
            if hasattr(runtime.config.device, "value")
            else str(runtime.config.device)
        )
        if runtime.engine_client is not None:
            await runtime.engine_client.start()
            logger.info("Gateway ready: device=%s, IPC connected", device_str)
        else:
            logger.info("Gateway ready: device=%s", device_str)
        yield
        if runtime.engine_client is not None:
            await runtime.engine_client.stop()
        logger.info("Shutting down gateway")

    return lifespan
