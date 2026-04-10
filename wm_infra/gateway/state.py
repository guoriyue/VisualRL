"""Shared Gateway runtime state and request-scoped accessors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fastapi import Request

from wm_infra.config import EngineConfig

if TYPE_CHECKING:
    from fastapi import FastAPI

    from wm_infra.engine.ipc.client import EngineIPCClient


@dataclass(slots=True)
class GatewayRuntime:
    """Owns Gateway-scoped runtime dependencies and lifecycle state."""

    config: EngineConfig
    engine_client: EngineIPCClient | None = None


def bind_gateway_runtime(app: FastAPI, runtime: GatewayRuntime) -> None:
    """Attach the assembled Gateway runtime to the FastAPI app."""
    app.state.gateway_runtime = runtime


def get_gateway_runtime(request: Request) -> GatewayRuntime:
    """Return the per-app Gateway runtime for the current request."""
    return cast("GatewayRuntime", request.app.state.gateway_runtime)
