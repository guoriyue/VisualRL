"""Gateway route registration helpers."""

from wm_infra.gateway.routes.samples import register_sample_routes
from wm_infra.gateway.routes.temporal import register_temporal_routes

__all__ = [
    "register_sample_routes",
    "register_temporal_routes",
]
