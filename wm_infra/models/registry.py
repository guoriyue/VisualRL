"""Unified model registry for staged temporal generation models.

All models — Wan, Cosmos, action-conditioned interactive video generators
(Matrix-Game-3), and future temporal models — implement the
``VideoGenerationModel`` contract and register here.
"""

from __future__ import annotations

from typing import Callable, Type

from wm_infra.models.video_generation import VideoGenerationModel

_REGISTRY: dict[str, Type[VideoGenerationModel]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model class implementing VideoGenerationModel."""
    def wrapper(cls: Type[VideoGenerationModel]) -> Type[VideoGenerationModel]:
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name: str) -> Type[VideoGenerationModel]:
    """Get a registered model class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return sorted(_REGISTRY.keys())


# Register built-in models
from wm_infra.models.cosmos_adapter import CosmosGenerationModel  # noqa: E402
from wm_infra.models.wan_official import OfficialWanModel  # noqa: E402
from wm_infra.models.wan_diffusers_i2v import DiffusersWanI2VModel  # noqa: E402

register_model("cosmos")(CosmosGenerationModel)
register_model("wan-official")(OfficialWanModel)
register_model("wan-diffusers-i2v")(DiffusersWanI2VModel)
