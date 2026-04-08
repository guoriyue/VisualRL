"""World-model rollout runtime exports.

This package owns the lower-level rollout-engine path that previously lived in
``wm_infra.core``. It remains distinct from ``wm_infra.env_runtime`` because it is still
the direct observation -> latent -> transition -> decode engine used by the
legacy rollout surface.
"""

from wm_infra.rollout_engine.engine import AsyncWorldModelEngine, RolloutJob, RolloutResult, WorldModelEngine
from wm_infra.rollout_engine.scheduler import (
    DEFAULT_RESOURCE_UNITS_PER_GB,
    DEFAULT_FRAME_COUNT,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    HIGH_QUALITY_MEMORY_MULTIPLIER,
    LOW_VRAM_MEMORY_MULTIPLIER,
    RolloutRequest,
    RolloutScheduler,
    ScheduledBatch,
)
from wm_infra.rollout_engine.state import LatentStateManager, RolloutState

__all__ = [
    "AsyncWorldModelEngine",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_HEIGHT",
    "DEFAULT_RESOURCE_UNITS_PER_GB",
    "DEFAULT_WIDTH",
    "HIGH_QUALITY_MEMORY_MULTIPLIER",
    "LOW_VRAM_MEMORY_MULTIPLIER",
    "LatentStateManager",
    "RolloutJob",
    "RolloutRequest",
    "RolloutResult",
    "RolloutScheduler",
    "RolloutState",
    "ScheduledBatch",
    "WorldModelEngine",
]
