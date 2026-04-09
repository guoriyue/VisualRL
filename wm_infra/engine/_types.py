"""Core types for the unified runtime engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Phase(IntEnum):
    """Lifecycle phase of an entity request inside the engine."""

    WAITING = 0       # queued, not yet admitted
    ENCODING = 1      # running the encoder (tokenizer / observation encoder)
    STEPPING = 2      # running dynamics model forward steps
    SWAPPED = 3       # preempted, latent pages swapped to CPU
    DONE = 4          # finished, awaiting result drain


@dataclass(frozen=True, slots=True)
class EngineRunConfig:
    """Static configuration for one engine instance."""

    max_num_blocks: int = 1024
    block_size: int = 1
    latent_tokens: int = 256
    latent_dim: int = 16
    max_batch_size: int = 64
    max_steps_per_entity: int = 128
    swap_enabled: bool = True
    device: str = "cpu"

    @property
    def pool_shape(self) -> tuple[int, int, int, int]:
        return (self.max_num_blocks, self.block_size, self.latent_tokens, self.latent_dim)


@dataclass(slots=True)
class EntityRequest:
    """One entity's request to the engine.

    Each entity corresponds to a single world-model rollout (e.g. one
    environment trajectory or one video generation request).
    """

    request_id: str
    num_steps: int
    action_sequence: list[Any] = field(default_factory=list)
    initial_latent: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    prefix_hash: str | None = None


@dataclass(slots=True)
class SchedulerOutput:
    """Decision produced by the scheduler on each iteration.

    Lists of request IDs partitioned by what action the engine should take.
    """

    encode_ids: list[str] = field(default_factory=list)
    step_ids: list[str] = field(default_factory=list)
    preempt_ids: list[str] = field(default_factory=list)
    swap_in_ids: list[str] = field(default_factory=list)
    done_ids: list[str] = field(default_factory=list)
    num_free_blocks: int = 0


@dataclass(slots=True)
class StepResult:
    """Result of one dynamics step for one entity."""

    request_id: str
    step_index: int
    output_latent: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass(frozen=True, slots=True)
class SwapHandle:
    """Opaque handle tracking a GPU <-> CPU swap for one entity's blocks."""

    request_id: str
    block_ids: tuple[int, ...]
    direction: str  # "out" or "in"


# ---------------------------------------------------------------------------
# Schemas that the engine owns but the control-plane re-exports
# ---------------------------------------------------------------------------

from enum import Enum  # noqa: E402


class VideoMemoryProfile(str, Enum):
    DEFAULT = "default"
    BALANCED = "balanced"
    LOW_VRAM = "low_vram"
    HIGH_QUALITY = "high_quality"


class RolloutTaskConfig(BaseModel):
    num_steps: int = Field(default=1, ge=1, description="Number of rollout or denoising steps to execute")
    frame_count: Optional[int] = Field(default=None, ge=1, description="Target frame count for video-like tasks")
    width: Optional[int] = Field(default=None, ge=1, description="Requested output width")
    height: Optional[int] = Field(default=None, ge=1, description="Requested output height")
    offload_model: Optional[bool] = Field(default=None, description="Whether model weights should be CPU/offload backed")
    convert_model_dtype: Optional[bool] = Field(default=None, description="Whether to enable reduced-precision model conversion")
    t5_cpu: Optional[bool] = Field(default=None, description="Whether text encoder work should stay on CPU")
    memory_profile: Optional[VideoMemoryProfile] = Field(default=None, description="Coarse memory/quality mode for schedulers and backends")
