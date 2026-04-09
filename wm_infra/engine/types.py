"""Core types for the unified runtime engine.

Consolidates all shared data types from:
- engine._types (Phase, EngineRunConfig, EntityRequest, SchedulerOutput, etc.)
- engine.execution.types (BatchSignature, ExecutionChunk, ExecutionStats, etc.)
- engine.execution.profiling (ExecutionRuntimeTrace, ExecutionStageRecord)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from statistics import mean
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase / engine config / scheduler types (from _types.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Execution types (from execution/types.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatchSignature:
    """Execution signature for grouping homogeneous rollout work."""

    stage: str
    latent_shape: tuple[int, ...]
    action_dim: int
    dtype: str
    device: str
    needs_decode: bool


@dataclass(slots=True)
class ExecutionEntity:
    """Transient runtime entity for one shard of rollout work."""

    entity_id: str
    rollout_id: str
    stage: str
    step_idx: int
    batch_signature: BatchSignature


@dataclass(slots=True)
class ExecutionWorkItem:
    """One schedulable unit before a homogeneous execution chunk is built."""

    entity: ExecutionEntity
    latent_item: Any
    action_item: Any


@dataclass(slots=True)
class ExecutionChunk:
    """Homogeneous execution chunk sent through one stage-local pass."""

    chunk_id: str
    signature: BatchSignature
    entities: list[ExecutionEntity]
    latent_batch: object
    action_batch: object

    @property
    def size(self) -> int:
        return len(self.entities)


@dataclass(frozen=True, slots=True)
class ExecutionBatchPolicy:
    """Batching policy for one homogeneous execution stage."""

    mode: str = "sync"
    max_chunk_size: int = 1
    min_ready_size: int = 1
    return_when_ready_count: int | None = None
    allow_partial_batch: bool = True

    def normalized_return_when_ready_count(self) -> int:
        if self.return_when_ready_count is None:
            return self.max_chunk_size
        return self.return_when_ready_count


def chunk_fill_ratio(size: int, max_chunk_size: int) -> float:
    """Return the occupancy ratio for one execution chunk."""

    if max_chunk_size <= 0:
        return 0.0
    return min(max(size / max_chunk_size, 0.0), 1.0)


def summarize_execution_chunks(
    chunks: list[ExecutionChunk],
    *,
    policy: ExecutionBatchPolicy,
) -> dict[str, Any]:
    """Summarize chunk formation under one execution policy."""

    chunk_sizes = [chunk.size for chunk in chunks]
    fill_ratios = [chunk_fill_ratio(chunk.size, policy.max_chunk_size) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "chunk_sizes": chunk_sizes,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
        "chunk_fill_ratios": fill_ratios,
        "avg_chunk_fill_ratio": (sum(fill_ratios) / len(fill_ratios)) if fill_ratios else 0.0,
        "batch_policy": {
            "mode": policy.mode,
            "max_chunk_size": policy.max_chunk_size,
            "min_ready_size": policy.min_ready_size,
            "return_when_ready_count": policy.normalized_return_when_ready_count(),
            "allow_partial_batch": policy.allow_partial_batch,
        },
    }


@dataclass(slots=True)
class ExecutionStats:
    """Lightweight in-memory stats for benchmark comparison."""

    mode: str
    transition_chunks: int = 0
    transition_entities: int = 0
    max_transition_chunk_size: int = 0
    chunk_sizes: list[int] | None = None

    def __post_init__(self) -> None:
        if self.chunk_sizes is None:
            self.chunk_sizes = []

    def record_transition_chunk(self, size: int) -> None:
        self.transition_chunks += 1
        self.transition_entities += size
        self.max_transition_chunk_size = max(self.max_transition_chunk_size, size)
        self.chunk_sizes.append(size)

    def snapshot(self) -> dict[str, float | int | str]:
        avg_chunk_size = mean(self.chunk_sizes) if self.chunk_sizes else 0.0
        fill_ratio = (self.transition_entities / self.transition_chunks) if self.transition_chunks else 0.0
        return {
            "mode": self.mode,
            "transition_chunks": self.transition_chunks,
            "transition_entities": self.transition_entities,
            "max_transition_chunk_size": self.max_transition_chunk_size,
            "avg_transition_chunk_size": avg_chunk_size,
            "avg_transition_fill_ratio": fill_ratio,
        }


# ---------------------------------------------------------------------------
# Profiling types (from execution/profiling.py)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionStageRecord:
    """One stage execution sample in a backend runtime trace."""

    stage: str
    entity_id: str
    queue_lane: str
    elapsed_ms: float
    started_at: float | None = None
    chunk_id: str | None = None
    chunk_size: int | None = None
    expected_occupancy: float | None = None
    estimated_transfer_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionRuntimeTrace:
    """Shared per-request runtime trace for staged backend execution."""

    include_estimated_transfer_bytes: bool = False

    records: list[ExecutionStageRecord] = field(default_factory=list)
    queue_lanes_seen: list[str] = field(default_factory=list)

    def record(self, record: ExecutionStageRecord) -> None:
        self.records.append(record)
        if record.queue_lane not in self.queue_lanes_seen:
            self.queue_lanes_seen.append(record.queue_lane)

    def stage_timings_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.records:
            totals[record.stage] = round(totals.get(record.stage, 0.0) + record.elapsed_ms, 3)
        return totals

    def chunk_summary(self) -> dict[str, Any]:
        chunk_count = 0
        chunk_sizes: list[int] = []
        occupancies: list[float] = []
        transfer_bytes = 0
        for record in self.records:
            if record.chunk_id is None:
                continue
            chunk_count += 1
            if record.chunk_size is not None:
                chunk_sizes.append(record.chunk_size)
            if record.expected_occupancy is not None:
                occupancies.append(record.expected_occupancy)
            if record.estimated_transfer_bytes is not None:
                transfer_bytes += record.estimated_transfer_bytes

        summary = {
            "chunk_count": chunk_count,
            "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_expected_occupancy": (sum(occupancies) / len(occupancies)) if occupancies else 0.0,
        }
        if self.include_estimated_transfer_bytes:
            summary["estimated_transfer_bytes"] = transfer_bytes
        return summary
