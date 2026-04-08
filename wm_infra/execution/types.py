"""Execution-plane types for homogeneous temporal work."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any


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
