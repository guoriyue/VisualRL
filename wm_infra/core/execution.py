"""Execution-plane objects for homogeneous temporal work."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


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
