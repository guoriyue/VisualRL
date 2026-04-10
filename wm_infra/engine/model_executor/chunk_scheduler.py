"""Shared schedulers for homogeneous temporal execution.

Merges:
- execution/scheduler.py (HomogeneousChunkScheduler, SchedulerDecision, etc.)
- execution/profiling.py (ExecutionRuntimeTrace, ExecutionStageRecord)

Execution-specific types (BatchSignature, ExecutionChunk, etc.) are defined
locally since they are tightly coupled to chunk scheduling and not part of
the model-agnostic engine types.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Generic, TypeVar

# ---------------------------------------------------------------------------
# Execution types (formerly in engine/types.py)
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
        fill_ratio = (
            (self.transition_entities / self.transition_chunks) if self.transition_chunks else 0.0
        )
        return {
            "mode": self.mode,
            "transition_chunks": self.transition_chunks,
            "transition_entities": self.transition_entities,
            "max_transition_chunk_size": self.max_transition_chunk_size,
            "avg_transition_chunk_size": avg_chunk_size,
            "avg_transition_fill_ratio": fill_ratio,
        }


# ---------------------------------------------------------------------------
# Profiling types (formerly in engine/types.py)
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
            "avg_expected_occupancy": (sum(occupancies) / len(occupancies))
            if occupancies
            else 0.0,
        }
        if self.include_estimated_transfer_bytes:
            summary["estimated_transfer_bytes"] = transfer_bytes
        return summary


# ---------------------------------------------------------------------------
# Chunk scheduler types
# ---------------------------------------------------------------------------

TEntity = TypeVar("TEntity")
TChunk = TypeVar("TChunk")
TGroupKey = TypeVar("TGroupKey", bound=Hashable)


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    """Batching decision for one homogeneous signature lane."""

    signature: BatchSignature
    ready_count: int
    chunk_count: int
    policy_mode: str


@dataclass(frozen=True, slots=True)
class GroupedChunkDecision(Generic[TChunk]):
    """One backend-specific chunk plus scheduler metadata.

    This keeps the repeated ``group -> sort -> chunk -> annotate`` substrate in
    one place while allowing each backend to keep its own chunk type and policy
    inputs.
    """

    chunk: TChunk
    scheduler_inputs: dict[str, float | int | str | bool]


class HomogeneousChunkScheduler:
    """Group work by signature, then emit fixed-size stage-local chunks.

    This keeps the scheduler substrate separate from env or backend-specific
    orchestration. The current implementation is synchronous and deterministic,
    but the boundary matches a future async send/recv style runtime.
    """

    def schedule(
        self,
        *,
        work_items: list[ExecutionWorkItem],
        policy: ExecutionBatchPolicy,
        chunk_id_prefix: str,
        latent_join: Callable[[list[Any]], Any],
        action_join: Callable[[list[Any]], Any],
    ) -> tuple[list[ExecutionChunk], list[SchedulerDecision]]:
        if not work_items:
            return [], []

        max_chunk_size = max(1, policy.max_chunk_size)
        grouped_items: OrderedDict[BatchSignature, list[ExecutionWorkItem]] = OrderedDict()
        for item in work_items:
            grouped_items.setdefault(item.entity.batch_signature, []).append(item)

        chunks: list[ExecutionChunk] = []
        decisions: list[SchedulerDecision] = []
        chunk_index = 0
        for signature, signature_items in grouped_items.items():
            signature_chunk_count = 0
            for offset in range(0, len(signature_items), max_chunk_size):
                chunk_items = signature_items[offset : offset + max_chunk_size]
                if len(chunk_items) < policy.min_ready_size and not policy.allow_partial_batch:
                    continue
                chunks.append(
                    ExecutionChunk(
                        chunk_id=f"{chunk_id_prefix}:{chunk_index}",
                        signature=signature,
                        entities=[item.entity for item in chunk_items],
                        latent_batch=latent_join([item.latent_item for item in chunk_items]),
                        action_batch=action_join([item.action_item for item in chunk_items]),
                    )
                )
                signature_chunk_count += 1
                chunk_index += 1
            decisions.append(
                SchedulerDecision(
                    signature=signature,
                    ready_count=len(signature_items),
                    chunk_count=signature_chunk_count,
                    policy_mode=policy.mode,
                )
            )
        return chunks, decisions


def schedule_grouped_chunks(
    *,
    entities: Iterable[TEntity],
    max_chunk_size: int,
    group_key: Callable[[TEntity], TGroupKey],
    entity_sort_key: Callable[[TEntity], Any],
    build_chunk: Callable[[TGroupKey, list[TEntity], int], TChunk],
    build_scheduler_inputs: Callable[
        [TGroupKey, list[TEntity], TChunk, int], dict[str, float | int | str | bool]
    ],
    decision_sort_key: Callable[[GroupedChunkDecision[TChunk]], Any] | None = None,
) -> list[GroupedChunkDecision[TChunk]]:
    """Build homogeneous backend chunks from grouped entities.

    Backend schedulers often share the same structural workflow even when their
    chunk payloads and prioritization inputs differ. This helper centralizes
    the framework behavior so backends only define grouping, chunk assembly,
    and policy-specific metadata.
    """

    grouped_entities: dict[TGroupKey, list[TEntity]] = defaultdict(list)
    for entity in entities:
        grouped_entities[group_key(entity)].append(entity)

    group_count = len(grouped_entities)
    if group_count == 0:
        return []

    chunk_capacity = max(1, max_chunk_size)
    decisions: list[GroupedChunkDecision[TChunk]] = []
    for key, items in grouped_entities.items():
        items.sort(key=entity_sort_key)
        for chunk_index, offset in enumerate(range(0, len(items), chunk_capacity)):
            chunk_entities = items[offset : offset + chunk_capacity]
            chunk = build_chunk(key, chunk_entities, chunk_index)
            scheduler_inputs = build_scheduler_inputs(key, chunk_entities, chunk, group_count)
            decisions.append(GroupedChunkDecision(chunk=chunk, scheduler_inputs=scheduler_inputs))

    if decision_sort_key is not None:
        decisions.sort(key=decision_sort_key)
    return decisions


def build_execution_chunks(
    *,
    signature: BatchSignature,
    entities: list[Any],
    latent_items: list[Any],
    action_items: list[Any],
    policy: ExecutionBatchPolicy,
    chunk_id_prefix: str,
    latent_join: Callable[[list[Any]], Any],
    action_join: Callable[[list[Any]], Any],
) -> list[ExecutionChunk]:
    """Compatibility helper for call sites that already precomputed one signature."""

    if len(entities) != len(latent_items) or len(entities) != len(action_items):
        raise ValueError("entities, latent_items, and action_items must have the same length")

    work_items = [
        ExecutionWorkItem(
            entity=entity,
            latent_item=latent_item,
            action_item=action_item,
        )
        for entity, latent_item, action_item in zip(
            entities, latent_items, action_items, strict=True
        )
    ]
    chunks, _ = HomogeneousChunkScheduler().schedule(
        work_items=work_items,
        policy=policy,
        chunk_id_prefix=chunk_id_prefix,
        latent_join=latent_join,
        action_join=action_join,
    )

    if any(chunk.signature != signature for chunk in chunks):
        raise ValueError(
            "build_execution_chunks received entities with mismatched batch signatures"
        )
    return chunks


__all__ = [
    "BatchSignature",
    "ExecutionBatchPolicy",
    "ExecutionChunk",
    "ExecutionEntity",
    "ExecutionRuntimeTrace",
    "ExecutionStageRecord",
    "ExecutionStats",
    "ExecutionWorkItem",
    "GroupedChunkDecision",
    "HomogeneousChunkScheduler",
    "SchedulerDecision",
    "build_execution_chunks",
    "chunk_fill_ratio",
    "schedule_grouped_chunks",
    "summarize_execution_chunks",
]
