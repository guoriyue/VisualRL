"""Execution-plane objects and helpers for Cosmos world-generation jobs."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wm_infra.execution import ExecutionRuntimeTrace, ExecutionStageRecord


class CosmosQueueLane(str, Enum):
    """Execution lane used for coarse Cosmos scheduling."""

    REFERENCE_REUSE = "reference_reuse"
    TEXT_ONLY = "text_only"
    INPUT_HEAVY = "input_heavy"


@dataclass(frozen=True, slots=True)
class CosmosBatchSignature:
    """Signature for homogeneous Cosmos sample-production work."""

    backend: str
    model: str
    stage: str
    runner_mode: str
    variant: str
    width: int
    height: int
    frame_count: int
    num_steps: int
    fps: int
    has_reference: bool


@dataclass(slots=True)
class CosmosExecutionEntity:
    """Runtime entity representing one Cosmos generation request."""

    entity_id: str
    sample_id: str
    stage: str
    priority: float
    batch_signature: CosmosBatchSignature
    queue_lane: CosmosQueueLane
    reference_key: str | None = None
    last_scheduled_at: float = 0.0


@dataclass(slots=True)
class CosmosExecutionChunk:
    """A stage-local homogeneous chunk for Cosmos execution."""

    chunk_id: str
    signature: CosmosBatchSignature
    entity_ids: list[str]
    expected_occupancy: float
    estimated_units: float

    @property
    def size(self) -> int:
        return len(self.entity_ids)


class CosmosStageRecord(ExecutionStageRecord):
    """One runtime stage sample for Cosmos profiling."""


class CosmosRuntimeTrace(ExecutionRuntimeTrace):
    """Per-request runtime trace for Cosmos stage scheduling."""


def prompt_reference_key(prompt: str, references: list[str]) -> str | None:
    if not prompt and not references:
        return None
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    for item in references:
        digest.update(item.encode("utf-8"))
    return digest.hexdigest()


def expected_occupancy(chunk_size: int, max_chunk_size: int) -> float:
    if max_chunk_size <= 0:
        return 0.0
    return min(max(chunk_size / max_chunk_size, 0.0), 1.0)


def queue_lane_for_request(has_reference: bool, reuse_hit: bool) -> CosmosQueueLane:
    if reuse_hit:
        return CosmosQueueLane.REFERENCE_REUSE
    if has_reference:
        return CosmosQueueLane.INPUT_HEAVY
    return CosmosQueueLane.TEXT_ONLY


@dataclass(slots=True)
class CachedCosmosInput:
    cache_key: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    reuse_count: int = 0


class CosmosInputCache:
    """Best-effort cache keyed by prompt + reference identity."""

    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self._entries: dict[str, CachedCosmosInput] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, cache_key: str | None) -> CachedCosmosInput | None:
        if cache_key is None:
            self._misses += 1
            return None
        entry = self._entries.get(cache_key)
        if entry is None:
            self._misses += 1
            return None
        entry.last_accessed = time.time()
        entry.reuse_count += 1
        self._hits += 1
        return entry

    def put(self, cache_key: str | None) -> None:
        if cache_key is None:
            return
        if cache_key in self._entries:
            self._entries[cache_key].last_accessed = time.time()
            return
        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda key: self._entries[key].last_accessed)
            self._entries.pop(oldest)
            self._evictions += 1
        self._entries[cache_key] = CachedCosmosInput(cache_key=cache_key)

    def snapshot(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }
