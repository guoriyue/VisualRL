"""Execution-plane objects and state helpers for stage-oriented Genie runtime."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional

import numpy as np


GENIE_STAGE_GRAPH = [
    "admission",
    "state_materialize",
    "prompt_prepare",
    "transition",
    "checkpoint",
    "artifact_persist",
    "controlplane_commit",
]


class GenieQueueLane(str, Enum):
    """Execution lane for Genie stage scheduling."""

    HOT_CONTINUATION = "hot_continuation"
    COLD_MATERIALIZE = "cold_materialize"
    CHECKPOINT_HEAVY = "checkpoint_heavy"
    PERSIST_ONLY = "persist_only"


class GenieResidencyTier(str, Enum):
    """Best-effort residency tier for prompt / token state."""

    HOT_GPU = "hot_gpu"
    WARM_PINNED_CPU = "warm_pinned_cpu"
    COLD_FILE = "cold_file"


@dataclass(frozen=True, slots=True)
class GenieBatchSignature:
    """Execution signature for grouping homogeneous Genie work."""

    backend: str
    model_name: str
    stage: str
    device: str
    dtype: str
    tokenizer_kind: str
    spatial_h: int
    spatial_w: int
    window_num_frames: int
    num_prompt_frames: int
    maskgit_steps: int
    temperature_bucket: str
    checkpoint_every_n_frames: int
    runner_mode: str
    needs_persist: bool


@dataclass(slots=True)
class GenieExecutionEntity:
    """A bounded schedulable window of Genie rollout work."""

    entity_id: str
    rollout_id: str
    episode_id: str
    branch_id: str | None
    sample_id: str
    input_state_handle_id: str | None
    current_stage: str
    next_stage: str | None
    window_start_frame: int
    window_num_frames: int
    total_frames: int
    num_prompt_frames: int
    checkpoint_every_n_frames: int
    priority: float
    deadline_s: float | None
    batch_signature: GenieBatchSignature
    queue_lane: GenieQueueLane
    stage_attempts: int = 0
    last_scheduled_at: float = 0.0

    @property
    def window_end_frame(self) -> int:
        return self.window_start_frame + self.window_num_frames


@dataclass(slots=True)
class GenieRuntimeState:
    """Hot execution state for a Genie rollout."""

    rollout_id: str
    prompt_tokens_ref: object | None = None
    generated_tokens_ref: object | None = None
    last_completed_frame: int = 0
    resident_tier: GenieResidencyTier = GenieResidencyTier.COLD_FILE
    ancestor_state_ref: str | None = None
    checkpoint_delta_ref: str | None = None
    materialized_bytes: int = 0
    dirty_since_checkpoint: bool = False
    prompt_reuse_hit: bool = False
    source_cache_key: str | None = None
    reuse_hits: int = 0
    reuse_misses: int = 0


@dataclass(slots=True)
class GenieExecutionChunk:
    """A stage-local homogeneous chunk sent to a Genie worker."""

    chunk_id: str
    signature: GenieBatchSignature
    entity_ids: list[str]
    runnable_stage: str
    frame_range: tuple[int, int]
    estimated_vram_bytes: int
    estimated_transfer_bytes: int
    estimated_flops: float
    queue_lane: GenieQueueLane
    expected_occupancy: float

    @property
    def size(self) -> int:
        return len(self.entity_ids)

    @property
    def fill_ratio(self) -> float:
        return self.expected_occupancy

    @property
    def scheduler_score(self) -> float:
        return self.expected_occupancy - (self.estimated_transfer_bytes / max(self.estimated_vram_bytes, 1))


@dataclass(slots=True)
class GenieStageRecord:
    """One stage execution sample for runtime introspection."""

    stage: str
    entity_id: str
    queue_lane: str
    started_at: float
    elapsed_ms: float
    chunk_id: str | None = None
    chunk_size: int | None = None
    expected_occupancy: float | None = None
    estimated_transfer_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenieRuntimeTrace:
    """Per-request runtime trace for stage-local profiling and debugging."""

    records: list[GenieStageRecord] = field(default_factory=list)
    queue_lanes_seen: list[str] = field(default_factory=list)
    chunk_signatures: list[dict[str, Any]] = field(default_factory=list)

    def record(self, record: GenieStageRecord) -> None:
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
        occupancy: list[float] = []
        transfer_bytes = 0
        for record in self.records:
            if record.chunk_id is None:
                continue
            chunk_count += 1
            if record.chunk_size is not None:
                chunk_sizes.append(record.chunk_size)
            if record.expected_occupancy is not None:
                occupancy.append(record.expected_occupancy)
            if record.estimated_transfer_bytes is not None:
                transfer_bytes += record.estimated_transfer_bytes
        return {
            "chunk_count": chunk_count,
            "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_expected_occupancy": (sum(occupancy) / len(occupancy)) if occupancy else 0.0,
            "estimated_transfer_bytes": transfer_bytes,
        }


@dataclass(slots=True)
class CachedPromptState:
    """Best-effort reusable prompt state entry."""

    cache_key: str
    tokens: np.ndarray
    source_state_handle_id: str | None
    resident_tier: GenieResidencyTier
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    reuse_count: int = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.tokens.nbytes)


def _stable_token_hash(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def prompt_cache_key(
    *,
    input_state_handle_id: str | None,
    input_tokens: np.ndarray | None,
    prompt: str,
    seed: int,
) -> str | None:
    """Build a stable cache key for prompt materialization."""

    if input_state_handle_id:
        return f"state_handle:{input_state_handle_id}"
    if input_tokens is not None:
        return f"tokens:{_stable_token_hash(np.asarray(input_tokens, dtype=np.uint32))}"
    if prompt:
        prompt_hash = hashlib.sha256(f"{prompt}:{seed}".encode("utf-8")).hexdigest()
        return f"prompt:{prompt_hash}"
    return None


class GeniePromptStateCache:
    """In-memory prompt state cache used for warm continuation and branch reuse."""

    def __init__(self, max_entries: int = 32):
        self.max_entries = max_entries
        self._entries: dict[str, CachedPromptState] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "bytes_hot": 0,
        }

    def get(self, cache_key: str | None) -> CachedPromptState | None:
        if cache_key is None:
            self._stats["misses"] += 1
            return None
        entry = self._entries.get(cache_key)
        if entry is None:
            self._stats["misses"] += 1
            return None
        entry.last_accessed = time.time()
        entry.reuse_count += 1
        self._stats["hits"] += 1
        return entry

    def put(
        self,
        cache_key: str | None,
        tokens: np.ndarray,
        *,
        source_state_handle_id: str | None,
        resident_tier: GenieResidencyTier = GenieResidencyTier.WARM_PINNED_CPU,
    ) -> CachedPromptState | None:
        if cache_key is None:
            return None
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            self._stats["bytes_hot"] -= entry.memory_bytes
        else:
            self._evict_if_needed()
        entry = CachedPromptState(
            cache_key=cache_key,
            tokens=np.asarray(tokens, dtype=np.uint32).copy(),
            source_state_handle_id=source_state_handle_id,
            resident_tier=resident_tier,
        )
        self._entries[cache_key] = entry
        self._stats["bytes_hot"] += entry.memory_bytes
        return entry

    def promote(self, cache_key: str | None, tier: GenieResidencyTier) -> None:
        if cache_key is None:
            return
        entry = self._entries.get(cache_key)
        if entry is not None:
            entry.resident_tier = tier
            entry.last_accessed = time.time()

    def snapshot(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "bytes_hot": self._stats["bytes_hot"],
        }

    def _evict_if_needed(self) -> None:
        if len(self._entries) < self.max_entries:
            return
        oldest_key = min(self._entries, key=lambda key: self._entries[key].last_accessed)
        oldest = self._entries.pop(oldest_key)
        self._stats["bytes_hot"] -= oldest.memory_bytes
        self._stats["evictions"] += 1


def default_window_num_frames(
    total_frames: int,
    num_prompt_frames: int,
    checkpoint_every_n_frames: int,
) -> int:
    """Choose a bounded frame-group size for one transition window."""

    remaining = max(total_frames - num_prompt_frames, 1)
    if checkpoint_every_n_frames > 0:
        return max(1, min(checkpoint_every_n_frames, remaining))
    return remaining


def default_window_size(
    *,
    total_frames: int,
    prompt_frames: int,
    checkpoint_every_n_frames: int,
) -> int:
    """Backward-compatible alias used by runtime tests."""

    return default_window_num_frames(
        total_frames=total_frames,
        num_prompt_frames=prompt_frames,
        checkpoint_every_n_frames=checkpoint_every_n_frames,
    )


def frame_windows(
    *,
    total_frames: int,
    num_prompt_frames: int,
    checkpoint_every_n_frames: int,
) -> list[tuple[int, int]]:
    """Split the generation tail into bounded execution windows."""

    start = min(max(num_prompt_frames, 0), max(total_frames - 1, 0))
    if total_frames <= start:
        return []
    window = default_window_num_frames(total_frames, num_prompt_frames, checkpoint_every_n_frames)
    windows: list[tuple[int, int]] = []
    cursor = start
    while cursor < total_frames:
        size = min(window, total_frames - cursor)
        windows.append((cursor, size))
        cursor += size
    return windows


def temperature_bucket(temperature: float) -> str:
    """Bucket temperature for execution signature grouping."""

    if temperature <= 0:
        return "argmax"
    if temperature < 0.5:
        return "low"
    if temperature < 1.0:
        return "medium"
    return "high"


def estimate_transition_flops(
    *,
    spatial_h: int,
    spatial_w: int,
    window_num_frames: int,
    maskgit_steps: int,
) -> float:
    """Very coarse relative flops estimate for scheduling."""

    tokens = spatial_h * spatial_w * max(window_num_frames, 1)
    return float(tokens * max(maskgit_steps, 1))


def estimate_expected_occupancy(chunk_size: int, max_chunk_size: int) -> float:
    """Best-effort occupancy estimate derived from chunk fill."""

    if max_chunk_size <= 0:
        return 0.0
    return min(max(chunk_size / max_chunk_size, 0.0), 1.0)


def lane_priority(lane: GenieQueueLane) -> int:
    """Lower number means higher scheduling priority."""

    if isinstance(lane, str):
        lane = GenieQueueLane(lane)
    priorities = {
        GenieQueueLane.CHECKPOINT_HEAVY: 0,
        GenieQueueLane.HOT_CONTINUATION: 1,
        GenieQueueLane.COLD_MATERIALIZE: 2,
        GenieQueueLane.PERSIST_ONLY: 3,
    }
    return priorities[lane]


def make_stage_signature(
    *,
    backend: str,
    model_name: str,
    stage: str,
    device: str,
    dtype: str,
    tokenizer_kind: str,
    spatial_h: int,
    spatial_w: int,
    window_num_frames: int,
    num_prompt_frames: int,
    maskgit_steps: int,
    temperature: float,
    checkpoint_every_n_frames: int,
    runner_mode: str,
    needs_persist: bool,
) -> GenieBatchSignature:
    """Build a stable stage signature from Genie request shape."""

    return GenieBatchSignature(
        backend=backend,
        model_name=model_name,
        stage=stage,
        device=device,
        dtype=dtype,
        tokenizer_kind=tokenizer_kind,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        window_num_frames=window_num_frames,
        num_prompt_frames=num_prompt_frames,
        maskgit_steps=maskgit_steps,
        temperature_bucket=temperature_bucket(temperature),
        checkpoint_every_n_frames=checkpoint_every_n_frames,
        runner_mode=runner_mode,
        needs_persist=needs_persist,
    )


def build_transition_entities(root_entity: GenieExecutionEntity) -> list[GenieExecutionEntity]:
    """Split one Genie rollout tail into schedulable frame-window entities."""

    windows = frame_windows(
        total_frames=root_entity.total_frames,
        num_prompt_frames=root_entity.num_prompt_frames,
        checkpoint_every_n_frames=root_entity.checkpoint_every_n_frames,
    )
    if not windows:
        return []

    entities: list[GenieExecutionEntity] = []
    for index, (window_start_frame, window_num_frames) in enumerate(windows):
        window_end_frame = window_start_frame + window_num_frames
        queue_lane = root_entity.queue_lane
        if (
            root_entity.checkpoint_every_n_frames > 0
            and window_end_frame < root_entity.total_frames
            and window_end_frame % root_entity.checkpoint_every_n_frames == 0
        ):
            queue_lane = GenieQueueLane.CHECKPOINT_HEAVY.value
        entities.append(
            GenieExecutionEntity(
                entity_id=f"{root_entity.entity_id}:{window_end_frame:04d}",
                rollout_id=root_entity.rollout_id,
                episode_id=root_entity.episode_id,
                branch_id=root_entity.branch_id,
                sample_id=root_entity.sample_id,
                input_state_handle_id=root_entity.input_state_handle_id,
                current_stage=root_entity.current_stage,
                next_stage=root_entity.next_stage,
                window_start_frame=window_start_frame,
                window_num_frames=window_num_frames,
                total_frames=root_entity.total_frames,
                num_prompt_frames=root_entity.num_prompt_frames,
                checkpoint_every_n_frames=root_entity.checkpoint_every_n_frames,
                priority=root_entity.priority - (index * 0.001),
                deadline_s=root_entity.deadline_s,
                batch_signature=replace(root_entity.batch_signature, window_num_frames=window_num_frames),
                queue_lane=queue_lane,
                stage_attempts=root_entity.stage_attempts,
                last_scheduled_at=root_entity.last_scheduled_at,
            )
        )
    return entities
