"""Chunk scheduler for stage-oriented Genie runtime."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from wm_infra.backends.genie_runtime import (
    GenieBatchSignature,
    GenieExecutionChunk,
    GenieExecutionEntity,
    GenieQueueLane,
    estimate_expected_occupancy,
    estimate_transition_flops,
    lane_priority,
)


@dataclass(slots=True)
class SchedulerDecision:
    """Single chunk decision plus scheduler metadata."""

    chunk: GenieExecutionChunk
    scheduler_inputs: dict[str, float | int | str | bool]


class GenieChunkScheduler:
    """Best-effort scheduler for homogeneous Genie execution chunks."""

    def __init__(self, max_chunk_size: int = 8) -> None:
        self.max_chunk_size = max_chunk_size

    def schedule(
        self,
        entities: Iterable[GenieExecutionEntity],
        *,
        persist_backlog: int = 0,
        prompt_state_hot: bool = False,
        estimated_transfer_bytes: int = 0,
    ) -> list[SchedulerDecision]:
        """Group ready entities by batch signature and return prioritized chunks."""

        groups: dict[tuple[GenieQueueLane | str, GenieBatchSignature, int, int], list[GenieExecutionEntity]] = defaultdict(list)
        for entity in entities:
            groups[
                (
                    entity.queue_lane,
                    entity.batch_signature,
                    entity.window_start_frame,
                    entity.window_num_frames,
                )
            ].append(entity)

        decisions: list[SchedulerDecision] = []
        now = time.time()
        for (lane, signature, _window_start, _window_size), items in groups.items():
            lane_value = lane.value if isinstance(lane, GenieQueueLane) else str(lane)
            items.sort(key=lambda item: (-item.priority, item.window_start_frame, item.entity_id))
            for offset in range(0, len(items), self.max_chunk_size):
                chunk_entities = items[offset:offset + self.max_chunk_size]
                for entity in chunk_entities:
                    entity.last_scheduled_at = now
                    entity.stage_attempts += 1
                chunk = self._build_chunk(
                    signature,
                    lane,
                    chunk_entities,
                    estimated_transfer_bytes=estimated_transfer_bytes,
                )
                scheduler_inputs = {
                    "batch_signature_cardinality": len(groups),
                    "expected_occupancy": chunk.expected_occupancy,
                    "estimated_transfer_bytes": chunk.estimated_transfer_bytes,
                    "prompt_state_hot": prompt_state_hot,
                    "continuation_locality": lane_value == GenieQueueLane.HOT_CONTINUATION.value,
                    "checkpoint_due": lane_value == GenieQueueLane.CHECKPOINT_HEAVY.value,
                    "artifact_persist_backlog": persist_backlog,
                    "queue_lane": lane_value,
                    "priority_max": max(entity.priority for entity in chunk_entities),
                    "has_deadline": any(entity.deadline_s is not None for entity in chunk_entities),
                }
                decisions.append(SchedulerDecision(chunk=chunk, scheduler_inputs=scheduler_inputs))

        decisions.sort(key=self._sort_key)
        return decisions

    def build_chunks(
        self,
        entities: Iterable[GenieExecutionEntity],
        runtime_state,
        *,
        persist_backlog: int = 0,
    ) -> list[GenieExecutionChunk]:
        """Backward-compatible wrapper used by backend and runtime tests."""

        prompt_state_hot = str(getattr(runtime_state, "resident_tier", "")) == GenieQueueLane.HOT_CONTINUATION or str(
            getattr(runtime_state, "resident_tier", "")
        ) == "hot_gpu"
        estimated_transfer_bytes = int(getattr(runtime_state, "materialized_bytes", 0))
        decisions = self.schedule(
            entities,
            persist_backlog=persist_backlog,
            prompt_state_hot=prompt_state_hot,
            estimated_transfer_bytes=estimated_transfer_bytes,
        )
        return [decision.chunk for decision in decisions]

    def _build_chunk(
        self,
        signature: GenieBatchSignature,
        lane: GenieQueueLane,
        entities: list[GenieExecutionEntity],
        *,
        estimated_transfer_bytes: int,
    ) -> GenieExecutionChunk:
        lane_value = lane.value if isinstance(lane, GenieQueueLane) else str(lane)
        first = entities[0]
        chunk_size = len(entities)
        frame_start = first.window_start_frame
        frame_end = max(entity.window_start_frame + entity.window_num_frames for entity in entities)
        return GenieExecutionChunk(
            chunk_id=f"{signature.stage}:{lane_value}:{frame_start}:{chunk_size}",
            signature=signature,
            entity_ids=[entity.entity_id for entity in entities],
            runnable_stage=signature.stage,
            frame_range=(frame_start, frame_end),
            estimated_vram_bytes=sum(signature.spatial_h * signature.spatial_w * entity.window_num_frames * 4 for entity in entities),
            estimated_transfer_bytes=estimated_transfer_bytes,
            estimated_flops=estimate_transition_flops(
                spatial_h=signature.spatial_h,
                spatial_w=signature.spatial_w,
                window_num_frames=sum(entity.window_num_frames for entity in entities),
                maskgit_steps=signature.maskgit_steps,
            ),
            queue_lane=lane_value,
            expected_occupancy=estimate_expected_occupancy(chunk_size, self.max_chunk_size),
        )

    def _sort_key(self, decision: SchedulerDecision) -> tuple[float, int, float]:
        chunk = decision.chunk
        stage_bias = 0 if chunk.runnable_stage == "transition" else 1
        return (
            stage_bias,
            lane_priority(chunk.queue_lane),
            -chunk.expected_occupancy,
        )


GenieScheduler = GenieChunkScheduler
