"""Chunk scheduler for homogeneous Cosmos sample-production work."""

from __future__ import annotations

from typing import Iterable

from wm_infra.backends.cosmos.runtime import CosmosExecutionChunk, CosmosExecutionEntity, expected_occupancy
from wm_infra.execution import GroupedChunkDecision, schedule_grouped_chunks

CosmosSchedulerDecision = GroupedChunkDecision[CosmosExecutionChunk]


class CosmosChunkScheduler:
    """Group homogeneous Cosmos entities into ECS-style chunks."""

    def __init__(self, max_chunk_size: int = 4) -> None:
        self.max_chunk_size = max_chunk_size

    def schedule(self, entities: Iterable[CosmosExecutionEntity], *, estimated_units: float) -> list[CosmosSchedulerDecision]:
        return schedule_grouped_chunks(
            entities=entities,
            max_chunk_size=self.max_chunk_size,
            group_key=lambda entity: (entity.queue_lane.value, entity.batch_signature),
            entity_sort_key=lambda item: (-item.priority, item.sample_id),
            build_chunk=lambda key, chunk_entities, chunk_index: CosmosExecutionChunk(
                chunk_id=f"{key[1].stage}:{key[0]}:{chunk_index}",
                signature=key[1],
                entity_ids=[entity.entity_id for entity in chunk_entities],
                expected_occupancy=expected_occupancy(len(chunk_entities), self.max_chunk_size),
                estimated_units=estimated_units,
            ),
            build_scheduler_inputs=lambda key, _chunk_entities, chunk, group_count: {
                "queue_lane": key[0],
                "batch_signature_cardinality": group_count,
                "expected_occupancy": chunk.expected_occupancy,
                "estimated_units": estimated_units,
                "has_reference": key[1].has_reference,
            },
            decision_sort_key=lambda item: (-item.chunk.expected_occupancy, item.chunk.chunk_id),
        )
