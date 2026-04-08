"""Tests for execution chunking (core and runtime)."""

from __future__ import annotations

import torch

from wm_infra.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionEntity,
    build_execution_chunks,
    summarize_execution_chunks,
)
from wm_infra.execution import (
    ExecutionRuntimeTrace,
    ExecutionStageRecord,
    ExecutionWorkItem,
    GroupedChunkDecision,
    HomogeneousChunkScheduler,
    schedule_grouped_chunks,
)


def test_build_execution_chunks_respects_shared_batch_policy() -> None:
    signature = BatchSignature(
        stage="env_step",
        latent_shape=(1, 1),
        action_dim=3,
        dtype="float32",
        device="cpu",
        needs_decode=False,
    )
    policy = ExecutionBatchPolicy(mode="sync", max_chunk_size=2, min_ready_size=1, return_when_ready_count=2)
    entities = [
        ExecutionEntity(
            entity_id=f"env-{index}:env_step:0",
            rollout_id=f"env-{index}",
            stage="env_step",
            step_idx=0,
            batch_signature=signature,
        )
        for index in range(5)
    ]
    latent_items = [torch.full((1, 1, 1), float(index)) for index in range(5)]
    action_items = [torch.eye(3, dtype=torch.float32)[index % 3:index % 3 + 1] for index in range(5)]

    chunks = build_execution_chunks(
        signature=signature,
        entities=entities,
        latent_items=latent_items,
        action_items=action_items,
        policy=policy,
        chunk_id_prefix="env_step",
        latent_join=lambda items: torch.cat(items, dim=0),
        action_join=lambda items: torch.cat(items, dim=0),
    )
    summary = summarize_execution_chunks(chunks, policy=policy)

    assert [chunk.size for chunk in chunks] == [2, 2, 1]
    assert chunks[0].latent_batch.shape == (2, 1, 1)
    assert chunks[0].action_batch.shape == (2, 3)
    assert summary["batch_policy"]["mode"] == "sync"
    assert summary["batch_policy"]["return_when_ready_count"] == 2
    assert summary["avg_chunk_fill_ratio"] == 5 / 6


# ---------------------------------------------------------------------------
# Runtime HomogeneousChunkScheduler tests (merged from test_runtime_execution.py)
# ---------------------------------------------------------------------------


def _entity(signature: BatchSignature, *, rollout_id: str, step_idx: int) -> ExecutionEntity:
    return ExecutionEntity(
        entity_id=f"{rollout_id}:{signature.stage}:{step_idx}",
        rollout_id=rollout_id,
        stage=signature.stage,
        step_idx=step_idx,
        batch_signature=signature,
    )


def test_homogeneous_chunk_scheduler_groups_by_signature() -> None:
    policy = ExecutionBatchPolicy(mode="sync", max_chunk_size=2, min_ready_size=1, return_when_ready_count=2)
    transition_sig = BatchSignature(
        stage="transition",
        latent_shape=(1, 1),
        action_dim=3,
        dtype="float32",
        device="cpu",
        needs_decode=False,
    )
    decode_sig = BatchSignature(
        stage="decode",
        latent_shape=(1, 1),
        action_dim=3,
        dtype="float32",
        device="cpu",
        needs_decode=True,
    )
    work_items = [
        ExecutionWorkItem(
            entity=_entity(transition_sig, rollout_id=f"traj-{index}", step_idx=index),
            latent_item=torch.full((1, 1, 1), float(index)),
            action_item=torch.eye(3, dtype=torch.float32)[index % 3:index % 3 + 1],
        )
        for index in range(3)
    ]
    work_items.append(
        ExecutionWorkItem(
            entity=_entity(decode_sig, rollout_id="traj-decode", step_idx=0),
            latent_item=torch.ones((1, 1, 1), dtype=torch.float32),
            action_item=torch.zeros((1, 3), dtype=torch.float32),
        )
    )

    chunks, decisions = HomogeneousChunkScheduler().schedule(
        work_items=work_items,
        policy=policy,
        chunk_id_prefix="runtime",
        latent_join=lambda items: torch.cat(items, dim=0),
        action_join=lambda items: torch.cat(items, dim=0),
    )

    assert [chunk.size for chunk in chunks] == [2, 1, 1]
    assert [chunk.signature.stage for chunk in chunks] == ["transition", "transition", "decode"]
    assert chunks[0].latent_batch.shape == (2, 1, 1)
    assert chunks[2].action_batch.shape == (1, 3)
    assert decisions[0].ready_count == 3
    assert decisions[0].chunk_count == 2
    assert decisions[1].ready_count == 1
    assert decisions[1].chunk_count == 1


def test_schedule_grouped_chunks_builds_backend_specific_metadata() -> None:
    signature = BatchSignature(
        stage="transition",
        latent_shape=(1, 1),
        action_dim=3,
        dtype="float32",
        device="cpu",
        needs_decode=False,
    )
    entities = [
        ExecutionEntity(
            entity_id=f"env-{index}:transition:0",
            rollout_id=f"env-{index}",
            stage="transition",
            step_idx=0,
            batch_signature=signature,
        )
        for index in range(5)
    ]
    priorities = {entity.entity_id: float(5 - index) for index, entity in enumerate(entities)}

    decisions = schedule_grouped_chunks(
        entities=entities,
        max_chunk_size=2,
        group_key=lambda entity: entity.stage,
        entity_sort_key=lambda entity: (-priorities[entity.entity_id], entity.entity_id),
        build_chunk=lambda group_key, chunk_entities, chunk_index: {
            "group": group_key,
            "chunk_index": chunk_index,
            "entity_ids": [entity.entity_id for entity in chunk_entities],
            "size": len(chunk_entities),
        },
        build_scheduler_inputs=lambda group_key, chunk_entities, chunk, group_count: {
            "group": group_key,
            "group_count": group_count,
            "chunk_size": len(chunk_entities),
            "first_entity": chunk["entity_ids"][0],
        },
        decision_sort_key=lambda decision: (-decision.scheduler_inputs["chunk_size"], decision.chunk["chunk_index"]),
    )

    assert all(isinstance(decision, GroupedChunkDecision) for decision in decisions)
    assert [decision.chunk["size"] for decision in decisions] == [2, 2, 1]
    assert decisions[0].scheduler_inputs["group_count"] == 1
    assert decisions[0].scheduler_inputs["first_entity"] == "env-0:transition:0"


def test_execution_runtime_trace_summarizes_stage_and_chunk_metrics() -> None:
    trace = ExecutionRuntimeTrace()
    trace.record(
        ExecutionStageRecord(
            stage="admission",
            entity_id="sample:0",
            queue_lane="hot",
            elapsed_ms=1.25,
            chunk_id="chunk-0",
            chunk_size=2,
            expected_occupancy=1.0,
        )
    )
    trace.record(
        ExecutionStageRecord(
            stage="transition",
            entity_id="sample:0",
            queue_lane="warm",
            elapsed_ms=2.5,
            chunk_id="chunk-1",
            chunk_size=1,
            expected_occupancy=0.5,
            estimated_transfer_bytes=4096,
        )
    )

    assert trace.queue_lanes_seen == ["hot", "warm"]
    assert trace.stage_timings_ms() == {"admission": 1.25, "transition": 2.5}
    assert trace.chunk_summary() == {
        "chunk_count": 2,
        "avg_chunk_size": 1.5,
        "max_chunk_size": 2,
        "avg_expected_occupancy": 0.75,
    }


def test_execution_runtime_trace_can_include_transfer_bytes() -> None:
    class TransferAwareTrace(ExecutionRuntimeTrace):
        include_estimated_transfer_bytes = True

    trace = TransferAwareTrace()
    trace.record(
        ExecutionStageRecord(
            stage="transition",
            entity_id="sample:1",
            queue_lane="hot",
            elapsed_ms=3.0,
            chunk_id="chunk-2",
            chunk_size=3,
            expected_occupancy=0.75,
            estimated_transfer_bytes=1024,
        )
    )

    assert trace.chunk_summary()["estimated_transfer_bytes"] == 1024
