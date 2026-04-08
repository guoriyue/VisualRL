from __future__ import annotations

import pytest

from wm_infra.controlplane import TemporalStore
from wm_infra.env_runtime.state import load_runtime_state_view, split_state_handle_refs
from wm_infra.workloads.reinforcement_learning.runtime import ReinforcementLearningEnvManager


def test_step_many_splits_into_multiple_chunks_when_batch_exceeds_limit(tmp_path) -> None:
    manager = ReinforcementLearningEnvManager(TemporalStore(tmp_path / "temporal"), max_chunk_size=2)
    sessions = [
        manager.create_session(
            env_name="toy-line-v0",
            task_id="toy-line-eval",
            seed=10 + index,
            policy_version="pi-runtime",
            max_episode_steps=4,
            labels={},
            metadata={},
        )
        for index in range(5)
    ]

    try:
        response = manager.step_many(
            sessions[0].env_id,
            env_ids=[item.env_id for item in sessions[1:]],
            actions=[[0.0, 0.0, 1.0] for _ in sessions],
            policy_version="pi-runtime",
            checkpoint=False,
            metadata={},
        )
    finally:
        for session in sessions:
            manager.delete_session(session.env_id)

    assert len(response.results) == 5
    assert response.runtime["chunk_count"] == 3
    assert response.runtime["chunk_sizes"] == [2, 2, 1]
    assert response.runtime["max_chunk_size"] == 2
    assert response.runtime["batch_policy"]["mode"] == "sync"
    assert response.runtime["batch_policy"]["max_chunk_size"] == 2
    assert response.runtime["schedule_ms"] >= 0.0
    assert response.runtime["transition_ms"] >= 0.0
    assert response.runtime["reward_ms"] >= 0.0
    assert response.runtime["persist_ms"] >= 0.0
    assert response.runtime["state_bytes"] > 0
    assert response.runtime["state_count"] == 5
    assert response.runtime["queue_wait_ms"] == 0.0
    assert response.runtime["step_semantics"] == "sync_step_many"
    assert response.runtime["northbound_reset_policy"] == "explicit_reset_required"
    assert response.runtime["stage_profile"]["stages"]["materialize"]["count"] == 1
    assert response.runtime["stage_profile"]["stages"]["transition"]["count"] == 3
    assert response.runtime["stage_profile"]["stages"]["reward"]["count"] == 3
    assert response.runtime["stage_profile"]["stages"]["persist"]["count"] == 3


def test_stateless_predict_many_splits_into_multiple_chunks_when_batch_exceeds_limit(tmp_path) -> None:
    manager = ReinforcementLearningEnvManager(TemporalStore(tmp_path / "temporal"), max_chunk_size=2)
    contexts = [
        manager.initialize_transition_context(
            env_name="toy-line-v0",
            task_id="toy-line-eval",
            seed=10 + index,
            policy_version="pi-stateless",
            max_episode_steps=4,
            branch_name=None,
            labels={},
            metadata={},
        )
        for index in range(5)
    ]

    response = manager.predict_many_transitions(
        items=[
            {
                "state_handle_id": context.state_handle_id,
                "trajectory_id": context.trajectory_id,
                "action": [0.0, 0.0, 1.0],
            }
            for context in contexts
        ],
        policy_version="pi-stateless",
        checkpoint=False,
        metadata={},
    )

    assert len(response.results) == 5
    assert response.runtime["chunk_count"] == 3
    assert response.runtime["chunk_sizes"] == [2, 2, 1]
    assert response.runtime["max_chunk_size"] == 2
    assert response.runtime["schedule_ms"] >= 0.0
    assert response.runtime["transition_ms"] >= 0.0
    assert response.runtime["reward_ms"] >= 0.0
    assert response.runtime["persist_ms"] >= 0.0
    assert response.runtime["state_bytes"] > 0
    assert response.runtime["state_count"] == 5
    assert response.runtime["queue_wait_ms"] == 0.0
    assert response.runtime["dispatch_mode"] == "sync_inline"
    assert response.runtime["stage_profile"]["stages"]["materialize"]["count"] == 1

def test_runtime_state_view_splits_execution_residency_from_lineage(tmp_path) -> None:
    store = TemporalStore(tmp_path / "temporal")
    manager = ReinforcementLearningEnvManager(store)
    context = manager.initialize_transition_context(
        env_name="toy-line-v0",
        task_id="toy-line-train",
        seed=123,
        policy_version="pi-state-split",
        max_episode_steps=4,
        branch_name="main",
        labels={},
        metadata={},
    )
    state_handle = store.state_handles.get(context.state_handle_id)
    assert state_handle is not None

    view = load_runtime_state_view(state_handle, dtype=manager.dtype, device=manager.device)
    refs = split_state_handle_refs(state_handle)

    assert view.execution_residency_ref.residency.value == "inline"
    assert view.lineage_ref.branch_id == context.branch_id
    assert refs.execution_residency_ref.residency.value == "inline"
    assert refs.lineage_ref.branch_id == context.branch_id
    assert refs.lineage_ref.trajectory_id == context.trajectory_id


@pytest.mark.asyncio
async def test_async_transition_dispatch_and_collect(tmp_path) -> None:
    manager = ReinforcementLearningEnvManager(TemporalStore(tmp_path / "temporal"), max_chunk_size=2)
    context = manager.initialize_transition_context(
        env_name="toy-line-v0",
        task_id="toy-line-eval",
        seed=41,
        policy_version="pi-async",
        max_episode_steps=4,
        branch_name=None,
        labels={},
        metadata={},
    )

    dispatch = manager.dispatch_transition_batch(
        items=[
            {
                "state_handle_id": context.state_handle_id,
                "trajectory_id": context.trajectory_id,
                "action": [0.0, 0.0, 1.0],
            }
        ],
        policy_version="pi-async",
        checkpoint=False,
        metadata={},
    )
    response = await manager.collect_transition_batch_async(dispatch.dispatch_id)

    assert len(response.results) == 1
    assert response.runtime["dispatch_mode"] == "async_dispatched"
    assert response.runtime["stage_profile"]["stages"]["transition"]["count"] == 1
