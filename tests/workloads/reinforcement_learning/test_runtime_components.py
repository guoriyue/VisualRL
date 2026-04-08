from __future__ import annotations

import torch

from wm_infra.controlplane import TemporalStore
from wm_infra.execution import ExecutionBatchPolicy
from wm_infra.env_runtime.async_runtime import AsyncTransitionDispatcher
from wm_infra.env_runtime.catalog import LearnedEnvCatalog
from wm_infra.env_runtime.persistence import (
    TransitionExecutionResult,
    TransitionPersistenceContext,
    TransitionPersistenceLayer,
    build_transition_persistence_plan,
)
from wm_infra.env_runtime.pipeline import TransitionStagePipeline
from wm_infra.env_runtime.transition_executor import TransitionExecutor
from wm_infra.workloads.reinforcement_learning.defaults import build_default_registry
from wm_infra.workloads.reinforcement_learning.runtime import ReinforcementLearningEnvManager


def test_async_transition_dispatcher_batches_homogeneous_payloads() -> None:
    dispatcher = AsyncTransitionDispatcher(
        max_batch_size=2,
        batch_runner=lambda batch, payloads: [payload["value"] * 2 for payload in payloads],
    )
    dispatches = dispatcher.send_many(
        batch_key=("toy-line-v0", (3,), "cpu"),
        payloads=[{"value": 1}, {"value": 3}],
    )

    try:
        assert dispatcher.collect(dispatches[0].dispatch_id) == 2
        assert dispatcher.collect(dispatches[1].dispatch_id) == 6
        snapshot = dispatcher.snapshot()
        assert snapshot["sealed_batches"]
    finally:
        dispatcher.close()


def _build_transition_executor(store: TemporalStore, *, max_chunk_size: int = 2) -> TransitionExecutor:
    device = torch.device("cpu")
    dtype = torch.float32
    catalog = LearnedEnvCatalog(
        store,
        device=device,
        dtype=dtype,
        registry=build_default_registry(device=device, dtype=dtype),
    )
    catalog.sync_to_store()
    return TransitionExecutor(
        temporal_store=store,
        catalog=catalog,
        dispatcher=AsyncTransitionDispatcher(),
        batch_policy=ExecutionBatchPolicy(
            mode="sync",
            max_chunk_size=max_chunk_size,
            min_ready_size=1,
            return_when_ready_count=max_chunk_size,
            allow_partial_batch=True,
        ),
        device=device,
        dtype=dtype,
    )


def test_transition_stage_pipeline_emits_persist_intent(tmp_path) -> None:
    store = TemporalStore(tmp_path / "temporal")
    manager = ReinforcementLearningEnvManager(store, max_chunk_size=2)
    executor = _build_transition_executor(store, max_chunk_size=2)
    initialized = manager.initialize_transition_context(
        env_name="toy-line-v0",
        task_id="toy-line-eval",
        seed=7,
        policy_version="pi-pipeline",
        max_episode_steps=4,
        branch_name="main",
        labels={},
        metadata={},
    )
    context = executor.load_stateless_context(
        state_handle_id=initialized.state_handle_id,
        trajectory_id=initialized.trajectory_id,
        max_episode_steps=initialized.max_episode_steps,
        policy_version=initialized.policy_version,
    )
    pipeline = TransitionStagePipeline(
        world_model=executor.catalog.world_model_for_env(context.env_name),
        reward_fn=executor.catalog.reward_fn_for_env(context.env_name),
        dtype=executor.dtype,
        device=executor.device,
        policy=executor.batch_policy,
    )

    prepared = pipeline.prepare(
        contexts=[context],
        action_tensor=torch.tensor([[0.0, 0.0, 1.0]], dtype=executor.dtype, device=executor.device),
    )
    run = pipeline.run_chunk(
        prepared.chunks[0],
        checkpoint=True,
        metadata={"source": "unit-test"},
        stage_profile=prepared.stage_profile,
    )

    assert len(run.persist.intents) == 1
    assert run.persist.intents[0].checkpoint_requested is True
    assert run.persist.intents[0].next_step_idx == context.step_idx + 1
    assert prepared.stage_profile.snapshot()["stages"]["materialize"]["count"] == 1
    assert prepared.stage_profile.snapshot()["stages"]["transition"]["count"] == 1
    assert prepared.stage_profile.snapshot()["stages"]["reward"]["count"] == 1


def test_transition_persistence_layer_commits_state_transition(tmp_path) -> None:
    store = TemporalStore(tmp_path / "temporal")
    manager = ReinforcementLearningEnvManager(store)
    executor = _build_transition_executor(store)
    initialized = manager.initialize_transition_context(
        env_name="toy-line-v0",
        task_id="toy-line-eval",
        seed=11,
        policy_version="pi-persist",
        max_episode_steps=4,
        branch_name="main",
        labels={},
        metadata={},
    )
    context = executor.load_stateless_context(
        state_handle_id=initialized.state_handle_id,
        trajectory_id=initialized.trajectory_id,
        max_episode_steps=initialized.max_episode_steps,
        policy_version=initialized.policy_version,
    )
    persistence = TransitionPersistenceLayer(store)
    plan = build_transition_persistence_plan(
        TransitionPersistenceContext(
            env_id=context.scope_id,
            env_name=context.env_name,
            task_id=context.task_id,
            episode_id=context.episode_id,
            branch_id=context.branch_id,
            trajectory_id=context.trajectory_id,
            state_handle_id=context.state_handle_id,
            state=context.state,
            goal=context.goal,
            step_idx=context.step_idx,
            max_episode_steps=context.max_episode_steps,
            policy_version=context.policy_version,
            checkpoint_id=context.checkpoint_id,
        ),
        TransitionExecutionResult(
            action=[0.0, 0.0, 1.0],
            next_state=context.state + 0.5,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={"source": "unit-test"},
            policy_version=context.policy_version,
        ),
        checkpoint_requested=True,
        metadata={"checkpoint_tag": "unit"},
    )

    committed = persistence.commit(plan)
    trajectory = store.trajectories.get(context.trajectory_id)

    assert committed.transition.reward == 1.0
    assert committed.checkpoint_id is not None
    assert committed.state_handle.checkpoint_id == committed.checkpoint_id
    assert trajectory is not None
    assert trajectory.num_steps == 1
    assert trajectory.transition_refs == [committed.transition_id]
