"""Learned environment stepping runtime for wm-infra.

This module owns environment/session stepping over learned temporal state.
Northbound RL and HTTP surfaces should depend on this runtime rather than
embedding transition execution logic directly.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from typing import Any

import torch

from wm_infra.api.protocol import (
    EnvironmentSessionResponse,
    EnvironmentStepManyResponse,
    EnvironmentStepResponse,
    TransitionContextResponse,
    TransitionPredictManyResponse,
    TransitionPredictResponse,
)
from wm_infra.controlplane import (
    BranchCreate,
    CheckpointCreate,
    ExecutionStateRef,
    EnvironmentSessionCreate,
    EnvironmentSessionRecord,
    EnvironmentSpec,
    EpisodeCreate,
    StateHandleCreate,
    StateHandleKind,
    StateLineageRef,
    TaskSpec,
    TemporalStatus,
    TemporalStore,
    TrajectoryCreate,
    TrajectoryRecord,
    TransitionCreate,
)
from wm_infra.runtime.execution import (
    ExecutionBatchPolicy,
    ExecutionChunk,
)
from wm_infra.runtime.env.async_runtime import AsyncTransitionDispatcher, TransitionDispatch
from wm_infra.runtime.env.catalog import (
    LearnedEnvCatalog,
    default_environment_specs as _catalog_default_environment_specs,
    default_task_specs as _catalog_default_task_specs,
)
from wm_infra.runtime.env.pipeline import TransitionStagePipeline
from wm_infra.runtime.env.state import build_inline_state_handle_create, load_runtime_state_view
from wm_infra.runtime.env.transition import StatelessTransitionContext, build_stateless_step_chunks


def _default_environment_specs() -> list[EnvironmentSpec]:
    return _catalog_default_environment_specs()


def _default_task_specs() -> list[TaskSpec]:
    return _catalog_default_task_specs()


class RLEnvironmentManager:
    """Environment registry + stateless transition manager for trainer-facing RL APIs."""

    def __init__(self, temporal_store: TemporalStore, *, max_chunk_size: int = 32) -> None:
        self.temporal_store = temporal_store
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.max_chunk_size = max(1, max_chunk_size)
        self.env_step_batch_policy = ExecutionBatchPolicy(
            mode="sync",
            max_chunk_size=self.max_chunk_size,
            min_ready_size=1,
            return_when_ready_count=self.max_chunk_size,
            allow_partial_batch=True,
        )
        self.catalog = LearnedEnvCatalog(temporal_store, device=self.device, dtype=self.dtype)
        self.world_model = self.catalog.world_model
        self.genie_world_model = self.catalog.genie_world_model
        self.dispatcher = AsyncTransitionDispatcher()
        self._register_catalog()

    def _register_catalog(self) -> None:
        self.catalog.register_defaults()

    def list_environment_specs(self) -> list[EnvironmentSpec]:
        return self.catalog.list_environment_specs()

    def list_task_specs(self, env_name: str | None = None) -> list[TaskSpec]:
        return self.catalog.list_task_specs(env_name=env_name)

    def action_dim_for_env(self, env_name: str) -> int:
        return self.catalog.action_dim_for_env(env_name)

    def backend_for_env(self, env_name: str) -> str:
        return self.catalog.backend_for_env(env_name)

    def get_session(self, env_id: str) -> EnvironmentSessionRecord:
        session = self.temporal_store.environment_sessions.get(env_id)
        if session is None:
            raise KeyError(env_id)
        return session

    def list_sessions(self) -> list[EnvironmentSessionRecord]:
        return self.temporal_store.environment_sessions.list()

    def create_session(
        self,
        *,
        env_name: str,
        task_id: str | None,
        seed: int | None,
        policy_version: str | None,
        max_episode_steps: int | None,
        labels: dict[str, str],
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        spec = self.catalog.resolve_env_spec(env_name)
        initialized = self.initialize_transition_context(
            env_name=env_name,
            task_id=task_id,
            seed=seed,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps,
            branch_name="main",
            labels=labels,
            metadata=metadata,
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=env_name,
                episode_id=initialized.episode_id,
                task_id=initialized.task_id,
                backend=spec.backend,
                current_step=0,
                state_handle_id=initialized.state_handle_id,
                checkpoint_id=None,
                trajectory_id=initialized.trajectory_id,
                branch_id=initialized.branch_id,
                policy_version=initialized.policy_version,
                labels=labels,
                metadata={
                    "env_name": env_name,
                    "task_split": self.catalog.resolve_task(initialized.task_id, env_name=env_name).split,
                    "max_episode_steps": initialized.max_episode_steps,
                    "needs_reset": False,
                    "compat_session": True,
                    **metadata,
                },
            )
        )
        trajectory = self.temporal_store.trajectories.get(initialized.trajectory_id)
        assert trajectory is not None
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        return self._session_response(session)

    def initialize_transition_context(
        self,
        *,
        env_name: str,
        task_id: str | None,
        seed: int | None,
        policy_version: str | None,
        max_episode_steps: int | None,
        branch_name: str | None,
        labels: dict[str, str],
        metadata: dict[str, Any],
    ) -> TransitionContextResponse:
        spec = self.catalog.resolve_env_spec(env_name)
        task = self.catalog.resolve_task(task_id or self.catalog.default_task_for_env(env_name), env_name=env_name)
        episode = self.temporal_store.create_episode(
            EpisodeCreate(
                title=f"{env_name}:{task.task_id}",
                labels=labels,
                metadata={"env_name": env_name, "task_id": task.task_id, "stateless": True, **metadata},
            )
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=episode.episode_id,
                name=branch_name or "main",
                labels=labels,
                metadata={"env_name": env_name, "task_id": task.task_id, "stateless": True, **metadata},
            )
        )
        trajectory = self._ensure_stateless_trajectory(
            env_name=env_name,
            task=task,
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=None,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            metadata={"seed": seed, **metadata},
        )
        state, goal = self._sample_initial_state(task, seed)
        state_handle = self._persist_state_handle(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            state=state,
            goal=goal,
            step_idx=0,
            task=task,
            env_name=env_name,
            trajectory_id=trajectory.trajectory_id,
        )
        return TransitionContextResponse(
            env_name=env_name,
            episode_id=episode.episode_id,
            task_id=task.task_id,
            branch_id=branch.branch_id,
            state_handle_id=state_handle.state_handle_id,
            checkpoint_id=None,
            trajectory_id=trajectory.trajectory_id,
            current_step=0,
            policy_version=policy_version,
            max_episode_steps=int(trajectory.metadata.get("max_episode_steps", spec.default_horizon)),
            observation=self._observation(state, goal),
            info=self.catalog.session_info_for_env(env_name, goal, 0),
        )

    def reset_session(
        self,
        env_id: str,
        *,
        seed: int | None,
        policy_version: str | None,
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        session = self.get_session(env_id)
        task = self.catalog.resolve_task(session.task_id, env_name=session.env_name)
        if session.trajectory_id:
            self._finalize_trajectory(session.trajectory_id, success=False)

        initialized = self.initialize_transition_context(
            env_name=session.env_name,
            task_id=session.task_id,
            seed=seed,
            policy_version=policy_version or session.policy_version,
            max_episode_steps=int(session.metadata.get("max_episode_steps", self.catalog.resolve_env_spec(session.env_name).default_horizon)),
            branch_name="main",
            labels=dict(session.labels),
            metadata={"env_id": env_id, "reset_from_episode_id": session.episode_id, **metadata},
        )
        trajectory = self.temporal_store.trajectories.get(initialized.trajectory_id)
        assert trajectory is not None
        trajectory.env_id = env_id
        self.temporal_store.update_trajectory(trajectory)

        session.episode_id = initialized.episode_id
        session.branch_id = initialized.branch_id
        session.current_step = 0
        session.state_handle_id = initialized.state_handle_id
        session.trajectory_id = initialized.trajectory_id
        session.checkpoint_id = None
        session.policy_version = initialized.policy_version
        session.completed_at = None
        session.status = TemporalStatus.ACTIVE
        session.metadata["needs_reset"] = False
        session.metadata["max_episode_steps"] = initialized.max_episode_steps
        session.metadata["task_split"] = task.split
        session.metadata.update(metadata)
        session = self.temporal_store.update_environment_session(session)
        return self._session_response(session)

    def step_session(
        self,
        env_id: str,
        *,
        action: list[float],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> EnvironmentStepResponse:
        batch_response = self.step_many(
            env_id,
            env_ids=[],
            actions=[action],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return batch_response.results[0]

    def step_many(
        self,
        env_id: str,
        *,
        env_ids: list[str],
        actions: list[list[float]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> EnvironmentStepManyResponse:
        ordered_env_ids = [env_id, *[item for item in env_ids if item != env_id]]
        if len(ordered_env_ids) != len(actions):
            raise ValueError("step_many requires one action per env_id")

        session_records = [self.get_session(item) for item in ordered_env_ids]
        if any(bool(session.metadata.get("needs_reset")) for session in session_records):
            raise ValueError("All sessions in step_many must be reset before further stepping")
        env_name = session_records[0].env_name
        if any(session.env_name != env_name for session in session_records):
            raise ValueError("step_many currently only supports batching the same env_name")

        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(len(actions), -1)
        expected_action_dim = self.catalog.action_dim_for_env(env_name)
        if action_tensor.shape[1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got {action_tensor.shape[1]}")

        prediction = self.predict_many_transitions(
            items=[
                {
                    "state_handle_id": session.state_handle_id,
                    "trajectory_id": session.trajectory_id,
                    "action": action,
                    "max_episode_steps": int(session.metadata.get("max_episode_steps", self.catalog.resolve_env_spec(session.env_name).default_horizon)),
                }
                for session, action in zip(session_records, actions, strict=True)
            ],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )

        responses: list[EnvironmentStepResponse] = []
        chunk_ids = [chunk["chunk_id"] for chunk in prediction.runtime.get("chunks", [])]
        for index, (session, result) in enumerate(zip(session_records, prediction.results, strict=True)):
            session.current_step = result.step_idx
            session.state_handle_id = result.state_handle_id
            session.checkpoint_id = result.checkpoint_id
            session.policy_version = result.policy_version
            session.metadata["needs_reset"] = result.terminated or result.truncated
            session.metadata["last_transition_id"] = result.transition_id
            if chunk_ids:
                session.metadata["last_chunk_id"] = chunk_ids[min(index // self.max_chunk_size, len(chunk_ids) - 1)]
            session = self.temporal_store.update_environment_session(session)
            responses.append(
                EnvironmentStepResponse(
                    env_id=session.env_id,
                    episode_id=result.episode_id,
                    task_id=result.task_id,
                    trajectory_id=result.trajectory_id,
                    state_handle_id=result.state_handle_id,
                    checkpoint_id=result.checkpoint_id,
                    transition_id=result.transition_id,
                    policy_version=result.policy_version,
                    step_idx=result.step_idx,
                    observation=result.observation,
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    info=result.info,
                )
            )

        runtime = dict(prediction.runtime)
        runtime["execution_path"] = "chunked_env_step"
        runtime["env_step_chunk_total"] = runtime.get("chunk_count", 0)
        runtime["state_locality_mode"] = "legacy_env_wrapper"
        runtime["step_semantics"] = "sync_step_many"
        runtime["northbound_reset_policy"] = "explicit_reset_required"
        return EnvironmentStepManyResponse(
            env_ids=ordered_env_ids,
            results=responses,
            runtime=runtime,
        )

    def predict_transition(
        self,
        *,
        state_handle_id: str,
        action: list[float],
        trajectory_id: str | None,
        policy_version: str | None,
        checkpoint: bool,
        max_episode_steps: int | None,
        metadata: dict[str, Any],
    ) -> TransitionPredictResponse:
        response = self.predict_many_transitions(
            items=[{
                "state_handle_id": state_handle_id,
                "action": action,
                "trajectory_id": trajectory_id,
                "max_episode_steps": max_episode_steps,
            }],
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return response.results[0]

    def predict_many_transitions(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        if not items:
            raise ValueError("predict_many requires at least one item")
        return self._execute_transition_batch(
            items=items,
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
            dispatch_mode="sync_inline",
        )

    async def predict_many_transitions_async(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        dispatch = self.dispatch_transition_batch(
            items=items,
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return await self.collect_transition_batch_async(dispatch.dispatch_id)

    def dispatch_transition_batch(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionDispatch:
        if not items:
            raise ValueError("dispatch_transition_batch requires at least one item")
        return self.dispatcher.dispatch(
            item_count=len(items),
            metadata={"policy_version": policy_version, "checkpoint": checkpoint},
            fn=self._execute_transition_batch,
            kwargs={
                "items": items,
                "policy_version": policy_version,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "dispatch_mode": "async_dispatched",
            },
        )

    def collect_transition_batch(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return self.dispatcher.collect(dispatch_id, timeout=timeout)

    async def collect_transition_batch_async(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return await self.dispatcher.collect_async(dispatch_id, timeout=timeout)

    def fork_session(
        self,
        env_id: str,
        *,
        branch_name: str | None,
        policy_version: str | None,
        metadata: dict[str, Any],
    ) -> EnvironmentSessionResponse:
        source_record = self.get_session(env_id)
        context = self._load_stateless_context(
            state_handle_id=source_record.state_handle_id,
            trajectory_id=source_record.trajectory_id,
            max_episode_steps=int(source_record.metadata.get("max_episode_steps", self.catalog.resolve_env_spec(source_record.env_name).default_horizon)),
            policy_version=policy_version or source_record.policy_version,
        )
        branch = self.temporal_store.create_branch(
            BranchCreate(
                episode_id=context.episode_id,
                parent_branch_id=context.branch_id,
                forked_from_checkpoint_id=source_record.checkpoint_id,
                name=branch_name or f"fork-{context.step_idx}",
                metadata={"source_env_id": env_id, **metadata},
            )
        )
        state_handle = self._persist_state_handle(
            episode_id=context.episode_id,
            branch_id=branch.branch_id,
            state=context.state.clone(),
            goal=context.goal.clone(),
            step_idx=context.step_idx,
            task=self.catalog.resolve_task(context.task_id, env_name=context.env_name),
            env_name=context.env_name,
            trajectory_id=None,
            parent_state_handle_id=context.state_handle_id,
        )
        trajectory = self._ensure_stateless_trajectory(
            env_name=context.env_name,
            task=self.catalog.resolve_task(context.task_id, env_name=context.env_name),
            episode_id=context.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=None,
            policy_version=policy_version or context.policy_version,
            max_episode_steps=context.max_episode_steps,
            metadata={"forked_from_env_id": env_id},
        )
        session = self.temporal_store.create_environment_session(
            EnvironmentSessionCreate(
                env_name=context.env_name,
                episode_id=context.episode_id,
                task_id=context.task_id,
                backend=source_record.backend,
                current_step=context.step_idx,
                state_handle_id=state_handle.state_handle_id,
                checkpoint_id=source_record.checkpoint_id,
                trajectory_id=trajectory.trajectory_id,
                branch_id=branch.branch_id,
                policy_version=policy_version or context.policy_version,
                labels=dict(source_record.labels),
                metadata={
                    "forked_from_env_id": env_id,
                    "max_episode_steps": context.max_episode_steps,
                    "needs_reset": False,
                    "compat_session": True,
                    **metadata,
                },
            )
        )
        trajectory.env_id = session.env_id
        self.temporal_store.update_trajectory(trajectory)
        return self._session_response(session)

    def checkpoint_session(self, env_id: str, *, tag: str | None, metadata: dict[str, Any]) -> str:
        session = self.get_session(env_id)
        return self._checkpoint_state_handle(
            state_handle_id=session.state_handle_id,
            episode_id=session.episode_id,
            branch_id=session.branch_id or "",
            step_idx=session.current_step,
            tag=tag or f"step-{session.current_step}",
            metadata={"env_name": session.env_name, "task_id": session.task_id, **metadata},
        )

    def delete_session(self, env_id: str) -> None:
        session = self.get_session(env_id)
        if session.trajectory_id:
            self._finalize_trajectory(session.trajectory_id, success=False)
        session.status = TemporalStatus.ARCHIVED
        session.completed_at = time.time()
        self.temporal_store.update_environment_session(session)

    def list_transitions(self, env_id: str | None = None, trajectory_id: str | None = None) -> list[Any]:
        items = self.temporal_store.transitions.list()
        if env_id is not None:
            items = [item for item in items if item.env_id == env_id]
        if trajectory_id is not None:
            items = [item for item in items if item.trajectory_id == trajectory_id]
        return items

    def list_trajectories(self, env_id: str | None = None, episode_id: str | None = None) -> list[TrajectoryRecord]:
        items = self.temporal_store.trajectories.list()
        if env_id is not None:
            items = [item for item in items if item.env_id == env_id]
        if episode_id is not None:
            items = [item for item in items if item.episode_id == episode_id]
        return items

    def list_evaluation_runs(self) -> list[Any]:
        return self.temporal_store.evaluation_runs.list()

    def _session_response(self, session: EnvironmentSessionRecord) -> EnvironmentSessionResponse:
        state, goal, _step_idx = self._load_state_goal_from_handle(session.state_handle_id)
        return EnvironmentSessionResponse(
            env_id=session.env_id,
            env_name=session.env_name,
            episode_id=session.episode_id,
            task_id=session.task_id,
            branch_id=session.branch_id,
            state_handle_id=session.state_handle_id,
            checkpoint_id=session.checkpoint_id,
            trajectory_id=session.trajectory_id,
            current_step=session.current_step,
            policy_version=session.policy_version,
            status=session.status.value,
            observation=self._observation(state, goal),
            info=self.catalog.session_info_for_env(session.env_name, goal, session.current_step),
        )

    def _sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.catalog.sample_initial_state(task, seed)

    def _persist_state_handle(
        self,
        *,
        episode_id: str,
        branch_id: str,
        state: torch.Tensor,
        goal: torch.Tensor,
        step_idx: int,
        task: TaskSpec,
        env_name: str,
        trajectory_id: str | None,
        parent_state_handle_id: str | None = None,
    ):
        return self.temporal_store.create_state_handle(
            build_inline_state_handle_create(
                episode_id=episode_id,
                branch_id=branch_id,
                state=state,
                goal=goal,
                step_idx=step_idx,
                env_name=env_name,
                task_id=task.task_id,
                trajectory_id=trajectory_id,
                parent_state_handle_id=parent_state_handle_id,
                kind=StateHandleKind.LATENT,
            )
        )

    def _finalize_trajectory(self, trajectory_id: str, *, success: bool) -> None:
        trajectory = self.temporal_store.trajectories.get(trajectory_id)
        if trajectory is None or trajectory.completed_at is not None:
            return
        trajectory.success = trajectory.success or success
        trajectory.status = TemporalStatus.SUCCEEDED if trajectory.success else TemporalStatus.ARCHIVED
        trajectory.completed_at = time.time()
        self.temporal_store.update_trajectory(trajectory)

    def _checkpoint_state_handle(
        self,
        *,
        state_handle_id: str,
        episode_id: str,
        branch_id: str | None,
        step_idx: int,
        tag: str,
        metadata: dict[str, Any],
    ) -> str:
        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=episode_id,
                branch_id=branch_id,
                state_handle_id=state_handle_id,
                step_index=step_idx,
                tag=tag,
                metadata=metadata,
            )
        )
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is not None:
            state_handle.checkpoint_id = checkpoint.checkpoint_id
            self.temporal_store.state_handles.put(state_handle)
        return checkpoint.checkpoint_id

    def _load_state_goal_from_handle(self, state_handle_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is None:
            raise KeyError(state_handle_id)
        resolved = load_runtime_state_view(state_handle, dtype=self.dtype, device=self.device)
        return resolved.state, resolved.goal, resolved.step_idx

    def _load_stateless_context(
        self,
        *,
        state_handle_id: str,
        trajectory_id: str | None,
        max_episode_steps: int | None,
        policy_version: str | None,
    ) -> StatelessTransitionContext:
        state_handle = self.temporal_store.state_handles.get(state_handle_id)
        if state_handle is None:
            raise KeyError(state_handle_id)
        resolved = load_runtime_state_view(state_handle, dtype=self.dtype, device=self.device)
        env_name = str(resolved.lineage_ref.env_name)
        task_id = str(resolved.lineage_ref.task_id)
        if not env_name or env_name == "None":
            raise ValueError(f"state_handle {state_handle_id} is missing metadata.env_name")
        if not task_id or task_id == "None":
            raise ValueError(f"state_handle {state_handle_id} is missing metadata.task_id")
        if state_handle.branch_id is None:
            raise ValueError(f"state_handle {state_handle_id} is missing branch_id")
        task = self.catalog.resolve_task(task_id, env_name=env_name)
        spec = self.catalog.resolve_env_spec(env_name)
        trajectory = self._ensure_stateless_trajectory(
            env_name=env_name,
            task=task,
            episode_id=state_handle.episode_id,
            branch_id=state_handle.branch_id,
            trajectory_id=trajectory_id,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            metadata={"source_state_handle_id": state_handle_id},
        )
        if state_handle.lineage_ref is None:
            state_handle.lineage_ref = StateLineageRef(
                env_name=env_name,
                task_id=task_id,
                trajectory_id=trajectory.trajectory_id,
                step_idx=resolved.step_idx,
            )
        elif state_handle.lineage_ref.trajectory_id is None:
            state_handle.lineage_ref.trajectory_id = trajectory.trajectory_id
        if state_handle.execution_state_ref is None:
            state_handle.execution_state_ref = ExecutionStateRef(device=str(self.device))
        self.temporal_store.state_handles.put(state_handle)
        return StatelessTransitionContext(
            env_name=env_name,
            task_id=task_id,
            episode_id=state_handle.episode_id,
            branch_id=state_handle.branch_id,
            trajectory_id=trajectory.trajectory_id,
            state_handle_id=state_handle_id,
            checkpoint_id=resolved.checkpoint_id,
            policy_version=policy_version or trajectory.policy_version,
            max_episode_steps=int(trajectory.metadata.get("max_episode_steps", max_episode_steps or spec.default_horizon)),
            state=resolved.state,
            goal=resolved.goal,
            step_idx=resolved.step_idx,
            scope_id=trajectory.env_id,
        )

    def _ensure_stateless_trajectory(
        self,
        *,
        env_name: str,
        task: TaskSpec,
        episode_id: str,
        branch_id: str,
        trajectory_id: str | None,
        policy_version: str | None,
        max_episode_steps: int,
        metadata: dict[str, Any],
    ) -> TrajectoryRecord:
        if trajectory_id is not None:
            trajectory = self.temporal_store.trajectories.get(trajectory_id)
            if trajectory is None:
                raise KeyError(trajectory_id)
            if trajectory.episode_id != episode_id:
                raise ValueError("trajectory_id does not belong to the requested episode/state")
            if trajectory.task_id != task.task_id:
                raise ValueError("trajectory_id does not match the requested task")
            if policy_version is not None and trajectory.policy_version != policy_version:
                trajectory.policy_version = policy_version
                self.temporal_store.update_trajectory(trajectory)
            if "max_episode_steps" not in trajectory.metadata:
                trajectory.metadata["max_episode_steps"] = max_episode_steps
                self.temporal_store.update_trajectory(trajectory)
            return trajectory

        scope_id = f"stateless:{episode_id}:{branch_id}"
        return self.temporal_store.create_trajectory(
            TrajectoryCreate(
                env_id=scope_id,
                episode_id=episode_id,
                task_id=task.task_id,
                policy_version=policy_version,
                metadata={
                    "env_name": env_name,
                    "branch_id": branch_id,
                    "task_split": task.split,
                    "max_episode_steps": max_episode_steps,
                    "stateless": True,
                    **metadata,
                },
            )
        )

    def _world_model_for_env(self, env_name: str):
        return self.catalog.world_model_for_env(env_name)

    def _reward_fn_for_env(self, env_name: str):
        return self.catalog.reward_fn_for_env(env_name)

    def _action_dim_for_env(self, env_name: str) -> int:
        return self.catalog.action_dim_for_env(env_name)

    def _session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        return self.catalog.session_info_for_env(env_name, goal, step_idx)

    def _transition_info_for_env(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        return self.catalog.transition_info_for_env(
            env_name,
            goal=goal,
            info_tensors=info_tensors,
            index=index,
            step_idx=step_idx,
        )

    def _build_stateless_step_chunks(
        self,
        contexts: list[StatelessTransitionContext],
        action_tensor: torch.Tensor,
    ) -> list[ExecutionChunk]:
        return build_stateless_step_chunks(
            contexts,
            action_tensor,
            dtype=self.dtype,
            device=self.device,
            policy=self.env_step_batch_policy,
        )

    def _resolve_env_spec(self, env_name: str) -> EnvironmentSpec:
        return self.catalog.resolve_env_spec(env_name)

    def _resolve_task(self, task_id: str, *, env_name: str) -> TaskSpec:
        return self.catalog.resolve_task(task_id, env_name=env_name)

    def _default_task_for_env(self, env_name: str) -> str:
        return self.catalog.default_task_for_env(env_name)

    @staticmethod
    def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
        return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()

    def _execute_transition_batch(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
        dispatch_mode: str,
    ) -> TransitionPredictManyResponse:
        contexts = [
            self._load_stateless_context(
                state_handle_id=str(item["state_handle_id"]),
                trajectory_id=item.get("trajectory_id"),
                max_episode_steps=item.get("max_episode_steps"),
                policy_version=policy_version,
            )
            for item in items
        ]
        env_name = contexts[0].env_name
        if any(context.env_name != env_name for context in contexts):
            raise ValueError("predict_many currently only supports batching the same env_name")

        actions = [item["action"] for item in items]
        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(len(actions), -1)
        expected_action_dim = self.catalog.action_dim_for_env(env_name)
        if action_tensor.shape[1] != expected_action_dim:
            raise ValueError(f"Expected action_dim={expected_action_dim}, got {action_tensor.shape[1]}")

        responses: list[TransitionPredictResponse] = []
        trajectory_persist_ms = 0.0
        pipeline = TransitionStagePipeline(
            world_model=self.catalog.world_model_for_env(env_name),
            reward_fn=self.catalog.reward_fn_for_env(env_name),
            dtype=self.dtype,
            device=self.device,
            policy=self.env_step_batch_policy,
        )
        prepared = pipeline.prepare(contexts=contexts, action_tensor=action_tensor)
        for executed in (pipeline.execute_chunk(chunk, stage_profile=prepared.stage_profile) for chunk in prepared.chunks):
            persist_started_at = time.perf_counter()
            chunk = executed.prepared.chunk
            for index, context in enumerate(executed.prepared.contexts):
                task = self.catalog.resolve_task(context.task_id, env_name=context.env_name)
                reward = float(executed.rewards[index].item())
                is_terminated = bool(executed.terminated[index].item())
                next_step_idx = context.step_idx + 1
                is_truncated = next_step_idx >= context.max_episode_steps
                next_state = executed.next_states[index:index + 1]
                next_handle = self._persist_state_handle(
                    episode_id=context.episode_id,
                    branch_id=context.branch_id,
                    state=next_state,
                    goal=context.goal,
                    step_idx=next_step_idx,
                    task=task,
                    env_name=context.env_name,
                    trajectory_id=context.trajectory_id,
                    parent_state_handle_id=context.state_handle_id,
                )
                info = self.catalog.transition_info_for_env(
                    context.env_name,
                    goal=context.goal,
                    info_tensors=executed.info_tensors,
                    index=index,
                    step_idx=next_step_idx,
                )
                transition = self.temporal_store.create_transition(
                    TransitionCreate(
                        env_id=context.scope_id,
                        episode_id=context.episode_id,
                        trajectory_id=context.trajectory_id,
                        task_id=context.task_id,
                        step_idx=context.step_idx,
                        observation_ref=context.state_handle_id,
                        action=list(chunk.action_batch[index].detach().cpu().numpy().tolist()),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        next_observation_ref=next_handle.state_handle_id,
                        info={**info, **metadata},
                        policy_version=policy_version or context.policy_version,
                        metadata={"batched": len(contexts) > 1, "chunk_id": chunk.chunk_id, "stateless": True},
                    )
                )
                checkpoint_id = None
                if checkpoint or is_terminated or is_truncated:
                    checkpoint_id = self._checkpoint_state_handle(
                        state_handle_id=next_handle.state_handle_id,
                        episode_id=context.episode_id,
                        branch_id=context.branch_id,
                        step_idx=next_step_idx,
                        tag=f"step-{next_step_idx}",
                        metadata={
                            "env_name": context.env_name,
                            "task_id": context.task_id,
                            "trajectory_id": context.trajectory_id,
                            "batched": len(contexts) > 1,
                            "chunk_id": chunk.chunk_id,
                            "stateless": True,
                            **metadata,
                        },
                    )
                trajectory = self.temporal_store.trajectories.get(context.trajectory_id)
                assert trajectory is not None
                trajectory.num_steps += 1
                trajectory.return_value += reward
                trajectory.success = trajectory.success or bool(info["success"])
                trajectory.transition_refs.append(transition.transition_id)
                if is_terminated:
                    trajectory.status = TemporalStatus.SUCCEEDED
                    trajectory.completed_at = time.time()
                elif is_truncated:
                    trajectory.status = TemporalStatus.ARCHIVED
                    trajectory.completed_at = time.time()
                self.temporal_store.update_trajectory(trajectory)
                responses.append(
                    TransitionPredictResponse(
                        env_name=context.env_name,
                        episode_id=context.episode_id,
                        task_id=context.task_id,
                        branch_id=context.branch_id,
                        trajectory_id=context.trajectory_id,
                        state_handle_id=next_handle.state_handle_id,
                        checkpoint_id=checkpoint_id,
                        transition_id=transition.transition_id,
                        policy_version=policy_version or context.policy_version,
                        step_idx=next_step_idx,
                        max_episode_steps=context.max_episode_steps,
                        observation=self._observation(next_state, context.goal),
                        reward=reward,
                        terminated=is_terminated,
                        truncated=is_truncated,
                        info=info,
                    )
                )
            persist_elapsed_ms = (time.perf_counter() - persist_started_at) * 1000.0
            trajectory_persist_ms += persist_elapsed_ms
            prepared.stage_profile.record("persist", persist_elapsed_ms)

        chunk_sizes = [chunk.chunk.size for chunk in prepared.chunks]
        return TransitionPredictManyResponse(
            results=responses,
            runtime={
                "execution_path": "chunked_stateless_transition",
                "dispatch_mode": dispatch_mode,
                "chunk_count": len(prepared.chunks),
                "chunk_sizes": chunk_sizes,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
                "reward_stage_ms": prepared.stage_profile.snapshot()["stages"].get("reward", {}).get("total_ms", 0.0),
                "trajectory_persist_ms": trajectory_persist_ms,
                "state_locality_hit_rate": 0.0,
                "state_locality_mode": "explicit_state_handle",
                "step_semantics": "explicit_state_transition",
                "northbound_reset_policy": "resource_reference_required",
                **prepared.chunk_summary,
                "chunks": prepared.chunk_history,
                "stage_profile": prepared.stage_profile.snapshot(),
            },
        )


# Preserve the old class name for compatibility while exposing a runtime-native
# name that better matches the module's role.
LearnedEnvRuntimeManager = RLEnvironmentManager


__all__ = ["LearnedEnvRuntimeManager", "RLEnvironmentManager"]
