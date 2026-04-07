"""Session CRUD and trajectory management for the learned env runtime.

This module owns the stateful session lifecycle: create, reset, fork,
checkpoint, delete.  It delegates transition execution to
:class:`TransitionExecutor`.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from wm_infra.api.protocol import (
    EnvironmentSessionResponse,
    TransitionContextResponse,
)
from wm_infra.controlplane import (
    BranchCreate,
    EnvironmentSessionCreate,
    EnvironmentSessionRecord,
    EpisodeCreate,
    StateHandleKind,
    TaskSpec,
    TemporalStatus,
    TemporalStore,
    TrajectoryCreate,
    TrajectoryRecord,
)
from wm_infra.runtime.env.catalog import LearnedEnvCatalog
from wm_infra.runtime.env.state import build_inline_state_handle_create, load_runtime_state_view
from wm_infra.runtime.env.transition import StatelessTransitionContext


class SessionStore:
    """Session lifecycle management — create, reset, fork, checkpoint, delete."""

    def __init__(
        self,
        *,
        temporal_store: TemporalStore,
        catalog: LearnedEnvCatalog,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.temporal_store = temporal_store
        self.catalog = catalog
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

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
        return self.session_response(session)

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
            finalize_trajectory(self.temporal_store, session.trajectory_id, success=False)
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
        return self.session_response(session)

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
        trajectory = ensure_stateless_trajectory(
            self.temporal_store,
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
        return self.session_response(session)

    def checkpoint_session(self, env_id: str, *, tag: str | None, metadata: dict[str, Any]) -> str:
        from wm_infra.controlplane import CheckpointCreate
        session = self.get_session(env_id)
        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=session.episode_id,
                branch_id=session.branch_id or "",
                state_handle_id=session.state_handle_id,
                step_index=session.current_step,
                tag=tag or f"step-{session.current_step}",
                metadata={"env_name": session.env_name, "task_id": session.task_id, **metadata},
            )
        )
        state_handle = self.temporal_store.state_handles.get(session.state_handle_id)
        if state_handle is not None:
            state_handle.checkpoint_id = checkpoint.checkpoint_id
            self.temporal_store.state_handles.put(state_handle)
        return checkpoint.checkpoint_id

    def delete_session(self, env_id: str) -> None:
        session = self.get_session(env_id)
        if session.trajectory_id:
            finalize_trajectory(self.temporal_store, session.trajectory_id, success=False)
        session.status = TemporalStatus.ARCHIVED
        session.completed_at = time.time()
        self.temporal_store.update_environment_session(session)

    # ------------------------------------------------------------------
    # Context initialization
    # ------------------------------------------------------------------

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
        trajectory = ensure_stateless_trajectory(
            self.temporal_store,
            env_name=env_name,
            task=task,
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            trajectory_id=None,
            policy_version=policy_version,
            max_episode_steps=max_episode_steps or spec.default_horizon,
            metadata={"seed": seed, **metadata},
        )
        state, goal = self.catalog.sample_initial_state(task, seed)
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
            observation=_observation(state, goal),
            info=self.catalog.session_info_for_env(env_name, goal, 0),
        )

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def session_response(self, session: EnvironmentSessionRecord) -> EnvironmentSessionResponse:
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
            observation=_observation(state, goal),
            info=self.catalog.session_info_for_env(session.env_name, goal, session.current_step),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        from wm_infra.runtime.env.transition_executor import TransitionExecutor
        # Reuse the TransitionExecutor's context loader via a lightweight instance
        from wm_infra.runtime.env.async_runtime import AsyncTransitionDispatcher
        from wm_infra.runtime.execution import ExecutionBatchPolicy
        executor = TransitionExecutor(
            temporal_store=self.temporal_store,
            catalog=self.catalog,
            dispatcher=AsyncTransitionDispatcher(),
            batch_policy=ExecutionBatchPolicy(),
            device=self.device,
            dtype=self.dtype,
        )
        return executor.load_stateless_context(
            state_handle_id=state_handle_id,
            trajectory_id=trajectory_id,
            max_episode_steps=max_episode_steps,
            policy_version=policy_version,
        )

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


# ------------------------------------------------------------------
# Shared free functions (used by both SessionStore and TransitionExecutor)
# ------------------------------------------------------------------

def ensure_stateless_trajectory(
    temporal_store: TemporalStore,
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
        trajectory = temporal_store.trajectories.get(trajectory_id)
        if trajectory is None:
            raise KeyError(trajectory_id)
        if trajectory.episode_id != episode_id:
            raise ValueError("trajectory_id does not belong to the requested episode/state")
        if trajectory.task_id != task.task_id:
            raise ValueError("trajectory_id does not match the requested task")
        if policy_version is not None and trajectory.policy_version != policy_version:
            trajectory.policy_version = policy_version
            temporal_store.update_trajectory(trajectory)
        if "max_episode_steps" not in trajectory.metadata:
            trajectory.metadata["max_episode_steps"] = max_episode_steps
            temporal_store.update_trajectory(trajectory)
        return trajectory

    scope_id = f"stateless:{episode_id}:{branch_id}"
    return temporal_store.create_trajectory(
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


def finalize_trajectory(temporal_store: TemporalStore, trajectory_id: str, *, success: bool) -> None:
    trajectory = temporal_store.trajectories.get(trajectory_id)
    if trajectory is None or trajectory.completed_at is not None:
        return
    trajectory.success = trajectory.success or success
    trajectory.status = TemporalStatus.SUCCEEDED if trajectory.success else TemporalStatus.ARCHIVED
    trajectory.completed_at = time.time()
    temporal_store.update_trajectory(trajectory)


def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
    return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()


__all__ = [
    "SessionStore",
    "ensure_stateless_trajectory",
    "finalize_trajectory",
]
