"""Learned environment stepping runtime for wm-infra.

This module is a thin facade that composes :class:`SessionStore` (session
CRUD) with :class:`TransitionExecutor` (pipeline execution) and
:class:`LearnedEnvCatalog` (env registry).

Northbound RL and HTTP surfaces should depend on this runtime rather than
embedding transition execution logic directly.
"""

from __future__ import annotations

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
    EnvironmentSessionRecord,
    EnvironmentSpec,
    TaskSpec,
    TemporalStore,
    TrajectoryRecord,
)
from wm_infra.runtime.execution import ExecutionBatchPolicy, ExecutionChunk
from wm_infra.runtime.env.async_runtime import AsyncTransitionDispatcher, TransitionDispatch
from wm_infra.runtime.env.catalog import LearnedEnvCatalog
from wm_infra.runtime.env.session_store import SessionStore
from wm_infra.runtime.env.transition import StatelessTransitionContext, build_stateless_step_chunks
from wm_infra.runtime.env.transition_executor import TransitionExecutor


class RLEnvironmentManager:
    """Environment registry + stateless transition manager for trainer-facing RL APIs.

    Composes SessionStore + TransitionExecutor + LearnedEnvCatalog.
    """

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

        # Composed substrate layers
        self._session_store = SessionStore(
            temporal_store=temporal_store,
            catalog=self.catalog,
            device=self.device,
            dtype=self.dtype,
        )
        self._executor = TransitionExecutor(
            temporal_store=temporal_store,
            catalog=self.catalog,
            dispatcher=self.dispatcher,
            batch_policy=self.env_step_batch_policy,
            device=self.device,
            dtype=self.dtype,
        )
        self._register_catalog()

    def _register_catalog(self) -> None:
        self.catalog.register_defaults()

    # ------------------------------------------------------------------
    # Catalog delegation
    # ------------------------------------------------------------------

    def list_environment_specs(self) -> list[EnvironmentSpec]:
        return self.catalog.list_environment_specs()

    def list_task_specs(self, env_name: str | None = None) -> list[TaskSpec]:
        return self.catalog.list_task_specs(env_name=env_name)

    def action_dim_for_env(self, env_name: str) -> int:
        return self.catalog.action_dim_for_env(env_name)

    def backend_for_env(self, env_name: str) -> str:
        return self.catalog.backend_for_env(env_name)

    # ------------------------------------------------------------------
    # Session delegation (→ SessionStore)
    # ------------------------------------------------------------------

    def get_session(self, env_id: str) -> EnvironmentSessionRecord:
        return self._session_store.get_session(env_id)

    def list_sessions(self) -> list[EnvironmentSessionRecord]:
        return self._session_store.list_sessions()

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
        return self._session_store.create_session(
            env_name=env_name, task_id=task_id, seed=seed,
            policy_version=policy_version, max_episode_steps=max_episode_steps,
            labels=labels, metadata=metadata,
        )

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
        return self._session_store.initialize_transition_context(
            env_name=env_name, task_id=task_id, seed=seed,
            policy_version=policy_version, max_episode_steps=max_episode_steps,
            branch_name=branch_name, labels=labels, metadata=metadata,
        )

    def reset_session(self, env_id: str, *, seed: int | None, policy_version: str | None, metadata: dict[str, Any]) -> EnvironmentSessionResponse:
        return self._session_store.reset_session(env_id, seed=seed, policy_version=policy_version, metadata=metadata)

    def fork_session(self, env_id: str, *, branch_name: str | None, policy_version: str | None, metadata: dict[str, Any]) -> EnvironmentSessionResponse:
        return self._session_store.fork_session(env_id, branch_name=branch_name, policy_version=policy_version, metadata=metadata)

    def checkpoint_session(self, env_id: str, *, tag: str | None, metadata: dict[str, Any]) -> str:
        return self._session_store.checkpoint_session(env_id, tag=tag, metadata=metadata)

    def delete_session(self, env_id: str) -> None:
        self._session_store.delete_session(env_id)

    # ------------------------------------------------------------------
    # Transition execution delegation (→ TransitionExecutor)
    # ------------------------------------------------------------------

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
        return self._executor.predict_transition(
            state_handle_id=state_handle_id, action=action,
            trajectory_id=trajectory_id, policy_version=policy_version,
            checkpoint=checkpoint, max_episode_steps=max_episode_steps,
            metadata=metadata,
        )

    def predict_many_transitions(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        return self._executor.predict_many(
            items=items, policy_version=policy_version,
            checkpoint=checkpoint, metadata=metadata,
        )

    async def predict_many_transitions_async(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        return await self._executor.predict_many_async(
            items=items, policy_version=policy_version,
            checkpoint=checkpoint, metadata=metadata,
        )

    def dispatch_transition_batch(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionDispatch:
        return self._executor.dispatch_batch(
            items=items, policy_version=policy_version,
            checkpoint=checkpoint, metadata=metadata,
        )

    def collect_transition_batch(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return self._executor.collect(dispatch_id, timeout=timeout)

    async def collect_transition_batch_async(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return await self._executor.collect_async(dispatch_id, timeout=timeout)

    # ------------------------------------------------------------------
    # step_session / step_many (compose session + executor)
    # ------------------------------------------------------------------

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
        runtime["state_locality_mode"] = "explicit_state_transition"
        runtime["step_semantics"] = "sync_step_many"
        runtime["northbound_reset_policy"] = "explicit_reset_required"
        return EnvironmentStepManyResponse(
            env_ids=ordered_env_ids,
            results=responses,
            runtime=runtime,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Backward-compat private methods (thin wrappers)
    # ------------------------------------------------------------------

    @staticmethod
    def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
        return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()

    def _sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.catalog.sample_initial_state(task, seed)

    def _world_model_for_env(self, env_name: str):
        return self.catalog.world_model_for_env(env_name)

    def _reward_fn_for_env(self, env_name: str):
        return self.catalog.reward_fn_for_env(env_name)

    def _action_dim_for_env(self, env_name: str) -> int:
        return self.catalog.action_dim_for_env(env_name)

    def _session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        return self.catalog.session_info_for_env(env_name, goal, step_idx)

    def _transition_info_for_env(self, env_name: str, *, goal: torch.Tensor, info_tensors: dict[str, torch.Tensor], index: int, step_idx: int) -> dict[str, Any]:
        return self.catalog.transition_info_for_env(env_name, goal=goal, info_tensors=info_tensors, index=index, step_idx=step_idx)

    def _build_stateless_step_chunks(self, contexts: list[StatelessTransitionContext], action_tensor: torch.Tensor) -> list[ExecutionChunk]:
        return build_stateless_step_chunks(contexts, action_tensor, dtype=self.dtype, device=self.device, policy=self.env_step_batch_policy)

    def _resolve_env_spec(self, env_name: str) -> EnvironmentSpec:
        return self.catalog.resolve_env_spec(env_name)

    def _resolve_task(self, task_id: str, *, env_name: str) -> TaskSpec:
        return self.catalog.resolve_task(task_id, env_name=env_name)

    def _default_task_for_env(self, env_name: str) -> str:
        return self.catalog.default_task_for_env(env_name)

    def _load_stateless_context(self, *, state_handle_id: str, trajectory_id: str | None, max_episode_steps: int | None, policy_version: str | None) -> StatelessTransitionContext:
        return self._executor.load_stateless_context(state_handle_id=state_handle_id, trajectory_id=trajectory_id, max_episode_steps=max_episode_steps, policy_version=policy_version)

    def _persist_state_handle(self, *, episode_id: str, branch_id: str, state: torch.Tensor, goal: torch.Tensor, step_idx: int, task: TaskSpec, env_name: str, trajectory_id: str | None, parent_state_handle_id: str | None = None):
        return self._executor._persist_state_handle(episode_id=episode_id, branch_id=branch_id, state=state, goal=goal, step_idx=step_idx, task=task, env_name=env_name, trajectory_id=trajectory_id, parent_state_handle_id=parent_state_handle_id)

    def _finalize_trajectory(self, trajectory_id: str, *, success: bool) -> None:
        from wm_infra.runtime.env.session_store import finalize_trajectory
        finalize_trajectory(self.temporal_store, trajectory_id, success=success)

    def _checkpoint_state_handle(self, *, state_handle_id: str, episode_id: str, branch_id: str | None, step_idx: int, tag: str, metadata: dict[str, Any]) -> str:
        return self._executor._checkpoint_state_handle(state_handle_id=state_handle_id, episode_id=episode_id, branch_id=branch_id, step_idx=step_idx, tag=tag, metadata=metadata)

    def _load_state_goal_from_handle(self, state_handle_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self._session_store._load_state_goal_from_handle(state_handle_id)

    def _ensure_stateless_trajectory(self, *, env_name: str, task: TaskSpec, episode_id: str, branch_id: str, trajectory_id: str | None, policy_version: str | None, max_episode_steps: int, metadata: dict[str, Any]) -> Any:
        from wm_infra.runtime.env.session_store import ensure_stateless_trajectory
        return ensure_stateless_trajectory(self.temporal_store, env_name=env_name, task=task, episode_id=episode_id, branch_id=branch_id, trajectory_id=trajectory_id, policy_version=policy_version, max_episode_steps=max_episode_steps, metadata=metadata)

    def _execute_transition_batch(self, *, items: list[dict[str, Any]], policy_version: str | None, checkpoint: bool, metadata: dict[str, Any], dispatch_mode: str) -> TransitionPredictManyResponse:
        return self._executor.execute_batch(items=items, policy_version=policy_version, checkpoint=checkpoint, metadata=metadata, dispatch_mode=dispatch_mode)


__all__ = ["RLEnvironmentManager"]
