"""Transition execution substrate — the Madrona-style system runner.

This module owns the core execution path: loading stateless contexts,
building the pipeline, running chunks through transition + reward stages,
and persisting results.  It knows nothing about sessions.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from wm_infra.api.protocol import (
    TransitionPredictManyResponse,
    TransitionPredictResponse,
)
from wm_infra.controlplane import (
    CheckpointCreate,
    ExecutionStateRef,
    StateHandleKind,
    StateLineageRef,
    TaskSpec,
    TemporalStatus,
    TemporalStore,
    TransitionCreate,
)
from wm_infra.runtime.execution import ExecutionBatchPolicy
from wm_infra.runtime.env.async_runtime import AsyncTransitionDispatcher, TransitionDispatch
from wm_infra.runtime.env.catalog import LearnedEnvCatalog
from wm_infra.runtime.env.pipeline import TransitionStagePipeline
from wm_infra.runtime.env.state import build_inline_state_handle_create, load_runtime_state_view
from wm_infra.runtime.env.transition import StatelessTransitionContext


class TransitionExecutor:
    """Pure transition execution substrate.

    Owns: context loading, pipeline construction, chunk execution,
    state-handle + transition + trajectory persistence.

    Does NOT own: sessions, env catalog construction, or dispatch lifecycle.
    Those are injected.
    """

    def __init__(
        self,
        *,
        temporal_store: TemporalStore,
        catalog: LearnedEnvCatalog,
        dispatcher: AsyncTransitionDispatcher,
        batch_policy: ExecutionBatchPolicy,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.temporal_store = temporal_store
        self.catalog = catalog
        self.dispatcher = dispatcher
        self.batch_policy = batch_policy
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Public execution API
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
        response = self.predict_many(
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

    def predict_many(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        if not items:
            raise ValueError("predict_many requires at least one item")
        return self.execute_batch(
            items=items,
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
            dispatch_mode="sync_inline",
        )

    async def predict_many_async(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPredictManyResponse:
        dispatch = self.dispatch_batch(
            items=items,
            policy_version=policy_version,
            checkpoint=checkpoint,
            metadata=metadata,
        )
        return await self.collect_async(dispatch.dispatch_id)

    def dispatch_batch(
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
            fn=self.execute_batch,
            kwargs={
                "items": items,
                "policy_version": policy_version,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "dispatch_mode": "async_dispatched",
            },
        )

    def collect(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return self.dispatcher.collect(dispatch_id, timeout=timeout)

    async def collect_async(self, dispatch_id: str, *, timeout: float | None = None) -> TransitionPredictManyResponse:
        return await self.dispatcher.collect_async(dispatch_id, timeout=timeout)

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        *,
        items: list[dict[str, Any]],
        policy_version: str | None,
        checkpoint: bool,
        metadata: dict[str, Any],
        dispatch_mode: str,
    ) -> TransitionPredictManyResponse:
        contexts = [
            self.load_stateless_context(
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
            policy=self.batch_policy,
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
                        observation=_observation(next_state, context.goal),
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

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def load_stateless_context(
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

        from wm_infra.runtime.env.session_store import ensure_stateless_trajectory
        trajectory = ensure_stateless_trajectory(
            self.temporal_store,
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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


def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
    return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()


__all__ = ["TransitionExecutor"]
