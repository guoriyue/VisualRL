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
    ExecutionStateRef,
    StateLineageRef,
    TemporalStore,
)
from wm_infra.execution import ExecutionBatchPolicy
from wm_infra.env_runtime.async_runtime import AsyncTransitionDispatcher, TransitionDispatch
from wm_infra.env_runtime.catalog import LearnedEnvCatalog
from wm_infra.env_runtime.persistence import (
    TransitionExecutionResult,
    TransitionPersistenceContext,
    TransitionPersistenceLayer,
    build_transition_persistence_plan,
)
from wm_infra.env_runtime.pipeline import TransitionStagePipeline
from wm_infra.env_runtime.state import load_runtime_state_view
from wm_infra.env_runtime.transition import StatelessTransitionContext


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
        self.persistence = TransitionPersistenceLayer(temporal_store)

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
        persist_ms_total = 0.0
        pipeline = TransitionStagePipeline(
            world_model=self.catalog.world_model_for_env(env_name),
            reward_fn=self.catalog.reward_fn_for_env(env_name),
            dtype=self.dtype,
            device=self.device,
            policy=self.batch_policy,
        )
        schedule_started_at = time.perf_counter()
        prepared = pipeline.prepare(contexts=contexts, action_tensor=action_tensor)
        schedule_ms = (time.perf_counter() - schedule_started_at) * 1000.0
        state_count = 0
        state_bytes = 0
        for run in (
            pipeline.run_chunk(
                chunk,
                checkpoint=checkpoint,
                metadata=metadata,
                stage_profile=prepared.stage_profile,
            )
            for chunk in prepared.chunks
        ):
            persist_started_at = time.perf_counter()
            chunk = run.persist.reward_stage.prepared.chunk
            plans = self._build_persistence_plans(
                run=run,
                policy_version=policy_version,
                request_metadata=metadata,
            )
            state_count += len(plans)
            state_bytes += sum(int(plan.result.next_state.nbytes) for plan in plans)
            committed = self.persistence.commit_many(plans)
            for plan, commit in zip(plans, committed, strict=True):
                responses.append(
                    TransitionPredictResponse(
                        env_name=plan.context.env_name,
                        episode_id=plan.context.episode_id,
                        task_id=plan.context.task_id,
                        branch_id=plan.context.branch_id,
                        trajectory_id=plan.context.trajectory_id,
                        state_handle_id=commit.next_state_handle_id,
                        checkpoint_id=commit.checkpoint_id,
                        transition_id=commit.transition_id,
                        policy_version=plan.result.policy_version or plan.context.policy_version,
                        step_idx=plan.next_step_idx,
                        max_episode_steps=plan.context.max_episode_steps,
                        observation=_observation(plan.result.next_state, plan.context.goal),
                        reward=plan.result.reward,
                        terminated=plan.result.terminated,
                        truncated=plan.result.truncated,
                        info=plan.result.info,
                    )
                )
            persist_elapsed_ms = (time.perf_counter() - persist_started_at) * 1000.0
            persist_ms_total += persist_elapsed_ms
            prepared.stage_profile.record("persist", persist_elapsed_ms)

        chunk_sizes = [chunk.chunk.size for chunk in prepared.chunks]
        stage_snapshot = prepared.stage_profile.snapshot()
        return TransitionPredictManyResponse(
            results=responses,
            runtime={
                "execution_path": "chunked_stateless_transition",
                "dispatch_mode": dispatch_mode,
                "schedule_ms": schedule_ms,
                "transition_ms": stage_snapshot["stages"].get("transition", {}).get("total_ms", 0.0),
                "reward_ms": stage_snapshot["stages"].get("reward", {}).get("total_ms", 0.0),
                "persist_ms": persist_ms_total,
                "chunk_count": len(prepared.chunks),
                "chunk_sizes": chunk_sizes,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
                "state_bytes": state_bytes,
                "state_count": state_count,
                "queue_wait_ms": 0.0,
                "state_locality_hit_rate": 0.0,
                "state_locality_mode": "explicit_state_handle",
                "step_semantics": "explicit_state_transition",
                "northbound_reset_policy": "resource_reference_required",
                **prepared.chunk_summary,
                "chunks": prepared.chunk_history,
                "stage_profile": stage_snapshot,
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

        from wm_infra.env_runtime.session_store import ensure_stateless_trajectory
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

    def _build_persistence_plans(
        self,
        *,
        run,
        policy_version: str | None,
        request_metadata: dict[str, Any],
    ) -> list:
        plans = []
        for index, (context, intent) in enumerate(zip(run.persist.reward_stage.prepared.contexts, run.persist.intents, strict=True)):
            info = self.catalog.transition_info_for_env(
                context.env_name,
                goal=context.goal,
                info_tensors=run.reward.info_tensors,
                index=index,
                step_idx=intent.next_step_idx,
            )
            plans.append(
                build_transition_persistence_plan(
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
                        policy_version=policy_version or context.policy_version,
                        checkpoint_id=context.checkpoint_id,
                    ),
                    TransitionExecutionResult(
                        action=intent.action,
                        next_state=intent.next_state,
                        reward=intent.reward,
                        terminated=intent.terminated,
                        truncated=intent.truncated,
                        info=info,
                        policy_version=policy_version or context.policy_version,
                    ),
                    checkpoint_requested=intent.checkpoint_requested,
                    metadata={
                        "checkpoint_tag": intent.checkpoint_tag,
                        "env_name": context.env_name,
                        "task_id": context.task_id,
                        "trajectory_id": context.trajectory_id,
                        "batched": run.persist.reward_stage.prepared.chunk.size > 1,
                        "chunk_id": intent.chunk_id,
                        "stateless": True,
                        **request_metadata,
                    },
                )
            )
        return plans


def _observation(state: torch.Tensor, goal: torch.Tensor) -> list[list[float]]:
    return torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()


__all__ = ["TransitionExecutor"]
