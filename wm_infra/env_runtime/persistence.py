"""Persistence commit layer for learned environment transitions.

This module sits below the runtime manager. It turns a transition execution
result plus a state continuity context into durable control-plane writes:
state handle, transition, checkpoint, and trajectory updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from wm_infra.controlplane import (
    CheckpointCreate,
    CheckpointRecord,
    StateHandleRecord,
    TemporalStatus,
    TemporalStore,
    TrajectoryRecord,
    TransitionCreate,
    TransitionRecord,
)
from wm_infra.env_runtime.state import build_inline_state_handle_create


@dataclass(slots=True)
class TransitionPersistenceContext:
    env_id: str
    env_name: str
    task_id: str
    episode_id: str
    branch_id: str
    trajectory_id: str
    state_handle_id: str
    state: torch.Tensor
    goal: torch.Tensor
    step_idx: int
    max_episode_steps: int
    policy_version: str | None = None
    checkpoint_id: str | None = None
    parent_state_handle_id: str | None = None


@dataclass(slots=True)
class TransitionExecutionResult:
    action: list[float]
    next_state: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    policy_version: str | None = None


@dataclass(slots=True)
class TransitionPersistencePlan:
    context: TransitionPersistenceContext
    result: TransitionExecutionResult
    checkpoint_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def next_step_idx(self) -> int:
        return self.context.step_idx + 1

    @property
    def should_checkpoint(self) -> bool:
        return self.checkpoint_requested or self.result.terminated or self.result.truncated

    @property
    def should_finalize_trajectory(self) -> bool:
        return self.result.terminated or self.result.truncated


@dataclass(slots=True)
class TransitionCommitResult:
    next_state_handle_id: str
    transition_id: str
    checkpoint_id: str | None
    trajectory_id: str
    state_handle: StateHandleRecord
    transition: TransitionRecord
    trajectory: TrajectoryRecord
    checkpoint: CheckpointRecord | None = None


def build_transition_persistence_plan(
    context: TransitionPersistenceContext,
    result: TransitionExecutionResult,
    *,
    checkpoint_requested: bool = False,
    metadata: dict[str, Any] | None = None,
) -> TransitionPersistencePlan:
    return TransitionPersistencePlan(
        context=context,
        result=result,
        checkpoint_requested=checkpoint_requested,
        metadata=dict(metadata or {}),
    )


class TransitionPersistenceLayer:
    """Commit learned-env transitions into the temporal control plane."""

    def __init__(self, temporal_store: TemporalStore) -> None:
        self.temporal_store = temporal_store

    def commit(self, plan: TransitionPersistencePlan) -> TransitionCommitResult:
        next_handle = self.temporal_store.create_state_handle(
            build_inline_state_handle_create(
                episode_id=plan.context.episode_id,
                branch_id=plan.context.branch_id,
                state=plan.result.next_state,
                goal=plan.context.goal,
                step_idx=plan.next_step_idx,
                env_name=plan.context.env_name,
                task_id=plan.context.task_id,
                trajectory_id=plan.context.trajectory_id,
                parent_state_handle_id=plan.context.state_handle_id,
                checkpoint_id=plan.context.checkpoint_id,
                is_terminal=plan.result.terminated,
            )
        )

        transition = self.temporal_store.create_transition(
            TransitionCreate(
                env_id=plan.context.env_id,
                episode_id=plan.context.episode_id,
                trajectory_id=plan.context.trajectory_id,
                task_id=plan.context.task_id,
                step_idx=plan.context.step_idx,
                observation_ref=plan.context.state_handle_id,
                action=list(plan.result.action),
                reward=float(plan.result.reward),
                terminated=bool(plan.result.terminated),
                truncated=bool(plan.result.truncated),
                next_observation_ref=next_handle.state_handle_id,
                info={
                    **plan.result.info,
                    "state_handle_id": plan.context.state_handle_id,
                    "next_state_handle_id": next_handle.state_handle_id,
                    "step_idx": plan.next_step_idx,
                    **plan.metadata,
                },
                policy_version=plan.result.policy_version or plan.context.policy_version,
                metadata={
                    "checkpoint_requested": plan.checkpoint_requested,
                    "should_checkpoint": plan.should_checkpoint,
                    "should_finalize_trajectory": plan.should_finalize_trajectory,
                    "state_handle_id": plan.context.state_handle_id,
                    "next_state_handle_id": next_handle.state_handle_id,
                    **plan.metadata,
                },
            )
        )

        trajectory = self.temporal_store.trajectories.get(plan.context.trajectory_id)
        if trajectory is None:
            raise KeyError(plan.context.trajectory_id)

        trajectory.num_steps += 1
        trajectory.return_value += float(plan.result.reward)
        trajectory.success = trajectory.success or bool(plan.result.info.get("success", False))
        trajectory.transition_refs.append(transition.transition_id)
        if plan.should_finalize_trajectory:
            trajectory.status = TemporalStatus.SUCCEEDED if trajectory.success else TemporalStatus.ARCHIVED
            trajectory.completed_at = trajectory.completed_at or transition.created_at
        self.temporal_store.update_trajectory(trajectory)

        checkpoint_record: CheckpointRecord | None = None
        if plan.should_checkpoint:
            checkpoint_tag = str(plan.metadata.get("checkpoint_tag", f"step-{plan.next_step_idx}"))
            checkpoint_record = self.temporal_store.create_checkpoint(
                CheckpointCreate(
                    episode_id=plan.context.episode_id,
                    rollout_id=plan.context.trajectory_id,
                    branch_id=plan.context.branch_id,
                    state_handle_id=next_handle.state_handle_id,
                    step_index=plan.next_step_idx,
                    tag=checkpoint_tag,
                    metadata={
                        "env_id": plan.context.env_id,
                        "env_name": plan.context.env_name,
                        "task_id": plan.context.task_id,
                        "trajectory_id": plan.context.trajectory_id,
                        **plan.metadata,
                    },
                )
            )
            next_handle.checkpoint_id = checkpoint_record.checkpoint_id
            next_handle = self.temporal_store.state_handles.put(next_handle)

        return TransitionCommitResult(
            next_state_handle_id=next_handle.state_handle_id,
            transition_id=transition.transition_id,
            checkpoint_id=checkpoint_record.checkpoint_id if checkpoint_record is not None else None,
            trajectory_id=trajectory.trajectory_id,
            state_handle=next_handle,
            transition=transition,
            trajectory=trajectory,
            checkpoint=checkpoint_record,
        )

    def commit_many(self, plans: list[TransitionPersistencePlan]) -> list[TransitionCommitResult]:
        return [self.commit(plan) for plan in plans]


__all__ = [
    "TransitionCommitResult",
    "TransitionExecutionResult",
    "TransitionPersistenceContext",
    "TransitionPersistenceLayer",
    "TransitionPersistencePlan",
    "build_transition_persistence_plan",
]
