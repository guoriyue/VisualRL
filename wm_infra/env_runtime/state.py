"""State-handle helpers for learned env runtime.

State handles remain a persisted control-plane surface, but runtime execution
state and lineage are modeled as separate concepts so scheduling and residency
policies can evolve independently from trajectory bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wm_infra.controlplane import (
    ExecutionResidencyRef,
    StateHandleCreate,
    StateHandleKind,
    StateHandleRecord,
    StateLineageRef,
    StateResidency,
)


@dataclass(slots=True)
class RuntimeStateView:
    state_handle_id: str
    state: torch.Tensor
    goal: torch.Tensor
    step_idx: int
    execution_state_ref: ExecutionResidencyRef
    lineage_ref: StateLineageRef
    checkpoint_id: str | None

    @property
    def execution_residency_ref(self) -> ExecutionResidencyRef:
        return self.execution_state_ref


@dataclass(slots=True)
class StateHandleRefs:
    execution_residency_ref: ExecutionResidencyRef
    lineage_ref: StateLineageRef


def _estimate_tensor_bytes(*tensors: torch.Tensor) -> int:
    total = 0
    for tensor in tensors:
        total += int(tensor.numel() * tensor.element_size())
    return total


def build_inline_state_handle_create(
    *,
    episode_id: str,
    branch_id: str,
    state: torch.Tensor,
    goal: torch.Tensor,
    step_idx: int,
    env_name: str,
    task_id: str,
    trajectory_id: str | None,
    parent_state_handle_id: str | None = None,
    rollout_id: str | None = None,
    checkpoint_id: str | None = None,
    is_terminal: bool = False,
    kind: StateHandleKind = StateHandleKind.LATENT,
) -> StateHandleCreate:
    observation = torch.cat([state, goal], dim=-1).squeeze(0).detach().cpu().numpy().tolist()
    execution_state_ref = ExecutionResidencyRef(
        residency=StateResidency.INLINE,
        storage_backend="state_handle_metadata",
        state_key="latent_state",
        goal_key="goal_state",
        step_key="step_idx",
        device=str(state.device),
        bytes_estimate=_estimate_tensor_bytes(state, goal),
        metadata={"observation_key": "observation"},
    )
    lineage_ref = StateLineageRef(
        env_name=env_name,
        task_id=task_id,
        branch_id=branch_id,
        rollout_id=rollout_id,
        checkpoint_id=checkpoint_id,
        trajectory_id=trajectory_id,
        step_idx=step_idx,
        parent_state_handle_id=parent_state_handle_id,
        metadata={"branch_id": branch_id},
    )
    return StateHandleCreate(
        episode_id=episode_id,
        branch_id=branch_id,
        rollout_id=rollout_id,
        checkpoint_id=checkpoint_id,
        kind=kind,
        shape=list(state.shape[1:]),
        dtype=str(state.dtype).replace("torch.", ""),
        is_terminal=is_terminal,
        execution_state_ref=execution_state_ref,
        lineage_ref=lineage_ref,
        metadata={
            "env_name": env_name,
            "task_id": task_id,
            "step_idx": step_idx,
            "latent_state": state.squeeze(0).detach().cpu().numpy().tolist(),
            "goal_state": goal.squeeze(0).detach().cpu().numpy().tolist(),
            "observation": observation,
        },
    )


def split_state_handle_refs(record: StateHandleRecord) -> StateHandleRefs:
    execution_state_ref = record.execution_state_ref or ExecutionResidencyRef(
        residency=StateResidency.INLINE,
        storage_backend="state_handle_metadata",
    )
    lineage_ref = record.lineage_ref or StateLineageRef(
        env_name=record.metadata.get("env_name"),
        task_id=record.metadata.get("task_id"),
        branch_id=record.branch_id,
        rollout_id=record.rollout_id,
        checkpoint_id=record.checkpoint_id,
        trajectory_id=record.metadata.get("trajectory_id"),
        step_idx=int(record.metadata.get("step_idx", 0)),
        parent_state_handle_id=record.metadata.get("parent_state_handle_id"),
    )
    if lineage_ref.branch_id is None:
        lineage_ref.branch_id = record.branch_id
    if lineage_ref.rollout_id is None:
        lineage_ref.rollout_id = record.rollout_id
    if lineage_ref.checkpoint_id is None:
        lineage_ref.checkpoint_id = record.checkpoint_id
    return StateHandleRefs(
        execution_residency_ref=execution_state_ref,
        lineage_ref=lineage_ref,
    )


def load_runtime_state_view(
    record: StateHandleRecord,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> RuntimeStateView:
    refs = split_state_handle_refs(record)
    execution_state_ref = refs.execution_residency_ref
    lineage_ref = refs.lineage_ref
    metadata = record.metadata
    state_key = execution_state_ref.state_key or "latent_state"
    goal_key = execution_state_ref.goal_key or "goal_state"
    step_key = execution_state_ref.step_key or "step_idx"
    if state_key not in metadata or goal_key not in metadata:
        raise ValueError(
            f"state_handle {record.state_handle_id} is missing execution payload "
            f"for keys {state_key!r}/{goal_key!r}"
        )
    state = torch.as_tensor(metadata[state_key], dtype=dtype, device=device).unsqueeze(0)
    goal = torch.as_tensor(metadata[goal_key], dtype=dtype, device=device).unsqueeze(0)
    step_idx = int(getattr(lineage_ref, "step_idx", metadata.get(step_key, 0)))
    if not lineage_ref.env_name:
        lineage_ref.env_name = metadata.get("env_name")
    if not lineage_ref.task_id:
        lineage_ref.task_id = metadata.get("task_id")
    return RuntimeStateView(
        state_handle_id=record.state_handle_id,
        state=state,
        goal=goal,
        step_idx=step_idx,
        execution_state_ref=execution_state_ref,
        lineage_ref=lineage_ref,
        checkpoint_id=record.checkpoint_id,
    )


__all__ = [
    "RuntimeStateView",
    "StateHandleRefs",
    "build_inline_state_handle_create",
    "load_runtime_state_view",
    "split_state_handle_refs",
]
