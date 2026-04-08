"""Transition execution helpers for learned environment runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wm_infra.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionEntity,
    ExecutionWorkItem,
    HomogeneousChunkScheduler,
)


@dataclass(slots=True)
class StatelessTransitionContext:
    env_name: str
    task_id: str
    episode_id: str
    branch_id: str
    trajectory_id: str
    state_handle_id: str
    checkpoint_id: str | None
    policy_version: str | None
    max_episode_steps: int
    state: torch.Tensor
    goal: torch.Tensor
    step_idx: int
    scope_id: str


def build_stateless_step_chunks(
    contexts: list[StatelessTransitionContext],
    action_tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    policy: ExecutionBatchPolicy,
) -> list[ExecutionChunk]:
    if not contexts:
        return []
    signature = BatchSignature(
        stage="stateless_transition",
        latent_shape=tuple(contexts[0].state.shape[-2:]),
        action_dim=int(action_tensor.shape[-1]),
        dtype=str(dtype).replace("torch.", ""),
        device=str(device),
        needs_decode=False,
    )
    work_items = [
        ExecutionWorkItem(
            entity=ExecutionEntity(
                entity_id=f"{context.state_handle_id}:transition:{context.step_idx}",
                rollout_id=context.state_handle_id,
                stage="stateless_transition",
                step_idx=context.step_idx,
                batch_signature=signature,
            ),
            latent_item=context.state,
            action_item=action_tensor[index:index + 1],
        )
        for index, context in enumerate(contexts)
    ]
    chunks, _ = HomogeneousChunkScheduler().schedule(
        work_items=work_items,
        policy=policy,
        chunk_id_prefix="stateless_transition",
        latent_join=lambda items: torch.cat(items, dim=0),
        action_join=lambda items: torch.cat(items, dim=0),
    )
    return chunks


__all__ = ["StatelessTransitionContext", "build_stateless_step_chunks"]
