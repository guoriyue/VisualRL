"""Explicit stage pipeline for learned env transitions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from wm_infra.runtime.execution import ExecutionBatchPolicy, ExecutionChunk, chunk_fill_ratio, summarize_execution_chunks
from wm_infra.runtime.env.transition import StatelessTransitionContext, build_stateless_step_chunks


@dataclass(slots=True)
class StageTiming:
    count: int = 0
    total_ms: float = 0.0

    def record(self, duration_ms: float) -> None:
        self.count += 1
        self.total_ms += duration_ms


@dataclass(slots=True)
class TransitionStageProfile:
    stages: dict[str, StageTiming] = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        self.stages.setdefault(stage, StageTiming()).record(duration_ms)

    def snapshot(self) -> dict[str, Any]:
        return {
            "stages": {
                name: {"count": timing.count, "total_ms": timing.total_ms}
                for name, timing in self.stages.items()
            }
        }


@dataclass(slots=True)
class PreparedTransitionChunk:
    chunk: ExecutionChunk
    contexts: list[StatelessTransitionContext]
    goal_batch: torch.Tensor


@dataclass(slots=True)
class ExecutedTransitionChunk:
    prepared: PreparedTransitionChunk
    next_states: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    info_tensors: dict[str, torch.Tensor]


@dataclass(slots=True)
class PreparedTransitionBatch:
    chunks: list[PreparedTransitionChunk]
    chunk_summary: dict[str, Any]
    chunk_history: list[dict[str, Any]]
    stage_profile: TransitionStageProfile


class TransitionStagePipeline:
    """Explicit transition pipeline with stage-local timing and boundaries."""

    def __init__(
        self,
        *,
        world_model: Any,
        reward_fn: Any,
        dtype: torch.dtype,
        device: torch.device,
        policy: ExecutionBatchPolicy,
    ) -> None:
        self.world_model = world_model
        self.reward_fn = reward_fn
        self.dtype = dtype
        self.device = device
        self.policy = policy

    def prepare(
        self,
        *,
        contexts: list[StatelessTransitionContext],
        action_tensor: torch.Tensor,
    ) -> PreparedTransitionBatch:
        profile = TransitionStageProfile()
        started_at = time.perf_counter()
        step_chunks = build_stateless_step_chunks(
            contexts,
            action_tensor,
            dtype=self.dtype,
            device=self.device,
            policy=self.policy,
        )
        profile.record("materialize", (time.perf_counter() - started_at) * 1000.0)
        context_by_id = {context.state_handle_id: context for context in contexts}
        prepared_chunks: list[PreparedTransitionChunk] = []
        chunk_history: list[dict[str, Any]] = []
        for chunk in step_chunks:
            chunk_contexts = [context_by_id[entity.rollout_id] for entity in chunk.entities]
            prepared_chunks.append(
                PreparedTransitionChunk(
                    chunk=chunk,
                    contexts=chunk_contexts,
                    goal_batch=torch.cat([context.goal for context in chunk_contexts], dim=0),
                )
            )
            chunk_history.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_size": chunk.size,
                    "signature": asdict(chunk.signature),
                    "fill_ratio": chunk_fill_ratio(chunk.size, self.policy.max_chunk_size),
                }
            )
        return PreparedTransitionBatch(
            chunks=prepared_chunks,
            chunk_summary=summarize_execution_chunks(step_chunks, policy=self.policy),
            chunk_history=chunk_history,
            stage_profile=profile,
        )

    def execute_chunk(self, prepared: PreparedTransitionChunk, *, stage_profile: TransitionStageProfile) -> ExecutedTransitionChunk:
        transition_started_at = time.perf_counter()
        next_states = self.world_model.predict_next(prepared.chunk.latent_batch, prepared.chunk.action_batch)
        stage_profile.record("transition", (time.perf_counter() - transition_started_at) * 1000.0)

        reward_started_at = time.perf_counter()
        rewards, terminated, info_tensors = self.reward_fn.evaluate(next_states, prepared.goal_batch)
        stage_profile.record("reward", (time.perf_counter() - reward_started_at) * 1000.0)
        return ExecutedTransitionChunk(
            prepared=prepared,
            next_states=next_states,
            rewards=rewards,
            terminated=terminated,
            info_tensors=info_tensors,
        )


__all__ = [
    "ExecutedTransitionChunk",
    "PreparedTransitionBatch",
    "PreparedTransitionChunk",
    "StageTiming",
    "TransitionStagePipeline",
    "TransitionStageProfile",
]
