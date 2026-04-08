"""Explicit stage pipeline for learned env transitions.

The pipeline is intentionally split into separate stage objects so future
integration can evolve independently at each boundary:

- encode/materialize
- transition
- reward
- persist
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from wm_infra.execution import ExecutionBatchPolicy, ExecutionChunk, chunk_fill_ratio, summarize_execution_chunks
from wm_infra.env_runtime.transition import StatelessTransitionContext, build_stateless_step_chunks


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
class TransitionMaterializedChunk:
    chunk: ExecutionChunk
    contexts: list[StatelessTransitionContext]
    goal_batch: torch.Tensor
    materialize_ms: float = 0.0


@dataclass(slots=True)
class TransitionMaterializationBatch:
    chunks: list["TransitionMaterializedChunk"]
    chunk_summary: dict[str, Any]
    chunk_history: list[dict[str, Any]]
    stage_profile: TransitionStageProfile
    materialize_ms: float = 0.0


@dataclass(slots=True)
class TransitionExecutionStage:
    prepared: TransitionMaterializedChunk
    next_states: torch.Tensor
    transition_ms: float = 0.0
    auxiliary_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransitionRewardStage:
    execution: TransitionExecutionStage
    rewards: torch.Tensor
    terminated: torch.Tensor
    info_tensors: dict[str, torch.Tensor]
    reward_ms: float = 0.0

    @property
    def prepared(self) -> TransitionMaterializedChunk:
        return self.execution.prepared

    @property
    def next_states(self) -> torch.Tensor:
        return self.execution.next_states

    @property
    def transition_ms(self) -> float:
        return self.execution.transition_ms


@dataclass(slots=True)
class TransitionPersistIntent:
    state_handle_id: str
    episode_id: str
    branch_id: str
    trajectory_id: str
    task_id: str
    env_name: str
    policy_version: str | None
    step_idx: int
    next_step_idx: int
    checkpoint_requested: bool
    checkpoint_tag: str | None
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    next_state: torch.Tensor
    goal: torch.Tensor
    info: dict[str, Any]
    chunk_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TransitionPersistStage:
    reward_stage: TransitionRewardStage
    intents: list[TransitionPersistIntent]
    checkpoint_requested: bool
    metadata: dict[str, Any]
    persist_ms: float = 0.0


@dataclass(slots=True)
class TransitionPipelineRun:
    materialization: TransitionMaterializedChunk
    execution: TransitionExecutionStage
    reward: TransitionRewardStage
    persist: TransitionPersistStage


@dataclass(slots=True)
class TransitionPipelineBatch:
    chunks: list[TransitionMaterializedChunk]
    chunk_summary: dict[str, Any]
    chunk_history: list[dict[str, Any]]
    stage_profile: TransitionStageProfile
    materialize_ms: float = 0.0


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
    ) -> TransitionPipelineBatch:
        profile = TransitionStageProfile()
        started_at = time.perf_counter()
        step_chunks = build_stateless_step_chunks(
            contexts,
            action_tensor,
            dtype=self.dtype,
            device=self.device,
            policy=self.policy,
        )
        materialize_ms = (time.perf_counter() - started_at) * 1000.0
        profile.record("materialize", materialize_ms)
        context_by_id = {context.state_handle_id: context for context in contexts}
        prepared_chunks: list[TransitionMaterializedChunk] = []
        chunk_history: list[dict[str, Any]] = []
        for chunk in step_chunks:
            chunk_contexts = [context_by_id[entity.rollout_id] for entity in chunk.entities]
            prepared_chunks.append(
                TransitionMaterializedChunk(
                    chunk=chunk,
                    contexts=chunk_contexts,
                    goal_batch=torch.cat([context.goal for context in chunk_contexts], dim=0),
                    materialize_ms=materialize_ms,
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
        return TransitionPipelineBatch(
            chunks=prepared_chunks,
            chunk_summary=summarize_execution_chunks(step_chunks, policy=self.policy),
            chunk_history=chunk_history,
            stage_profile=profile,
            materialize_ms=materialize_ms,
        )

    def materialize(
        self,
        *,
        contexts: list[StatelessTransitionContext],
        action_tensor: torch.Tensor,
    ) -> TransitionPipelineBatch:
        return self.prepare(contexts=contexts, action_tensor=action_tensor)

    def transition(
        self,
        prepared: TransitionMaterializedChunk,
        *,
        stage_profile: TransitionStageProfile | None = None,
    ) -> TransitionExecutionStage:
        profile = stage_profile or TransitionStageProfile()
        transition_started_at = time.perf_counter()
        next_states = self.world_model.predict_next(prepared.chunk.latent_batch, prepared.chunk.action_batch)
        transition_ms = (time.perf_counter() - transition_started_at) * 1000.0
        profile.record("transition", transition_ms)
        return TransitionExecutionStage(
            prepared=prepared,
            next_states=next_states,
            transition_ms=transition_ms,
        )

    def reward(
        self,
        execution: TransitionExecutionStage,
        *,
        stage_profile: TransitionStageProfile | None = None,
    ) -> TransitionRewardStage:
        profile = stage_profile or TransitionStageProfile()
        reward_started_at = time.perf_counter()
        rewards, terminated, info_tensors = self.reward_fn.evaluate(execution.next_states, execution.prepared.goal_batch)
        reward_ms = (time.perf_counter() - reward_started_at) * 1000.0
        profile.record("reward", reward_ms)
        return TransitionRewardStage(
            execution=execution,
            rewards=rewards,
            terminated=terminated,
            info_tensors=info_tensors,
            reward_ms=reward_ms,
        )

    def build_persist_plan(
        self,
        reward_stage: TransitionRewardStage,
        *,
        checkpoint: bool,
        metadata: dict[str, Any],
    ) -> TransitionPersistStage:
        started_at = time.perf_counter()
        intents: list[TransitionPersistIntent] = []
        for index, context in enumerate(reward_stage.prepared.contexts):
            next_step_idx = context.step_idx + 1
            truncated = next_step_idx >= context.max_episode_steps
            intents.append(
                TransitionPersistIntent(
                    state_handle_id=context.state_handle_id,
                    episode_id=context.episode_id,
                    branch_id=context.branch_id,
                    trajectory_id=context.trajectory_id,
                    task_id=context.task_id,
                    env_name=context.env_name,
                    policy_version=context.policy_version,
                    step_idx=context.step_idx,
                    next_step_idx=next_step_idx,
                    checkpoint_requested=bool(checkpoint or bool(reward_stage.terminated[index].item()) or truncated),
                    checkpoint_tag=f"step-{next_step_idx}",
                    action=reward_stage.prepared.chunk.action_batch[index].detach().cpu().tolist(),
                    reward=float(reward_stage.rewards[index].item()),
                    terminated=bool(reward_stage.terminated[index].item()),
                    truncated=truncated,
                    next_state=reward_stage.next_states[index : index + 1],
                    goal=context.goal,
                    info={
                        "step_idx": next_step_idx,
                        "env_name": context.env_name,
                        "task_id": context.task_id,
                        "trajectory_id": context.trajectory_id,
                        "policy_version": context.policy_version,
                        **metadata,
                    },
                    chunk_id=reward_stage.prepared.chunk.chunk_id,
                    metadata={
                        "scope_id": context.scope_id,
                        "state_handle_id": context.state_handle_id,
                        "chunk_size": reward_stage.prepared.chunk.size,
                        "batched": reward_stage.prepared.chunk.size > 1,
                        **metadata,
                    },
                )
            )
        return TransitionPersistStage(
            reward_stage=reward_stage,
            intents=intents,
            checkpoint_requested=bool(checkpoint or any(intent.checkpoint_requested for intent in intents)),
            metadata=dict(metadata),
            persist_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def run_chunk(
        self,
        prepared: TransitionMaterializedChunk,
        *,
        checkpoint: bool,
        metadata: dict[str, Any],
        stage_profile: TransitionStageProfile | None = None,
    ) -> TransitionPipelineRun:
        execution = self.transition(prepared, stage_profile=stage_profile)
        reward_stage = self.reward(execution, stage_profile=stage_profile)
        persist = self.build_persist_plan(reward_stage, checkpoint=checkpoint, metadata=metadata)
        return TransitionPipelineRun(
            materialization=prepared,
            execution=execution,
            reward=reward_stage,
            persist=persist,
        )

    def execute_chunk(
        self,
        prepared: TransitionMaterializedChunk,
        *,
        stage_profile: TransitionStageProfile,
    ) -> TransitionRewardStage:
        execution = self.transition(prepared, stage_profile=stage_profile)
        return self.reward(execution, stage_profile=stage_profile)


__all__ = [
    "StageTiming",
    "TransitionExecutionStage",
    "TransitionMaterializationBatch",
    "TransitionMaterializedChunk",
    "TransitionPersistIntent",
    "TransitionPersistStage",
    "TransitionPipelineBatch",
    "TransitionPipelineRun",
    "TransitionRewardStage",
    "TransitionStagePipeline",
    "TransitionStageProfile",
]
