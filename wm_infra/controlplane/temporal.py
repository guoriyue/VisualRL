"""First-class temporal control-plane entities and persistence.

This module gives wm-infra explicit, persisted world-model concepts instead of
forcing temporal state into loose sample metadata.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError


class TemporalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ARCHIVED = "archived"


class StateHandleKind(str, Enum):
    LATENT = "latent"
    VIDEO_LATENT = "video_latent"
    TOKEN_CACHE = "token_cache"
    FRAME = "frame"
    ACTION_TRACE = "action_trace"
    METADATA = "metadata"


class StateResidency(str, Enum):
    INLINE = "inline"
    CPU = "cpu"
    GPU = "gpu"
    DISK = "disk"


class ExecutionStateRef(BaseModel):
    residency: StateResidency = StateResidency.INLINE
    storage_backend: str = "state_handle_metadata"
    state_key: str = "latent_state"
    goal_key: str = "goal_state"
    step_key: str = "step_idx"
    device: Optional[str] = None
    bytes_estimate: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateLineageRef(BaseModel):
    env_name: Optional[str] = None
    task_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    step_idx: int = 0
    parent_state_handle_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRecord(BaseModel):
    episode_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: TemporalStatus = TemporalStatus.ACTIVE
    labels: dict[str, str] = Field(default_factory=dict)
    seed: Optional[int] = None
    initial_prompt: Optional[str] = None
    parent_episode_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchRecord(BaseModel):
    branch_id: str
    episode_id: str
    parent_branch_id: Optional[str] = None
    forked_from_rollout_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    name: str
    status: TemporalStatus = TemporalStatus.ACTIVE
    labels: dict[str, str] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateHandleRecord(BaseModel):
    state_handle_id: str
    episode_id: str
    branch_id: Optional[str] = None
    rollout_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    kind: StateHandleKind = StateHandleKind.LATENT
    uri: Optional[str] = None
    shape: list[int] = Field(default_factory=list)
    dtype: Optional[str] = None
    version: int = 1
    is_terminal: bool = False
    execution_state_ref: Optional[ExecutionStateRef] = None
    lineage_ref: Optional[StateLineageRef] = None
    created_at: float = Field(default_factory=time.time)
    artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutRecord(BaseModel):
    rollout_id: str
    episode_id: str
    branch_id: Optional[str] = None
    backend: str
    model: str
    status: TemporalStatus = TemporalStatus.PENDING
    sample_id: Optional[str] = None
    request_id: Optional[str] = None
    input_state_handle_id: Optional[str] = None
    output_state_handle_id: Optional[str] = None
    checkpoint_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    step_count: int = 0
    priority: float = 0.0
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    updated_at: float = Field(default_factory=time.time)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    episode_id: str
    rollout_id: Optional[str] = None
    branch_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_index: int = 0
    tag: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    seed: Optional[int] = None
    initial_prompt: Optional[str] = None
    parent_episode_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BranchCreate(BaseModel):
    episode_id: str
    parent_branch_id: Optional[str] = None
    forked_from_rollout_id: Optional[str] = None
    forked_from_checkpoint_id: Optional[str] = None
    name: str
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateHandleCreate(BaseModel):
    episode_id: str
    branch_id: Optional[str] = None
    rollout_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    kind: StateHandleKind = StateHandleKind.LATENT
    uri: Optional[str] = None
    shape: list[int] = Field(default_factory=list)
    dtype: Optional[str] = None
    version: int = 1
    is_terminal: bool = False
    execution_state_ref: Optional[ExecutionStateRef] = None
    lineage_ref: Optional[StateLineageRef] = None
    artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutCreate(BaseModel):
    episode_id: str
    branch_id: Optional[str] = None
    backend: str
    model: str
    sample_id: Optional[str] = None
    request_id: Optional[str] = None
    input_state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_count: int = 0
    priority: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointCreate(BaseModel):
    episode_id: str
    rollout_id: Optional[str] = None
    branch_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    artifact_ids: list[str] = Field(default_factory=list)
    step_index: int = 0
    tag: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSpec(BaseModel):
    env_name: str
    backend: str
    observation_mode: str
    action_space: dict[str, Any] = Field(default_factory=dict)
    reward_schema: dict[str, Any] = Field(default_factory=dict)
    default_horizon: int = 1
    supports_batch_step: bool = False
    supports_fork: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    task_id: str
    env_name: str
    task_family: str
    goal_spec: dict[str, Any] = Field(default_factory=dict)
    seed_policy: str = "explicit"
    difficulty: str = "default"
    split: str = "train"
    reward_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSessionRecord(BaseModel):
    env_id: str
    env_name: str
    episode_id: str
    task_id: str
    backend: str
    status: TemporalStatus = TemporalStatus.ACTIVE
    current_step: int = 0
    state_handle_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    branch_id: Optional[str] = None
    policy_version: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionRecord(BaseModel):
    transition_id: str
    env_id: str
    episode_id: str
    trajectory_id: str
    task_id: str
    step_idx: int
    observation_ref: str
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    next_observation_ref: str
    info: dict[str, Any] = Field(default_factory=dict)
    policy_version: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    trajectory_id: str
    env_id: str
    episode_id: str
    task_id: str
    policy_version: Optional[str] = None
    status: TemporalStatus = TemporalStatus.ACTIVE
    num_steps: int = 0
    return_value: float = 0.0
    success: bool = False
    transition_refs: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunRecord(BaseModel):
    eval_run_id: str
    policy_version: str
    task_split: str
    status: TemporalStatus = TemporalStatus.PENDING
    trajectory_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayShardManifest(BaseModel):
    replay_shard_id: str
    policy_version: str
    task_split: str
    trajectory_ids: list[str] = Field(default_factory=list)
    transition_ids: list[str] = Field(default_factory=list)
    num_trajectories: int = 0
    num_transitions: int = 0
    uri: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSessionCreate(BaseModel):
    env_name: str
    episode_id: str
    task_id: str
    backend: str
    current_step: int = 0
    state_handle_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    branch_id: Optional[str] = None
    policy_version: Optional[str] = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionCreate(BaseModel):
    env_id: str
    episode_id: str
    trajectory_id: str
    task_id: str
    step_idx: int
    observation_ref: str
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    next_observation_ref: str
    info: dict[str, Any] = Field(default_factory=dict)
    policy_version: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryCreate(BaseModel):
    env_id: str
    episode_id: str
    task_id: str
    policy_version: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunCreate(BaseModel):
    policy_version: str
    task_split: str
    trajectory_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayShardCreate(BaseModel):
    policy_version: str
    task_split: str
    trajectory_ids: list[str] = Field(default_factory=list)
    transition_ids: list[str] = Field(default_factory=list)
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


class _EntityStore(Generic[T]):
    def __init__(self, root: Path, bucket: str, model_cls: type[T], id_field: str) -> None:
        self.root = root / bucket
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_cls = model_cls
        self.id_field = id_field

    def _path(self, entity_id: str) -> Path:
        return self.root / f"{entity_id}.json"

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _read_record(self, path: Path) -> T | None:
        try:
            payload = path.read_text(encoding="utf-8")
            return self.model_cls.model_validate_json(payload)
        except (OSError, ValidationError, json.JSONDecodeError):
            return None

    def put(self, record: T) -> T:
        self._atomic_write_text(
            self._path(getattr(record, self.id_field)),
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True),
        )
        return record

    def get(self, entity_id: str) -> T | None:
        path = self._path(entity_id)
        if not path.exists():
            return None
        return self._read_record(path)

    def list(self) -> list[T]:
        records: list[T] = []
        for path in sorted(self.root.glob("*.json")):
            record = self._read_record(path)
            if record is not None:
                records.append(record)
        return records


class TemporalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes = _EntityStore(self.root, "episodes", EpisodeRecord, "episode_id")
        self.rollouts = _EntityStore(self.root, "rollouts", RolloutRecord, "rollout_id")
        self.state_handles = _EntityStore(self.root, "state_handles", StateHandleRecord, "state_handle_id")
        self.branches = _EntityStore(self.root, "branches", BranchRecord, "branch_id")
        self.checkpoints = _EntityStore(self.root, "checkpoints", CheckpointRecord, "checkpoint_id")
        self.environment_specs = _EntityStore(self.root, "environment_specs", EnvironmentSpec, "env_name")
        self.task_specs = _EntityStore(self.root, "task_specs", TaskSpec, "task_id")
        self.environment_sessions = _EntityStore(self.root, "environment_sessions", EnvironmentSessionRecord, "env_id")
        self.transitions = _EntityStore(self.root, "transitions", TransitionRecord, "transition_id")
        self.trajectories = _EntityStore(self.root, "trajectories", TrajectoryRecord, "trajectory_id")
        self.evaluation_runs = _EntityStore(self.root, "evaluation_runs", EvaluationRunRecord, "eval_run_id")
        self.replay_shards = _EntityStore(self.root, "replay_shards", ReplayShardManifest, "replay_shard_id")

    def create_episode(self, request: EpisodeCreate) -> EpisodeRecord:
        now = time.time()
        record = EpisodeRecord(episode_id=str(uuid.uuid4()), updated_at=now, created_at=now, **request.model_dump())
        return self.episodes.put(record)

    def create_branch(self, request: BranchCreate) -> BranchRecord:
        now = time.time()
        record = BranchRecord(branch_id=str(uuid.uuid4()), updated_at=now, created_at=now, **request.model_dump())
        return self.branches.put(record)

    def create_state_handle(self, request: StateHandleCreate) -> StateHandleRecord:
        record = StateHandleRecord(state_handle_id=str(uuid.uuid4()), **request.model_dump())
        return self.state_handles.put(record)

    def create_rollout(self, request: RolloutCreate, *, status: TemporalStatus = TemporalStatus.PENDING) -> RolloutRecord:
        now = time.time()
        record = RolloutRecord(
            rollout_id=str(uuid.uuid4()),
            status=status,
            created_at=now,
            updated_at=now,
            started_at=now if status == TemporalStatus.ACTIVE else None,
            **request.model_dump(),
        )
        return self.rollouts.put(record)

    def update_rollout(self, rollout: RolloutRecord) -> RolloutRecord:
        rollout.updated_at = time.time()
        return self.rollouts.put(rollout)

    def create_checkpoint(self, request: CheckpointCreate) -> CheckpointRecord:
        record = CheckpointRecord(checkpoint_id=str(uuid.uuid4()), **request.model_dump())
        return self.checkpoints.put(record)

    def attach_checkpoint_to_rollout(self, rollout_id: str, checkpoint_id: str) -> RolloutRecord | None:
        rollout = self.rollouts.get(rollout_id)
        if rollout is None:
            return None
        if checkpoint_id not in rollout.checkpoint_ids:
            rollout.checkpoint_ids.append(checkpoint_id)
        return self.update_rollout(rollout)

    def upsert_environment_spec(self, spec: EnvironmentSpec) -> EnvironmentSpec:
        return self.environment_specs.put(spec)

    def upsert_task_spec(self, task: TaskSpec) -> TaskSpec:
        return self.task_specs.put(task)

    def create_environment_session(
        self,
        request: EnvironmentSessionCreate,
        *,
        status: TemporalStatus = TemporalStatus.ACTIVE,
    ) -> EnvironmentSessionRecord:
        now = time.time()
        record = EnvironmentSessionRecord(
            env_id=str(uuid.uuid4()),
            status=status,
            created_at=now,
            updated_at=now,
            **request.model_dump(),
        )
        return self.environment_sessions.put(record)

    def update_environment_session(self, session: EnvironmentSessionRecord) -> EnvironmentSessionRecord:
        session.updated_at = time.time()
        return self.environment_sessions.put(session)

    def create_transition(self, request: TransitionCreate) -> TransitionRecord:
        record = TransitionRecord(transition_id=str(uuid.uuid4()), **request.model_dump())
        return self.transitions.put(record)

    def create_trajectory(
        self,
        request: TrajectoryCreate,
        *,
        status: TemporalStatus = TemporalStatus.ACTIVE,
    ) -> TrajectoryRecord:
        now = time.time()
        record = TrajectoryRecord(
            trajectory_id=str(uuid.uuid4()),
            status=status,
            created_at=now,
            updated_at=now,
            **request.model_dump(),
        )
        return self.trajectories.put(record)

    def update_trajectory(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        trajectory.updated_at = time.time()
        return self.trajectories.put(trajectory)

    def create_evaluation_run(
        self,
        request: EvaluationRunCreate,
        *,
        status: TemporalStatus = TemporalStatus.PENDING,
    ) -> EvaluationRunRecord:
        now = time.time()
        record = EvaluationRunRecord(
            eval_run_id=str(uuid.uuid4()),
            status=status,
            created_at=now,
            updated_at=now,
            **request.model_dump(),
        )
        return self.evaluation_runs.put(record)

    def update_evaluation_run(self, evaluation_run: EvaluationRunRecord) -> EvaluationRunRecord:
        evaluation_run.updated_at = time.time()
        return self.evaluation_runs.put(evaluation_run)

    def create_replay_shard(self, request: ReplayShardCreate) -> ReplayShardManifest:
        now = time.time()
        record = ReplayShardManifest(
            replay_shard_id=str(uuid.uuid4()),
            created_at=now,
            updated_at=now,
            num_trajectories=len(request.trajectory_ids),
            num_transitions=len(request.transition_ids),
            **request.model_dump(),
        )
        return self.replay_shards.put(record)

    def update_replay_shard(self, replay_shard: ReplayShardManifest) -> ReplayShardManifest:
        replay_shard.updated_at = time.time()
        replay_shard.num_trajectories = len(replay_shard.trajectory_ids)
        replay_shard.num_transitions = len(replay_shard.transition_ids)
        return self.replay_shards.put(replay_shard)
