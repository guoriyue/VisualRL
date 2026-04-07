"""Control-plane schemas and interfaces for sample production.

This package deliberately sits above the runtime layer.
The runtime executes model workloads.
The control plane tracks what was requested, what was produced,
and how outputs can flow into evaluation and training.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .schemas import (
    ArtifactKind,
    ArtifactRecord,
    CosmosTaskConfig,
    CosmosVariant,
    EvaluationRecord,
    EvaluationStatus,
    ExperimentRef,
    FailureTag,
    GenieTaskConfig,
    ProduceSampleRequest,
    ResourceEstimate,
    RolloutTaskConfig,
    SampleRecord,
    SampleSpec,
    SampleStatus,
    TaskType,
    TemporalRefs,
    TokenInputSource,
    TokenInputSpec,
    TokenizerFamily,
    TokenizerKind,
    TrainingExportRecord,
    VideoMemoryProfile,
    WanTaskConfig,
)
from .storage import SampleManifestStore
from .temporal import (
    BranchCreate,
    BranchRecord,
    CheckpointCreate,
    CheckpointRecord,
    ExecutionStateRef,
    EnvironmentSessionCreate,
    EnvironmentSessionRecord,
    EnvironmentSpec,
    EvaluationRunCreate,
    EvaluationRunRecord,
    EpisodeCreate,
    EpisodeRecord,
    ReplayShardCreate,
    ReplayShardManifest,
    RolloutCreate,
    RolloutRecord,
    StateHandleCreate,
    StateHandleKind,
    StateHandleRecord,
    StateLineageRef,
    StateResidency,
    TaskSpec,
    TemporalStatus,
    TemporalStore,
    TrajectoryCreate,
    TrajectoryRecord,
    TransitionCreate,
    TransitionRecord,
)

if TYPE_CHECKING:
    from .resource_estimator import estimate_cosmos_request, estimate_rollout_request, estimate_wan_request

_RESOURCE_ESTIMATOR_EXPORTS = {"estimate_cosmos_request", "estimate_rollout_request", "estimate_wan_request"}


def __getattr__(name: str):
    if name in _RESOURCE_ESTIMATOR_EXPORTS:
        module = import_module(".resource_estimator", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "BranchCreate",
    "BranchRecord",
    "CheckpointCreate",
    "CheckpointRecord",
    "CosmosTaskConfig",
    "CosmosVariant",
    "ExecutionStateRef",
    "EnvironmentSessionCreate",
    "EnvironmentSessionRecord",
    "EnvironmentSpec",
    "EvaluationRunCreate",
    "EvaluationRunRecord",
    "EpisodeCreate",
    "EpisodeRecord",
    "EvaluationRecord",
    "EvaluationStatus",
    "ExperimentRef",
    "FailureTag",
    "GenieTaskConfig",
    "ProduceSampleRequest",
    "ReplayShardCreate",
    "ReplayShardManifest",
    "ResourceEstimate",
    "RolloutCreate",
    "RolloutRecord",
    "RolloutTaskConfig",
    "SampleRecord",
    "SampleManifestStore",
    "SampleSpec",
    "SampleStatus",
    "StateHandleCreate",
    "StateHandleKind",
    "StateHandleRecord",
    "StateLineageRef",
    "StateResidency",
    "TaskSpec",
    "TaskType",
    "TemporalRefs",
    "TemporalStatus",
    "TemporalStore",
    "TokenInputSource",
    "TokenInputSpec",
    "TokenizerFamily",
    "TokenizerKind",
    "TrainingExportRecord",
    "TrajectoryCreate",
    "TrajectoryRecord",
    "TransitionCreate",
    "TransitionRecord",
    "VideoMemoryProfile",
    "WanTaskConfig",
    "estimate_cosmos_request",
    "estimate_rollout_request",
    "estimate_wan_request",
]
