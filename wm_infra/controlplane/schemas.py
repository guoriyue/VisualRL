"""Schema-first control-plane models for video data production.

These models are intentionally higher level than the low-level rollout API.
They define the entities needed to turn inference into reproducible sample
production, evaluation, and export.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# VideoMemoryProfile and RolloutTaskConfig are owned by the engine layer
# and re-exported here for backward compatibility.
from wm_infra.engine._types import RolloutTaskConfig, VideoMemoryProfile  # noqa: F401


class TaskType(str, Enum):
    """Product-level task semantics for persisted sample requests.

    Backend identifiers such as ``matrix-game`` belong in the separate
    ``backend`` field, not in this enum.
    """

    TEMPORAL_ROLLOUT = "temporal_rollout"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"


class WorldModelKind(str, Enum):
    """Internal world-model family classification for runtime and control-plane routing."""

    DYNAMICS = "dynamics"
    GENERATION = "generation"


class SampleStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    AUTO_PASSED = "auto_passed"
    AUTO_FAILED = "auto_failed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_ACCEPTED = "human_accepted"
    HUMAN_REJECTED = "human_rejected"


class ArtifactKind(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    LATENT = "latent"
    THUMBNAIL = "thumbnail"
    METADATA = "metadata"
    LOG = "log"
    EMBEDDING = "embedding"
    CHECKPOINT = "checkpoint"


class CosmosVariant(str, Enum):
    """Cosmos world-generation family variants."""

    PREDICT1_TEXT2WORLD = "predict1_text2world"
    PREDICT1_VIDEO2WORLD = "predict1_video2world"
    PREDICT2_VIDEO2WORLD = "predict2_video2world"


class FailureTag(str, Enum):
    IDENTITY_DRIFT = "identity_drift"
    TEMPORAL_FLICKER = "temporal_flicker"
    PROMPT_MISMATCH = "prompt_mismatch"
    CAMERA_INSTABILITY = "camera_instability"
    UNSAFE_CONTENT = "unsafe_content"
    DECODE_FAILURE = "decode_failure"
    LOW_MOTION_QUALITY = "low_motion_quality"
    BROKEN_PHYSICS = "broken_physics"
    UNKNOWN = "unknown"


class ExperimentRef(BaseModel):
    experiment_id: str = Field(..., description="Stable experiment identifier")
    run_id: Optional[str] = Field(default=None, description="Specific run within experiment")
    tags: list[str] = Field(default_factory=list)


class TemporalRefs(BaseModel):
    episode_id: Optional[str] = None
    rollout_id: Optional[str] = None
    branch_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    state_handle_id: Optional[str] = None
    parent_state_handle_id: Optional[str] = None


class TokenInputSource(str, Enum):
    INLINE = "inline"
    URI = "uri"


class TokenizerFamily(str, Enum):
    RAW = "raw"
    MAGVIT2 = "magvit2"


class TokenInputSpec(BaseModel):
    source: TokenInputSource = TokenInputSource.INLINE
    tokenizer_family: TokenizerFamily = TokenizerFamily.RAW
    layout: Literal["thw", "flat"] = "thw"
    uri: Optional[str] = None
    inline_tokens: list[int] = Field(default_factory=list)
    shape: list[int] = Field(default_factory=list, description="Expected token tensor shape, e.g. [T, H, W]")
    dtype: str = Field(default="uint32")
    token_count: Optional[int] = Field(default=None, ge=0)
    tokenizer_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WanTaskConfig(BaseModel):
    """Wan 2.2 video generation config for `wan-video`.

    The current concrete model identifiers used by the backend are
    `wan2.2-t2v-A14B` and `wan2.2-i2v-A14B`.
    """

    num_steps: int = Field(default=4, ge=1, description="Number of denoising steps (sample_steps)")
    frame_count: int = Field(default=9, ge=1, description="Target frame count (frame_num)")
    width: int = Field(default=832, ge=1, description="Output width")
    height: int = Field(default=480, ge=1, description="Output height")
    guidance_scale: float = Field(default=4.0, ge=0, description="Classifier-free guidance scale (sample_guide_scale)")
    high_noise_guidance_scale: Optional[float] = Field(
        default=None,
        ge=0,
        description="Optional high-noise guidance scale override for official Wan I2V",
    )
    shift: float = Field(default=12.0, ge=0, description="Noise schedule shift (sample_shift)")
    sample_solver: str = Field(default="unipc", description="Sampling solver (sample_solver)")
    offload_model: bool = Field(default=True, description="Offload model weights to CPU between stages")
    convert_model_dtype: bool = Field(default=True, description="Enable reduced-precision model conversion")
    t5_cpu: bool = Field(default=True, description="Keep T5 text encoder on CPU")
    memory_profile: VideoMemoryProfile = Field(default=VideoMemoryProfile.LOW_VRAM, description="Coarse memory/quality mode")
    model_size: str = Field(default="A14B", description="Wan 2.2 model-size suffix used in model ids such as wan2.2-t2v-A14B")
    ckpt_dir: Optional[str] = Field(default=None, description="Path to the Wan 2.2 checkpoint directory")


class CosmosTaskConfig(BaseModel):
    """Cosmos-family world generation config for NIM or local shell runners."""

    variant: CosmosVariant = Field(
        default=CosmosVariant.PREDICT1_VIDEO2WORLD,
        description="Cosmos pipeline variant to execute",
    )
    model_size: str = Field(default="7B", description="Model size or variant label, e.g. 2B, 7B, 14B")
    guidance_scale: float = Field(default=7.0, ge=0, description="Guidance scale when supported by the runner")
    checkpoint_every_n_frames: int = Field(default=0, ge=0, description="Reserved for future chunked continuation scheduling")
    frames_per_second: int = Field(default=16, ge=1, description="Output video FPS")
    negative_prompt: Optional[str] = Field(default=None, description="Optional backend-specific negative prompt override")
    seed: Optional[int] = Field(default=None, description="Optional backend-specific seed override")


class ResourceEstimate(BaseModel):
    estimated_units: float = Field(..., ge=0, description="Relative resource score for admission/scheduling")
    estimated_vram_gb: Optional[float] = Field(default=None, ge=0)
    bottleneck: str = Field(default="unknown")
    notes: list[str] = Field(default_factory=list)


class SampleSpec(BaseModel):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    fps: Optional[int] = Field(default=None, ge=1)
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None
    references: list[str] = Field(default_factory=list, description="URIs or asset IDs")
    controls: dict[str, Any] = Field(default_factory=dict, description="Camera/motion/control inputs")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    artifact_id: str
    kind: ArtifactKind
    uri: str
    mime_type: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    evaluator: str = Field(..., description="Name of auto scorer or human review queue")
    status: EvaluationStatus = EvaluationStatus.PENDING
    score: Optional[float] = None
    failure_tags: list[FailureTag] = Field(default_factory=list)
    notes: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingExportRecord(BaseModel):
    export_id: str
    export_format: str = Field(..., description="pairwise_ranking | scorer_training | lora_finetune_manifest")
    dataset_name: Optional[str] = None
    split: Optional[str] = None
    uri: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProduceSampleRequest(BaseModel):
    task_type: TaskType
    backend: str = Field(..., description="Backend/runtime identifier")
    model: str = Field(..., description="Logical model identifier")
    world_model_kind: Optional[WorldModelKind] = Field(
        default=None,
        description=(
            "Optional client-declared world-model family. "
            "`dynamics` is for state/action-conditioned rollout systems such as Matrix/Genie-style models; "
            "`generation` is for world/video generation systems such as Cosmos-style models."
        ),
    )
    model_revision: Optional[str] = None
    experiment: Optional[ExperimentRef] = None
    sample_spec: SampleSpec
    temporal: Optional[TemporalRefs] = None
    token_input: Optional[TokenInputSpec] = Field(
        default=None,
        description=(
            "Optional tokenized-video input scaffold for temporal backends. Supports direct raw token input now "
            "(inline or .npy URI) and reserves tokenizer_family='magvit2' for forward-compatible tokenized inputs."
        ),
    )
    task_config: Optional[RolloutTaskConfig] = Field(
        default=None,
        description=(
            "Task-specific execution config for dynamics backends such as rollout-engine or matrix-game. "
            "Video-relevant execution knobs such as num_steps, frame_count, width, height, and memory/offload mode belong here."
        ),
    )
    wan_config: Optional[WanTaskConfig] = Field(
        default=None,
        description=(
            "Wan 2.2 video generation config. Used when backend is the wan-video runtime "
            "(text_to_video, image_to_video). Ignored for rollout-engine jobs."
        ),
    )
    cosmos_config: Optional[CosmosTaskConfig] = Field(
        default=None,
        description=(
            "Cosmos-family world generation config. Used when backend is cosmos-predict. "
            "Controls the Cosmos variant, FPS, guidance, and runner-specific options."
        ),
    )
    return_artifacts: list[ArtifactKind] = Field(default_factory=lambda: [ArtifactKind.VIDEO])
    evaluation_policy: Optional[str] = Field(default=None, description="Policy name for auto-QC / review")
    priority: float = 0.0
    labels: dict[str, str] = Field(default_factory=dict)


class SampleRecord(BaseModel):
    sample_id: str
    task_type: TaskType
    backend: str
    model: str
    world_model_kind: Optional[WorldModelKind] = None
    model_revision: Optional[str] = None
    status: SampleStatus = SampleStatus.QUEUED
    experiment: Optional[ExperimentRef] = None
    sample_spec: SampleSpec
    temporal: Optional[TemporalRefs] = None
    token_input: Optional[TokenInputSpec] = None
    task_config: Optional[RolloutTaskConfig] = None
    wan_config: Optional[WanTaskConfig] = None
    cosmos_config: Optional[CosmosTaskConfig] = None
    resource_estimate: Optional[ResourceEstimate] = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    exports: list[TrainingExportRecord] = Field(default_factory=list)
    lineage_parent_ids: list[str] = Field(default_factory=list)
    runtime: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
