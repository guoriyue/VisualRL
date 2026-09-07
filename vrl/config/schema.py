"""Pydantic typed boundary for merged training configs.

OmegaConf handles YAML defaults, interpolation, and CLI overrides.
Pydantic validates the fully-resolved, merged container after OmegaConf finishes.
Every section is a closed pydantic model, so an unknown YAML key — a typo, a
dead key, a removed legacy key — fails here with one error naming the dotted
path. ``parse_config`` is the one seam, so every entrypoint that parses a
config (training, eval, perf, encode tools) gets the same gate.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import (
    ConfigDict,
    Field,
    SerializeAsAny,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.config.algorithm import algorithm_config_class, resolve_kl_reward_coef
from vrl.config.base import ConfigBase, _extract_error_message
from vrl.config.data import DataLoaderName, manifest_sources, resolve_data_loader
from vrl.config.model_schema import ModelSection
from vrl.config.precision import PrecisionConfig
from vrl.config.reward_inference import (
    RewardInferenceConfig,
)
from vrl.config.sampling_schema import SamplingSection
from vrl.generation.execution.types import BatchPlacementStrategy
from vrl.models.families.names import normalize_model_family
from vrl.models.families.registry import get_model_family_entry
from vrl.ray.resources import (
    DistributedResourceConfig,
)
from vrl.trainers.core.types import (
    DebugConfig,
    EMAConfig,
    OptimConfig,
    PrecisionDriftGuardConfig,
    ReplayParityConfig,
    RolloutOrchestrationConfig,
)
from vrl.trainers.data.prompt_sampler import PromptSamplingStrategy
from vrl.trajectory.storage import TrajectoryStoragePolicy
from vrl.utils.config import import_from_path
from vrl.utils.profiling import TorchProfilerConfig

# ── Reward section ────────────────────────────────────────────────────────────


class RewardConfig(ConfigBase):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # reward names are user-chosen — open by design
    components: dict[str, Any]
    # each reward's kwargs contract is owned and validated by the reward class
    # itself at construction (vrl/rewards/), same as model families — the
    # config layer does not duplicate per-reward knowledge
    kwargs: dict[str, Any] = Field(default_factory=dict)
    # Per-component transport/deployment, keyed by the same user-chosen names.
    # A component without an entry executes in-process.
    inference: dict[str, RewardInferenceConfig] = Field(default_factory=dict)

    @field_validator("inference", mode="before")
    @classmethod
    def _parse_inference(cls, value: object) -> object:
        # RewardInferenceConfig is a frozen dataclass, not a pydantic model;
        # parse entries here so errors name the reward.inference.<name> path.
        if not isinstance(value, Mapping):
            return value
        return {
            str(name): RewardInferenceConfig.from_mapping(
                entry,
                context=f"reward.inference.{name}",
            )
            for name, entry in value.items()
        }

    @model_validator(mode="after")
    def _validate_reward(self) -> RewardConfig:
        # All kwargs entries must be mappings (or null)
        for name, sub in self.kwargs.items():
            if sub is not None and not isinstance(sub, dict):
                raise ValueError(
                    f"reward.kwargs.{name} must be a mapping, got {type(sub).__name__}",
                )

        unknown_inference = sorted(set(self.inference) - set(self.components))
        if unknown_inference:
            raise ValueError(
                f"reward.inference configured for unknown component(s): {unknown_inference}",
            )

        # Zero keeps a scorer observation-only: it is still computed and logged
        # but contributes nothing to the optimization reward.
        for name, weight_raw in self.components.items():
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"reward.components.{name} must be numeric, got {weight_raw!r}",
                ) from exc
            if weight < 0:
                raise ValueError(f"reward.components.{name} must be >= 0, got {weight}")
        return self

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> RewardConfig:
        """Build the reward section from ``cfg.reward``, rejecting malformed components."""
        if "reward" not in cfg:
            raise ValueError("config missing required field: reward")
        reward_raw = OmegaConf.to_container(cfg.reward, resolve=True, throw_on_missing=True) or {}
        try:
            return cls.model_validate(reward_raw)
        except ValidationError as exc:
            raise ValueError(_extract_error_message(exc, section="reward")) from exc


# ── Algorithm section ─────────────────────────────────────────────────────────


class AlgorithmConfig(ConfigBase):
    """Public ``algorithm`` section.

    ``kind`` selects the runtime hyper-parameter dataclass
    (``vrl.config.algorithm.algorithm_config_class``); every other YAML key is
    validated against that dataclass — unknown keys, missing required fields,
    its ``__post_init__`` — and the built instance lands in ``hyperparameters``.
    ``kl_reward_coef`` is collector-owned rather than an algorithm field, so it
    stays on the section.
    """

    kind: Literal[
        "grpo",
        "dance_grpo",
        "flash_grpo",
        "flow_dppo",
        "grpo_guard",
        "token_grpo",
        "token_grpo_multisegment",
        "diffusion_dpo",
        "diffusion_nft",
        "v_grpo",
    ]
    # Collector-owned diffusion reward-shaping coefficient. Token trajectories
    # do not carry the per-step KL tensor needed to consume a positive value.
    kl_reward_coef: float | None = None
    # The runtime dataclass selected by ``kind`` (e.g. GRPOConfig), built from
    # the remaining keys of this section. ``build_configs`` hands it to the
    # trainer as ``BuiltConfigs.algorithm``.
    hyperparameters: Any = None

    @field_validator("kl_reward_coef", mode="before")
    @classmethod
    def _validate_kl_reward_coef(cls, value: object | None) -> float | None:
        if value is None:
            return None
        return resolve_kl_reward_coef(value)

    @model_validator(mode="before")
    @classmethod
    def _select_hyperparameters(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        try:
            hyper_cls = algorithm_config_class(str(payload.get("kind")))
        except ValueError:
            return payload  # the ``kind`` Literal reports the authoritative error
        section = {key: payload[key] for key in ("kind", "kl_reward_coef") if key in payload}
        rest = {key: inner for key, inner in payload.items() if key not in section}
        prebuilt = rest.pop("hyperparameters", None)
        if isinstance(prebuilt, hyper_cls) and not rest:
            return {**section, "hyperparameters": prebuilt}
        if isinstance(prebuilt, Mapping):
            rest = {**dict(prebuilt), **rest}
        # A stdlib dataclass validated on its own ignores extra keys; the
        # dataclass's init fields are the complete vocabulary for this kind.
        unknown = sorted(set(rest) - {f.name for f in dataclass_fields(hyper_cls) if f.init})
        if unknown:
            raise ValueError("unknown " + ", ".join(f"algorithm.{key}" for key in unknown))
        try:
            hyperparameters = TypeAdapter(hyper_cls).validate_python(rest)
        except ValidationError as exc:
            raise ValueError(_extract_error_message(exc, section="algorithm")) from exc
        return {**section, "hyperparameters": hyperparameters}


# ── Data section ──────────────────────────────────────────────────────────────


class DataPreprocessingSection(ConfigBase):
    """``data.preprocessing``: loader-side sample shaping.

    readers: DataConfig._validate_data, vrl/trainers/data/prompts.py
    load_prompt_examples_from_config, the offline DPO entrypoint, and the
    i2v/v2w reference-image checks in the online recipe.
    """

    resolution: StrictInt | None = None
    random_crop: StrictBool | None = None
    horizontal_flip: StrictBool | None = None
    format: str | None = None
    image_field: str | None = None
    caption_field: str | None = None
    conditioning: str | None = None
    reference_image: str | None = None


class DataSamplerSection(ConfigBase):
    """``data.sampler``: prompt-batch sampling (online) or DataLoader knobs (offline)."""

    # PromptSamplingStrategy value; validated by DataConfig for the prompt loaders.
    type: str | None = None
    shuffle: StrictBool | None = None
    drop_last: StrictBool | None = None
    dataloader_num_workers: StrictInt | None = None


class DataConfig(ConfigBase):
    loader: DataLoaderName | None = None
    # A path, or a {manifest path: prompt count} mixture whose keys are file
    # paths chosen by the recipe (an open mapping). reader: manifest_sources.
    manifest: str | dict[str, Any] | None = None
    # Draw seed for a manifest mixture, required whenever one is declared: every
    # rank draws the mixture itself and then indexes into it.
    # reader: load_prompt_examples_from_config
    mix_seed: StrictInt | None = None
    eval_manifest: str | None = None
    preprocessing: DataPreprocessingSection | None = None
    sampler: DataSamplerSection | None = None
    dataset_name: str | None = None
    split: str | None = None
    cache_dir: str | None = None
    # Precomputed clean-latents shard for the GRPO diffusion-loss regularizer
    # (algorithm.sft_weight > 0): {target image/video -> VAE latents} written by
    # vrl/scripts/denoise/encode_targets.py. reader: run_online_recipe.
    sft_latents: str | None = None
    max_train_samples: StrictInt | None = None
    task_type: str | None = None
    # readers: data/eval tooling and the production Kling gate (validation.py).
    allow_absolute_artifact_paths: StrictBool | None = None
    artifact_data_root: str | None = None
    source_report: str | None = None

    @model_validator(mode="after")
    def _validate_data(self) -> DataConfig:
        preprocessing = self.preprocessing
        self.loader = resolve_data_loader(
            self.loader,
            None if preprocessing is None else preprocessing.format,
        )
        if self.loader == "prompt_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if len(manifest_sources(self.manifest)) > 1 and self.mix_seed is None:
                raise ValueError("config missing required field: data.mix_seed")
            if preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            self._validate_sampler_type()

        if self.loader == "prompt_image_manifest":
            if not self.manifest:
                raise ValueError("config missing required field: data.manifest")
            if not isinstance(self.manifest, str):
                raise ValueError(
                    "data.loader='prompt_image_manifest' takes a single data.manifest path, "
                    "not a mixture mapping",
                )
            if not self.eval_manifest:
                raise ValueError("config missing required field: data.eval_manifest")
            if preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("format", "image_field", "caption_field", "conditioning"):
                if getattr(preprocessing, field) is None:
                    raise ValueError(f"config missing required field: data.preprocessing.{field}")
            self._validate_sampler_type()

        if self.loader == "pickapic_preference":
            for field in ("dataset_name", "split", "cache_dir"):
                # Allow empty strings (only None/absent is invalid)
                if getattr(self, field) is None:
                    raise ValueError(f"config missing required field: data.{field}")
            if preprocessing is None:
                raise ValueError("config missing required field: data.preprocessing")
            for field in ("resolution", "random_crop", "horizontal_flip"):
                if getattr(preprocessing, field) is None:
                    raise ValueError(f"config missing required field: data.preprocessing.{field}")
            sampler = self.sampler
            for field in ("shuffle", "drop_last", "dataloader_num_workers"):
                if sampler is None or getattr(sampler, field) is None:
                    raise ValueError(f"config missing required field: data.sampler.{field}")

        return self

    def _validate_sampler_type(self) -> None:
        """Shared sampler.type check for the prompt-manifest loaders."""
        sampler_type = "" if self.sampler is None else str(self.sampler.type or "")
        if not sampler_type:
            raise ValueError("config missing required field: data.sampler.type")
        try:
            PromptSamplingStrategy(sampler_type)
        except ValueError as exc:
            expected = " / ".join(strategy.value for strategy in PromptSamplingStrategy)
            raise ValueError(
                f"unknown data.sampler.type={sampler_type!r}; expected {expected}",
            ) from exc


# ── Supporting sections for cross-field validation ────────────────────────────


class SdeConfig(ConfigBase):
    """Typed rollout.sde block. ``type`` is the user-facing allow-list, replacing
    hand-written membership checks previously duplicated in the
    schema cross-validator, layout, and flow_matching. The layout request-boundary
    guard stays for over-the-wire request dicts; ``window_*`` stay permissive.

    ``type`` names the replay/sampling transition distribution (flow_grpo, ddim,
    or cps); it is orthogonal to ``denoise_mode`` (native/sde), which owns the
    word ``sde``."""

    type: Literal["flow_grpo", "ddim", "cps"]
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py, which
    # owns the range/size checks against num_steps at the request boundary.
    window_size: StrictInt | None = None
    window_range: list[int] | None = None


class RolloutConfig(ConfigBase):
    # readers: vrl/math/denoise/flow_matching.py window + RootConfig check
    sde: SdeConfig | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    noise_level: float | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # janus_pro R1 only; the sole source for final_image_policy. Validated for
    # legality in RootConfig._cross_field_validate (which requires it for that kind).
    final_image_policy: Literal["always_generate", "use_selfcheck"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # Strict: a bool is not a batch dimension (OnlineBatchPlan rejects it too).
    n_samples_per_prompt: StrictInt | None = None
    prompts_per_batch: StrictInt | None = None
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py
    # _parse_denoise_mode (request boundary).
    # Allowed set is the type; the layout guard stays for over-the-wire request dicts.
    denoise_mode: Literal["native", "sde"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py — opt-in to storing
    # each denoise step's rollout proposal mean for trust-region replay.
    return_prev_sample_mean: StrictBool | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: vrl/generation/bindings/full_sequence_denoise/layout.py — opt-in to caching
    # the frozen reference (LoRA-disabled) noise_pred at collect, so KL replay never
    # reruns the ref forward. Lossless: replay applies the same sde_step_with_logprob.
    cache_ref_noise_pred: StrictBool | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    # reader: generation planner (batch_placement.py) + diffusion layout. int =
    # fixed batch size; "auto" = the Ray runtime's startup batch-size probe
    # resolves it before the first request (SPRINT_chunk_size_probe; Ray-only,
    # the planner rejects "auto" on other runtimes); null = samples_per_prompt.
    samples_per_generation_batch: int | Literal["auto"] | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )
    torch_profiler: TorchProfilerConfig | None = None
    trajectory_storage: TrajectoryStoragePolicy | None = Field(
        default=None,
        json_schema_extra={"runtime_owner": "generation_request"},
    )

    @field_validator("samples_per_generation_batch", mode="before")
    @classmethod
    def _validate_samples_per_generation_batch(cls, value: Any) -> Any:
        """Keep fixed generation batches positive; the runtime owns ``auto``."""

        if value is None or value == "auto":
            return value
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "rollout.samples_per_generation_batch must be a positive integer, 'auto', or null",
            )
        return value


def generation_request_rollout_fields() -> frozenset[str]:
    """Derive rollout keys allowed to cross the generation request boundary."""

    return frozenset(
        name
        for name, model_field in RolloutConfig.model_fields.items()
        if (model_field.json_schema_extra or {}).get("runtime_owner") == "generation_request"
    )


@functools.cache
def _model_section_class_from_path(path: str) -> type[ModelSection]:
    return import_from_path(path)


def _model_section_class_for_family(family: Any) -> type[ModelSection]:
    if family is None or not str(family).strip():
        raise ValueError("config missing required field: model.family")
    entry = get_model_family_entry(str(family))
    return _model_section_class_from_path(entry.model_section_cls)


def _parse_model_section(value: Any) -> ModelSection | None:
    if value is None:
        return None
    if isinstance(value, ModelSection):
        family = value.family
        section_cls = _model_section_class_for_family(family)
        if isinstance(value, section_cls):
            parsed = value
            payload = None
        else:
            payload = value.model_dump(exclude_unset=True)
    elif isinstance(value, Mapping):
        payload = dict(value)
        section_cls = _model_section_class_for_family(payload.get("family"))
    else:
        raise ValueError("model must be a mapping")

    if payload is not None:
        parsed = section_cls.revalidate(payload, section="model")

    entry = get_model_family_entry(str(parsed.family))
    entry.validate_model_runtime_sections(
        executor_config=parsed.executor,
        memory_config=parsed.memory,
    )
    return parsed


@functools.cache
def _sampling_section_class_from_path(path: str) -> type[SamplingSection]:
    return import_from_path(path)


def sampling_section_class_for_family(family: Any) -> type[SamplingSection]:
    if family is None or not str(family).strip():
        raise ValueError("sampling requires model.family")
    entry = get_model_family_entry(str(family))
    return _sampling_section_class_from_path(entry.sampling_section_cls)


def _parse_sampling_section(
    value: Any,
    *,
    model: ModelSection | None,
) -> SamplingSection | None:
    if value is None:
        return None
    section_cls = sampling_section_class_for_family(
        None if model is None else model.family,
    )
    if isinstance(value, SamplingSection):
        if isinstance(value, section_cls):
            return value
        payload = value.model_dump(exclude_unset=True)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ValueError("sampling must be a mapping")

    return section_cls.revalidate(payload, section="sampling")


# ── actor / trainer sections ──────────────────────────────────────────────────


class ActorSection(ConfigBase):
    """Public ``actor`` section: the online trainer's optimizer/loop knobs plus
    the offline-DPO entrypoint's inputs.

    Nested blocks are typed with the runtime dataclass that consumes them, so
    their keys, defaults, and range checks live once, on the consumer. Every
    scalar is optional at this boundary; requiredness is decided by the
    projection (``TrainerConfig.from_root``: a field without a default is
    required) because one public section feeds more than one runtime owner.
    """

    optim: OptimConfig | None = None
    ema: EMAConfig | None = None
    timestep_fraction: float | None = None
    drop_zero_advantage: StrictBool | None = None
    max_norm: float | None = None
    timestep_selection: Literal["strided", "random", "stratified", "sde_window"] | None = None
    ppo_epochs: StrictInt | None = None
    # OnlineBatchPlan inputs: the microstep count or the microbatch size (the
    # size derives the count); the plan is bridged, not stored here.
    gradient_accumulation_steps: StrictInt | None = None
    microbatch_size: StrictInt | None = None
    samples_per_replay_batch: StrictInt | None = None
    host_memory_budget_fraction: float | None = None
    # reader: vrl/trainers/activation_checkpointing.py (bool: true=full, false=off)
    gradient_checkpointing: Literal["off", "full", "selective"] | StrictBool | None = None
    # offline DPO entrypoint (vrl/scripts/families/wan_2_1/train_dpo.py)
    prediction_type: str | None = None
    scale_lr: StrictBool | None = None
    train_batch_size: StrictInt | None = None
    use_adafactor: StrictBool | None = None


class TrainerSection(ConfigBase):
    """Public ``trainer`` section: run lifecycle/IO plus trainer-level policies."""

    entrypoint: str | None = None
    output_dir: str | None = None
    total_epochs: StrictInt | None = None
    save_freq: StrictInt | None = None
    seed: StrictInt | None = None
    resume_from: str | None = None
    resume_strict: StrictBool | None = None
    profile: StrictBool | None = None
    debug: DebugConfig | None = None
    replay_parity: ReplayParityConfig | None = None
    precision_drift_guard: PrecisionDriftGuardConfig | None = None
    precision_correction: PrecisionCorrectionConfig | None = None
    rollout_orchestration: RolloutOrchestrationConfig | None = None
    torch_profiler: TorchProfilerConfig | None = None
    # offline DPO entrypoint (vrl/scripts/families/wan_2_1/train_dpo.py)
    checkpointing_steps: StrictInt | None = None
    log_interval: StrictInt | None = None
    max_train_steps: StrictInt | None = None


# Entrypoint-specific schema boundary. These are the only shared actor/trainer
# keys consumed by train_wan_2_1_dpo (plus trainer.entrypoint, consumed by the
# public dispatcher). Keeping the sets here makes inherited online-only knobs
# fail loudly before model/data construction.
_OFFLINE_DPO_ACTOR_FIELDS = frozenset(
    {
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "max_norm",
        "optim",
        "prediction_type",
        "scale_lr",
        "train_batch_size",
        "use_adafactor",
    },
)
_OFFLINE_DPO_TRAINER_FIELDS = frozenset(
    {
        "checkpointing_steps",
        "entrypoint",
        "log_interval",
        "max_train_steps",
        "output_dir",
        "resume_from",
        "resume_strict",
    },
)


class FSDPConfig(ConfigBase):
    """distributed.training.fsdp: the FSDP2 knobs ``build_strategy`` reads.

    Only the fields a reader consumes are declared (the same philosophy as
    ``TrainingSection``): ``vrl/trainers/strategy.py`` build_strategy +
    ``vrl/trainers/fsdp.py`` read ``mesh`` / ``precision_policy`` /
    ``reshard_after_forward`` / ``cpu_offload``. The remaining FSDP2 knobs from
    SPRINT_multi_gpu_training.md §3 (activation_checkpointing, state_dict,
    process-group backend/init) land here when their readers do —
    declaring an unread knob is a user-facing no-op footgun.
    """

    # 1D ZeRO-3 over the whole world; 2D HSDP is the multi-node follow-on.
    mesh: list[str] = Field(default_factory=lambda: ["dp_shard"])
    # actor -> MixedPrecisionPolicy(param=bf16, reduce=fp32); none -> full precision.
    # Named precision_policy (not mixed_precision) to avoid colliding with the
    # training-forward dtype `train_precision` (fp32/bf16/fp16); this is a
    # param/reduce policy, not a dtype.
    precision_policy: Literal["actor", "none"] = "actor"
    # True = re-gather params after forward (ZeRO-3, lowest memory).
    reshard_after_forward: bool = True
    # Keep parameter/gradient shards on CPU between forwards. This is slower but
    # lets timestep-routed multi-root models materialize only the active expert.
    cpu_offload: bool = False


class DDPConfig(ConfigBase):
    """distributed.training.ddp: the one DDP knob ``build_strategy`` reads.

    DDP replicates the full module per rank (right when the model fits on one
    card, e.g. a 2B transformer + LoRA), so unlike ``fsdp`` there are no shard/mesh
    knobs. ``find_unused_parameters`` is the only DDP-specific lever a reader
    consumes; declare nothing the strategy doesn't read.
    """

    # DDP's reducer expects every requires_grad param to get a grad each backward;
    # set True only if a LoRA/grad-checkpoint forward conditionally skips a wrapped
    # branch (slower — adds an extra graph traversal).
    find_unused_parameters: bool = False


class TrainingSection(ConfigBase):
    """distributed.training: how the trainer process maps onto GPUs.

    Only the fields a reader actually consumes are declared: ``strategy`` (the
    single source of truth for the allowed backends — resource validation and the
    training context dispatch on it; the Literal is the only allow-list), the
    ``num_nodes``/``gpus_per_node`` topology the fsdp/ddp context cross-checks
    against ``WORLD_SIZE``, and the per-strategy knob block (``fsdp`` / ``ddp``)
    read by ``build_strategy``.
    """

    strategy: Literal["single_process", "fsdp", "ddp"] = "single_process"
    num_nodes: int = 1
    gpus_per_node: int = 1
    fsdp: FSDPConfig | None = None
    ddp: DDPConfig | None = None

    @model_validator(mode="after")
    def _resolve_strategy_defaults(self) -> TrainingSection:
        """Resolve defaults here so runtime constructors cannot redeclare them.

        The unselected block stays absent: it has no runtime consumer and should
        not become duplicated resolved state merely because another strategy was
        chosen.
        """
        if self.fsdp is not None and self.strategy != "fsdp":
            raise ValueError(
                "distributed.training.fsdp requires distributed.training.strategy=fsdp",
            )
        if self.ddp is not None and self.strategy != "ddp":
            raise ValueError(
                "distributed.training.ddp requires distributed.training.strategy=ddp",
            )
        if self.strategy == "fsdp" and self.fsdp is None:
            self.fsdp = FSDPConfig()
        elif self.strategy == "ddp" and self.ddp is None:
            self.ddp = DDPConfig()
        return self


class RolloutRuntimeSection(ConfigBase):
    """distributed.rollout: rollout runtime knobs (engine level and rank level).

    reader: vrl/generation/ray/config.py RayGenerationConfig.from_root. Release
    scheduling and colocation are NOT declared here: colocation lives in
    distributed.resources.rollout.gpu_pool=trainer (mirrors reward.gpu_pool),
    and release scheduling is derived from GPU topology.
    batch_placement_strategy is a user-facing allow-list
    Literal: RayGenerationConfig is a plain dataclass whose annotations do not
    enforce, so this typed boundary is where a bad value is rejected (the runtime
    DistributedExecutionPlanner guard covers direct construction). sync_trainable_state
    is a plain on/off: True keeps rollout engines resynced to the trained policy
    (the syncer flattens whatever is trainable — lora or full-param), False
    disables the weight syncer.

    Each knob belongs to one layer of the engine/rank split (an engine is one
    replica; a rank is one per-GPU worker actor inside it):
    rank level: cpus_per_worker, health_check_*, worker_rpc_timeout_s.
    engine level: generation_stall_timeout_s, batch_placement_strategy,
    sync_trainable_state, pipelined.
    """

    # rank level: CPU grant per rank actor (Ray num_cpus).
    cpus_per_worker: float = 1.0
    # rank level: background liveness probing of rank actors. interval <= 0
    # disables it; a rank that stops answering kills the owned actors so active
    # or subsequent foreground work fails closed. A failed verdict then enters
    # the supervisor's bounded restart policy.
    health_check_interval_s: float = 30.0
    health_check_timeout_s: float = 30.0
    health_check_first_wait_s: float = 0.0
    # rank level: opaque control-plane calls expose no useful progress. Bound
    # startup, metadata, capability, and weight acknowledgements independently.
    worker_rpc_timeout_s: float = 600.0
    # engine level: generation has a separate stall budget — a completed batch
    # is real progress, and the pipelined path reports the same progress. One
    # hour covers the observed ~30-minute cold compile plus a 733-second Cosmos
    # batch with margin; opaque control calls retain their tighter budget above.
    generation_stall_timeout_s: float = 3600.0
    # engine level: batch -> engine binding strategy.
    batch_placement_strategy: BatchPlacementStrategy = "round_robin"
    sync_trainable_state: bool = True
    # engine level: opt-in single-engine pipelined rollout. Config resolution
    # rejects multiple engines; requests with fewer than two batches use the
    # standard per-batch path, and a pipeline OOM falls back to that path's
    # split-and-retry behavior.
    pipelined: bool = False

    @model_validator(mode="after")
    def _validate_health_check(self) -> RolloutRuntimeSection:
        if not math.isfinite(self.health_check_interval_s):
            raise ValueError(
                "distributed.rollout.health_check_interval_s must be finite",
            )
        if self.health_check_interval_s > 0 and (
            not math.isfinite(self.health_check_timeout_s) or self.health_check_timeout_s <= 0
        ):
            raise ValueError(
                "distributed.rollout.health_check_timeout_s must be finite and > 0 "
                "when health checking is enabled",
            )
        if not math.isfinite(self.health_check_first_wait_s) or self.health_check_first_wait_s < 0:
            raise ValueError(
                "distributed.rollout.health_check_first_wait_s must be finite and >= 0",
            )
        if not math.isfinite(self.worker_rpc_timeout_s) or self.worker_rpc_timeout_s <= 0:
            raise ValueError(
                "distributed.rollout.worker_rpc_timeout_s must be finite and > 0",
            )

        if (
            not math.isfinite(self.generation_stall_timeout_s)
            or self.generation_stall_timeout_s <= 0
        ):
            raise ValueError(
                "distributed.rollout.generation_stall_timeout_s must be finite and > 0",
            )
        return self


class DistributedSection(ConfigBase):
    """Key registry for distributed.*; values validated by vrl.ray.resources."""

    # reader: vrl/ray/resources.py ResolvedDistributedResources.from_root(root); the
    # consuming dataclass is the section type, so pydantic validates it here.
    resources: DistributedResourceConfig | None = None
    # reader: vrl/generation/ray/config.py RayGenerationConfig.from_root (worker
    # runtime knobs). batch_placement_strategy / sync_trainable_state Literals reject
    # bad values here at parse time. Colocation lives in resources.rollout.gpu_pool.
    rollout: RolloutRuntimeSection | None = None
    # readers: vrl/trainers/distributed.py DistributedTrainingContext.from_root (rank/device)
    # + vrl/ray/resources.py strategy-aware trainer GPU validation
    training: TrainingSection | None = None


# ── Root config ───────────────────────────────────────────────────────────────


class KlingVideoRewardProductionConfig(ConfigBase):
    """Production enablement for the Kling VideoReward contract gate."""

    enabled: bool = False


class ProductionSection(ConfigBase):
    """Closed registry of production gates with runtime consumers."""

    kling_video_reward: KlingVideoRewardProductionConfig = Field(
        default_factory=KlingVideoRewardProductionConfig,
    )


class RootConfig(ConfigBase):
    """Top-level typed boundary for all training config sections.

    model/sampling are family-selected; actor/trainer/precision are fully
    typed sections; distributed is a key registry whose values are validated
    by vrl.ray.resources. An unknown key anywhere fails at parse_config.
    """

    algorithm: AlgorithmConfig | None = None
    data: DataConfig | None = None
    reward: RewardConfig | None = None
    rollout: RolloutConfig | None = None
    # Family-selected: the ``model.family`` picks the concrete section class
    # (``_parse_model_section``); sampling follows the model's family.
    model: SerializeAsAny[ModelSection] | None = None
    sampling: SerializeAsAny[SamplingSection] | None = None
    # Per-component production gates; contract checks live in
    # vrl/config/validation.py validate_production_* (raw-cfg checks)
    production: ProductionSection | None = None
    trainer: TrainerSection | None = None
    actor: ActorSection | None = None
    distributed: DistributedSection | None = None
    # resolver: vrl/config/precision.py PrecisionPolicy.from_section(root.precision)
    precision: PrecisionConfig | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _select_model_section(cls, value: Any) -> ModelSection | None:
        return _parse_model_section(value)

    @field_validator("sampling", mode="before")
    @classmethod
    def _select_sampling_section(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> SamplingSection | None:
        model = info.data.get("model")
        return _parse_sampling_section(
            value,
            model=model if isinstance(model, ModelSection) else None,
        )

    @model_validator(mode="after")
    def _cross_field_validate(self) -> RootConfig:
        algo = self.algorithm
        if algo is None:
            return self

        kind = algo.kind
        kl_reward_coef = resolve_kl_reward_coef(algo.kl_reward_coef)
        if kl_reward_coef > 0.0 and kind in {
            "token_grpo",
            "token_grpo_multisegment",
            "diffusion_dpo",
        }:
            raise ValueError(
                "algorithm.kl_reward_coef > 0 requires a diffusion rollout "
                "trajectory with collected per-step KL; "
                f"algorithm.kind={kind!r} does not provide one",
            )
        rollout = self.rollout
        model_family = normalize_model_family(
            (self.model.family or "") if self.model else "",
        )

        if kind == "diffusion_dpo":
            self._validate_offline_dpo_surface()

        # The SFT term belongs to continuous diffusion GRPO and offline
        # Diffusion-DPO. Validate the numeric domain and algorithm ownership
        # here so an inherited token-GRPO field cannot silently become a no-op.
        raw_sft_weight = getattr(algo.hyperparameters, "sft_weight", None)
        if raw_sft_weight is not None:
            try:
                sft_weight = float(raw_sft_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError("algorithm.sft_weight must be a finite number >= 0") from exc
            if not math.isfinite(sft_weight) or sft_weight < 0:
                raise ValueError("algorithm.sft_weight must be a finite number >= 0")
            if sft_weight > 0:
                if kind in {"grpo", "dance_grpo"}:
                    if self.data is None or not self.data.sft_latents:
                        raise ValueError(
                            "algorithm.sft_weight > 0 requires data.sft_latents "
                            "(the precomputed clean-latents shard; see "
                            "vrl/scripts/denoise/encode_targets.py)",
                        )
                elif kind != "diffusion_dpo":
                    raise ValueError(
                        "algorithm.sft_weight > 0 is supported only for diffusion "
                        "grpo/dance_grpo or diffusion_dpo",
                    )

        # grpo / diffusion_nft: SDE type must be sde or cps
        # grpo / diffusion_nft require an sde block; sde.type membership is now
        # enforced by the SdeConfig Literal (for every kind, not just these two —
        # the runtime layout guard remains the wire-boundary check).
        if kind in {
            "grpo",
            "dance_grpo",
            "flash_grpo",
            "flow_dppo",
            "grpo_guard",
            "diffusion_nft",
        } and (rollout is None or rollout.sde is None):
            raise ValueError("config missing required field: rollout.sde.type")

        # token_grpo: nextstep_1 family requires rollout.noise_level
        if (
            kind == "token_grpo"
            and model_family == "nextstep_1"
            and (rollout is None or rollout.noise_level is None)
        ):
            raise ValueError("config missing required field: rollout.noise_level")

        if model_family == "janus_pro_r1" and kind != "token_grpo_multisegment":
            raise ValueError(
                "model.family=janus_pro_r1 requires algorithm.kind=token_grpo_multisegment",
            )

        # token_grpo_multisegment: explicit Janus R1 protocol; final_image_policy
        # remains owned by rollout.
        if kind == "token_grpo_multisegment":
            if model_family != "janus_pro_r1":
                raise ValueError(
                    "token_grpo_multisegment currently requires model.family=janus_pro_r1",
                )
            # Single source: final_image_policy lives on rollout only. Validate it
            # here as a legality check (the collector reads rollout.final_image_policy).
            policy = (rollout.final_image_policy or "") if rollout else ""
            if policy not in {"always_generate", "use_selfcheck"}:
                raise ValueError(
                    "rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'"
                )

        return self

    def _validate_offline_dpo_surface(self) -> None:
        for section_name, section, allowed in (
            ("actor", self.actor, _OFFLINE_DPO_ACTOR_FIELDS),
            ("trainer", self.trainer, _OFFLINE_DPO_TRAINER_FIELDS),
        ):
            if section is None:
                continue
            unsupported = sorted(section.model_fields_set - allowed)
            if unsupported:
                fields = ", ".join(f"{section_name}.{name}" for name in unsupported)
                raise ValueError(
                    f"diffusion_dpo does not consume config field(s): {fields}",
                )

        if self.rollout is not None and self.rollout.model_fields_set:
            fields = ", ".join(f"rollout.{name}" for name in sorted(self.rollout.model_fields_set))
            raise ValueError(
                f"diffusion_dpo does not consume config field(s): {fields}",
            )
        if self.reward is not None:
            raise ValueError("diffusion_dpo does not consume the reward config section")


# ── Parse boundary ────────────────────────────────────────────────────────────


def parse_config(cfg: DictConfig) -> RootConfig:
    """Validate a fully-merged, resolved DictConfig through the typed schema.

    OmegaConf resolves interpolations and enforces ??? missing-value semantics;
    Pydantic validates structure (every section is closed, so unknown keys fail
    here), enum discriminators, and cross-field rules.
    """
    try:
        raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as exc:
        missing_path = getattr(exc, "full_key", None) or str(exc)
        raise ValueError(f"config missing required field: {missing_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("config must be a top-level mapping")

    try:
        return RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(_extract_error_message(exc)) from exc


__all__ = [
    "ActorSection",
    "AlgorithmConfig",
    "DataConfig",
    "DataPreprocessingSection",
    "DataSamplerSection",
    "ModelSection",
    "RewardConfig",
    "RolloutConfig",
    "RootConfig",
    "TrainerSection",
    "generation_request_rollout_fields",
    "parse_config",
    "sampling_section_class_for_family",
]
