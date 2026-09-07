"""Build typed runtime config objects from merged YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from vrl.config.precision import (
    PrecisionPolicy,
)
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.config.schema import RewardConfig, RootConfig
from vrl.config.validation import require_training_config

if TYPE_CHECKING:
    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.checkpointing import TrainingResumeConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig
    from vrl.trainers.online.config import TrainerConfig


@dataclass(frozen=True, slots=True)
class RewardRuntimeConfig:
    """Resolved reward weights and per-component runtime kwargs."""

    weights: dict[str, float]
    kwargs: dict[str, dict[str, Any]]
    # Per-component inference deployment, resolved once here (compute-once) so GPU
    # placement reads it off this bundle instead of re-walking the raw reward cfg.
    inference_configs: dict[str, RewardInferenceConfig]

    def __post_init__(self) -> None:
        for name, component_kwargs in self.kwargs.items():
            for key in ("sleep_offload", "memory_parking_residual_bytes_limit"):
                if key in component_kwargs:
                    raise ValueError(
                        f"reward.kwargs.{name}.{key} is topology-derived and cannot "
                        "be set in YAML; remove it and select shared or dedicated "
                        "reward GPU ownership under distributed.resources.reward",
                    )

    @classmethod
    def from_cfg(cls, cfg: DictConfig | RewardConfig) -> RewardRuntimeConfig:
        """Resolve one public reward section into its runtime config.

        Zero-weight components remain present so they can be scored and logged
        as observation-only safeguards without changing the optimization
        reward.
        """

        reward = cfg if isinstance(cfg, RewardConfig) else RewardConfig.from_cfg(cfg)
        weights = {name: float(weight) for name, weight in reward.components.items()}
        unknown_kwargs = sorted(set(reward.kwargs) - set(weights))
        if unknown_kwargs:
            keys = ", ".join(f"reward.kwargs.{name}" for name in unknown_kwargs)
            raise ValueError(f"reward kwargs configured for unknown component(s): {keys}")
        kwargs = {name: dict(reward.kwargs.get(name) or {}) for name in weights}
        # RewardConfig already parsed the typed inference section; absent
        # entries execute in-process.
        inference_configs = {
            name: reward.inference.get(name, RewardInferenceConfig()) for name in weights
        }
        return cls(
            weights=weights,
            kwargs=kwargs,
            inference_configs=inference_configs,
        )

    @property
    def all_external_inference(self) -> bool:
        """Whether every configured component executes through an HTTP service."""

        return bool(self.inference_configs) and all(
            inference.kind == "http" for inference in self.inference_configs.values()
        )


@dataclass(frozen=True, slots=True)
class BuiltConfigs:
    """Named outputs derived from one validated public config."""

    root: RootConfig
    algorithm: Any
    precision: PrecisionPolicy
    trainer: TrainerConfig | None
    reward: RewardRuntimeConfig | None
    resume: TrainingResumeConfig


def build_precision_split_safety_configs() -> tuple[
    PrecisionCorrectionConfig,
    PrecisionDriftGuardConfig,
]:
    """Build the production correction and guard policy for a precision split.

    Hardware validation probes consume this same typed source so a measured gate
    cannot silently validate thresholds different from live training.
    """

    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig

    return (
        PrecisionCorrectionConfig(
            tis_mode="truncate",
            rs_mode="seq_mean_k1",
        ),
        PrecisionDriftGuardConfig(
            mode="fail",
            max_abs_log_ratio=math.log(10.0),
            max_ratio_abs_dev=9.0,
            fail_on_nonfinite=True,
        ),
    )


def build_offline_dpo_trainer_config(
    root: RootConfig,
    dpo_config: DiffusionDPOConfig,
) -> OfflineDPOTrainerConfig:
    """Project the parsed ``actor`` section into ``OfflineDPOTrainerConfig``.

    The offline twin of ``TrainerConfig.from_root``: the same public ``actor``
    section, projected into the offline trainer instead. It stays a free
    builder because ``vrl.trainers.offline`` deliberately holds no YAML
    knowledge, and it takes two sources — the typed root plus the already-built
    algorithm config.
    """

    from vrl.trainers.core.types import OptimConfig
    from vrl.trainers.offline import OfflineDPOTrainerConfig

    actor = root.actor

    def required(name: str) -> Any:
        value = None if actor is None else getattr(actor, name)
        if value is None:
            raise ValueError(f"config missing required field: actor.{name}")
        return value

    train_batch_size = int(required("train_batch_size"))
    gradient_accumulation_steps = int(required("gradient_accumulation_steps"))
    optim: OptimConfig = required("optim")
    if optim.optim_8bit:
        raise ValueError(
            "actor.optim.optim_8bit=true is not supported by OfflineDPOTrainer; "
            "use AdamW/Adafactor without 8-bit optimizer state",
        )
    use_adafactor = bool(required("use_adafactor"))
    if use_adafactor:
        # An AdamW-only knob moved off its default would be silently ignored
        # under Adafactor; refuse rather than train with a no-op setting.
        defaults = OptimConfig(lr=optim.lr)
        adam_only_keys = sorted(
            key
            for key in ("adam_beta1", "adam_beta2", "eps")
            if getattr(optim, key) != getattr(defaults, key)
        )
        if adam_only_keys:
            paths = ", ".join(f"actor.optim.{key}" for key in adam_only_keys)
            raise ValueError(
                f"actor.use_adafactor=true does not consume AdamW-only key(s): {paths}",
            )

    scale_lr = bool(required("scale_lr"))
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    lr = float(optim.lr) * effective_batch_size if scale_lr else float(optim.lr)
    max_grad_norm = actor.max_norm if actor is not None else None
    if max_grad_norm is None:
        max_grad_norm = OfflineDPOTrainerConfig().max_grad_norm
    return OfflineDPOTrainerConfig(
        beta=float(dpo_config.beta),
        sft_weight=float(dpo_config.sft_weight),
        lr=lr,
        adam_beta1=float(optim.adam_beta1),
        adam_beta2=float(optim.adam_beta2),
        adam_weight_decay=float(optim.weight_decay),
        adam_epsilon=float(optim.eps),
        max_grad_norm=float(max_grad_norm),
        gradient_accumulation_steps=gradient_accumulation_steps,
        prediction_type=str(required("prediction_type")),
        use_adafactor=use_adafactor,
    )


def build_configs(cfg: DictConfig) -> BuiltConfigs:
    """Bundle typed configs for downstream training scripts."""

    from vrl.trainers.checkpointing import (
        TrainingResumeConfig,
        prepare_model_config_for_training_resume,
    )
    from vrl.trainers.online.config import TrainerConfig

    root, precision = require_training_config(cfg)
    resume = TrainingResumeConfig.from_root(root)
    # A full checkpoint, not model.lora.path, owns trainable state on resume.
    # Clear it in both the parsed root and the merged source so persisted
    # config and all runtime consumers receive one truthful model tree.
    prepare_model_config_for_training_resume(cfg, root, resume)
    if root.algorithm is None:
        raise ValueError("config missing `algorithm` section")
    algorithm = root.algorithm.hyperparameters
    is_offline_dpo = root.algorithm is not None and root.algorithm.kind == "diffusion_dpo"
    trainer = None if is_offline_dpo else TrainerConfig.from_root(root, precision=precision)
    reward = RewardRuntimeConfig.from_cfg(root.reward) if root.reward is not None else None
    if not is_offline_dpo:
        if reward is None:
            raise ValueError("online recipe requires a reward section")
        if not any(weight > 0 for weight in reward.weights.values()):
            raise ValueError("At least one reward component must have weight > 0.")
    return BuiltConfigs(
        root=root,
        algorithm=algorithm,
        precision=precision,
        trainer=trainer,
        reward=reward,
        resume=resume,
    )


__all__ = [
    "BuiltConfigs",
    "RewardRuntimeConfig",
    "build_configs",
    "build_offline_dpo_trainer_config",
]
