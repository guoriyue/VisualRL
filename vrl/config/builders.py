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
    from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
    from vrl.trainers.checkpointing import TrainingResumeConfig
    from vrl.trainers.core.types import PrecisionDriftGuardConfig
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
]
