"""RL data containers for trajectories, rollouts, and training metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrajectoryStep:
    """Single denoising step within a trajectory."""

    timestep: int
    log_prob: Any  # Tensor — log-probability under the policy
    noise_pred: Any  # Tensor — predicted noise at this step
    new_log_prob: Any = None  # Refreshed log-prob under current policy
    ref_log_prob: Any = None  # Log-prob under reference policy


@dataclass(slots=True)
class Trajectory:
    """Full denoising trajectory for one generation."""

    prompt: str
    seed: int
    steps: list[TrajectoryStep]
    output: Any  # Final generated output (frames / latents)


@dataclass(slots=True)
class Rollout:
    """A single generation paired with its reward."""

    request: Any  # VideoGenerationRequest or similar
    trajectory: Trajectory
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainStepMetrics:
    """Metrics produced by a single training step."""

    loss: float = 0.0
    policy_loss: float = 0.0
    kl_penalty: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    advantage_mean: float = 0.0
    clip_fraction: float = 0.0
    approx_kl: float = 0.0
    grad_norm: float = 0.0
    adv_saturation: float = 0.0
    adv_zero_rate: float = 0.0
    lr: float = 0.0
    phase_times: dict[str, float] = field(default_factory=dict)
