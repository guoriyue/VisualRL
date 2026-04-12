"""Trainer configuration and training state.

Config fields ported from flow_grpo/config/base.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the training loop.

    Mirrors flow_grpo config.train.* with the same defaults.
    """

    # --- optimizer ---
    lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False

    # --- gradient ---
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # --- batch ---
    batch_size: int = 1
    group_size: int = 4
    num_inner_epochs: int = 1

    # --- PPO / GRPO ---
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    beta: float = 0.0  # KL coefficient (flow_grpo: config.train.beta)

    # --- timestep ---
    timestep_fraction: float = 1.0

    # --- EMA ---
    ema: bool = False
    ema_decay: float = 0.9999
    ema_update_interval: int = 1

    # --- mixed precision ---
    mixed_precision: str = "fp16"  # "fp16", "bf16", "no"
    allow_tf32: bool = True

    # --- CFG during training ---
    cfg: bool = True

    # --- misc ---
    epochs_per_step: int = 1


@dataclass(slots=True)
class TrainState:
    """Mutable training state tracked across steps."""

    step: int = 0
    global_step: int = 0
    total_reward: float = 0.0
    total_loss: float = 0.0
