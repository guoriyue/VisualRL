"""Training signal types for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalBatch:
    """Distribution-family-specific training signals.

    Produced by an ``Evaluator`` from model forward results.
    Consumed by ``Algorithm.compute_loss()``.
    """

    log_prob: Any                               # [B, T] log pi(a|s)
    ref_log_prob: Any | None = None             # [B, T] log pi_ref(a|s)
    entropy: Any | None = None
    # Flow-matching specific (for latent-space KL)
    prev_sample_mean: Any | None = None
    ref_prev_sample_mean: Any | None = None
    std_dev_t: Any | None = None
    dt: Any | None = None
    dist_family: str = "flow_matching"          # or "categorical", etc.
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRequest:
    """What the algorithm needs the evaluator to compute."""

    need_ref: bool = False
    need_entropy: bool = False
    need_kl_intermediates: bool = False  # for latent-space KL
