"""Generic training experience collected from rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperienceBatch:
    """Generic training experience collected from rollouts.

    Model-agnostic container: the collector fills in observations/actions
    from the denoising trajectory, and model-specific extras go in ``extras``.
    """

    observations: Any   # x_t — current state [B, T, ...]
    actions: Any         # x_{t-1} — next state (denoised) [B, T, ...]
    rewards: Any         # [B] scalar rewards per sample
    dones: Any           # [B] episode termination flags
    group_ids: Any       # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    videos: Any | None = None      # [B, C, T, H, W] decoded frames (for reward scoring)
    prompts: list[str] | None = None
