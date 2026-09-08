"""Import-light algorithm facts consumed by cross-section config rules."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class AlgorithmConfigContract:
    """Declared behavior, not user-overridable hyperparameters.

    Every independent algorithm config must choose its rollout, reward-shaping,
    and SFT semantics. A restricted surface lists consumed actor/trainer/rollout fields for
    offline training (which consumes no reward section); None leaves section
    ownership to the ordinary online validation rules.
    """

    needs_sde_rollout: bool
    supports_step_kl_reward: bool
    sft_source: Literal["unsupported", "latents", "preference_winner"]
    consumed_sections: tuple[tuple[str, frozenset[str]], ...] | None = None
