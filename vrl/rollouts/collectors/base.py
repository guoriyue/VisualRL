"""Collector protocol — collects training experience from model rollouts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.rollouts.types import ExperienceBatch


@runtime_checkable
class Collector(Protocol):
    """Collects training experience from model rollouts.

    Each model family implements both ``collect()`` (rollout) and
    ``forward_step()`` (single-timestep forward for training).
    """

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        ...

    def forward_step(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Run model forward for one timestep.

        Returns dict with at least 'noise_pred' key.
        Model-specific keys also allowed.
        """
        ...
