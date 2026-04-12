"""ModelAdapter protocol — model-specific forward ABI standardization."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.experience.types import ExperienceBatch


@runtime_checkable
class ModelAdapter(Protocol):
    """Model-specific forward ABI -> standardized result dict.

    Each model family (Wan, SD3, Cosmos) implements this protocol
    to handle its unique forward pass (dual-expert, CFG, etc.).
    """

    def forward_step(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Run model forward for one timestep.

        Returns dict with at least 'noise_pred' key.
        Model-specific keys (e.g., 'noise_pred_cond', 'noise_pred_uncond')
        also allowed.
        """
        ...
