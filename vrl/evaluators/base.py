"""Evaluator protocol — extract training signals from model forward results."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.adapters.base import ModelAdapter
from vrl.evaluators.types import SignalBatch, SignalRequest
from vrl.experience.types import ExperienceBatch


@runtime_checkable
class Evaluator(Protocol):
    """Extract training signals from model forward results.

    Uses the adapter for model-specific forward passes and extracts
    distribution-family-specific signals (log_prob, KL, etc.).
    """

    def evaluate(
        self,
        adapter: ModelAdapter,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Run adapter.forward_step() -> extract log_prob, KL, etc."""
        ...
