"""Algorithm ABC — advantage computation and policy loss."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vrl.algorithms.types import Advantages, RolloutBatch, RolloutGroup, TrainStepMetrics
from vrl.evaluators.types import SignalBatch


class Algorithm(ABC):
    """Base class for RL algorithms (GRPO, REINFORCE, etc.).

    Supports two interfaces:
    - Legacy: compute_advantages(group) + compute_loss(batch, policy, ref_policy)
    - New (4-layer): compute_advantages_from_tensors(rewards, group_ids)
                   + compute_signal_loss(signals, advantages, old_log_probs)
    """

    # ------------------------------------------------------------------
    # Legacy interface (RolloutGroup / RolloutBatch)
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_advantages(self, group: RolloutGroup) -> Advantages:
        """Compute per-rollout advantages for a single prompt group."""

    @abstractmethod
    def compute_loss(
        self,
        batch: RolloutBatch,
        policy: Any,
        ref_policy: Any = None,
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute the policy gradient loss and metrics (legacy path)."""

    # ------------------------------------------------------------------
    # New 4-layer interface (tensor-based)
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_advantages_from_tensors(
        self,
        rewards: Any,        # [B] tensor
        group_ids: Any,      # [B] tensor — prompt group assignment
    ) -> Any:                # [B] tensor of advantages
        """Compute per-sample advantages from reward tensors."""

    @abstractmethod
    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,          # [B] or [B, T] advantages
        old_log_probs: Any,       # [B] old log-probs from collection
    ) -> tuple[Any, TrainStepMetrics]:
        """Compute loss from evaluator signals. Returns (loss, metrics)."""
