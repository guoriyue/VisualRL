"""Multi-segment token GRPO for Janus-Pro-R1 style rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Torch is a call-time dependency, not an import-time one: this module's config
# dataclass is what ``algorithm.kind`` dispatch loads during config parsing,
# and every annotation is a string under PEP 563.
if TYPE_CHECKING:
    import torch

from vrl.algorithms.advantages import GroupAdvantageEstimator
from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics


@dataclass(slots=True)
class MultiSegmentTokenGRPOConfig(TokenGRPOConfig):
    """TokenGRPO config with weighted segment losses."""

    segment_weights: dict[str, float] = field(
        default_factory=lambda: {
            "initial_image": 1.0,
            "selfcheck_text": 0.0,
            "final_image": 1.0,
        },
    )
    train_segments: dict[str, bool] = field(
        default_factory=lambda: {
            "initial_image": True,
            "selfcheck_text": False,
            "final_image": True,
        },
    )


class MultiSegmentTokenGRPO(TokenGRPO):
    """Apply TokenGRPO independently per segment, then average by weight."""

    def __init__(
        self,
        config: MultiSegmentTokenGRPOConfig | None = None,
        *,
        advantage_estimator: GroupAdvantageEstimator | None = None,
    ) -> None:
        cfg = config or MultiSegmentTokenGRPOConfig()
        super().__init__(cfg, advantage_estimator=advantage_estimator)
        self.config: MultiSegmentTokenGRPOConfig = cfg

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        # Lazy: keep vrl.algorithms torch-free at import time so config parsing
        # (algorithm.kind dispatch) never pulls the rollout evaluator stack.
        from vrl.rollouts.evaluators.types import TrajectorySignalBatch

        # signals presence + required_signal_keys are enforced upstream by
        # AlgorithmAdapter.validate_inputs (inherited from GRPO).
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for MultiSegmentTokenGRPO")
        signals = inputs.signals

        total_loss: torch.Tensor | None = None
        policy_losses: list[float] = []
        kl_penalties: list[float] = []
        weighted_kl_losses: list[float] = []
        update_stats: list[PolicyUpdateStats] = []
        metric_weights: list[float] = []
        total_weight = 0.0
        train_segments = dict(self.config.train_segments or {})
        weights = dict(self.config.segment_weights or {})
        missing_weighted = [
            name
            for name, weight in weights.items()
            if bool(train_segments.get(name, True))
            and float(weight) > 0
            and name not in signals.segments
        ]
        if missing_weighted:
            raise RuntimeError(
                "missing multi-segment GRPO segment: " + ", ".join(missing_weighted),
            )
        segment_names = list(signals.segments)

        for name in segment_names:
            if not bool(train_segments.get(name, True)):
                continue
            weight = float(weights.get(name, 1.0))
            if weight <= 0:
                continue
            segment_signal = signals.segments.get(name)
            if segment_signal is None:
                raise RuntimeError(f"missing multi-segment GRPO segment: {name}")
            segment_advantages = inputs.advantages
            if isinstance(segment_advantages, dict):
                if name in segment_advantages:
                    segment_advantages = segment_advantages[name]
                elif "__default__" in segment_advantages:
                    segment_advantages = segment_advantages["__default__"]
                else:
                    raise RuntimeError(f"missing multi-segment advantages for segment: {name}")
            loss, metrics = super().compute_loss(
                AlgorithmInput(
                    signals=TrajectorySignalBatch(
                        segments={name: segment_signal},
                        group_ids=signals.group_ids,
                        primary_segment=name,
                    ),
                    advantages=segment_advantages,
                ),
            )
            weighted = loss * weight
            total_loss = weighted if total_loss is None else total_loss + weighted
            total_weight += weight
            policy_losses.append(metrics.policy_loss)
            kl_penalties.append(metrics.kl_penalty)
            weighted_kl_losses.append(metrics.weighted_kl_loss)
            update_stats.append(metrics.update)
            metric_weights.append(weight)

        if total_loss is None or total_weight <= 0:
            zero = signals.primary.log_prob.sum() * 0.0
            return zero, TrainStepMetrics()

        total_loss = total_loss / total_weight

        def _weighted_avg(values: list[float]) -> float:
            if not values:
                return 0.0
            return (
                sum(value * weight for value, weight in zip(values, metric_weights, strict=True))
                / total_weight
            )

        return total_loss, TrainStepMetrics(
            loss=float(total_loss.item()),
            policy_loss=_weighted_avg(policy_losses),
            kl_penalty=_weighted_avg(kl_penalties),
            weighted_kl_loss=_weighted_avg(weighted_kl_losses),
            update=PolicyUpdateStats.weighted_mean(update_stats, metric_weights),
        )
