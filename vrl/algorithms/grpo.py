"""GRPO — Group Relative Policy Optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import (
    Advantages,
    RolloutBatch,
    RolloutGroup,
    TrainStepMetrics,
)
from vrl.rollouts.evaluators.types import SignalBatch


@dataclass(slots=True)
class GRPOConfig:
    """Hyper-parameters for GRPO."""

    clip_eps: float = 0.2
    kl_coeff: float = 0.0
    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False


class GRPO(Algorithm):
    """Group Relative Policy Optimization.

    Advantages are normalised within each prompt group:
        a_i = (r_i - mean(r)) / max(std(r), eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    pre-computed log-probabilities stored in ``TrajectoryStep.log_prob``.
    """

    def __init__(self, config: GRPOConfig | None = None) -> None:
        self.config = config or GRPOConfig()

    # ------------------------------------------------------------------
    # Legacy: Advantages from RolloutGroup
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        group: RolloutGroup,
        global_rewards: list[float] | None = None,
    ) -> Advantages:
        """Compute per-rollout advantages for a single prompt group.

        When ``global_std=True`` (or ``global_rewards`` is provided), the
        standard deviation is computed across all rewards in the batch,
        not just within this group.  This matches the flow_grpo
        ``PerPromptStatTracker(global_std=True)`` behavior.
        """
        rewards = [r.reward for r in group.rollouts]
        n = len(rewards)
        if n == 0:
            return Advantages(values=[], method="grpo")

        mean = sum(rewards) / n

        if self.config.global_std and global_rewards is not None:
            g_n = len(global_rewards)
            g_mean = sum(global_rewards) / max(g_n, 1)
            g_var = sum((r - g_mean) ** 2 for r in global_rewards) / max(g_n, 1)
            std = math.sqrt(g_var)
        else:
            var = sum((r - mean) ** 2 for r in rewards) / max(n, 1)
            std = math.sqrt(var)

        denom = max(std, self.config.eps)
        values = [(r - mean) / denom for r in rewards]

        # Clip advantages (from flow_grpo: torch.clamp(adv, -adv_clip_max, adv_clip_max))
        clip = self.config.adv_clip_max
        values = [max(-clip, min(clip, v)) for v in values]

        return Advantages(
            values=values,
            method="grpo",
            stats={"reward_mean": mean, "reward_std": std},
        )

    # ------------------------------------------------------------------
    # Legacy: Loss from RolloutBatch
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        batch: RolloutBatch,
        policy: Any,
        ref_policy: Any = None,
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss over all rollout groups (legacy path)."""
        cfg = self.config
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_steps = 0
        clip_count = 0

        all_rewards: list[float] = []
        all_advantages: list[float] = []

        for group in batch.groups:
            if group.advantages is None:
                continue
            for rollout, adv in zip(group.rollouts, group.advantages.values):
                all_rewards.append(rollout.reward)
                all_advantages.append(adv)
                for step in rollout.trajectory.steps:
                    old_lp = step.log_prob
                    new_lp = step.new_log_prob if step.new_log_prob is not None else old_lp

                    ratio = math.exp(new_lp - old_lp)
                    clipped = max(min(ratio, 1.0 + cfg.clip_eps), 1.0 - cfg.clip_eps)
                    surrogate = min(ratio * adv, clipped * adv)
                    total_policy_loss += -surrogate

                    if ratio != clipped:
                        clip_count += 1

                    if cfg.kl_coeff > 0 and ref_policy is not None:
                        ref_lp = step.ref_log_prob if step.ref_log_prob is not None else old_lp
                        kl = old_lp - ref_lp
                        total_kl += cfg.kl_coeff * kl

                    total_steps += 1

        if total_steps > 0:
            total_policy_loss /= total_steps
            total_kl /= total_steps

        total_loss = total_policy_loss + total_kl

        n_rewards = len(all_rewards)
        reward_mean = sum(all_rewards) / max(n_rewards, 1)
        reward_var = (
            sum((r - reward_mean) ** 2 for r in all_rewards) / max(n_rewards, 1)
        )
        adv_mean = sum(all_advantages) / max(len(all_advantages), 1)

        metrics = TrainStepMetrics(
            loss=total_loss,
            policy_loss=total_policy_loss,
            kl_penalty=total_kl,
            reward_mean=reward_mean,
            reward_std=math.sqrt(reward_var),
            advantage_mean=adv_mean,
            clip_fraction=clip_count / max(total_steps, 1),
        )
        return total_loss, metrics

    # ------------------------------------------------------------------
    # New 4-layer: Advantages from tensors
    # ------------------------------------------------------------------

    def compute_advantages_from_tensors(
        self,
        rewards: Any,     # [B] tensor
        group_ids: Any,   # [B] tensor
    ) -> Any:
        """Per-group advantage normalization on tensors.

        Groups are identified by ``group_ids`` — samples sharing the same
        group_id are normalized together (GRPO per-prompt normalization).
        """
        import torch

        cfg = self.config
        advantages = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)

        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]
            mean = group_rewards.mean()

            if cfg.global_std:
                std = rewards.std()
            else:
                std = group_rewards.std()

            denom = torch.clamp(std, min=cfg.eps)
            group_adv = (group_rewards - mean) / denom
            group_adv = torch.clamp(group_adv, -cfg.adv_clip_max, cfg.adv_clip_max)
            advantages[mask] = group_adv

        return advantages

    # ------------------------------------------------------------------
    # New 4-layer: Loss from SignalBatch
    # ------------------------------------------------------------------

    def compute_signal_loss(
        self,
        signals: SignalBatch,
        advantages: Any,       # [B] advantages
        old_log_probs: Any,    # [B] old log-probs from collection
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss from evaluator signals.

        Handles both flow-matching (latent-space KL) and generic (log-prob KL).
        """
        import torch

        from vrl.rollouts.evaluators.diffusion.flow_matching import compute_kl_divergence

        cfg = self.config

        ratio = torch.exp(signals.log_prob - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        # KL penalty
        if cfg.kl_coeff > 0 and signals.ref_log_prob is not None:
            if (
                signals.dist_family == "flow_matching"
                and signals.prev_sample_mean is not None
                and signals.ref_prev_sample_mean is not None
            ):
                # Latent-space KL (more principled for continuous diffusion)
                kl = compute_kl_divergence(
                    signals.prev_sample_mean,
                    signals.ref_prev_sample_mean,
                    signals.std_dev_t,
                    signals.dt,
                )
                kl_loss = torch.mean(kl)
            else:
                # Log-prob KL fallback
                kl_loss = torch.mean(signals.log_prob - signals.ref_log_prob)
            loss = policy_loss + cfg.kl_coeff * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=signals.log_prob.device)
            loss = policy_loss

        # Metrics
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_eps).float()).item()
        approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            clip_fraction=clip_fraction,
        )
        metrics.approx_kl = approx_kl  # type: ignore[attr-defined]

        return loss, metrics
