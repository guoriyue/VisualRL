"""Reward functions for learned environment runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class GoalReward:
    """Dense goal-reaching reward over latent state."""

    success_threshold: float = 0.05
    reward_scale: float = 1.0

    def evaluate(
        self,
        next_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        error = (next_state - goal_state).pow(2).mean(dim=(1, 2))
        reward = -error * self.reward_scale
        terminated = error <= self.success_threshold
        info = {
            "goal_mse": error,
            "success": terminated.to(torch.float32),
        }
        return reward, terminated, info


__all__ = ["GoalReward"]
