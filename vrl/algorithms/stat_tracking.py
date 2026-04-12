"""Per-prompt reward stat tracking for GRPO advantage computation.

Ported from flow_grpo/stat_tracking.py.  Tracks per-prompt reward
history and computes group-relative advantages with optional global std.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PerPromptStatTracker:
    """Track per-prompt reward statistics for advantage normalization.

    When ``global_std=True``, the standard deviation is computed across
    *all* rewards in the batch (not just same-prompt), which is better
    for diverse prompt sets.

    Supported advantage methods (matching flow_grpo/stat_tracking.py):
    - ``grpo``: (reward - mean) / std per prompt group
    - ``rwr``:  raw rewards (reward-weighted regression)
    - ``sft``:  one-hot: 1 for the best sample in group, 0 otherwise
    - ``dpo``:  +1 for best, -1 for worst, 0 otherwise
    """

    def __init__(self, global_std: bool = False) -> None:
        self.global_std = global_std
        self.stats: dict[str, list[float]] = {}
        self.history_prompts: set[int] = set()

    def update(
        self,
        prompts: list[str] | Any,
        rewards: list[float] | Any,
        method: str = "grpo",
    ) -> Any:
        """Compute advantages for the given prompts and rewards.

        Returns a numpy array of advantages, same shape as ``rewards``.
        """
        import torch

        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards.tolist())
            self.history_prompts.add(hash(prompt))

        for prompt in unique:
            self.stats[prompt] = list(np.stack(self.stats[prompt]))
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            if method == "grpo":
                advantages[prompts == prompt] = (prompt_rewards - mean) / std

            elif method == "rwr":
                advantages[prompts == prompt] = prompt_rewards

            elif method == "sft":
                # One-hot: 1.0 for the best sample(s) in group, 0.0 otherwise
                # Port from flow_grpo/stat_tracking.py:37
                t = torch.tensor(prompt_rewards)
                advantages[prompts == prompt] = (
                    t == torch.max(t)
                ).float().numpy()

            elif method == "dpo":
                # Best sample = +1, worst = -1, others = 0
                # Port from flow_grpo/stat_tracking.py:39-52
                t = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(t)
                min_idx = torch.argmin(t)
                # If all rewards are identical, arbitrarily pick indices
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = min(1, len(t) - 1)
                result = torch.zeros_like(t).float()
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()

        return advantages

    def get_stats(self) -> tuple[float, int]:
        """Return (average group size, number of unique prompts seen)."""
        avg = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        return avg, len(self.history_prompts)

    def clear(self) -> None:
        """Clear accumulated stats (call at end of each epoch)."""
        self.stats = {}
