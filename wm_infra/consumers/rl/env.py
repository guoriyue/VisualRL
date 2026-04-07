"""Trainer-facing environment adapters built on top of the world-model interface."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from wm_infra.models.base import WorldModel
from wm_infra.envs.rewards import GoalReward


TensorSampler = Callable[[int, torch.device, torch.dtype, Optional[torch.Generator]], torch.Tensor]


class WorldModelEnv:
    """Single-environment RL adapter with a Gym-like reset/step contract."""

    def __init__(
        self,
        world_model: WorldModel,
        *,
        initial_state_sampler: TensorSampler,
        goal_state_sampler: TensorSampler,
        reward_fn: GoalReward,
        action_dim: int,
        max_episode_steps: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.world_model = world_model
        self.initial_state_sampler = initial_state_sampler
        self.goal_state_sampler = goal_state_sampler
        self.reward_fn = reward_fn
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.dtype = dtype
        self._generator: torch.Generator | None = None
        self._state: torch.Tensor | None = None
        self._goal: torch.Tensor | None = None
        self._step_idx = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(seed)
        self._state = self.initial_state_sampler(1, self.device, self.dtype, self._generator)
        self._goal = self.goal_state_sampler(1, self.device, self.dtype, self._generator)
        self._step_idx = 0
        return self._observation(), self._info_template()

    @torch.inference_mode()
    def step(self, action: np.ndarray | torch.Tensor) -> tuple[np.ndarray, float, bool, bool, dict[str, np.ndarray | float | int]]:
        if self._state is None or self._goal is None:
            raise RuntimeError("Environment must be reset before step().")
        action_tensor = torch.as_tensor(action, dtype=self.dtype, device=self.device).view(1, self.action_dim)
        next_state = self.world_model.predict_next(self._state, action_tensor)
        reward, terminated, info_tensors = self.reward_fn.evaluate(next_state, self._goal)
        self._state = next_state
        self._step_idx += 1
        truncated = self._step_idx >= self.max_episode_steps
        info = self._info_template()
        info.update(
            {
                "goal_mse": float(info_tensors["goal_mse"].item()),
                "success": bool(info_tensors["success"].item() > 0),
                "step": self._step_idx,
            }
        )
        return self._observation(), float(reward.item()), bool(terminated.item()), bool(truncated), info

    def _observation(self) -> np.ndarray:
        assert self._state is not None and self._goal is not None
        return torch.cat([self._state, self._goal], dim=-1).squeeze(0).detach().cpu().numpy()

    def _info_template(self) -> dict[str, np.ndarray | int]:
        assert self._goal is not None
        return {
            "goal": self._goal.squeeze(0).detach().cpu().numpy(),
            "step": self._step_idx,
        }


class WorldModelVectorEnv:
    """Vectorized RL environment that batches one-step world-model transitions."""

    def __init__(
        self,
        world_model: WorldModel,
        *,
        num_envs: int,
        initial_state_sampler: TensorSampler,
        goal_state_sampler: TensorSampler,
        reward_fn: GoalReward,
        action_dim: int,
        max_episode_steps: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        auto_reset: bool = True,
    ) -> None:
        self.world_model = world_model
        self.num_envs = num_envs
        self.initial_state_sampler = initial_state_sampler
        self.goal_state_sampler = goal_state_sampler
        self.reward_fn = reward_fn
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.dtype = dtype
        self.auto_reset = auto_reset
        self._generator: torch.Generator | None = None
        self._states: torch.Tensor | None = None
        self._goals: torch.Tensor | None = None
        self._step_idx = torch.zeros(num_envs, dtype=torch.int64)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(seed)
        self._states = self.initial_state_sampler(self.num_envs, self.device, self.dtype, self._generator)
        self._goals = self.goal_state_sampler(self.num_envs, self.device, self.dtype, self._generator)
        self._step_idx.zero_()
        return self._observation(), {"goal": self._goals.detach().cpu().numpy()}

    @torch.inference_mode()
    def step(
        self,
        actions: np.ndarray | torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if self._states is None or self._goals is None:
            raise RuntimeError("Environment must be reset before step().")
        action_tensor = torch.as_tensor(actions, dtype=self.dtype, device=self.device).view(self.num_envs, self.action_dim)
        next_states = self.world_model.predict_next(self._states, action_tensor)
        rewards, terminated, info_tensors = self.reward_fn.evaluate(next_states, self._goals)
        self._states = next_states
        self._step_idx += 1
        truncated = self._step_idx >= self.max_episode_steps

        info = {
            "goal": self._goals.detach().cpu().numpy(),
            "goal_mse": info_tensors["goal_mse"].detach().cpu().numpy(),
            "success": info_tensors["success"].detach().cpu().numpy(),
            "step": self._step_idx.detach().cpu().numpy(),
        }

        done_mask = terminated | truncated.to(terminated.device)
        if self.auto_reset and torch.any(done_mask):
            final_observation = torch.cat([self._states, self._goals], dim=-1).detach().cpu().numpy()
            info["final_observation"] = final_observation
            self._reset_mask(done_mask)

        return (
            self._observation(),
            rewards.detach().cpu().numpy(),
            terminated.detach().cpu().numpy().astype(np.bool_),
            truncated.detach().cpu().numpy().astype(np.bool_),
            info,
        )

    def _reset_mask(self, mask: torch.Tensor) -> None:
        assert self._states is not None and self._goals is not None
        count = int(mask.sum().item())
        if count == 0:
            return
        new_states = self.initial_state_sampler(count, self.device, self.dtype, self._generator)
        new_goals = self.goal_state_sampler(count, self.device, self.dtype, self._generator)
        self._states[mask] = new_states
        self._goals[mask] = new_goals
        self._step_idx[mask.cpu()] = 0

    def _observation(self) -> np.ndarray:
        assert self._states is not None and self._goals is not None
        return torch.cat([self._states, self._goals], dim=-1).detach().cpu().numpy()
