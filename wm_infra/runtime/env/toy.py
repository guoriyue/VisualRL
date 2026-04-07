"""Deterministic toy world models used by runtime bring-up and RL examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wm_infra.models.base import RolloutInput, RolloutOutput, WorldModel


@dataclass(slots=True)
class ToyWorldSpec:
    latent_tokens: int = 1
    latent_dim: int = 4
    action_dim: int = 2
    damping: float = 0.9


@dataclass(slots=True)
class ToyLineWorldSpec:
    latent_tokens: int = 1
    latent_dim: int = 1
    action_dim: int = 3
    step_size: float = 0.2


class ToyContinuousWorldModel(WorldModel):
    """A simple deterministic latent dynamics model with learnable structure."""

    def __init__(self, spec: ToyWorldSpec | None = None, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.spec = spec or ToyWorldSpec()
        self.device = torch.device(device)
        self.dtype = dtype

        action_to_latent = torch.zeros(self.spec.action_dim, self.spec.latent_dim, dtype=dtype, device=self.device)
        action_to_latent[0, 0] = 1.0
        action_to_latent[0, 2] = 0.25
        action_to_latent[1, 1] = 1.0
        action_to_latent[1, 3] = -0.25
        self.action_to_latent = action_to_latent

    @torch.inference_mode()
    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = latent_state.to(self.device, self.dtype)
        action = action.to(self.device, self.dtype)
        delta = torch.matmul(action, self.action_to_latent).unsqueeze(1).expand(-1, self.spec.latent_tokens, -1)
        return (state * self.spec.damping) + delta

    @torch.inference_mode()
    def rollout(self, input: RolloutInput) -> RolloutOutput:
        current = input.latent_state.to(self.device, self.dtype)
        actions = input.actions.to(self.device, self.dtype)
        predicted_states = []
        for step_idx in range(input.num_steps):
            current = self.predict_next(current, actions[:, step_idx, :])
            predicted_states.append(current)
        return RolloutOutput(predicted_states=torch.stack(predicted_states, dim=1))

    def get_initial_state(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.to(self.device, self.dtype)


class ToyLineWorldModel(WorldModel):
    """Small 1D line-world dynamics for stable runtime and RL examples."""

    def __init__(self, spec: ToyLineWorldSpec | None = None, *, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.spec = spec or ToyLineWorldSpec()
        self.device = torch.device(device)
        self.dtype = dtype
        self.action_deltas = torch.tensor(
            [-self.spec.step_size, 0.0, self.spec.step_size],
            device=self.device,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = latent_state.to(self.device, self.dtype)
        action = action.to(self.device, self.dtype)
        delta = torch.matmul(action, self.action_deltas.view(-1, 1)).view(-1, 1, 1)
        return torch.clamp(state + delta, -1.0, 1.0)

    @torch.inference_mode()
    def rollout(self, input: RolloutInput) -> RolloutOutput:
        current = input.latent_state.to(self.device, self.dtype)
        actions = input.actions.to(self.device, self.dtype)
        predicted_states = []
        for step_idx in range(input.num_steps):
            current = self.predict_next(current, actions[:, step_idx, :])
            predicted_states.append(current)
        return RolloutOutput(predicted_states=torch.stack(predicted_states, dim=1))

    def get_initial_state(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.to(self.device, self.dtype)


__all__ = [
    "ToyContinuousWorldModel",
    "ToyLineWorldModel",
    "ToyLineWorldSpec",
    "ToyWorldSpec",
]
