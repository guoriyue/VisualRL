"""Action-conditioned Genie runtime adapter for learned env stepping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from wm_infra.backends.genie_runner import GeniePreparedRun, GenieRunner
from wm_infra.models.base import RolloutInput, RolloutOutput, WorldModel


@dataclass(slots=True)
class GenieRLSpec:
    history_frames: int = 4
    spatial_h: int = 16
    spatial_w: int = 16
    vocab_size: int = 262144
    action_dim: int = 5
    maskgit_steps: int = 2
    temperature: float = 0.0
    token_delta: int = 17

    @property
    def frame_token_count(self) -> int:
        return self.spatial_h * self.spatial_w

    @property
    def state_token_count(self) -> int:
        return self.history_frames * self.frame_token_count


class GenieTokenReward:
    """Dense reward on the generated token frame against a goal token frame."""

    def __init__(self, spec: GenieRLSpec, *, success_threshold: float = 0.01, reward_scale: float = 4.0) -> None:
        self.spec = spec
        self.success_threshold = success_threshold
        self.reward_scale = reward_scale

    def evaluate(
        self,
        next_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        frame_tokens = self.spec.frame_token_count
        current_frame = next_state[:, -frame_tokens:, :]
        goal_frame = goal_state[:, -frame_tokens:, :]
        error = (current_frame - goal_frame).abs().mean(dim=(1, 2)) / max(float(self.spec.vocab_size), 1.0)
        reward = -error * self.reward_scale
        terminated = error <= self.success_threshold
        return reward, terminated, {"token_l1": error, "success": terminated.to(torch.float32)}


class GenieWorldModelAdapter(WorldModel):
    """WorldModel-compatible adapter that conditions Genie on explicit token controls."""

    def __init__(
        self,
        runner: GenieRunner | None = None,
        *,
        spec: GenieRLSpec | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.spec = spec or GenieRLSpec()
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.runner = runner or GenieRunner(device=str(self.device))
        if runner is None and self.device.type != "cuda":
            self.runner._mode = "stub"
        else:
            self.runner.load()

    def sample_initial_state(self, *, seed: int | None = None) -> torch.Tensor:
        prepared = self.runner.prepare_inputs(
            prompt="genie_rl_reset",
            seed=0 if seed is None else seed,
            num_frames=self.spec.history_frames + 1,
            num_prompt_frames=self.spec.history_frames,
            maskgit_steps=self.spec.maskgit_steps,
            temperature=self.spec.temperature,
        )
        self._refresh_spec_from_prepared(prepared)
        tokens = prepared.current_tokens_numpy()[: self.spec.history_frames]
        return self._tokens_to_state(tokens)

    def sample_goal_state(self, *, seed: int | None = None) -> torch.Tensor:
        goal_seed = 13 if seed is None else (seed + 997)
        prepared = self.runner.prepare_inputs(
            prompt="genie_rl_goal",
            seed=goal_seed,
            num_frames=self.spec.history_frames + 1,
            num_prompt_frames=self.spec.history_frames,
            maskgit_steps=self.spec.maskgit_steps,
            temperature=self.spec.temperature,
        )
        self._refresh_spec_from_prepared(prepared)
        goal_frame = prepared.current_tokens_numpy()[self.spec.history_frames - 1]
        repeated = np.repeat(goal_frame[None, :, :], self.spec.history_frames, axis=0)
        return self._tokens_to_state(repeated)

    @torch.inference_mode()
    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch_state = latent_state.to(self.device, self.dtype)
        batch_action = action.to(self.device, self.dtype)
        if batch_state.ndim != 3:
            raise ValueError(f"Expected latent_state [B, N, D], got {tuple(batch_state.shape)}")
        if batch_action.ndim != 2 or batch_action.shape[-1] != self.spec.action_dim:
            raise ValueError(f"Expected action [B, {self.spec.action_dim}], got {tuple(batch_action.shape)}")

        prepared_runs: list[GeniePreparedRun] = []
        for index in range(batch_state.shape[0]):
            state_tokens = self._state_to_tokens(batch_state[index:index + 1])
            conditioned_tokens = self._apply_action_control(state_tokens, batch_action[index])
            prepared_runs.append(
                self.runner.prepare_inputs(
                    prompt=f"genie_rl_step_{index}",
                    seed=index,
                    num_frames=self.spec.history_frames + 1,
                    input_tokens=conditioned_tokens,
                    num_prompt_frames=self.spec.history_frames,
                    maskgit_steps=self.spec.maskgit_steps,
                    temperature=self.spec.temperature,
                )
            )

        for prepared in prepared_runs:
            self._refresh_spec_from_prepared(prepared)

        self.runner.run_window_batch(
            prepared_runs,
            frame_start=self.spec.history_frames,
            frame_end=self.spec.history_frames + 1,
        )

        next_states = []
        for prepared in prepared_runs:
            output_tokens = prepared.current_tokens_numpy()
            next_history = output_tokens[1 : self.spec.history_frames + 1]
            next_states.append(self._tokens_to_state(next_history))
        return torch.cat(next_states, dim=0).to(self.device, self.dtype)

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
        if observation.ndim == 4:
            tokens = observation.detach().cpu().numpy()
        elif observation.ndim == 3:
            tokens = observation.unsqueeze(0).detach().cpu().numpy()
        else:
            raise ValueError("GenieWorldModelAdapter expects observation as [T,H,W] or [B,T,H,W]")
        if tokens.shape[-3:] != (self.spec.history_frames, self.spec.spatial_h, self.spec.spatial_w):
            raise ValueError(
                "Genie observation must match [history_frames, spatial_h, spatial_w] = "
                f"[{self.spec.history_frames}, {self.spec.spatial_h}, {self.spec.spatial_w}]"
            )
        tensor = torch.from_numpy(tokens.reshape(tokens.shape[0], self.spec.state_token_count, 1))
        return tensor.to(self.device, self.dtype)

    def _apply_action_control(self, tokens: np.ndarray, action: torch.Tensor) -> np.ndarray:
        controlled = np.array(tokens, copy=True, dtype=np.uint32)
        action_idx = int(torch.argmax(action).item())
        latest = controlled[self.spec.history_frames - 1]
        if action_idx == 1:
            latest = np.roll(latest, shift=-1, axis=1)
        elif action_idx == 2:
            latest = np.roll(latest, shift=1, axis=1)
        elif action_idx == 3:
            latest = ((latest.astype(np.int64) + self.spec.token_delta) % self.spec.vocab_size).astype(np.uint32)
        elif action_idx == 4:
            latest = ((latest.astype(np.int64) - self.spec.token_delta) % self.spec.vocab_size).astype(np.uint32)
        controlled[self.spec.history_frames - 1] = latest
        return controlled

    def _state_to_tokens(self, state: torch.Tensor) -> np.ndarray:
        flattened = state.detach().cpu().numpy().reshape(-1)
        expected = self.spec.state_token_count
        if flattened.size != expected:
            if flattened.size % self.spec.history_frames != 0:
                raise ValueError(
                    f"State token count {flattened.size} is not divisible by history_frames={self.spec.history_frames}"
                )
            frame_token_count = flattened.size // self.spec.history_frames
            inferred_spatial = int(round(frame_token_count ** 0.5))
            if inferred_spatial * inferred_spatial != frame_token_count:
                raise ValueError(f"Cannot infer square Genie grid from frame_token_count={frame_token_count}")
            self.spec.spatial_h = inferred_spatial
            self.spec.spatial_w = inferred_spatial
        tokens = flattened.reshape(self.spec.history_frames, self.spec.spatial_h, self.spec.spatial_w)
        clipped = np.clip(tokens, 0, self.spec.vocab_size - 1).astype(np.uint32)
        return clipped

    def _tokens_to_state(self, tokens: np.ndarray) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected Genie tokens as [T,H,W], got {tuple(tokens.shape)}")
        if tokens.shape[0] != self.spec.history_frames:
            raise ValueError(f"Expected {self.spec.history_frames} history frames, got {tokens.shape[0]}")
        self.spec.spatial_h = int(tokens.shape[1])
        self.spec.spatial_w = int(tokens.shape[2])
        flat = torch.from_numpy(tokens.reshape(1, self.spec.state_token_count, 1).astype(np.float32))
        return flat.to(self.device, self.dtype)

    def _refresh_spec_from_prepared(self, prepared: GeniePreparedRun) -> None:
        self.spec.spatial_h = int(prepared.spatial_h)
        self.spec.spatial_w = int(prepared.spatial_w)
        self.spec.vocab_size = int(prepared.vocab_size)


__all__ = [
    "GenieRLSpec",
    "GenieTokenReward",
    "GenieWorldModelAdapter",
]
