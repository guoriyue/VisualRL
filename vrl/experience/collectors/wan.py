"""Wan-specific experience collector for RL training.

Collects rollouts from a Wan model with per-step log-probabilities
using sde_step_with_logprob instead of the standard scheduler.step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from vrl.algorithms.types import Rollout, Trajectory
from vrl.evaluators.diffusion.flow_matching import sde_step_with_logprob
from vrl.experience.types import ExperienceBatch
from vrl.models.families.wan.state import WanDenoiseState
from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WanCollectorConfig:
    """Configuration for WanCollector."""

    num_steps: int = 50
    guidance_scale: float = 5.0
    high_noise_guidance_scale: float | None = None
    sample_solver: str = "dpmpp"
    shift: float = 5.0
    offload_model: bool = False
    cfg: bool = True


class WanCollector:
    """Collect Wan rollouts with per-step log-probabilities for training.

    Uses sde_step_with_logprob instead of scheduler.step to record
    log-probabilities at each denoising step, which are needed for
    the GRPO clipped surrogate loss.

    Reuses ``OfficialWanModel.encode_text()``, ``denoise_init()``, and
    ``decode_vae()`` — only the inner denoise loop is custom.
    """

    def __init__(
        self,
        wan_model: Any,  # OfficialWanModel
        reward_fn: RewardFunction,
        config: WanCollectorConfig | None = None,
    ) -> None:
        self.wan_model = wan_model
        self.reward_fn = reward_fn
        self.config = config or WanCollectorConfig()

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        """Collect Wan rollouts with per-step log-probabilities.

        Steps:
        1. encode_text() for each prompt
        2. denoise_init() -> WanDenoiseState
        3. Custom denoise loop with sde_step_with_logprob
        4. decode_vae() -> videos
        5. reward_fn.score() -> rewards
        6. Stack into ExperienceBatch
        """
        import torch

        wan = self.wan_model
        cfg = self.config

        # Build a minimal request for encode_text / denoise_init
        request = kwargs.get("request")
        if request is None:
            raise ValueError(
                "WanCollector.collect() requires a 'request' kwarg "
                "(VideoGenerationRequest) for encode_text/denoise_init."
            )

        # 1. Encode text
        state: dict[str, Any] = {}
        encode_result = await wan.encode_text(request, state)
        state.update(encode_result.state_updates or {})

        # Encode conditioning if i2v
        if request.task_type == "image_to_video":
            cond_result = await wan.encode_conditioning(request, state)
            state.update(cond_result.state_updates or {})

        # 2. Initialize denoising
        denoise_loop = await wan.denoise_init(request, state)
        ms: WanDenoiseState = denoise_loop.model_state

        pipeline = ms.pipeline
        total_steps = denoise_loop.total_steps

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []  # x_t at each step
        all_actions = []       # x_{t-1} at each step
        all_log_probs = []     # log_prob at each step
        all_timesteps = []     # timestep values

        latents = ms.latents
        is_list_latents = isinstance(latents, list)

        with (
            torch.amp.autocast("cuda", dtype=pipeline.param_dtype),
            torch.no_grad(),
        ):
            for step_idx in range(total_steps):
                t = ms.timesteps[step_idx]
                timestep = torch.stack([t]).to(pipeline.device)

                # Save current latents as observation
                current_latents = latents[0] if is_list_latents else latents

                # Select expert model by boundary
                if t.item() >= ms.boundary:
                    model = pipeline.high_noise_model
                    sample_guide_scale = ms.high_noise_guidance_scale
                else:
                    model = pipeline.low_noise_model
                    sample_guide_scale = ms.low_noise_guidance_scale

                # Ensure model is on correct device
                if next(model.parameters()).device.type == "cpu":
                    model.to(pipeline.device)

                # Forward pass with CFG
                if ms.task_key.startswith("t2v-"):
                    model_input = latents
                    noise_pred_cond = model(model_input, t=timestep, **ms.arg_c)[0]
                    noise_pred_uncond = model(model_input, t=timestep, **ms.arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    # SDE step with log-prob (replaces scheduler.step)
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.unsqueeze(0),
                        t.unsqueeze(0),
                        current_latents.unsqueeze(0),
                        return_dt=True,
                    )
                    next_latents = sde_result.prev_sample.squeeze(0)
                    latents = [next_latents]
                else:
                    latent_model_input = [current_latents.to(pipeline.device)]
                    noise_pred_cond = model(latent_model_input, t=timestep, **ms.arg_c)[0]
                    noise_pred_uncond = model(latent_model_input, t=timestep, **ms.arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.unsqueeze(0),
                        t.unsqueeze(0),
                        current_latents.unsqueeze(0),
                        return_dt=True,
                    )
                    next_latents = sde_result.prev_sample.squeeze(0)
                    latents = next_latents

                all_observations.append(current_latents.detach())
                all_actions.append(next_latents.detach())
                all_log_probs.append(sde_result.log_prob.detach())
                all_timesteps.append(t.detach())

        # Stack trajectories: [T, ...] -> add batch dim [1, T, ...]
        observations = torch.stack(all_observations, dim=0).unsqueeze(0)  # [1, T, ...]
        actions = torch.stack(all_actions, dim=0).unsqueeze(0)            # [1, T, ...]
        log_probs = torch.stack(all_log_probs, dim=0).unsqueeze(0)       # [1, T]
        timesteps_tensor = torch.stack(all_timesteps, dim=0).unsqueeze(0) # [1, T]

        # 4. Decode VAE -> video
        final_latents = latents if isinstance(latents, list) else [latents]
        state["latents"] = final_latents
        state["fps"] = pipeline.config.sample_fps
        decode_result = await wan.decode_vae(request, state)
        state.update(decode_result.state_updates or {})
        video = state["video_tensor"]  # [C, T, H, W]

        # 5. Score with reward function
        # Wrap into a Rollout for backward-compatible reward scoring
        dummy_trajectory = Trajectory(
            prompt=prompts[0],
            seed=ms.seed,
            steps=[],
            output=video,
        )
        dummy_rollout = Rollout(
            request=request,
            trajectory=dummy_trajectory,
        )
        reward = await self.reward_fn.score(dummy_rollout)

        # 6. Build ExperienceBatch
        rewards = torch.tensor([reward], dtype=torch.float32, device=observations.device)
        dones = torch.ones(1, dtype=torch.bool, device=observations.device)
        group_ids = torch.zeros(1, dtype=torch.long, device=observations.device)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            group_ids=group_ids,
            extras={
                "log_probs": log_probs,
                "timesteps": timesteps_tensor,
                "scheduler": ms.scheduler,
                "arg_c": ms.arg_c,
                "arg_null": ms.arg_null,
                "boundary": ms.boundary,
                "task_key": ms.task_key,
                "pipeline": pipeline,
                "guidance_scale": ms.high_noise_guidance_scale,
            },
            videos=video.unsqueeze(0),  # [1, C, T, H, W]
            prompts=prompts,
        )
