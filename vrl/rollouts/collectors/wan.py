"""Wan-specific experience collector for RL training.

Collects rollouts from a Wan model with per-step log-probabilities
using sde_step_with_logprob instead of the standard scheduler.step.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import copy
from dataclasses import replace

from vrl.algorithms.types import Rollout, Trajectory
from vrl.models.base import VideoGenerationRequest
from vrl.rollouts.evaluators.diffusion.flow_matching import SDEStepResult, sde_step_with_logprob
from vrl.rollouts.types import ExperienceBatch
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

    # Gap 3: KL reward — subtract kl_reward * kl from rewards before advantages.
    # Port from flow_grpo train_wan2_1.py:788:
    #   samples["rewards"]["avg"] = ... - config.sample.kl_reward * samples["kl"]
    kl_reward: float = 0.0

    # Gap 4: SDE window — only inject SDE noise for steps within the window.
    # Port from flow_grpo sd3_pipeline_with_logprob_fast.py:135-142.
    # sde_window_size=0 means all steps use SDE.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Gap 5: Same latent — reuse the same noise for samples sharing a prompt.
    # Port from flow_grpo config.sample.same_latent.
    same_latent: bool = False


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
        request_template: VideoGenerationRequest | None = None,
    ) -> None:
        self.wan_model = wan_model
        self.reward_fn = reward_fn
        self.config = config or WanCollectorConfig()
        self.request_template = request_template

    def _get_sde_window(self) -> tuple[int, int] | None:
        """Compute random SDE window for this collection.

        Returns None if sde_window_size=0 (all steps use SDE).
        """
        cfg = self.config
        if cfg.sde_window_size <= 0:
            return None
        lo, hi = cfg.sde_window_range
        start = random.randint(lo, max(lo, hi - cfg.sde_window_size))
        end = start + cfg.sde_window_size
        return (start, end)

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
        6. (Gap 3) Subtract kl_reward * kl from rewards
        7. Stack into ExperienceBatch
        """
        import torch

        wan = self.wan_model
        cfg = self.config

        # Build a request: use explicit kwarg, or clone from template with prompt
        request = kwargs.get("request")
        if request is None and self.request_template is not None:
            request = replace(self.request_template, prompt=prompts[0])
        if request is None:
            raise ValueError(
                "WanCollector.collect() requires either a 'request' kwarg "
                "or a request_template at init time."
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

        # Gap 4: Compute SDE window
        sde_window = self._get_sde_window()

        # Gap 5: Same-latent generator
        if cfg.same_latent:
            latent_generator = torch.Generator(device=pipeline.device)
            latent_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            latent_generator = None

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []  # x_t at each step
        all_actions = []       # x_{t-1} at each step
        all_log_probs = []     # log_prob at each step
        all_kls = []           # per-step KL (for kl_reward)
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
                else:
                    latent_model_input = [current_latents.to(pipeline.device)]
                    noise_pred_cond = model(latent_model_input, t=timestep, **ms.arg_c)[0]
                    noise_pred_uncond = model(latent_model_input, t=timestep, **ms.arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # Gap 4: Check if this step is inside the SDE window
                in_sde_window = sde_window is None or (sde_window[0] <= step_idx < sde_window[1])

                if in_sde_window:
                    # SDE step with log-prob (replaces scheduler.step)
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.unsqueeze(0),
                        t.unsqueeze(0),
                        current_latents.unsqueeze(0),
                        generator=latent_generator,
                        return_dt=cfg.kl_reward > 0,
                    )
                else:
                    # Deterministic step outside SDE window — no noise, zero log_prob
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.unsqueeze(0),
                        t.unsqueeze(0),
                        current_latents.unsqueeze(0),
                        deterministic=True,
                        return_dt=cfg.kl_reward > 0,
                    )

                next_latents = sde_result.prev_sample.squeeze(0)

                if ms.task_key.startswith("t2v-"):
                    latents = [next_latents]
                else:
                    latents = next_latents

                all_observations.append(current_latents.detach())
                all_actions.append(next_latents.detach())
                all_log_probs.append(sde_result.log_prob.detach())
                all_timesteps.append(t.detach())

                # Gap 3: Track per-step KL for kl_reward
                # The KL is approximated as the log_prob itself (||noise||^2 term)
                all_kls.append(sde_result.log_prob.detach().abs())

        # Stack trajectories: [T, ...] -> add batch dim [1, T, ...]
        observations = torch.stack(all_observations, dim=0).unsqueeze(0)  # [1, T, ...]
        actions = torch.stack(all_actions, dim=0).unsqueeze(0)            # [1, T, ...]
        log_probs = torch.stack(all_log_probs, dim=0).unsqueeze(0)       # [1, T]
        timesteps_tensor = torch.stack(all_timesteps, dim=0).unsqueeze(0) # [1, T]
        kl_tensor = torch.stack(all_kls, dim=0).unsqueeze(0)             # [1, T]

        # 4. Decode VAE -> video
        final_latents = latents if isinstance(latents, list) else [latents]
        state["latents"] = final_latents
        state["fps"] = pipeline.config.sample_fps
        decode_result = await wan.decode_vae(request, state)
        state.update(decode_result.state_updates or {})
        video = state["video_tensor"]  # [C, T, H, W]

        # 5. Score with reward function
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

        # 6. Gap 3: Subtract kl_reward * kl from reward
        # Port from flow_grpo train_wan2_1.py:788:
        #   rewards = rewards - kl_reward * kl_values
        kl_total = kl_tensor.sum().item()
        reward_adjusted = reward - cfg.kl_reward * kl_total

        # 7. Build ExperienceBatch
        rewards = torch.tensor([reward_adjusted], dtype=torch.float32, device=observations.device)
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
                "kl": kl_tensor,
                "reward_before_kl": torch.tensor([reward], dtype=torch.float32),
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

    # ------------------------------------------------------------------
    # forward_step — used by Evaluator during training
    # ------------------------------------------------------------------

    def forward_step(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Wan-specific forward: dual-expert selection + CFG.

        Used by the evaluator to compute fresh log-probs under the
        current policy during training.
        """
        import torch

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps
        timestep_value = t[0].item() if t.ndim > 0 else t.item()

        boundary = batch.extras["boundary"]
        task_key = batch.extras["task_key"]
        arg_c = batch.extras["arg_c"]
        arg_null = batch.extras["arg_null"]
        guidance_scale = batch.extras["guidance_scale"]
        pipeline = batch.extras["pipeline"]

        # Select expert by boundary
        if hasattr(model, "high_noise_model"):
            active_model = model.high_noise_model if timestep_value >= boundary else model.low_noise_model
        else:
            active_model = model

        # Prepare latents
        latents = batch.observations[:, timestep_idx]
        timestep_tensor = torch.stack([t[0]]).to(latents.device) if t.ndim > 0 else torch.stack([t]).to(latents.device)

        if task_key.startswith("t2v-"):
            model_input = [latents[i] for i in range(latents.shape[0])] if latents.ndim > 4 else latents
            noise_pred_cond = active_model(model_input, t=timestep_tensor, **arg_c)[0]
            noise_pred_uncond = active_model(model_input, t=timestep_tensor, **arg_null)[0]
        else:
            model_input = [latents.to(pipeline.device)]
            noise_pred_cond = active_model(model_input, t=timestep_tensor, **arg_c)[0]
            noise_pred_uncond = active_model(model_input, t=timestep_tensor, **arg_null)[0]

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }
