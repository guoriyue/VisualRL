"""Wan 1.3B Diffusers-based collector for RL training.

Uses HuggingFace ``diffusers.WanPipeline`` (single transformer, no dual-expert)
with ``sde_step_with_logprob`` for per-step log-probability tracking.

Targets ``Wan2.1-T2V-1.3B-Diffusers`` for single-GPU training.

Ported from flow_grpo/scripts/train_wan2_1.py +
flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vrl.rollouts.types import ExperienceBatch

if TYPE_CHECKING:
    from vrl.rewards.base import RewardFunction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WanDiffusersCollectorConfig:
    """Configuration for WanDiffusersCollector."""

    num_steps: int = 20
    guidance_scale: float = 4.5
    height: int = 240
    width: int = 416
    num_frames: int = 33
    max_sequence_length: int = 512

    # CFG during sampling
    cfg: bool = True

    # KL reward — subtract kl_reward * kl from rewards before advantages.
    kl_reward: float = 0.0

    # SDE window — only inject SDE noise for steps within the window.
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)

    # Same latent — reuse the same noise for samples sharing a prompt.
    same_latent: bool = False


class WanDiffusersCollector:
    """Collect rollouts from a Diffusers WanPipeline with per-step log-probabilities.

    Unlike ``WanCollector`` (which uses the official Wan repo with dual-expert
    models), this collector uses the HuggingFace ``diffusers.WanPipeline`` with
    a single transformer. This targets the smaller ``Wan2.1-T2V-1.3B-Diffusers``
    model that fits on a single 5090 GPU.

    Implements both ``collect()`` (rollout) and ``forward_step()`` (single-timestep
    forward for training evaluator).
    """

    def __init__(
        self,
        pipeline: Any,  # diffusers.WanPipeline
        reward_fn: Any,  # RewardFunction instance
        config: WanDiffusersCollectorConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.reward_fn = reward_fn
        self.config = config or WanDiffusersCollectorConfig()

    def _get_sde_window(self) -> tuple[int, int] | None:
        """Compute random SDE window for this collection."""
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
        """Collect Wan Diffusers rollouts with per-step log-probabilities.

        Steps:
        1. Encode text (prompt + negative prompt)
        2. Prepare latents and scheduler timesteps
        3. Custom denoise loop with sde_step_with_logprob
        4. Decode VAE -> video
        5. Reward scoring
        6. Stack into ExperienceBatch
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        pipe = self.pipeline
        cfg = self.config
        device = pipe.device

        # 1. Encode text
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=[""] * len(prompts),
            do_classifier_free_guidance=cfg.cfg and cfg.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            max_sequence_length=cfg.max_sequence_length,
            device=device,
        )
        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        batch_size = len(prompts)

        # 2. Prepare scheduler + latents
        pipe.scheduler.set_timesteps(cfg.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            cfg.height,
            cfg.width,
            cfg.num_frames,
            torch.float32,
            device,
            None,  # generator
            None,  # latents
        )

        # SDE window
        sde_window = self._get_sde_window()

        # Same-latent generator
        if cfg.same_latent:
            latent_generator = torch.Generator(device=device)
            latent_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            latent_generator = None

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []  # x_t at each step
        all_actions = []       # x_{t-1} at each step
        all_log_probs = []     # log_prob at each step
        all_kls = []           # per-step KL (for kl_reward)
        all_timestep_values = []

        do_cfg = cfg.cfg and cfg.guidance_scale > 1.0

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx, t in enumerate(timesteps):
                    latents_ori = latents.clone()
                    latent_model_input = latents.to(transformer_dtype)
                    timestep_batch = t.expand(batch_size)

                    # Forward pass: cond
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep_batch,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.to(prompt_embeds.dtype)

                    # CFG: uncond pass
                    if do_cfg:
                        noise_uncond = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep_batch,
                            encoder_hidden_states=negative_prompt_embeds,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + cfg.guidance_scale * (
                            noise_pred - noise_uncond
                        )

                    # Check SDE window
                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    # SDE step with log-prob
                    sde_result = sde_step_with_logprob(
                        pipe.scheduler,
                        noise_pred.float(),
                        t.unsqueeze(0),
                        latents.float(),
                        generator=latent_generator if in_sde_window else None,
                        deterministic=not in_sde_window,
                        return_dt=cfg.kl_reward > 0,
                    )
                    prev_latents = sde_result.prev_sample
                    latents = prev_latents

                    all_observations.append(latents_ori.detach())
                    all_actions.append(prev_latents.detach())
                    all_log_probs.append(sde_result.log_prob.detach())
                    all_timestep_values.append(t.detach())

                    # Per-step KL tracking for kl_reward
                    if cfg.kl_reward > 0:
                        all_kls.append(sde_result.log_prob.detach().abs())
                    else:
                        all_kls.append(
                            torch.zeros(batch_size, device=device)
                        )

        # Stack: [T, B, ...] -> [B, T, ...]
        observations = torch.stack(all_observations, dim=1)    # [B, T, C, D, H, W]
        actions = torch.stack(all_actions, dim=1)              # [B, T, C, D, H, W]
        log_probs = torch.stack(all_log_probs, dim=1)         # [B, T]
        timesteps_tensor = torch.stack(
            [tv.expand(batch_size) for tv in all_timestep_values], dim=1
        )  # [B, T]
        kl_tensor = torch.stack(all_kls, dim=1)               # [B, T]

        # 4. Decode VAE -> video (matching flow_grpo wan_pipeline_with_logprob)
        latents_for_decode = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents_for_decode.device, latents_for_decode.dtype)
        )
        latents_for_decode = latents_for_decode / latents_std + latents_mean
        video = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
        # Postprocess to [0, 1] range (matching flow_grpo output_type="pt")
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        # video: [B, T, C, H, W] after postprocess — transpose to [B, C, T, H, W]
        video = video.permute(0, 2, 1, 3, 4)

        # 5. Score with reward function
        rewards_list = []
        for i in range(batch_size):
            dummy_trajectory = Trajectory(
                prompt=prompts[i],
                seed=0,
                steps=[],
                output=video[i],
            )
            dummy_rollout = Rollout(
                request=None,
                trajectory=dummy_trajectory,
            )
            r = await self.reward_fn.score(dummy_rollout)
            rewards_list.append(r)

        # 6. Gap 3: Subtract kl_reward * kl from rewards
        rewards_raw = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)  # [B]
            rewards_adjusted = rewards_raw - cfg.kl_reward * kl_total
        else:
            rewards_adjusted = rewards_raw

        # 7. Build ExperienceBatch
        dones = torch.ones(batch_size, dtype=torch.bool, device=device)
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        return ExperienceBatch(
            observations=observations,
            actions=actions,
            rewards=rewards_adjusted,
            dones=dones,
            group_ids=group_ids,
            extras={
                "log_probs": log_probs,
                "timesteps": timesteps_tensor,
                "kl": kl_tensor,
                "reward_before_kl": rewards_raw,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": cfg.guidance_scale,
                "cfg": cfg.cfg,
            },
            videos=video,  # [B, C, T, H, W]
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
        """Wan Diffusers forward: single transformer + optional CFG.

        Used by the evaluator to compute fresh log-probs under the
        current policy during training.
        """
        import torch

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        prompt_embeds = batch.extras["prompt_embeds"]
        negative_prompt_embeds = batch.extras["negative_prompt_embeds"]
        guidance_scale = batch.extras["guidance_scale"]
        do_cfg = batch.extras["cfg"] and guidance_scale > 1.0

        # Prepare latents
        latents = batch.observations[:, timestep_idx]
        latent_input = latents.to(prompt_embeds.dtype)

        # Forward pass: cond
        noise_pred_cond = model(
            hidden_states=latent_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred_cond = noise_pred_cond.to(prompt_embeds.dtype)

        if do_cfg:
            noise_pred_uncond = model(
                hidden_states=latent_input,
                timestep=t,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }
