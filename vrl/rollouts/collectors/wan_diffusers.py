"""Wan 1.3B Diffusers-based collector for RL training.

Uses the model-family step runtime (``denoise_init`` / ``predict_noise`` /
``decode_vae_for_latents``) instead of raw pipeline access, so that
model-specific knowledge (CFG kwargs, VAE normalization) stays inside
the model layer.

Follows the same architecture as ``CosmosDiffusersCollector``:
  Collector → FlowMatchingEvaluator → GRPO → OnlineTrainer
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
    """Collect rollouts from Wan 1.3B with per-step log-probabilities.

    Delegates model-specific forward passes to the model family's
    ``denoise_init`` / ``predict_noise`` / ``decode_vae_for_latents``
    methods.  The collector only owns the SDE-step loop, reward scoring,
    and ``ExperienceBatch`` assembly.

    Implements both ``collect()`` (rollout) and ``forward_step()``
    (single-timestep forward for training evaluator).
    """

    def __init__(
        self,
        model: Any,  # DiffusersWanT2VModel
        reward_fn: Any,  # RewardFunction instance
        config: WanDiffusersCollectorConfig | None = None,
    ) -> None:
        self.model = model
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
        1. Encode text via model family
        2. denoise_init via model family (prepares latents + scheduler)
        3. Custom SDE loop: model.predict_noise per step + sde_step_with_logprob
        4. Decode VAE via model family
        5. Reward scoring
        6. Stack into ExperienceBatch
        """
        import torch

        from vrl.algorithms.types import Rollout, Trajectory
        from vrl.models.base import VideoGenerationRequest
        from vrl.rollouts.evaluators.diffusion.flow_matching import sde_step_with_logprob

        cfg = self.config
        batch_size = len(prompts)

        # Build request
        request = VideoGenerationRequest(
            prompt=prompts[0] if len(prompts) == 1 else prompts[0],
            num_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            frame_count=cfg.num_frames,
            extra={"max_sequence_length": cfg.max_sequence_length},
        )

        # 1. Encode text via model family
        state: dict[str, Any] = {}
        encode_result = await self.model.encode_text(request, state)
        state.update(encode_result.state_updates)

        # 2. denoise_init via model family
        denoise_loop = await self.model.denoise_init(request, state)
        ms = denoise_loop.model_state

        # SDE window
        sde_window = self._get_sde_window()

        # Same-latent generator
        device = ms.latents.device
        if cfg.same_latent:
            latent_generator = torch.Generator(device=device)
            latent_generator.manual_seed(hash(prompts[0]) % (2**32))
        else:
            latent_generator = None

        # 3. Custom denoise loop with log-prob tracking
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_kls = []
        all_timestep_values = []

        do_cfg = cfg.cfg and cfg.guidance_scale > 1.0
        transformer_dtype = ms.prompt_embeds.dtype

        with torch.amp.autocast("cuda", dtype=transformer_dtype):
            with torch.no_grad():
                for step_idx in range(denoise_loop.total_steps):
                    latents_ori = ms.latents.clone()
                    t = ms.timesteps[step_idx]

                    # Forward pass via model family
                    fwd = await self.model.predict_noise(denoise_loop, step_idx)
                    noise_pred = fwd["noise_pred"]

                    # Check SDE window
                    in_sde_window = sde_window is None or (
                        sde_window[0] <= step_idx < sde_window[1]
                    )

                    # SDE step with log-prob
                    sde_result = sde_step_with_logprob(
                        ms.scheduler,
                        noise_pred.float(),
                        t.unsqueeze(0),
                        ms.latents.float(),
                        generator=latent_generator if in_sde_window else None,
                        deterministic=not in_sde_window,
                        return_dt=cfg.kl_reward > 0,
                    )
                    prev_latents = sde_result.prev_sample
                    ms.latents = prev_latents

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

                    denoise_loop.current_step = step_idx + 1

        # Stack: [T, B, ...] -> [B, T, ...]
        observations = torch.stack(all_observations, dim=1)
        actions = torch.stack(all_actions, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        timesteps_tensor = torch.stack(
            [tv.expand(batch_size) for tv in all_timestep_values], dim=1
        )
        kl_tensor = torch.stack(all_kls, dim=1)

        # 4. Decode VAE via model family
        video = await self.model.decode_vae_for_latents(ms.latents)

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

        # 6. Subtract kl_reward * kl from rewards
        rewards_raw = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        if cfg.kl_reward > 0:
            kl_total = kl_tensor.sum(dim=1)
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
                "prompt_embeds": ms.prompt_embeds,
                "negative_prompt_embeds": ms.negative_prompt_embeds,
            },
            context={
                "guidance_scale": ms.guidance_scale,
                "cfg": do_cfg,
                "model_family": "wan-diffusers-t2v",
            },
            videos=video,
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
        current policy during training.  Delegates to the model family's
        ``_predict_noise_with_model``.
        """
        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        # Read per-sample tensors from extras
        timesteps = batch.extras["timesteps"]
        prompt_embeds = batch.extras["prompt_embeds"]
        negative_prompt_embeds = batch.extras["negative_prompt_embeds"]
        latents = batch.observations[:, timestep_idx]

        # Read shared metadata from context
        ctx = batch.context
        guidance_scale = ctx["guidance_scale"]
        do_cfg = ctx["cfg"]

        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        # Reconstruct DiffusersDenoiseState for the model family
        ms = DiffusersDenoiseState(
            latents=latents,
            timesteps=t,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg and guidance_scale > 1.0,
            model_family="wan-diffusers-t2v",
        )

        ds = DenoiseLoopState(current_step=0, total_steps=1, model_state=ms)
        return self.model._predict_noise_with_model(model, ds, step_idx=0)
