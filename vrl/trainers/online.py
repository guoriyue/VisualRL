"""Online RL trainer — collect -> evaluate -> advantage -> loss -> backward -> step.

Supports two modes:
- Legacy: RolloutSource + LogProbComputer (original flow_grpo port)
- New 4-layer: Collector + Adapter + Evaluator + Algorithm pipeline
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import RolloutBatch, RolloutGroup, TrainStepMetrics
from vrl.rewards.base import RewardFunction
from vrl.trainers.base import Trainer
from vrl.trainers.ema import EMAModuleWrapper
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import WeightSyncer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy Protocols (kept for backward compat)
# ---------------------------------------------------------------------------

@runtime_checkable
class RolloutSource(Protocol):
    """Protocol for rollout collectors (implemented in ``rollout/`` later)."""

    async def collect(self, prompts: list[str], **kwargs: Any) -> list[RolloutGroup]:
        ...


class LogProbComputer(Protocol):
    """Protocol for computing fresh log-probs under the current policy."""

    def compute_log_prob(
        self,
        model: nn.Module,
        samples: dict[str, torch.Tensor],
        timestep_idx: int,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, prev_sample_mean, std_dev_t, dt)."""
        ...


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def _create_optimizer(
    parameters: Any,
    config: TrainerConfig,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer matching flow_grpo defaults."""
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Install bitsandbytes for 8-bit Adam: pip install bitsandbytes"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
        parameters,
        lr=config.lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------

def _get_autocast(config: TrainerConfig, device: torch.device) -> Any:
    """Return an autocast context manager matching flow_grpo's mixed precision."""
    if config.mixed_precision == "fp16":
        return torch.amp.autocast(str(device), dtype=torch.float16)
    elif config.mixed_precision == "bf16":
        return torch.amp.autocast(str(device), dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# OnlineTrainer
# ---------------------------------------------------------------------------

class OnlineTrainer(Trainer):
    """Orchestrates the full online RL loop.

    Supports two modes:

    **New 4-layer mode** (preferred): pass ``collector``,
    ``evaluator``, and ``algorithm`` — the trainer just orchestrates
    collect -> evaluate -> advantage -> loss -> backward -> step.

    **Legacy mode**: pass ``rollout_source``, ``reward_fn``, and optionally
    ``log_prob_computer`` for the original flow_grpo-style training loop.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        # -- New 4-layer mode --
        collector: Any | None = None,      # Collector protocol
        evaluator: Any | None = None,      # Evaluator protocol
        # -- Legacy mode --
        reward_fn: RewardFunction | None = None,
        rollout_source: RolloutSource | None = None,
        log_prob_computer: LogProbComputer | None = None,
        # -- Common --
        model: nn.Module | None = None,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        config: TrainerConfig | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.algorithm = algorithm
        # New 4-layer components
        self.collector = collector
        self.evaluator = evaluator
        # Legacy components
        self.reward_fn = reward_fn
        self.rollout_source = rollout_source
        self.log_prob_computer = log_prob_computer
        # Common
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        self.config = config or TrainerConfig()
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()

        # Optimizer — created lazily when model is available
        self._optimizer: torch.optim.Optimizer | None = None

        # EMA — created lazily
        self._ema: EMAModuleWrapper | None = None

        # TF32
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            if self.model is None:
                raise RuntimeError("Cannot train without a model — pass model= to OnlineTrainer")
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = _create_optimizer(trainable, self.config)
        return self._optimizer

    def _ensure_ema(self) -> EMAModuleWrapper | None:
        if not self.config.ema:
            return None
        if self._ema is None:
            if self.model is None:
                return None
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._ema = EMAModuleWrapper(
                trainable,
                decay=self.config.ema_decay,
                update_step_interval=self.config.ema_update_interval,
                device=self.device,
            )
        return self._ema

    @property
    def _is_4layer_mode(self) -> bool:
        """Check if we're in new 4-layer mode."""
        return self.collector is not None and self.evaluator is not None

    # ------------------------------------------------------------------
    # Training step (dispatches to new or legacy path)
    # ------------------------------------------------------------------

    async def step(self, prompts: list[str] | None = None) -> TrainStepMetrics:
        """Run one full training step.

        In 4-layer mode: collect -> evaluate -> advantage -> loss -> backward -> step.
        In legacy mode: collect -> reward -> advantage -> train (original flow_grpo).
        """
        if prompts is not None:
            self.prompts = prompts

        if self._is_4layer_mode:
            metrics = await self._step_4layer()
        else:
            metrics = await self._step_legacy()

        # Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        # Sync weights
        if self.weight_syncer is not None and self.model is not None:
            state_dict = self.model.state_dict()
            await self.weight_syncer.push(state_dict)

        return metrics

    # ------------------------------------------------------------------
    # New 4-layer training step
    # ------------------------------------------------------------------

    async def _step_4layer(self) -> TrainStepMetrics:
        """Pure orchestrator: collect -> evaluate -> advantage -> loss."""
        from vrl.rollouts.evaluators.types import SignalRequest

        cfg = self.config
        assert self.model is not None
        assert self.collector is not None
        assert self.evaluator is not None

        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        # 1. Collect experience
        batch = await self.collector.collect(self.prompts)

        # 2. Compute advantages
        advantages = self.algorithm.compute_advantages_from_tensors(
            batch.rewards, batch.group_ids
        )

        # 3. Train loop
        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device)
        agg_metrics: dict[str, list[float]] = defaultdict(list)

        num_timesteps = batch.observations.shape[1]
        # Select timesteps to train on (fraction)
        train_timestep_count = max(1, int(num_timesteps * cfg.timestep_fraction))
        # Evenly spaced timestep indices
        if train_timestep_count < num_timesteps:
            step_size = num_timesteps / train_timestep_count
            train_indices = [int(i * step_size) for i in range(train_timestep_count)]
        else:
            train_indices = list(range(num_timesteps))

        old_log_probs = batch.extras["log_probs"]

        for _inner_epoch in range(cfg.num_inner_epochs):
            for j in train_indices:
                with autocast_ctx:
                    signals = self.evaluator.evaluate(
                        self.collector,
                        self.model,
                        batch,
                        j,
                        ref_model=self.ref_model,
                        signal_request=SignalRequest(
                            need_ref=cfg.beta > 0,
                            need_kl_intermediates=cfg.beta > 0,
                        ),
                    )

                    old_lp_j = old_log_probs[:, j] if old_log_probs.ndim > 1 else old_log_probs
                    loss, metrics = self.algorithm.compute_signal_loss(
                        signals, advantages, old_lp_j
                    )

                # backward
                loss.backward()

                agg_metrics["loss"].append(metrics.loss)
                agg_metrics["policy_loss"].append(metrics.policy_loss)
                agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                agg_metrics["clip_fraction"].append(metrics.clip_fraction)

            # gradient clipping + optimizer step
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # EMA update
            if ema is not None:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                ema.step(trainable, self.state.global_step)

            self.state.global_step += 1

        # Aggregate metrics
        n_steps = max(len(agg_metrics.get("loss", [])), 1)
        avg = lambda key: sum(agg_metrics.get(key, [0.0])) / n_steps

        reward_mean = batch.rewards.mean().item()
        reward_std = batch.rewards.std().item() if batch.rewards.numel() > 1 else 0.0
        adv_mean = advantages.mean().item()

        return TrainStepMetrics(
            loss=avg("loss"),
            policy_loss=avg("policy_loss"),
            kl_penalty=avg("kl_penalty"),
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=adv_mean,
            clip_fraction=avg("clip_fraction"),
        )

    # ------------------------------------------------------------------
    # Legacy training step
    # ------------------------------------------------------------------

    async def _step_legacy(self) -> TrainStepMetrics:
        """Original flow_grpo-style training loop."""
        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        assert self.rollout_source is not None
        assert self.reward_fn is not None

        # 1. Collect rollouts
        groups = await self.rollout_source.collect(self.prompts)

        # 2. Score each rollout
        for group in groups:
            scores = await self.reward_fn.score_batch(group.rollouts)
            for rollout, score in zip(group.rollouts, scores):
                rollout.reward = score

        # 3. Compute advantages
        for group in groups:
            group.advantages = self.algorithm.compute_advantages(group)

        # 4. Train
        batch = RolloutBatch(groups=groups)
        metrics = self._train_on_batch(batch, optimizer, ema)

        return metrics

    # ------------------------------------------------------------------
    # Inner training loop — legacy path (ported from train_wan2_1.py)
    # ------------------------------------------------------------------

    def _train_on_batch(
        self,
        batch: RolloutBatch,
        optimizer: torch.optim.Optimizer,
        ema: EMAModuleWrapper | None,
    ) -> TrainStepMetrics:
        """Backward + step loop over a RolloutBatch (legacy path)."""
        cfg = self.config
        assert self.model is not None

        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device)
        info: dict[str, list[float]] = defaultdict(list)

        all_rewards: list[float] = []
        all_advantages: list[float] = []

        for _inner_epoch in range(cfg.num_inner_epochs):
            for group in batch.groups:
                if group.advantages is None:
                    continue
                for rollout, adv_val in zip(group.rollouts, group.advantages.values):
                    all_rewards.append(rollout.reward)
                    all_advantages.append(adv_val)

                    adv = torch.tensor(adv_val, device=self.device)
                    adv = torch.clamp(adv, -cfg.adv_clip_max, cfg.adv_clip_max)

                    for step in rollout.trajectory.steps:
                        with autocast_ctx:
                            old_lp = torch.tensor(step.log_prob, device=self.device)

                            if step.new_log_prob is not None:
                                new_lp = torch.tensor(
                                    step.new_log_prob, device=self.device,
                                    requires_grad=True,
                                )
                            else:
                                new_lp = old_lp.clone().requires_grad_(True)

                            ratio = torch.exp(new_lp - old_lp)
                            clipped_ratio = torch.clamp(
                                ratio,
                                1.0 - cfg.clip_range,
                                1.0 + cfg.clip_range,
                            )
                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * clipped_ratio
                            policy_loss = torch.maximum(unclipped_loss, clipped_loss)

                            kl_loss = torch.tensor(0.0, device=self.device)
                            if cfg.beta > 0 and self.ref_model is not None:
                                ref_lp_val = (
                                    step.ref_log_prob
                                    if step.ref_log_prob is not None
                                    else step.log_prob
                                )
                                ref_lp = torch.tensor(ref_lp_val, device=self.device)
                                kl_loss = old_lp - ref_lp

                            loss = policy_loss + cfg.beta * kl_loss

                        loss.backward()

                        info["policy_loss"].append(policy_loss.item())
                        info["loss"].append(loss.item())
                        if cfg.beta > 0:
                            info["kl_loss"].append(kl_loss.item())
                        info["approx_kl"].append(
                            0.5 * (new_lp - old_lp).detach().pow(2).item()
                        )
                        info["clipfrac"].append(
                            float(torch.abs(ratio.detach() - 1.0) > cfg.clip_range)
                        )

            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                ema.step(trainable, self.state.global_step)

            self.state.global_step += 1

        # aggregate metrics
        n_rewards = len(all_rewards)
        reward_mean = sum(all_rewards) / max(n_rewards, 1)
        reward_var = sum((r - reward_mean) ** 2 for r in all_rewards) / max(n_rewards, 1)
        adv_mean = sum(all_advantages) / max(len(all_advantages), 1)

        n_steps = max(len(info.get("loss", [])), 1)
        avg = lambda key: sum(info.get(key, [0.0])) / n_steps

        return TrainStepMetrics(
            loss=avg("loss"),
            policy_loss=avg("policy_loss"),
            kl_penalty=avg("kl_loss"),
            reward_mean=reward_mean,
            reward_std=math.sqrt(reward_var),
            advantage_mean=adv_mean,
            clip_fraction=avg("clipfrac"),
        )

    # ------------------------------------------------------------------
    # Tensor-based training (legacy — for real model forward passes)
    # ------------------------------------------------------------------

    def train_on_samples(
        self,
        samples: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        train_timesteps: list[int],
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Low-level training loop matching flow_grpo exactly (legacy path).

        Call this if you've already collected samples as tensors
        (e.g., from a pipeline_with_logprob call).
        """
        cfg = self.config
        assert self.model is not None
        assert self.log_prob_computer is not None

        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device)
        info: dict[str, list[torch.Tensor]] = defaultdict(list)

        for j in train_timesteps:
            with autocast_ctx:
                log_prob, prev_sample_mean, std_dev_t, dt = (
                    self.log_prob_computer.compute_log_prob(
                        self.model, samples, j, prompt_embeds, negative_embeds,
                    )
                )

                if cfg.beta > 0 and self.ref_model is not None:
                    with torch.no_grad():
                        _, prev_sample_mean_ref, _, dt_ref = (
                            self.log_prob_computer.compute_log_prob(
                                self.ref_model, samples, j,
                                prompt_embeds, negative_embeds,
                            )
                        )

                adv = torch.clamp(
                    advantages[:, j], -cfg.adv_clip_max, cfg.adv_clip_max
                )
                ratio = torch.exp(log_prob - samples["log_probs"][:, j])
                unclipped_loss = -adv * ratio
                clipped_loss = -adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
                )
                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                if cfg.beta > 0 and self.ref_model is not None:
                    kl = (
                        (prev_sample_mean - prev_sample_mean_ref) ** 2
                    ).mean(dim=tuple(range(1, prev_sample_mean.ndim)), keepdim=True) / (
                        2 * (std_dev_t * dt_ref) ** 2
                    )
                    kl_loss = torch.mean(kl)
                    loss = policy_loss + cfg.beta * kl_loss
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    loss = policy_loss

            loss.backward()

            info["approx_kl"].append(
                0.5 * torch.mean((log_prob - samples["log_probs"][:, j]) ** 2)
            )
            info["clipfrac"].append(
                torch.mean((torch.abs(ratio - 1.0) > cfg.clip_range).float())
            )
            info["policy_loss"].append(policy_loss)
            if cfg.beta > 0:
                info["kl_loss"].append(kl_loss)
            info["loss"].append(loss)

        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if ema is not None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            ema.step(trainable, self.state.global_step)
        self.state.global_step += 1

        result = {}
        for k, v_list in info.items():
            result[k] = torch.mean(torch.stack(v_list)).item()
        return result

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict[str, Any] = {
            "step": self.state.step,
            "global_step": self.state.global_step,
            "total_reward": self.state.total_reward,
            "total_loss": self.state.total_loss,
        }
        if self._optimizer is not None:
            d["optimizer"] = self._optimizer.state_dict()
        if self._ema is not None:
            d["ema"] = self._ema.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.state.step = state.get("step", 0)
        self.state.global_step = state.get("global_step", 0)
        self.state.total_reward = state.get("total_reward", 0.0)
        self.state.total_loss = state.get("total_loss", 0.0)
        if "optimizer" in state and self._optimizer is not None:
            self._optimizer.load_state_dict(state["optimizer"])
        if "ema" in state and self._ema is not None:
            self._ema.load_state_dict(state["ema"])
