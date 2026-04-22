"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from vrl.algorithms.base import Algorithm
from vrl.algorithms.types import TrainStepMetrics
from vrl.rollouts.types import ExperienceBatch, stack_batches
from vrl.trainers.base import Trainer
from vrl.trainers.ema import EMAModuleWrapper
from vrl.trainers.types import TrainerConfig, TrainState
from vrl.trainers.weight_sync import WeightSyncer

logger = logging.getLogger(__name__)


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
# Phase profiler
# ---------------------------------------------------------------------------

class PhaseTimer:
    """Accumulating phase timer with optional CUDA sync.

    Each ``time(name)`` call returns a context manager whose wall time is
    added to ``self.times[name]``. When ``sync=True`` and CUDA is available,
    ``torch.cuda.synchronize()`` is called on both ends so async GPU kernels
    are captured.
    """

    def __init__(self, enabled: bool = False, sync: bool = True) -> None:
        self.enabled = enabled
        self.sync = sync and torch.cuda.is_available()
        self.times: dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.times[name] += time.perf_counter() - t0


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
    """Orchestrates the CEA online RL loop.

    Pipeline: collect -> evaluate -> advantage -> loss -> backward -> step.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        collector: Any,
        evaluator: Any,
        model: nn.Module,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        config: TrainerConfig | None = None,
        prompts: list[str] | None = None,
        device: torch.device | str = "cuda",
        accelerator: Any | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.collector = collector
        self.evaluator = evaluator
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        self.config = config or TrainerConfig()
        self.prompts = prompts or []
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        self.accelerator = accelerator

        self._optimizer: torch.optim.Optimizer | None = None
        self._ema: EMAModuleWrapper | None = None

        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._optimizer = _create_optimizer(trainable, self.config)
        return self._optimizer

    def _ensure_ema(self) -> EMAModuleWrapper | None:
        if not self.config.ema:
            return None
        if self._ema is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._ema = EMAModuleWrapper(
                trainable,
                decay=self.config.ema_decay,
                update_step_interval=self.config.ema_update_interval,
                device=self.device,
            )
        return self._ema

    # ------------------------------------------------------------------
    # Accelerator-aware backward/step helpers
    # ------------------------------------------------------------------

    def _backward(self, loss: Any) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def _clip_and_step(self, optimizer: Any) -> float:
        """Clip grads, step optimizer, return pre-clip total grad-norm (float)."""
        cfg = self.config
        grad_norm: Any = 0.0
        if self.accelerator is not None:
            if self.accelerator.sync_gradients and cfg.max_grad_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
        else:
            if cfg.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
            else:
                # no clip — compute norm manually for diagnostic
                sq_sum = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        sq_sum += float(p.grad.detach().pow(2).sum().item())
                grad_norm = sq_sum ** 0.5
        optimizer.step()
        optimizer.zero_grad()
        return float(grad_norm) if hasattr(grad_norm, "__float__") else float(grad_norm)

    # ------------------------------------------------------------------
    # Training step — CEA pipeline
    # ------------------------------------------------------------------

    async def step(self, prompts: list[str] | None = None) -> TrainStepMetrics:
        """Run one full training step: collect -> evaluate -> advantage -> loss -> backward -> step."""
        from vrl.rollouts.evaluators.types import SignalRequest
        from vrl.trainers.data import PromptExample

        if prompts is not None:
            self.prompts = prompts

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()

        timer = PhaseTimer(enabled=cfg.profile)

        # 1. Collect group_size samples per prompt
        all_batches: list[ExperienceBatch] = []
        with timer.time("collect"):
            for prompt_idx, item in enumerate(self.prompts):
                if isinstance(item, PromptExample):
                    prompt_str = item.prompt
                    collect_kwargs: dict[str, Any] = {
                        "target_text": item.target_text,
                        "references": item.references,
                        "task_type": item.task_type,
                        "request_overrides": item.request_overrides,
                        "sample_metadata": item.metadata,
                    }
                else:
                    prompt_str = str(item)
                    collect_kwargs = {}

                # Group-batched collect: one call produces cfg.group_size samples.
                b = await self.collector.collect(
                    [prompt_str],
                    group_size=cfg.group_size,
                    **collect_kwargs,
                )
                b.group_ids[:] = prompt_idx
                all_batches.append(b)

            batch = stack_batches(all_batches)

        # 2. Compute advantages via GRPO algorithm (group-relative, with adv_clip)
        with timer.time("advantage"):
            advantages = self.algorithm.compute_advantages_from_tensors(
                batch.rewards, batch.group_ids,
            )

        # Advantage diagnostics: zero_rate = |adv|<1e-6 (group collapse),
        # saturation = |adv| at adv_clip_max (reward outlier clipped).
        _adv_abs = advantages.detach().abs()
        _total = max(advantages.numel(), 1)
        adv_zero_rate = float((_adv_abs < 1e-6).sum().item()) / _total
        _clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
        adv_saturation = (
            float((_adv_abs >= _clip_max - 1e-6).sum().item()) / _total
            if _clip_max is not None else 0.0
        )

        # 3. Train loop
        self.model.train()
        autocast_ctx = _get_autocast(cfg, self.device)
        agg_metrics: dict[str, list[float]] = defaultdict(list)

        num_timesteps = batch.observations.shape[1]
        train_timestep_count = max(1, int(num_timesteps * cfg.timestep_fraction))
        if train_timestep_count < num_timesteps:
            step_size = num_timesteps / train_timestep_count
            train_indices = [int(i * step_size) for i in range(train_timestep_count)]
        else:
            train_indices = list(range(num_timesteps))

        old_log_probs = batch.extras["log_probs"]

        # Debug first step: compare old vs fresh log-probs on first timestep
        if cfg.debug_first_step and self.state.step == 0:
            with autocast_ctx:
                _dbg_signals = self.evaluator.evaluate(
                    self.collector,
                    self.model,
                    batch,
                    0,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            _old_lp_0 = old_log_probs[:, 0] if old_log_probs.ndim > 1 else old_log_probs
            _diff = (_dbg_signals.log_prob - _old_lp_0).abs()
            logger.info(
                "DEBUG first-step log-prob diff: mean=%.6f max=%.6f | "
                "old_lp[0]=%.6f fresh_lp[0]=%.6f",
                _diff.mean().item(), _diff.max().item(),
                _old_lp_0[0].item(), _dbg_signals.log_prob[0].item(),
            )

        for _inner_epoch in range(cfg.num_inner_epochs):
            for j in train_indices:
                with timer.time("evaluate"):
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

                with timer.time("backward"):
                    self._backward(loss)

                agg_metrics["loss"].append(metrics.loss)
                agg_metrics["policy_loss"].append(metrics.policy_loss)
                agg_metrics["kl_penalty"].append(metrics.kl_penalty)
                agg_metrics["clip_fraction"].append(metrics.clip_fraction)
                agg_metrics["approx_kl"].append(metrics.approx_kl)

            # gradient clipping + optimizer step
            with timer.time("optim_step"):
                _gn = self._clip_and_step(optimizer)
                agg_metrics["grad_norm"].append(_gn)

            # EMA update
            if ema is not None:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                ema.step(trainable, self.state.global_step)

            self.state.global_step += 1

        # Aggregate metrics — each metric averages over its own count (loss/policy
        # appended per-timestep, grad_norm appended per-inner-epoch).
        def avg(key: str) -> float:
            vals = agg_metrics.get(key, [])
            return sum(vals) / len(vals) if vals else 0.0

        reward_mean = batch.rewards.mean().item()
        reward_std = batch.rewards.std().item() if batch.rewards.numel() > 1 else 0.0
        adv_mean = advantages.mean().item()

        phase_times = dict(timer.times)
        if cfg.profile and phase_times:
            total = sum(phase_times.values())
            parts = " | ".join(
                f"{k}={v:.3f}s ({100*v/total:.1f}%)" for k, v in phase_times.items()
            )
            logger.info("phase_times[step=%d] total=%.3fs | %s",
                        self.state.step, total, parts)

        metrics = TrainStepMetrics(
            loss=avg("loss"),
            policy_loss=avg("policy_loss"),
            kl_penalty=avg("kl_penalty"),
            reward_mean=reward_mean,
            reward_std=reward_std,
            advantage_mean=adv_mean,
            clip_fraction=avg("clip_fraction"),
            approx_kl=avg("approx_kl"),
            grad_norm=avg("grad_norm"),
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            phase_times=phase_times,
        )

        # Update state
        self.state.step += 1
        self.state.total_reward += metrics.reward_mean
        self.state.total_loss += metrics.loss

        # Sync weights
        if self.weight_syncer is not None:
            state_dict = self.model.state_dict()
            await self.weight_syncer.push(state_dict)

        return metrics

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
