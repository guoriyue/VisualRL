"""Online RL trainer — CEA pipeline (Collector + Evaluator + Algorithm).

collect -> evaluate -> advantage -> loss -> backward -> step.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vrl.algorithms.advantages import all_reduce_sufficient_stats
from vrl.algorithms.base import Algorithm, ComponentAdvantageAlgorithm
from vrl.algorithms.logprob_mismatch import (
    LogprobMismatchStats,
)
from vrl.algorithms.types import InitialReplayStats, PolicyUpdateStats, TrainStepMetrics
from vrl.models.precision import (
    apply_float32_precision,
    float32_precision_state,
    model_precision,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import (
    move_training_batch_to_device,
    nonzero_advantage_mask,
    select_batch,
)
from vrl.rollouts.evaluators.base import Evaluator
from vrl.rollouts.orchestration import build_rollout_schedule
from vrl.rollouts.stats import (
    JsonlStatsSink,
    LoggingStatsSink,
    MultiStatsSink,
    RolloutStats,
    StatsSink,
)
from vrl.trainers.core.types import TrainState
from vrl.trainers.diagnostics import (
    append_jsonl_record,
    parameter_state_summary,
    tensor_stats,
    trainable_state_digest,
)
from vrl.trainers.online.config import TrainerConfig
from vrl.trainers.online.ema import EMAModuleWrapper
from vrl.trainers.online.precision_guard import (
    enforce_precision_drift,
    measure_precision_drift,
)
from vrl.trainers.optimizer import FP32MasterWeightOptimizer, build_optimizer
from vrl.trainers.strategy import SingleProcessStrategy, Strategy, TrainingMemoryState
from vrl.trainers.weight_sync import TrainableStateGetter, WeightSyncer

if TYPE_CHECKING:
    from vrl.algorithms.trajectory import AlgorithmAdapter

logger = logging.getLogger(__name__)


def _global_reward_stats(rewards: Any) -> tuple[float, float]:
    """Population (mean, std) of ``rewards`` over **all DDP ranks**.

    The logged reward curve is written by rank0 only (online.py is_primary), and
    each rank holds just its own prompt slice (16 of the 32 global prompts for
    rbs=16), so a plain ``rewards.mean()`` reports a per-rank metric — half the
    true optimization objective. All-reduce the sufficient statistics (sum,
    sum-of-squares, count) so the logged mean/std reflect the full cross-rank
    batch. Falls back to local stats with no process group or world_size==1
    (single-GPU), where local already equals global. Unconditional on every rank
    (the writer-gating happens upstream), so the collective stays balanced.
    """

    # Sufficient-statistics derivation on every path (a POPULATION std): one
    # formula at world_size 1 and >1, so the same config reports the same
    # reward_std either way (torch's unbiased default was 15% apart on a
    # 4-sample step).
    g_sum, g_sumsq, g_count = all_reduce_sufficient_stats(rewards)
    if g_count < 1:
        return 0.0, 0.0
    g_mean = g_sum / g_count
    mean = float(g_mean.item())
    if g_count <= 1:
        return mean, 0.0
    g_var = (g_sumsq / g_count) - g_mean * g_mean
    std = float(torch.sqrt(torch.clamp(g_var, min=0.0)).item())
    return mean, std


def _all_ranks_have_work(has_work: bool, device: torch.device) -> bool:
    """True iff EVERY training rank has a non-empty (post-filter) microbatch.

    A backward pass fires cross-rank collectives — FSDP2 per-layer all-gather +
    reduce-scatter, or DDP's gradient all-reduce. If one rank skips backward on an
    all-filtered (zero-advantage) microbatch while another rank runs it, those
    collectives mismatch and the job DEADLOCKS: an unrecoverable NCCL hang, not an
    exception. So the skip decision must be unanimous. All-reduce the local
    ``has_work`` flag with MIN, so every rank takes the SAME branch — the
    microbatch runs only when all ranks have work, otherwise all ranks skip it
    together (matched: no rank issues backward collectives). Dropping a microbatch
    because one rank's slice came back empty wastes the other ranks' work for that
    slice, but empty slices are rare (reward spread) and a dropped slice beats a
    hung run.

    No process group / world_size==1 (single-GPU) returns the local value
    unchanged. Must be called UNCONDITIONALLY on every rank, the same number of
    times, so this collective itself stays balanced (the recipe runs a fixed
    microbatch/step count per rank regardless of filtering).
    """

    return _distributed_all_true(has_work, device)


# ---------------------------------------------------------------------------
# Phase profiler
# ---------------------------------------------------------------------------


class PhaseTimer:
    """Accumulating phase timer with optional CUDA sync.

    Each ``time(name)`` call returns a context manager whose wall time is
    added to ``self.times[name]``. When CUDA is available,
    ``torch.cuda.synchronize()`` is called on both ends so async GPU kernels
    are captured.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.sync = torch.cuda.is_available()
        self.times: dict[str, float] = defaultdict(float)
        self.events: list[tuple[str, float, float]] = []

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wall0 = time.time()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.times[name] += t1 - t0
            self.events.append((name, wall0, time.time()))


# ---------------------------------------------------------------------------
# Precision execution
# ---------------------------------------------------------------------------


def _create_grad_scaler(
    device: torch.device,
    model: Any,
) -> torch.amp.GradScaler | None:
    """Build the CUDA scaler for either FP16 compute or native FP16 grads."""

    if device.type != "cuda":
        return None
    precision = model_precision(model)
    uses_fp16_autocast = precision.outer_autocast and precision.dtype == "fp16"
    has_native_fp16_gradients = bool(
        model is not None
        and any(
            parameter.requires_grad and parameter.dtype == torch.float16
            for parameter in model.parameters()
        )
    )
    if not (uses_fp16_autocast or has_native_fp16_gradients):
        return None
    return torch.amp.GradScaler("cuda")


def _requires_fp32_master_weights(model: Any) -> bool:
    """Return whether the optimizer must preserve sub-ULP update residuals."""

    return any(
        parameter.requires_grad and parameter.dtype in {torch.float16, torch.bfloat16}
        for parameter in model.parameters()
    )


def _precision_label(value: Any) -> str:
    token = str(value or "").strip().lower().removeprefix("torch.")
    return {
        "": "fp32",
        "no": "fp32",
        "float32": "fp32",
        "bfloat16": "bf16",
        "float16": "fp16",
    }.get(token, token)


def _dtype_label(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _model_transformer_dtype(model: Any) -> str | None:
    getter = getattr(model, "_transformer_dtype", None)
    if callable(getter):
        try:
            return _dtype_label(getter())
        except Exception:
            pass
    transformer = getattr(model, "transformer", None)
    dtype = getattr(transformer, "dtype", None)
    if dtype is not None:
        return _dtype_label(dtype)
    parameters = getattr(transformer if transformer is not None else model, "parameters", None)
    if callable(parameters):
        try:
            return _dtype_label(next(parameters()).dtype)
        except (StopIteration, RuntimeError, TypeError):
            return None
    return None


def _trainer_precision_metadata(
    config: TrainerConfig,
    model: Any,
    evaluator: Evaluator | None,
) -> dict[str, Any]:
    training_precision = _precision_label(config.train_precision)
    rollout_precision = _precision_label(config.rollout_precision or training_precision)
    return {
        "training_precision": training_precision,
        "rollout_precision": rollout_precision,
        # Report the dtype the evaluator actually consumes instead of carrying a
        # duplicate TrainerConfig projection of the public precision policy.
        "math_precision": _precision_label(
            getattr(evaluator, "math_dtype", None) or torch.float32,
        ),
        "effective_float32_precision": float32_precision_state(),
        "trainer_transformer_dtype": _model_transformer_dtype(model),
    }


# ---------------------------------------------------------------------------
# OnlineTrainer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """The data half of one step: filtered rollouts + advantages + diagnostics.

    ``collect_training_batch`` produces it and ``train_on_rollout_batch`` consumes
    it; everything the train half reads — the filtered batches/advantages, the
    pre-filter reward/advantage diagnostics, and the shared timer/iteration whose
    timings both halves accumulate into — travels in here. The split is the seam
    a future rank split would collect/train across.
    """

    iteration: Any
    timer: PhaseTimer
    batches: list[RolloutBatch]
    advantages: list[torch.Tensor]
    group_size: float
    trained_prompt_num: int
    adv_zero_rate: float
    adv_saturation: float
    pre_filter_reward_mean: float
    pre_filter_reward_std: float
    pre_filter_adv_mean: float
    reward_components: dict[str, list[float]]


@dataclass(slots=True)
class _ReplayMetrics:
    """Typed aggregation shared by streaming and full-batch replay paths."""

    losses: list[float] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    kl_penalties: list[float] = field(default_factory=list)
    weighted_kl_losses: list[float] = field(default_factory=list)
    updates: list[PolicyUpdateStats] = field(default_factory=list)
    mismatches: list[LogprobMismatchStats] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    initial_updates: list[PolicyUpdateStats] = field(default_factory=list)
    initial_mismatches: list[LogprobMismatchStats] = field(default_factory=list)
    initial_weights: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    weighted_sft_losses: list[float] = field(default_factory=list)
    sft_weights: list[float] = field(default_factory=list)

    def add(
        self,
        metrics: TrainStepMetrics,
        *,
        weight: float,
        capture_initial_replay: bool,
    ) -> None:
        """Record one replay evaluation with its objective-normalization weight."""

        resolved_weight = float(weight)
        if resolved_weight <= 0:
            raise ValueError("real replay metric weight must be positive")

        self.losses.append(metrics.loss)
        self.policy_losses.append(metrics.policy_loss)
        self.kl_penalties.append(metrics.kl_penalty)
        self.weighted_kl_losses.append(metrics.weighted_kl_loss)
        self.updates.append(metrics.update)
        self.mismatches.append(metrics.logprob_mismatch)
        self.weights.append(resolved_weight)
        if capture_initial_replay:
            self.initial_updates.append(metrics.update)
            self.initial_mismatches.append(metrics.logprob_mismatch)
            self.initial_weights.append(resolved_weight)

    def initial_replay_snapshot(self) -> tuple[InitialReplayStats, float]:
        """Build the local snapshot captured before the first optimizer boundary."""

        update = PolicyUpdateStats.weighted_mean(
            self.initial_updates,
            self.initial_weights,
        )
        mismatch = LogprobMismatchStats.aggregate(
            self.initial_mismatches,
            self.initial_weights,
        )
        finite = bool(self.initial_mismatches) and mismatch.finite
        return (
            InitialReplayStats(
                clip_fraction=update.clip_fraction,
                active_clip_fraction=update.active_clip_fraction,
                logprob_abs_diff_max=(mismatch.logprob_abs_diff_max if finite else float("inf")),
                finite=finite,
            ),
            sum(self.initial_weights),
        )

    def add_sft(self, loss: float, weight: float) -> None:
        self.weighted_sft_losses.append(float(loss) * float(weight))
        self.sft_weights.append(float(weight))

    @property
    def sft_loss(self) -> float:
        denominator = sum(self.sft_weights)
        return sum(self.weighted_sft_losses) / denominator if denominator > 0 else 0.0

    def build(
        self,
        *,
        reward_mean: float,
        reward_std: float,
        reward_components: dict[str, float],
        advantage_mean: float,
        adv_saturation: float,
        adv_zero_rate: float,
        group_size: float,
        trained_prompt_num: int,
        phase_times: dict[str, float],
        initial_replay: InitialReplayStats,
    ) -> TrainStepMetrics:
        """Build the public step result while preserving each reduction rule."""

        def mean(values: Sequence[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def weighted_mean(values: Sequence[float]) -> float:
            total_weight = sum(self.weights)
            if not values or total_weight <= 0:
                return 0.0
            if len(values) != len(self.weights):
                raise ValueError("replay metric values/weights length mismatch")
            return (
                sum(
                    float(value) * weight
                    for value, weight in zip(values, self.weights, strict=True)
                )
                / total_weight
            )

        sft_loss = self.sft_loss
        return TrainStepMetrics(
            loss=weighted_mean(self.losses) + sft_loss,
            policy_loss=weighted_mean(self.policy_losses),
            sft_loss=sft_loss,
            kl_penalty=weighted_mean(self.kl_penalties),
            weighted_kl_loss=weighted_mean(self.weighted_kl_losses),
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_components=reward_components,
            advantage_mean=advantage_mean,
            update=PolicyUpdateStats.weighted_mean(self.updates, self.weights),
            logprob_mismatch=LogprobMismatchStats.aggregate(
                self.mismatches,
                self.weights,
            ),
            initial_replay=initial_replay,
            grad_norm=mean(self.grad_norms),
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            phase_times=dict(phase_times),
        )


def _rollout_reward_components(
    batches: list[RolloutBatch],
) -> dict[str, list[float]]:
    """Extract batch-aligned component scores before sample filtering."""

    components: dict[str, list[float]] = {}
    for batch in batches:
        payload = batch.extras.get("reward_components")
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise TypeError("rollout reward_components must be a mapping")
        batch_size = int(batch.rewards.shape[0])
        for raw_name, raw_values in payload.items():
            name = str(raw_name)
            if isinstance(raw_values, torch.Tensor):
                values = raw_values.detach().cpu().reshape(-1).tolist()
            else:
                values = list(raw_values)
            if len(values) != batch_size:
                raise ValueError(
                    "rollout reward component/sample mismatch: "
                    f"component={name!r}, scores={len(values)}, expected={batch_size}",
                )
            components.setdefault(name, []).extend(float(value) for value in values)
    return components


def _mean_reward_components(
    components: Mapping[str, list[float]],
) -> dict[str, float]:
    return {name: sum(values) / len(values) for name, values in components.items() if values}


@dataclass(frozen=True, slots=True)
class _TrainingGenerationSampleBatch:
    batch: RolloutBatch
    advantages: torch.Tensor
    loss_weight: float
    is_dummy: bool = False


def _training_sample_batches(
    batch: RolloutBatch,
    advantages: torch.Tensor,
    samples_per_replay_batch: int,
) -> list[_TrainingGenerationSampleBatch]:
    """Split one prompt group for replay without changing full-group loss math."""

    batch_size = int(batch.rewards.shape[0])
    if batch_size != int(advantages.shape[0]):
        raise ValueError(
            "rollout batch and advantages must have the same sample count "
            f"({batch_size} != {int(advantages.shape[0])})",
        )
    if batch_size <= 0:
        return []
    slice_size = int(samples_per_replay_batch)
    if slice_size <= 0 or slice_size >= batch_size:
        return [
            _TrainingGenerationSampleBatch(batch=batch, advantages=advantages, loss_weight=1.0)
        ]

    batches: list[_TrainingGenerationSampleBatch] = []
    for start in range(0, batch_size, slice_size):
        stop = min(start + slice_size, batch_size)
        selector = torch.arange(start, stop, device=batch.rewards.device)
        batches.append(
            _TrainingGenerationSampleBatch(
                batch=select_batch(batch, selector),
                advantages=advantages[selector.to(advantages.device)],
                loss_weight=float(stop - start) / float(batch_size),
            ),
        )
    return batches


def _all_reduce_scalar(
    value: float,
    *,
    dtype: torch.dtype,
    op: Any,
    device: torch.device,
) -> float:
    """One scalar collective (nccl needs the tensor on the rank's GPU)."""

    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return value
    tensor = torch.tensor([value], dtype=dtype)
    if dist.get_backend() == "nccl":
        tensor = tensor.to(device)
    dist.all_reduce(tensor, op=op)
    return tensor.item()


def _distributed_max_int(value: int, device: torch.device) -> int:
    """Return the maximum integer value across training ranks."""

    op = torch.distributed.ReduceOp.MAX
    return int(_all_reduce_scalar(int(value), dtype=torch.int64, op=op, device=device))


def _distributed_max_float(value: float, device: torch.device) -> float:
    """Return the maximum float across ranks without changing single-rank runs."""

    op = torch.distributed.ReduceOp.MAX
    return float(_all_reduce_scalar(float(value), dtype=torch.float64, op=op, device=device))


def _distributed_all_true(value: bool, device: torch.device) -> bool:
    """Return True only when every training rank reports True (MIN over {0,1})."""

    op = torch.distributed.ReduceOp.MIN
    return bool(_all_reduce_scalar(int(bool(value)), dtype=torch.int32, op=op, device=device))


def _distributed_parity_verdict(
    *,
    local_finite: bool,
    local_max_abs_diff: float,
    limit: float,
    device: torch.device,
) -> tuple[bool, float, bool]:
    """Return one rank-consistent parity verdict for every training process."""

    finite = _distributed_all_true(local_finite, device)
    max_abs_diff = _distributed_max_float(
        local_max_abs_diff if local_finite else float("inf"),
        device,
    )
    return finite, max_abs_diff, finite and max_abs_diff <= limit


def _distributed_initial_replay_stats(
    local: InitialReplayStats,
    *,
    local_weight: float,
    device: torch.device,
) -> tuple[InitialReplayStats, bool]:
    """Resolve the global snapshot and whether any rank measured replay."""

    weight = float(local_weight)
    if weight < 0:
        raise ValueError("initial replay weight must be non-negative")
    has_local_measurements = weight > 0
    # Zero-weight ranks execute the same collectives as real ranks, but none of
    # their placeholder values may enter a weighted sum or max reduction. The
    # conditional also prevents IEEE ``nan * 0`` from poisoning the sum.
    weighted_clip_fraction = local.clip_fraction * weight if has_local_measurements else 0.0
    weighted_active_clip_fraction = (
        local.active_clip_fraction * weight if has_local_measurements else 0.0
    )
    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if distributed:
        totals = torch.tensor(
            [
                weighted_clip_fraction,
                weighted_active_clip_fraction,
                weight,
            ],
            dtype=torch.float64,
        )
        if dist.get_backend() == "nccl":
            totals = totals.to(device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_weight = float(totals[2].item())
        clip_fraction = float(totals[0].item()) / total_weight if total_weight > 0 else 0.0
        active_clip_fraction = float(totals[1].item()) / total_weight if total_weight > 0 else 0.0
    else:
        total_weight = weight
        clip_fraction = local.clip_fraction if total_weight > 0 else 0.0
        active_clip_fraction = local.active_clip_fraction if total_weight > 0 else 0.0

    # A rank with nothing to measure is neutral, not a failure: dummy batches
    # exist precisely so an all-filtered rank still runs matching collectives.
    # Whether ANY rank measured something is the gate's decision (it skips a
    # globally empty first update), not a per-rank finiteness verdict.
    finite = _distributed_all_true(local.finite or not has_local_measurements, device)
    if not has_local_measurements:
        local_max_abs_diff = 0.0
    elif local.finite:
        local_max_abs_diff = local.logprob_abs_diff_max
    else:
        local_max_abs_diff = float("inf")
    max_abs_diff = _distributed_max_float(
        local_max_abs_diff,
        device,
    )
    return (
        InitialReplayStats(
            clip_fraction=clip_fraction,
            active_clip_fraction=active_clip_fraction,
            logprob_abs_diff_max=max_abs_diff if finite else float("inf"),
            finite=finite,
        ),
        total_weight > 0,
    )


def _balanced_training_sample_batches(
    batches: list[RolloutBatch],
    advantages: list[torch.Tensor],
    samples_per_replay_batch: int,
    device: torch.device,
) -> list[_TrainingGenerationSampleBatch]:
    """Plan replay execution slots with equal slot counts across ranks.

    Local zero-advantage filtering can leave different ranks with different
    numbers of prompt groups or sample batches. DDP/FSDP forward/backward issue
    collectives, so the number of replay slots must be globally balanced even
    when only some slots carry training signal.
    """

    sample_batches: list[_TrainingGenerationSampleBatch] = []
    for batch, adv in zip(batches, advantages, strict=True):
        sample_batches.extend(_training_sample_batches(batch, adv, samples_per_replay_batch))

    target_count = _distributed_max_int(len(sample_batches), device)
    if target_count == len(sample_batches):
        return sample_batches
    if target_count <= 0:
        return []
    if not sample_batches:
        raise RuntimeError(
            "distributed replay planner cannot synthesize dummy slots without a "
            "local real batch; call _all_ranks_have_work before planning replay batches",
        )
    # Use the smallest available local batch as the dummy template to minimize
    # the extra zero-loss forward/backward work needed for collective balance.
    template = min(sample_batches, key=lambda batch: int(batch.batch.rewards.shape[0]))
    sample_batches.extend(
        _TrainingGenerationSampleBatch(
            batch=template.batch,
            advantages=torch.zeros_like(template.advantages),
            loss_weight=0.0,
            is_dummy=True,
        )
        for _ in range(target_count - len(sample_batches))
    )
    return sample_batches


class OnlineTrainer:
    """Orchestrates the CEA online RL loop.

    Pipeline: collect -> evaluate -> advantage -> loss -> backward -> step.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        collector: Any,
        evaluator: Evaluator | None,
        model: nn.Module,
        config: TrainerConfig,
        ref_model: nn.Module | None = None,
        weight_syncer: WeightSyncer | None = None,
        sync_state_getter: TrainableStateGetter | None = None,
        device: torch.device | str = "cuda",
        strategy: Strategy | None = None,
        sft_latents: Mapping[str, Any] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.collector = collector
        self.evaluator = evaluator
        self.model = model
        self.ref_model = ref_model
        self.weight_syncer = weight_syncer
        if weight_syncer is not None and sync_state_getter is None:
            raise ValueError(
                "OnlineTrainer weight sync requires an explicit trainable-state "
                "getter; syncing model.state_dict() would send frozen modules.",
            )
        self.sync_state_getter = sync_state_getter
        self.config = config
        # RuntimeBundle stamps the selected role policy on the model so trainer
        # process settings cannot be paired with a different build.
        self.precision = model_precision(model)
        # Precision correction (TIS) is a trainer-level precision-drift concern, not
        # an algorithm hyperparameter; inject it into algorithms that apply it
        # (importance-ratio algorithms hold a `precision_correction` slot).
        if hasattr(algorithm, "precision_correction"):
            algorithm.precision_correction = config.precision_correction
        # Clean fine-tuning latents ({target artifact -> [C,T,H,W]}) for the GRPO
        # diffusion-loss regularizer; the recipe loads data.sft_latents and the
        # config layer already rejected sft_weight>0 without it.
        self._sft_latents = dict(sft_latents) if sft_latents else None
        if self._sft_weight > 0 and self._sft_latents is None:
            raise ValueError(
                "algorithm.sft_weight > 0 but no sft_latents were provided to "
                "OnlineTrainer (wire data.sft_latents)",
            )
        self.device = torch.device(device) if isinstance(device, str) else device
        self.state = TrainState()
        # How a step runs on the hardware (backward / clip / state export). The
        # default keeps current single-GPU behavior; FSDP2 swaps this in later
        # without the trainer loop changing. See vrl/trainers/strategy.py.
        self._strategy: Strategy = strategy or SingleProcessStrategy()
        if self._strategy.context.strategy != "single_process" and any(
            parameter.requires_grad and parameter.dtype == torch.float16
            for parameter in self.model.parameters()
        ):
            # BF16 masters follow the FSDP/DTensor shard layout and need no loss
            # scaling. FP16 additionally needs a cross-rank found-inf decision;
            # letting independent scalers skip different optimizer steps would
            # desynchronize the policy even though gradients were reduced.
            raise NotImplementedError(
                "distributed FP16 trainable parameters require cross-rank GradScaler "
                "found-inf agreement; use BF16 or single_process until that collective "
                "decision is implemented",
            )
        # Route the model through the strategy once: identity for single process,
        # fully_shard wrapping for FSDP2. Done before optimizer / grad-scaler / EMA
        # so they bind to the (possibly sharded) parameters the strategy returns.
        self.model = self._strategy.prepare_model(self.model)
        # Sinks for the per-step phase timings (recording decoupled from
        # emitting). The log line stays the human-facing view; the jsonl file is
        # the complete machine-readable view. metrics.csv exposes only the
        # stable continuous-health subset, not arbitrary collect.* phases.
        self._stats_sink: StatsSink = MultiStatsSink(
            LoggingStatsSink(logger),
            JsonlStatsSink(f"{self.config.output_dir}/rollout_stats.jsonl"),
        )
        self._grad_scaler = _create_grad_scaler(
            self.device,
            self.model,
        )

        self._optimizer: torch.optim.Optimizer | None = None
        self._ema: EMAModuleWrapper | None = None
        self._rollout_weights_initialized = False
        # Recheck rollout/replay parity in each process; a checkpoint's previous
        # pass does not cover changes to kernels, compilation, or batch geometry.
        self._replay_parity_passed = False
        self._precision_drift_guard_pending = True
        self._update_phase_timers: list[PhaseTimer] = []
        self.rollout_schedule = build_rollout_schedule(
            self.config.rollout_orchestration,
            collector=self.collector,
            strategy=self._strategy,
            training_state_getter=self._training_memory_state,
            weight_syncer=self.weight_syncer,
            sync_state_getter=self.sync_state_getter,
            weights_initialized=lambda: self._rollout_weights_initialized,
            set_weights_initialized=self._set_rollout_weights_initialized,
            # Likelihood-free objectives (DiffusionNFT) opt out, which makes a
            # continuous max_stale>0 config fail fast as unsound.
            algorithm_tolerates_off_policy_staleness=(
                self.algorithm.tolerates_off_policy_staleness
            ),
        )
        self._validate_trust_region_engages()

        # The backend setting is process-global, so every trainer construction
        # projects both the IEEE and TF32 paths instead of inheriting stale state.
        apply_float32_precision(self.precision.float32_precision)

    def _validate_trust_region_engages(self) -> None:
        """Refuse configs where a trust-region algorithm's ratio term is inert.

        Flow-DPPO / GRPO-Guard are *defined* by a clipped/guarded importance ratio
        ``r = pi_new / pi_old``. With ``strict_on_policy`` + ``ppo_epochs == 1`` the
        behavior and target policy are identical on the single replay pass, so
        ``r == 1``, the trust-region term is identically zero, and the run is
        byte-equivalent to plain GRPO — the documented flat-curve root cause. Fail
        fast instead of silently training a no-op mechanism.

        Continuous always permits a bounded behavior-policy lag; its typed
        configuration rejects a zero-version window because that execution is
        the ``strict_on_policy`` schedule.
        """
        if not self.algorithm.requires_active_trust_region:
            return
        cfg = self.config
        orchestration = cfg.rollout_orchestration
        schedule_mode = orchestration.schedule_mode
        max_stale = (
            int(orchestration.continuous.max_stale_policy_versions)
            if schedule_mode == "continuous"
            else 0
        )
        if int(cfg.ppo_epochs) <= 1 and max_stale <= 0:
            raise ValueError(
                f"{type(self.algorithm).__name__} is defined by its importance-ratio "
                "trust region, but the rollout schedule permits no behavior-policy "
                "staleness and actor.ppo_epochs=1 makes the ratio identically 1 (behavior == "
                "target on the single replay pass), so the clip/guard term is a no-op "
                "and the run is equivalent to plain GRPO. Set actor.ppo_epochs>1 — which "
                "needs the legacy full-batch path (actor.gradient_accumulation_steps=0 "
                "and actor.microbatch_size=0, since streaming releases each microbatch "
                "and cannot replay it across epochs) — or use schedule_mode='continuous' "
                "with continuous.max_stale_policy_versions>0 for an off-policy ratio."
            )

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if _requires_fp32_master_weights(self.model):
                self._optimizer = FP32MasterWeightOptimizer(
                    trainable,
                    lambda masters: build_optimizer(masters, self.config),
                )
            else:
                self._optimizer = build_optimizer(trainable, self.config)
        return self._optimizer

    def _ensure_ema(self) -> EMAModuleWrapper | None:
        if not self.config.ema.enable:
            return None
        if self._ema is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self._ema = EMAModuleWrapper(
                trainable,
                decay=self.config.ema.decay,
                update_step_interval=self.config.ema.update_interval,
                device=self.device,
            )
        return self._ema

    def _set_rollout_weights_initialized(self, value: bool) -> None:
        self._rollout_weights_initialized = bool(value)

    def _training_memory_state(self) -> TrainingMemoryState:
        """Resolve the trainer-owned objects live at the rollout phase boundary."""

        return TrainingMemoryState(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self._optimizer,
            ema=self._ema,
            grad_scaler=self._grad_scaler,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Accelerator-aware backward/step helpers
    # ------------------------------------------------------------------

    def _backward(self, loss: Any) -> None:
        from vrl.utils.profiling import profile_range

        with profile_range("trainer.backward"):
            self._strategy.backward(loss, grad_scaler=self._grad_scaler)

    def _compute_replay_loss(
        self,
        group_batch: RolloutBatch,
        batch_advantages: torch.Tensor,
        timestep_index: int,
        *,
        algorithm_adapter: AlgorithmAdapter,
    ) -> tuple[torch.Tensor, TrainStepMetrics]:
        """Run the shared evaluator/algorithm decision for one replay timestep."""

        from vrl.algorithms.trajectory import AlgorithmInput
        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import profile_range

        if not self.algorithm.uses_evaluator:
            # DiffusionNFT owns its raw transformer boundaries and applies the
            # resolved contract there; its objective reductions remain outside.
            with profile_range("trainer.loss"):
                return algorithm_adapter.compute_loss(
                    self.algorithm,
                    AlgorithmInput(
                        rewards=group_batch.rewards,
                        group_ids=group_batch.group_ids,
                        advantages=batch_advantages,
                        model=self.model,
                        rollout_batch=group_batch,
                        timestep_index=timestep_index,
                    ),
                )

        if self.evaluator is None:
            raise RuntimeError(f"{type(self.algorithm).__name__} requires an evaluator")
        kl_coef = float(getattr(self.algorithm.config, "kl_coef", 0.0))
        need_kl_intermediates = kl_coef > 0 or self.algorithm.needs_kl_intermediates
        # The evaluator scopes only replay_forward; SDE/log-prob/gather math
        # and the algorithm loss remain outside autocast.
        with profile_range("trainer.replay"):
            signals = self.evaluator.evaluate(
                self.model,
                group_batch,
                timestep_index,
                ref_model=self.ref_model,
                signal_request=SignalRequest(
                    need_ref=kl_coef > 0,
                    need_kl_intermediates=need_kl_intermediates,
                ),
            )
        with profile_range("trainer.loss"):
            if not isinstance(signals, TrajectorySignalBatch):
                raise TypeError(
                    "evaluator output must be TrajectorySignalBatch; "
                    f"got {type(signals).__name__}",
                )
            loss, metrics = algorithm_adapter.compute_loss(
                self.algorithm,
                AlgorithmInput(
                    signals=signals,
                    advantages=batch_advantages,
                    group_ids=group_batch.group_ids,
                ),
            )

        # Parity is a trainer/evaluator fact, not an objective-specific metric.
        # Measure every replayed segment here so TokenGRPO, trust-region variants,
        # and multi-segment objectives cannot accidentally leave a false zero that
        # lets the pre-optimizer correctness gate pass.
        fresh_parts: list[torch.Tensor] = []
        old_parts: list[torch.Tensor] = []
        for segment in signals.segments.values():
            fresh = segment.log_prob
            old = segment.old_log_prob
            if fresh is None or old is None:
                continue
            if fresh.shape != old.shape:
                raise ValueError(
                    "replay/rollout log-prob shape mismatch while measuring parity: "
                    f"fresh={tuple(fresh.shape)} old={tuple(old.shape)}",
                )
            mask = segment.mask
            if isinstance(mask, torch.Tensor) and mask.numel() == fresh.numel():
                valid = mask.reshape(-1).to(device=fresh.device, dtype=torch.bool)
                fresh_parts.append(fresh.reshape(-1)[valid])
                old_parts.append(old.reshape(-1)[valid])
            else:
                fresh_parts.append(fresh.reshape(-1))
                old_parts.append(old.reshape(-1))
        if not fresh_parts:
            raise RuntimeError(
                "evaluator-backed policy loss produced no replay/rollout log-prob pair",
            )
        metrics.logprob_mismatch = LogprobMismatchStats.compute(
            torch.cat(fresh_parts),
            torch.cat(old_parts),
        )
        return loss, metrics

    def _clip_and_step(self, optimizer: Any) -> tuple[float, bool]:
        """Clip grads and step the optimizer.

        Returns ``(pre_clip_grad_norm, stepped)``. ``stepped`` is False only when
        the fp16 ``GradScaler`` skipped the step because it found inf/nan grads;
        callers must not run EMA / ``after_optimizer_step`` on a skipped step (it
        would fold an update that never happened into the averaged/adapter state).
        """
        cfg = self.config
        grad_norm: Any = 0.0
        if isinstance(optimizer, FP32MasterWeightOptimizer):
            optimizer.prepare_gradients()
            parameters_for_clip = list(optimizer.parameters())
        else:
            parameters_for_clip = list(self.model.parameters())
        if self._grad_scaler is not None:
            self._grad_scaler.unscale_(optimizer)
        if cfg.max_norm > 0:
            grad_norm = self._strategy.clip_grad_norm(parameters_for_clip, cfg.max_norm)
        else:
            # No clip, but still route through the strategy: under FSDP the grads
            # are DTensors holding only this rank's shard, so summing them locally
            # would under-report the global norm and — worse — let the non-finite
            # check below pass on ranks whose shard happens to be clean while the
            # rank owning the NaN raises, diverging the ranks mid-step.
            # max_norm=inf makes the clip coefficient a no-op while still
            # computing (and cross-mesh reducing) the true global norm.
            grad_norm = self._strategy.clip_grad_norm(parameters_for_clip, math.inf)
        grad_norm_value = float(grad_norm)
        if self._grad_scaler is None and not math.isfinite(grad_norm_value):
            # BF16 training has no GradScaler to skip a poisoned update. Refuse
            # the optimizer step here so an external metrics monitor never sees
            # NaN only after corrupt weights have already been committed. With a
            # GradScaler, inf grads are a designed event (scale overshoot): the
            # scaler skips the step and backs off, so it must not raise here.
            optimizer.zero_grad()
            raise FloatingPointError(
                f"non-finite gradient norm {grad_norm_value}; optimizer step refused",
            )
        stepped = True
        if self._grad_scaler is not None:
            scale_before = self._grad_scaler.get_scale()
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            # The scaler lowers its scale by the backoff factor only when it
            # skipped the step on inf/nan grads; an unchanged or grown scale
            # means the optimizer actually applied the update.
            stepped = self._grad_scaler.get_scale() >= scale_before
        else:
            optimizer.step()
        optimizer.zero_grad()
        return grad_norm_value, stepped

    # ------------------------------------------------------------------
    # Training step — CEA pipeline
    # ------------------------------------------------------------------

    async def step(
        self,
        prompts: list[Any],
        *,
        next_prompts: list[Any] | None = None,
    ) -> TrainStepMetrics:
        """Run one full training step: collect -> evaluate -> advantage -> loss -> backward -> step."""
        from vrl.utils.profiling import capture_torch_trace

        with capture_torch_trace(
            self.config.torch_profiler,
            output_dir=self.config.output_dir,
            step=self.state.step,
            device=self.device,
            worker_name="online_trainer",
        ):
            return await self._step_impl(prompts, next_prompts=next_prompts)

    async def _step_impl(
        self,
        prompts: list[Any],
        *,
        next_prompts: list[Any] | None = None,
    ) -> TrainStepMetrics:
        """Run one full training step without profiler wrapping."""
        batch = await self.collect_training_batch(prompts, next_prompts=next_prompts)
        return await self.train_on_rollout_batch(batch)

    async def collect_training_batch(
        self,
        prompts: list[Any],
        *,
        next_prompts: list[Any] | None = None,
    ) -> TrainingBatch:
        """Collect rollouts and compute + filter advantages — the data half.

        Returns everything ``train_on_rollout_batch`` needs; in single-process
        ``step()`` the two run back-to-back. The shared ``timer`` is created here
        and carried through the batch so both halves' phase timings land in one
        accumulator, exactly as the previous single method did.

        ``next_prompts`` is the recipe loop's lookahead: only that loop knows both
        the gradient-accumulation split and the next epoch's draw (previewed off
        the checkpointed RNG), so the value cannot be reconstructed downstream and
        is forwarded verbatim. Continuous rollout installs it as the producer's
        next prompt batch so generation overlaps this step's training; strict
        on-policy discards it.
        """
        cfg = self.config

        timer = PhaseTimer(enabled=cfg.profile)
        runtime_debug_collect = bool(cfg.debug.first_step and self.state.step == 0)

        # 1. The rollout schedule owns collect/offload/release/sync timing.
        iteration = await self.rollout_schedule.next_iteration(
            prompts,
            group_size=cfg.batch_plan.n_samples_per_prompt,
            runtime_debug=runtime_debug_collect,
            next_prompts=next_prompts,
        )
        all_batches: list[RolloutBatch] = iteration.batches
        reward_components = _rollout_reward_components(all_batches)

        # 2. Compute advantages (per-prompt normalization).
        # Rewards are concatenated across all collected batches, normalized
        # per prompt-group by the algorithm (the single source of truth for
        # advantage math), then split back per-batch for the training loop.
        with timer.time("advantage"):
            all_rewards = torch.cat([b.rewards for b in all_batches])
            all_group_ids = torch.cat([b.group_ids for b in all_batches])
            if isinstance(self.algorithm, ComponentAdvantageAlgorithm):
                component_tensors = {
                    name: torch.as_tensor(
                        values,
                        dtype=all_rewards.dtype,
                        device=all_rewards.device,
                    )
                    for name, values in reward_components.items()
                }
                advantages_all = self.algorithm.compute_advantages_from_components(
                    all_rewards,
                    component_tensors,
                    all_group_ids,
                )
            else:
                advantages_all = self.algorithm.compute_advantages_from_tensors(
                    all_rewards,
                    all_group_ids,
                )
            # Per-prompt grouping stats for logging (mean group size + unique prompts).
            _unique_groups, _group_counts = torch.unique(all_group_ids, return_counts=True)
            group_size = (
                float(_group_counts.float().mean().item()) if _group_counts.numel() else 0.0
            )
            trained_prompt_num = int(_unique_groups.numel())

        # Advantage diagnostics on the full (pre-filter) advantages.
        _adv_abs = advantages_all.detach().abs()
        _total = max(advantages_all.numel(), 1)
        adv_zero_rate = float((_adv_abs < 1e-6).sum().item()) / _total
        _clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
        adv_saturation = (
            float((_adv_abs >= _clip_max - 1e-6).sum().item()) / _total
            if _clip_max is not None
            else 0.0
        )

        # Cross-rank so the logged curve is the full 32-prompt objective, not
        # rank0's local 16-prompt slice (see _global_reward_stats).
        pre_filter_reward_mean, pre_filter_reward_std = _global_reward_stats(all_rewards)
        pre_filter_adv_mean = advantages_all.mean().item()

        # Split advantages back per-batch for the gradient-accumulation loop.
        split_sizes = [b.rewards.shape[0] for b in all_batches]
        adv_split = list(torch.split(advantages_all, split_sizes))

        # Zero-advantage sample filter. Flow-GRPO applies this globally across
        # the outer rollout epoch, pads enough zero-advantage rows back in for
        # exact rebatching, then shuffles/rebatches into
        # num_batches_per_epoch training microbatches.
        # Per-group batches (one RolloutBatch per prompt group). Streaming
        # accumulation (gradient_accumulation_steps>0) splits the optimizer
        # target into prompt microbatches in the recipe loop, so the collector
        # no longer rebatches internally; advantage stays per-group either way.
        filtered_batches: list[RolloutBatch] = []
        filtered_advs: list[torch.Tensor] = []
        if cfg.drop_zero_advantage:
            for b, adv_b in zip(all_batches, adv_split, strict=True):
                mask = nonzero_advantage_mask(adv_b)
                if not bool(mask.any()):
                    continue
                if not bool(mask.all()):
                    b = select_batch(b, mask)
                    adv_b = adv_b[mask.to(adv_b.device)]
                if b.rewards.shape[0] > 0:
                    filtered_batches.append(b)
                    filtered_advs.append(adv_b)
        else:
            filtered_batches = list(all_batches)
            filtered_advs = adv_split

        return TrainingBatch(
            iteration=iteration,
            timer=timer,
            batches=filtered_batches,
            advantages=filtered_advs,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            adv_zero_rate=adv_zero_rate,
            adv_saturation=adv_saturation,
            pre_filter_reward_mean=pre_filter_reward_mean,
            pre_filter_reward_std=pre_filter_reward_std,
            pre_filter_adv_mean=pre_filter_adv_mean,
            reward_components=reward_components,
        )

    @staticmethod
    def _train_timestep_indices(
        num_timesteps: int,
        timestep_fraction: float,
        selection: str = "strided",
    ) -> list[int]:
        """Denoise timesteps that receive loss (single source of truth).

        ``selection`` controls which subset gets gradient when
        ``timestep_fraction < 1``:
        - ``"strided"`` — a fixed evenly-spaced subset, identical every update.
        - ``"random"`` (DanceGRPO) — a fresh random subset resampled each call,
          so gradient coverage of the denoise trajectory decorrelates across
          updates instead of always landing on the same strided steps. Returned
          sorted; the loss loop sums over the indices, so order is irrelevant.
        - ``"stratified"`` (V-GRPO) — the grid is cut into ``count`` equal-length
          intervals and one index is drawn uniformly from each, resampled each
          call: every region of the schedule is covered on every update, with
          none of ``"strided"``'s fixed positions.
        """
        train_timestep_count = max(1, int(num_timesteps * timestep_fraction))
        if train_timestep_count >= num_timesteps:
            return list(range(num_timesteps))
        if selection == "random":
            import torch

            picks = torch.randperm(num_timesteps)[:train_timestep_count].tolist()
            return sorted(int(i) for i in picks)
        if selection == "stratified":
            import torch

            bounds = [
                round(i * num_timesteps / train_timestep_count)
                for i in range(train_timestep_count + 1)
            ]
            return [
                int(torch.randint(lo, hi, (1,)).item()) for lo, hi in itertools.pairwise(bounds)
            ]
        step_size = num_timesteps / train_timestep_count
        return [int(i * step_size) for i in range(train_timestep_count)]

    def _train_replay_indices(
        self,
        batch: RolloutBatch,
        timestep_fraction: float,
        selection: str = "strided",
    ) -> list[int]:
        """Return evaluator invocations for one replay batch.

        Existing evaluators replay one denoise step per invocation and retain
        the configured timestep subset.  Causal policies may instead require
        one ordered pass over the complete trajectory; index ``0`` is then an
        API-compatible sentinel and must not be interpreted as a selected
        temporal chunk or denoise transition.
        """

        granularity = getattr(self.evaluator, "replay_granularity", "step")
        if granularity == "trajectory":
            return [0]
        if granularity != "step":
            raise ValueError(
                "evaluator.replay_granularity must be 'step' or 'trajectory', "
                f"got {granularity!r}",
            )
        if selection == "sde_window":
            # Flash-GRPO: train exactly the steps the rollout made stochastic.
            # The window is a GENERATION-time fact recorded in the trajectory,
            # not a trainer-side draw — training any step outside it would put
            # surrogate loss on a deterministic ODE transition that was never a
            # policy action.
            return self._sde_window_indices(batch)

        from vrl.trajectory import TrajectoryResolver

        resolver = TrajectoryResolver.from_batch(batch)
        primary_segment = resolver.primary_trainable_segment_name()
        action = resolver.role_tensor(primary_segment, "action")
        action_axes = tuple(axis for axis in action.axes if axis != "sample")
        if len(action_axes) != 1:
            raise ValueError(
                "step replay requires exactly one non-sample axis on the primary "
                f"action tensor; {primary_segment}.{action.name} declares "
                f"{action.axes!r}",
            )
        action_axis_name = action_axes[0]
        action_axis = resolver.trajectory.axes[action_axis_name]
        num_timesteps = action_axis.length
        if num_timesteps is None:
            action_shape = getattr(action.value, "shape", None)
            if action_shape is None:
                raise ValueError(
                    f"step replay action axis {action_axis_name!r} has no declared "
                    "length and its tensor has no shape",
                )
            num_timesteps = int(action_shape[action.axes.index(action_axis_name)])
        if num_timesteps <= 0:
            raise ValueError(
                f"step replay action axis {action_axis_name!r} must be non-empty",
            )
        return self._train_timestep_indices(
            num_timesteps,
            timestep_fraction,
            selection,
        )

    @staticmethod
    def _sde_window_indices(batch: RolloutBatch) -> list[int]:
        """Replay indices from the trajectory's recorded stochastic window.

        Generation records ``[lo, hi)`` per sample (denoise replay tensor
        ``sde_window``); one request owns one window, so every row of a
        group's batch must agree — a mismatch means groups were merged across
        requests and iso-temporal grouping is already broken upstream.
        """

        from vrl.trajectory import TrajectoryResolver

        replay = TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise")
        window = replay.get("sde_window")
        if window is None:
            raise ValueError(
                "actor.timestep_selection='sde_window' requires the rollout to "
                "record its stochastic window: set rollout.sde.window_size > 0 "
                "so generation stores the 'sde_window' replay tensor "
                "(absent on this batch).",
            )
        lo = int(window[:, 0].min().item())
        hi = int(window[:, 1].max().item())
        if int(window[:, 0].max().item()) != lo or int(window[:, 1].min().item()) != hi:
            raise ValueError(
                "sde_window differs across rows of one replay batch; a prompt "
                "group's samples must come from one generation request so they "
                "share the stochastic timestep",
            )
        if hi <= lo:
            raise ValueError(f"recorded sde_window [{lo}, {hi}) is empty")
        return list(range(lo, hi))

    def _sample_batch_train_indices(
        self,
        sample_batch: Any,
        default_indices: list[int],
        selection: str,
    ) -> list[int]:
        """Per-microbatch replay indices; only ``sde_window`` varies by batch.

        Each group's window sits at a different (iso-temporal) position, so the
        first-batch indices cannot stand in for the others. The WIDTH must
        match everywhere — loss_scale and the cross-rank microbatch counts are
        derived from it, and sampling config fixes window_size for the run.
        """

        if selection != "sde_window":
            return default_indices
        indices = self._sde_window_indices(sample_batch.batch)
        if len(indices) != len(default_indices):
            raise ValueError(
                f"sde_window width {len(indices)} differs from the first "
                f"batch's {len(default_indices)}; sde_window_size must be "
                "uniform across a run",
            )
        return indices

    # ------------------------------------------------------------------
    # Streaming accumulation boundary (gradient_accumulation_steps>0):
    # begin -> backward(microbatch)* -> finish. One optimizer update spans
    # several separately-collected prompt microbatches so the whole target
    # batch never has to be materialized at once (SPRINT_streaming_rollout_
    # accumulation). The legacy full-batch path keeps train_on_rollout_batch.
    # ------------------------------------------------------------------

    def begin_optimizer_update(self) -> None:
        """Start one streaming optimizer update: reset grads + metric accumulator."""
        self._update_optimizer = self._ensure_optimizer()
        self._update_ema = self._ensure_ema()
        self._update_phase_timers.clear()
        self.model.train()
        self._update_agg_metrics = _ReplayMetrics()
        self._update_had_training_work = False
        self._update_optimizer.zero_grad(set_to_none=True)

    def _run_replay_pass(
        self,
        batches: list[Any],
        advantages: Any,
        *,
        total_groups: int,
        train_indices: list[int],
        algorithm_adapter: Any,
        agg: Any,
        capture_initial_replay: bool,
        defer_replay_tensors: bool,
        timer: PhaseTimer | None = None,
    ) -> None:
        """One replay/backward sweep over the update's sample batches.

        The single body behind both the streaming microbatch path and the
        full-batch PPO-epoch path — the two hand-copies drifted once
        (the all-filtered initial_replay crash), so the sweep lives here and
        the callers keep only their own orchestration. Both paths pass their
        batch profiler so replay and backward timings share the same boundary.
        """

        cfg = self.config
        loss_scale = int(total_groups) * len(train_indices)
        samples_per_replay_batch = cfg.batch_plan.samples_per_replay_batch
        for sample_batch in _balanced_training_sample_batches(
            batches,
            advantages,
            samples_per_replay_batch,
            self.device,
        ):
            group_batch = move_training_batch_to_device(
                sample_batch.batch,
                self.device,
                defer_replay_tensors=defer_replay_tensors,
            )
            batch_adv = sample_batch.advantages.to(self.device)
            self._backward_sft_regularizer(
                group_batch,
                loss_weight=sample_batch.loss_weight,
                total_groups=total_groups,
                is_dummy=sample_batch.is_dummy,
                agg=agg,
            )
            for j in self._sample_batch_train_indices(
                sample_batch,
                train_indices,
                cfg.timestep_selection,
            ):
                with timer.time("evaluate") if timer else contextlib.nullcontext():
                    loss, metrics = self._compute_replay_loss(
                        group_batch,
                        batch_adv,
                        j,
                        algorithm_adapter=algorithm_adapter,
                    )
                    # Average across the complete optimizer target batch;
                    # step evaluators retain the per-denoise-step surrogate.
                    loss = loss * sample_batch.loss_weight / loss_scale
                with timer.time("backward") if timer else contextlib.nullcontext():
                    self._backward(loss)
                if not sample_batch.is_dummy:
                    agg.add(
                        metrics,
                        weight=sample_batch.loss_weight,
                        capture_initial_replay=capture_initial_replay,
                    )

    def backward_on_training_batch(
        self,
        batch: TrainingBatch,
        *,
        total_groups: int,
    ) -> None:
        """Evaluate + loss + backward for ONE collected microbatch; no optimizer step.

        ``total_groups`` is the global prompt-group count across the whole
        optimizer update (= prompts_per_batch); loss divides by
        ``total_groups * num_replay_units`` so the accumulated gradient over
        all microbatches equals the legacy full-batch path.
        """
        from vrl.algorithms.trajectory import AlgorithmAdapter

        cfg = self.config
        if cfg.profile:
            self._update_phase_timers.append(batch.timer)
        # Unanimous skip across ranks: a backward fires cross-rank collectives, so
        # one rank skipping an empty microbatch while another runs it deadlocks
        # (see _all_ranks_have_work). Called once per microbatch on every rank, in
        # lockstep with the fixed gradient-accumulation count, so this collective
        # is balanced.
        if not _all_ranks_have_work(bool(batch.batches), self.device):
            return
        self._update_had_training_work = True
        uses_evaluator = self.algorithm.uses_evaluator
        algorithm_adapter = AlgorithmAdapter()
        defer = uses_evaluator and bool(
            getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
        )
        train_indices = self._train_replay_indices(
            batch.batches[0],
            cfg.timestep_fraction,
            cfg.timestep_selection,
        )
        samples_per_replay_batch = cfg.batch_plan.samples_per_replay_batch
        first_batch = _training_sample_batches(
            batch.batches[0],
            batch.advantages[0],
            samples_per_replay_batch,
        )[0]
        self._check_initial_precision_drift(first_batch.batch, train_indices)
        self._run_replay_pass(
            batch.batches,
            batch.advantages,
            total_groups=int(total_groups),
            train_indices=train_indices,
            algorithm_adapter=algorithm_adapter,
            agg=self._update_agg_metrics,
            capture_initial_replay=True,
            defer_replay_tensors=defer,
            timer=batch.timer,
        )

    async def finish_optimizer_update(
        self,
        *,
        stats: RolloutStats,
        reward_mean: float,
        reward_std: float,
        adv_mean: float,
        adv_zero_rate: float,
        adv_saturation: float,
        group_size: float,
        trained_prompt_num: int,
        reward_components: dict[str, float],
    ) -> TrainStepMetrics:
        """Clip + optimizer.step + EMA/NFT (once) and build the one update's metrics.

        ``stats`` is the caller-aggregated typed accumulator across the update's
        microbatches. The streaming recipe owns aggregation so each microbatch's
        tensors are released before this runs.
        """
        optimizer = self._update_optimizer
        agg = self._update_agg_metrics
        sync_stats = RolloutStats()
        optimizer_timer = PhaseTimer(enabled=self.config.profile)
        if self._update_had_training_work:
            with optimizer_timer.time("optim_step"):
                local_initial, local_weight = agg.initial_replay_snapshot()
                initial_replay = self._validate_first_update_parity(
                    local_initial,
                    local_weight=local_weight,
                )
                grad_norm, stepped = self._clip_and_step(optimizer)
            agg.grad_norms.append(grad_norm)
            if stepped:
                after_optimizer_step = getattr(self.algorithm, "after_optimizer_step", None)
                if callable(after_optimizer_step):
                    after_optimizer_step(self.model, self.state.global_step)
                if self._update_ema is not None:
                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    self._update_ema.step(trainable, self.state.global_step)
            self.state.global_step += 1
            if stepped:
                sync_stats = await self.rollout_schedule.after_train_step()
        else:
            # No replay evaluations ran, so report the same zero-stats snapshot
            # the non-streaming path uses; build() and the metrics CSV require
            # a non-None InitialReplayStats.
            initial_replay = InitialReplayStats()
            agg.grad_norms.append(0.0)
            logger.info(
                "step %d: every streamed microbatch was filtered; skipping "
                "optimizer and rollout weight sync",
                self.state.step,
            )

        metric_step = self.state.step
        self.state.step += 1
        stats.add_phases(optimizer_timer.times)
        stats.merge(sync_stats)
        if self.config.profile:
            self._stats_sink.record(metric_step, stats)
            for timer in (*self._update_phase_timers, optimizer_timer):
                self._write_phase_events(timer, step=metric_step)
            self._update_phase_timers.clear()
        phase_times = stats.as_phase_dict()
        metrics = agg.build(
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_components=reward_components,
            advantage_mean=adv_mean,
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            phase_times=phase_times,
            initial_replay=initial_replay,
        )
        return metrics

    async def train_on_rollout_batch(self, batch: TrainingBatch) -> TrainStepMetrics:
        """Train on a collected batch — the compute half of one step.

        The schedule's post-update weight publication is asynchronous even though
        replay evaluation, backward, and the optimizer step are synchronous.
        """
        from vrl.algorithms.trajectory import AlgorithmAdapter
        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import profile_range

        cfg = self.config
        optimizer = self._ensure_optimizer()
        ema = self._ensure_ema()
        iteration = batch.iteration
        timer = batch.timer
        filtered_batches = batch.batches
        filtered_advs = batch.advantages
        group_size = batch.group_size
        trained_prompt_num = batch.trained_prompt_num
        adv_zero_rate = batch.adv_zero_rate
        adv_saturation = batch.adv_saturation
        pre_filter_reward_mean = batch.pre_filter_reward_mean
        pre_filter_reward_std = batch.pre_filter_reward_std
        pre_filter_adv_mean = batch.pre_filter_adv_mean
        reward_components = _mean_reward_components(batch.reward_components)

        # 3. Train loop — gradient accumulation across per-prompt batches.
        self.model.train()
        agg_metrics = _ReplayMetrics()
        uses_evaluator = self.algorithm.uses_evaluator
        algorithm_adapter = AlgorithmAdapter()

        # If every batch was filtered out (all dead), skip training this step.
        # Unanimous across ranks (see _all_ranks_have_work): a backward fires
        # cross-rank collectives, so the skip must be agreed or the ranks that did
        # vs. did not run backward deadlock. Called once per step on every rank.
        if not _all_ranks_have_work(bool(filtered_batches), self.device):
            logger.info(
                "step %d: all batches filtered (zero advantages) on this or a peer "
                "rank; skipping backward",
                self.state.step,
            )
            # Early exit — still advance state + return metrics with zeros.
            self.state.step += 1
            reward_mean = pre_filter_reward_mean
            reward_std = pre_filter_reward_std
            return TrainStepMetrics(
                loss=0.0,
                policy_loss=0.0,
                kl_penalty=0.0,
                reward_mean=reward_mean,
                reward_std=reward_std,
                reward_components=reward_components,
                advantage_mean=pre_filter_adv_mean,
                grad_norm=0.0,
                adv_saturation=adv_saturation,
                adv_zero_rate=adv_zero_rate,
                group_size=group_size,
                trained_prompt_num=trained_prompt_num,
                phase_times=self._step_stats(iteration, timer).as_phase_dict(),
            )

        defer_replay_tensor_move = uses_evaluator and bool(
            getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
        )

        # Replay schedule — step evaluators use the configured denoise subset;
        # trajectory evaluators perform one causal pass over all policy axes.
        # Trajectory schemas are consistent across batches, so inspect the first batch.
        train_indices = self._train_replay_indices(
            filtered_batches[0],
            cfg.timestep_fraction,
            cfg.timestep_selection,
        )

        samples_per_replay_batch = cfg.batch_plan.samples_per_replay_batch

        # Debug first step: compare old vs fresh log-probs on first timestep
        # (using first filtered batch so memory footprint is bounded).
        first_step_debug_record: dict[str, Any] | None = None
        precision_metadata = _trainer_precision_metadata(
            cfg,
            self.model,
            self.evaluator,
        )
        first_debug_batch = _training_sample_batches(
            filtered_batches[0],
            filtered_advs[0],
            samples_per_replay_batch,
        )[0]
        if cfg.debug.first_step and self.state.step == 0 and uses_evaluator:
            _dbg_batch = move_training_batch_to_device(
                first_debug_batch.batch,
                self.device,
                defer_replay_tensors=defer_replay_tensor_move,
            )
            with (
                torch.no_grad(),
                profile_range("trainer.replay"),
            ):
                _dbg_signals = self.evaluator.evaluate(
                    self.model,
                    _dbg_batch,
                    0,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            if not isinstance(_dbg_signals, TrajectorySignalBatch):
                raise TypeError(
                    "evaluator output must be TrajectorySignalBatch; "
                    f"got {type(_dbg_signals).__name__}",
                )
            _dbg_trajectory_signals = _dbg_signals
            _dbg_log_prob = _dbg_trajectory_signals.primary.log_prob
            _old_lp_0 = _dbg_trajectory_signals.primary.old_log_prob
            _diff = (_dbg_log_prob - _old_lp_0).abs()
            _ratio = torch.exp(_dbg_log_prob - _old_lp_0)
            _old_lp_first = _old_lp_0.reshape(-1)[0]
            _fresh_lp_first = _dbg_log_prob.reshape(-1)[0]
            _parity_limit = float(cfg.replay_parity.max_abs_logprob_diff)
            _local_parity_finite = bool(
                torch.isfinite(_old_lp_0).all().item()
                and torch.isfinite(_dbg_log_prob).all().item()
                and torch.isfinite(_diff).all().item()
                and torch.isfinite(_ratio).all().item()
            )
            _local_parity_max = float(_diff.max().item())
            _parity_finite, _parity_max, _parity_passed = _distributed_parity_verdict(
                local_finite=_local_parity_finite,
                local_max_abs_diff=_local_parity_max,
                limit=_parity_limit,
                device=self.device,
            )
            logger.info(
                "DEBUG first-step log-prob diff: mean=%.6f max=%.6f | "
                "old_lp[0]=%.6f fresh_lp[0]=%.6f",
                _diff.mean().item(),
                _diff.max().item(),
                _old_lp_first.item(),
                _fresh_lp_first.item(),
            )
            first_step_debug_record = {
                "event": "first_step_logprob_parity",
                "passed": _parity_passed,
                "finite": _parity_finite,
                "max_abs_diff": _parity_max,
                "local_finite": _local_parity_finite,
                "local_max_abs_diff": _local_parity_max,
                "max_abs_diff_limit": _parity_limit,
                "trainer_step": int(self.state.step),
                "global_step": int(self.state.global_step),
                "device": str(self.device),
                "precision_policy": precision_metadata,
                "old_log_prob": tensor_stats(_old_lp_0),
                "fresh_log_prob": tensor_stats(_dbg_log_prob),
                "abs_diff": tensor_stats(_diff),
                "ratio": tensor_stats(_ratio),
                "driver_trainable_before_step": trainable_state_digest(self.model),
                "driver_parameter_state_before_step": parameter_state_summary(self.model),
                "rollout_context": {
                    key: value
                    for key, value in _dbg_batch.context.items()
                    if key != "runtime_debug"
                },
                "runtime_debug": _dbg_batch.context.get("runtime_debug"),
            }
            # Persist a failing probe immediately: the mandatory full-update
            # gate below owns enforcement, while this optional probe owns the
            # tensor/provenance evidence needed to diagnose that failure.
            if not _parity_passed and self._strategy.context.is_primary:
                append_jsonl_record(
                    f"{cfg.output_dir}/training_debug.jsonl",
                    first_step_debug_record,
                )
        elif cfg.debug.first_step and self.state.step == 0:
            # Non-evaluator algorithms (NFT) compute no log-prob ratio, so the
            # parity probe above is blind to them. Ask the algorithm for its
            # own lr=0 invariant through the optional protocol method instead
            # of hardcoding algorithm checks here.
            _invariant_check = getattr(self.algorithm, "first_step_invariant_check", None)
            if callable(_invariant_check):
                _dbg_batch = move_training_batch_to_device(
                    first_debug_batch.batch,
                    self.device,
                    defer_replay_tensors=defer_replay_tensor_move,
                )
                _dbg_adv = first_debug_batch.advantages.to(self.device)
                with (
                    torch.no_grad(),
                    profile_range("trainer.replay"),
                ):
                    _invariant = _invariant_check(
                        model=self.model,
                        batch=_dbg_batch,
                        advantages=_dbg_adv,
                        timestep_index=0,
                    )
                logger.info(
                    "DEBUG first-step %s: abs_diff=%.3e (threshold %.1e)",
                    _invariant["event"],
                    _invariant["abs_diff"],
                    _invariant["threshold"],
                )
                if not _invariant.get("passed", True):
                    logger.warning(
                        "first-step %s invariant violated: abs_diff %.3e > %.1e. "
                        "The collection-time training signal is untrustworthy; "
                        "suspect replay-side conditioning/scheduler-domain drift.",
                        _invariant["invariant"],
                        _invariant["abs_diff"],
                        _invariant["threshold"],
                    )
                first_step_debug_record = {
                    **_invariant,
                    "trainer_step": int(self.state.step),
                    "global_step": int(self.state.global_step),
                    "device": str(self.device),
                    "precision_policy": precision_metadata,
                    "rollout_context": {
                        key: value
                        for key, value in _dbg_batch.context.items()
                        if key != "runtime_debug"
                    },
                }

        guard_record = self._check_initial_precision_drift(first_debug_batch.batch, train_indices)
        if guard_record is not None and first_step_debug_record is not None:
            first_step_debug_record["precision_drift_guard"] = guard_record

        initial_replay = InitialReplayStats()
        policy_updated = False
        for _ppo_epoch in range(cfg.ppo_epochs):
            # ``train_on_rollout_batch`` receives one already-collected optimizer
            # target batch. Streaming accumulation is owned by the recipe and
            # calls begin/backward/finish instead; interpreting the same count
            # again here would turn one target batch into several optimizer steps.
            capture_initial_replay = _ppo_epoch == 0
            self._run_replay_pass(
                filtered_batches,
                filtered_advs,
                total_groups=len(filtered_batches),
                train_indices=train_indices,
                algorithm_adapter=algorithm_adapter,
                agg=agg_metrics,
                capture_initial_replay=capture_initial_replay,
                defer_replay_tensors=defer_replay_tensor_move,
                timer=timer,
            )

            with timer.time("optim_step"):
                if capture_initial_replay:
                    local_initial, local_weight = agg_metrics.initial_replay_snapshot()
                    initial_replay = self._validate_first_update_parity(
                        local_initial,
                        local_weight=local_weight,
                    )
                _gn, _stepped = self._clip_and_step(optimizer)
                agg_metrics.grad_norms.append(_gn)

            # A scaler-skipped step (inf/nan grads) left the weights unchanged —
            # do not fold a non-update into EMA or the algorithm adapter.
            if _stepped:
                policy_updated = True
                after_optimizer_step = getattr(self.algorithm, "after_optimizer_step", None)
                if callable(after_optimizer_step):
                    after_optimizer_step(self.model, self.state.global_step)

                if ema is not None:
                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    ema.step(trainable, self.state.global_step)

            self.state.global_step += 1

        reward_mean = pre_filter_reward_mean
        reward_std = pre_filter_reward_std
        adv_mean = pre_filter_adv_mean

        # collect.* phase timings arrive inside iteration.stats: each collect
        # call owns its timings (no shared collector state), and the
        # schedule/consumer aggregates them per iteration. The trainer phases
        # (timer) merge on top into one typed accumulator, emitted via the sink.
        step_stats = self._step_stats(iteration, timer)
        metric_step = self.state.step

        # Update state
        self.state.step += 1

        if policy_updated:
            step_stats.merge(await self.rollout_schedule.after_train_step())
        phase_times = step_stats.as_phase_dict()
        if cfg.profile and phase_times:
            self._stats_sink.record(metric_step, step_stats)
            self._write_phase_events(timer, step=metric_step)

        metrics = agg_metrics.build(
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_components=reward_components,
            advantage_mean=adv_mean,
            adv_saturation=adv_saturation,
            adv_zero_rate=adv_zero_rate,
            group_size=group_size,
            trained_prompt_num=trained_prompt_num,
            phase_times=phase_times,
            initial_replay=initial_replay,
        )
        if first_step_debug_record is not None:
            first_step_debug_record["driver_trainable_after_step"] = trainable_state_digest(
                self.model
            )
            first_step_debug_record["driver_parameter_state_after_step"] = parameter_state_summary(
                self.model
            )
            first_step_debug_record["post_step_global_step"] = int(self.state.global_step)
            if self._strategy.context.is_primary:
                append_jsonl_record(
                    f"{cfg.output_dir}/training_debug.jsonl",
                    first_step_debug_record,
                )

        return metrics

    def _step_stats(self, iteration: Any, timer: PhaseTimer) -> RolloutStats:
        """Typed per-step stats: the iteration's accumulator + trainer phases."""

        stats = RolloutStats()
        stats.merge(iteration.stats)
        stats.add_phases(timer.times)
        return stats

    def _write_phase_events(self, timer: PhaseTimer, *, step: int) -> None:
        """Append the per-phase start/end events to phase_events.jsonl."""

        try:
            import json
            import os

            evt_path = os.path.join(self.config.output_dir, "phase_events.jsonl")
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(evt_path, "a") as handle:
                for name, start, end in timer.events:
                    handle.write(
                        json.dumps(
                            {
                                "step": int(step),
                                "phase": name,
                                "start": start,
                                "end": end,
                            },
                        )
                        + "\n",
                    )
            timer.events.clear()
        except Exception:
            pass

    def _backward_sft_regularizer(
        self,
        group_batch: Any,
        *,
        loss_weight: float,
        total_groups: int,
        is_dummy: bool,
        agg: _ReplayMetrics,
    ) -> None:
        """Backpropagate one correctly normalized, rank-balanced SFT term.

        The policy objective accumulates over denoise timesteps; the diffusion
        pretraining regularizer does not. It therefore uses only the group and
        sample-batch normalization here, outside the timestep loop. Distributed
        dummy slots still run the forward and a zero-weight backward so FSDP/DDP
        execute the same collective sequence on every rank.
        """

        if self._sft_weight <= 0:
            return
        if total_groups <= 0:
            raise ValueError("sft regularizer total_groups must be positive")

        from vrl.utils.profiling import profile_range

        with profile_range("trainer.sft_regularizer"):
            sft_term = self._sft_regularizer_loss(group_batch)
            scaled_term = sft_term * float(loss_weight) / int(total_groups)
        self._backward(scaled_term)
        if not is_dummy:
            agg.add_sft(float(sft_term.detach()), float(loss_weight))

    def _check_initial_precision_drift(
        self,
        batch: RolloutBatch,
        timestep_indices: Sequence[int],
    ) -> dict[str, Any] | None:
        """Enforce the same first-trainable-batch guard on both update paths.

        Precision correction deliberately permits non-exact replay, so its
        bounded drift guard must also run during streaming accumulation. Callers
        agree across ranks that training work exists before entering this gate.
        """

        from vrl.rollouts.evaluators.types import SignalRequest, TrajectorySignalBatch
        from vrl.utils.profiling import profile_range

        if (
            not self._precision_drift_guard_pending
            or not self.algorithm.uses_evaluator
            or self.evaluator is None
        ):
            return None
        cfg = self.config
        precision_metadata = _trainer_precision_metadata(cfg, self.model, self.evaluator)
        guard_batch = move_training_batch_to_device(
            batch,
            self.device,
            defer_replay_tensors=bool(
                getattr(self.evaluator, "supports_deferred_replay_tensor_move", False),
            ),
        )

        def evaluate(timestep_idx: int) -> TrajectorySignalBatch:
            with torch.no_grad(), profile_range("trainer.replay"):
                signals = self.evaluator.evaluate(
                    self.model,
                    guard_batch,
                    timestep_idx,
                    ref_model=self.ref_model,
                    signal_request=SignalRequest(need_ref=False, need_kl_intermediates=False),
                )
            if not isinstance(signals, TrajectorySignalBatch):
                raise TypeError(
                    "evaluator output must be TrajectorySignalBatch; "
                    f"got {type(signals).__name__}",
                )
            return signals

        record = measure_precision_drift(
            cfg.precision_drift_guard,
            training_precision=precision_metadata["training_precision"],
            rollout_precision=precision_metadata["rollout_precision"],
            math_precision=precision_metadata["math_precision"],
            timestep_indices=timestep_indices,
            evaluate_fn=evaluate,
            metadata=precision_metadata,
        )
        if record is not None:
            worst = dict(record.get("worst_stats") or {})
            worst["logprob_abs_diff_max"] = _distributed_max_float(
                float(worst.get("logprob_abs_diff_max", 0.0)),
                self.device,
            )
            worst["ratio_abs_dev_max"] = _distributed_max_float(
                float(worst.get("ratio_abs_dev_max", 0.0)),
                self.device,
            )
            worst["finite"] = _distributed_all_true(
                bool(worst.get("finite", True)),
                self.device,
            )
            record["worst_stats"] = worst
            record["violated"] = not _distributed_all_true(
                not bool(record["violated"]),
                self.device,
            )
            # Fail on every rank; warn and persist evidence only on the writer.
            if record["mode"] == "fail" or self._strategy.context.is_primary:
                enforce_precision_drift(record, logger=logger)
            if self._strategy.context.is_primary:
                append_jsonl_record(f"{cfg.output_dir}/training_debug.jsonl", record)
        self._precision_drift_guard_pending = False
        return record

    def _validate_first_update_parity(
        self,
        local: InitialReplayStats,
        *,
        local_weight: float,
    ) -> InitialReplayStats:
        """Gate this process's first measured exact replay before optimizer.step."""

        cfg = self.config
        resolved, has_measurements = _distributed_initial_replay_stats(
            local,
            local_weight=local_weight,
            device=self.device,
        )
        correction = getattr(self.algorithm, "precision_correction", None)
        intentional_correction = correction is not None and (
            correction.tis_mode != "off"
            or correction.rs_mode != "off"
            or correction.recompute_old_logprob != "off"
        )
        if (
            self._replay_parity_passed
            or self.evaluator is None
            or not self.algorithm.uses_evaluator
            or intentional_correction
        ):
            return resolved

        # The weight sum is already reduced with the metric numerators. Reusing
        # it avoids a fourth collective solely to count local evaluations.
        if not has_measurements:
            return resolved

        limit = float(cfg.replay_parity.max_abs_logprob_diff)
        passed = resolved.finite and resolved.logprob_abs_diff_max <= limit
        record = {
            "event": "replay_parity_gate",
            "passed": passed,
            "finite": resolved.finite,
            "max_abs_diff": resolved.logprob_abs_diff_max,
            "max_abs_diff_limit": limit,
            "trainer_step": int(self.state.step),
            "global_step": int(self.state.global_step),
            "driver_trainable_before_step": trainable_state_digest(self.model),
        }
        if self._strategy.context.is_primary:
            append_jsonl_record(f"{cfg.output_dir}/training_debug.jsonl", record)
        if not passed:
            raise RuntimeError(
                "replay parity failed before this process's first optimizer update: "
                f"finite={resolved.finite}, "
                f"max_abs_diff={resolved.logprob_abs_diff_max:.6g}, "
                f"limit={limit:.6g}. The first-sample probe is insufficient; "
                "align rollout/replay precision and batch shape or recompute "
                "the old policy with the replay backend.",
            )
        self._replay_parity_passed = True
        return resolved

    @property
    def _sft_weight(self) -> float:
        """Read the regularizer weight from its algorithm-config source of truth."""

        return float(getattr(self.algorithm.config, "sft_weight", 0.0) or 0.0)

    def _sft_regularizer_loss(self, group_batch: Any) -> torch.Tensor:
        """Weighted diffusion pretraining loss on clean fine-tuning latents.

        The Cosmos-Predict2.5 paper's anti-reward-hacking regularizer
        (§4.2.2): noise this batch's CLEAN target latents at a
        random step of the replay schedule and penalize the policy's
        prediction error against the family's pretraining target. Everything
        family-specific is reused, not re-derived: the noising runs through
        the evaluator's own scheduler (``diffusion_pretraining_pair``), and
        the forward through ``model.replay_forward_with_latents`` — the exact
        conditioning + timestep the log-prob replay uses, so no second
        sigma-domain conversion path exists (the EDM-domain trap).
        """

        import torch.nn.functional as F

        from vrl.math.denoise.flow_matching import diffusion_pretraining_pair
        from vrl.trainers.data.sft_latents import CleanTargetRef
        from vrl.trajectory import TrajectoryResolver

        assert self._sft_latents is not None  # ctor validated
        reward_metadata = group_batch.context.get("reward_metadata", {})
        target_key = CleanTargetRef.from_source(reward_metadata).key
        if target_key not in self._sft_latents:
            raise ValueError(
                f"data.sft_latents has no entry for clean target {target_key!r}; "
                "re-encode the shard over the full training manifest",
            )
        scheduler = getattr(self.evaluator, "scheduler", None)
        if scheduler is None:
            raise ValueError(
                "algorithm.sft_weight > 0 needs the diffusion SDE evaluator "
                "(its scheduler owns the noising domain); this evaluator has "
                "no scheduler",
            )

        resolver = TrajectoryResolver.from_batch(group_batch)
        primary_segment = resolver.primary_trainable_segment_name()
        observations = resolver.role_value(primary_segment, "observation")
        x0 = (
            self._sft_latents[target_key]
            .unsqueeze(0)
            .expand(
                int(observations.shape[0]),
                *self._sft_latents[target_key].shape,
            )
            .to(
                device=self.device,
                dtype=observations.dtype if hasattr(observations, "dtype") else None,
            )
        )
        expected_shape = tuple(observations.shape[0:1]) + tuple(observations.shape[2:])
        if tuple(x0.shape) != expected_shape:
            raise ValueError(
                f"sft latents shape {tuple(x0.shape)} does not match the "
                f"rollout latent shape {expected_shape}; encode the targets "
                "at the training sampling geometry",
            )

        timesteps = resolver.tensor_value(
            primary_segment,
            "timesteps",
        )
        num_steps = timesteps.shape[1] if timesteps.ndim > 1 else timesteps.shape[0]
        step_idx = self._shared_sft_step_index(int(num_steps))
        t = (
            timesteps[:, step_idx]
            if timesteps.ndim > 1
            else timesteps[step_idx].expand(x0.shape[0])
        ).to(x0.device)

        noise = torch.randn_like(x0)
        noisy, target = diffusion_pretraining_pair(scheduler, x0, noise, t)
        values = self.model.replay_forward_with_latents(group_batch, step_idx, noisy)
        model_pred = self.model.diffusion_pretraining_prediction(values)
        return self._sft_weight * F.mse_loss(model_pred.float(), target.float())

    def _shared_sft_step_index(self, num_steps: int) -> int:
        """Draw one denoise step index that every training rank agrees on.

        The regularizer is averaged across ranks by the gradient all-reduce, so an
        unsynchronized draw would have each rank penalize a different noise level.
        The estimator stays unbiased either way, but its variance would become
        world-size dependent: the same ``sft_weight`` would behave differently at 1
        vs 8 GPUs, and the run would not replay from the checkpointed RNG state
        (which snapshots only the local process). Rank 0's draw is authoritative.
        """

        step = torch.randint(0, num_steps, (1,), dtype=torch.int64)
        dist = torch.distributed
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return int(step.item())
        if dist.get_backend() == "nccl":
            step = step.cuda()
        dist.broadcast(step, src=0)
        return int(step.item())

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict(checkpoint_primary_only=False)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Build a resume snapshot without full state copies on every FSDP rank."""

        return self._state_dict(checkpoint_primary_only=True)

    def _state_dict(self, *, checkpoint_primary_only: bool) -> dict[str, Any]:
        # COLLECTIVE under fsdp (optimizer-moment + EMA-shadow gathers): the
        # checkpoint writer must call this on EVERY rank before its
        # primary-only file write, like the trainable-state gather.
        d: dict[str, Any] = {
            "step": self.state.step,
            "global_step": self.state.global_step,
        }
        if self._optimizer is not None:
            checkpoint_exporter = getattr(
                self._strategy,
                "export_checkpoint_optimizer_state",
                None,
            )
            exporter = (
                checkpoint_exporter
                if checkpoint_primary_only and callable(checkpoint_exporter)
                else self._strategy.export_optimizer_state
            )
            d["optimizer"] = exporter(
                self.model,
                self._optimizer,
            )
            d["optimizer_parameter_manifest"] = self._optimizer_parameter_manifest()
        if self._grad_scaler is not None:
            d["grad_scaler"] = self._grad_scaler.state_dict()
        ema = self._ensure_ema()
        if ema is not None:
            checkpoint_ema = getattr(ema, "checkpoint_state_dict", None)
            if checkpoint_primary_only and callable(checkpoint_ema):
                d["ema"] = checkpoint_ema(
                    is_primary=self._strategy.context.is_primary,
                )
            else:
                d["ema"] = ema.state_dict()
        return d

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if not isinstance(state, dict):
            raise TypeError("OnlineTrainer.load_state_dict expects a dict")

        self.state.step = int(state.get("step", 0))
        self.state.global_step = int(state.get("global_step", 0))

        nonzero_checkpoint = max(self.state.step, self.state.global_step) > 0
        if (
            nonzero_checkpoint
            and _requires_fp32_master_weights(self.model)
            and "optimizer" not in state
        ):
            message = (
                "nonzero low-precision checkpoint is missing optimizer state; "
                "FP32 master residuals cannot be reconstructed from rounded model weights"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s; initializing fresh masters from model weights", message)

        if "optimizer" in state:
            saved_manifest = state.get("optimizer_parameter_manifest")
            if saved_manifest is not None:
                current_manifest = self._optimizer_parameter_manifest()
                if saved_manifest != current_manifest:
                    message = (
                        "checkpoint optimizer parameter identity mismatch: "
                        f"checkpoint={saved_manifest!r}, runtime={current_manifest!r}"
                    )
                    if strict:
                        raise ValueError(message)
                    logger.warning("%s", message)
            optimizer = self._ensure_optimizer()
            optimizer_state = state["optimizer"]
            checkpoint_uses_fp32_master = isinstance(optimizer_state, Mapping) and (
                "fp32_master_weights" in optimizer_state
            )
            trainer_uses_fp32_master = isinstance(
                optimizer,
                FP32MasterWeightOptimizer,
            )
            if checkpoint_uses_fp32_master != trainer_uses_fp32_master:
                message = (
                    "checkpoint FP32 master-weight state does not match the current "
                    f"optimizer policy (checkpoint={checkpoint_uses_fp32_master}, "
                    f"current={trainer_uses_fp32_master})"
                )
                if strict:
                    raise ValueError(message)
                logger.warning("%s; loading only compatible optimizer state", message)
            try:
                self._strategy.load_optimizer_state(
                    self.model,
                    optimizer,
                    optimizer_state,
                )
            except Exception:
                if strict:
                    raise
                logger.warning("Skipping incompatible optimizer state during non-strict load")

        if "grad_scaler" in state:
            if self._grad_scaler is None:
                if strict:
                    raise ValueError(
                        "checkpoint contains GradScaler state but trainer fp16 scaling is disabled",
                    )
                logger.warning("Skipping GradScaler state because fp16 scaling is disabled")
            else:
                try:
                    self._grad_scaler.load_state_dict(state["grad_scaler"])
                except Exception:
                    if strict:
                        raise
                    logger.warning("Skipping incompatible GradScaler state during non-strict load")
        elif self._grad_scaler is not None and nonzero_checkpoint:
            if strict:
                raise ValueError(
                    "nonzero fp16 checkpoint is missing GradScaler state; strict resume "
                    "cannot reconstruct its loss-scale history",
                )
            logger.warning(
                "Resuming a nonzero fp16 checkpoint without GradScaler state; "
                "using a fresh dynamic loss scale",
            )

        if "ema" in state:
            if not self.config.ema.enable:
                if strict:
                    raise ValueError("checkpoint contains EMA state but trainer.ema.enable=false")
                logger.warning("Skipping EMA state because trainer.ema.enable=false")
            else:
                ema = self._ensure_ema()
                assert ema is not None
                _validate_ema_state_shapes(
                    state["ema"],
                    self.model,
                    strict=strict,
                )
                try:
                    ema.load_state_dict(state["ema"])
                except Exception:
                    if strict:
                        raise
                    logger.warning("Skipping incompatible EMA state during non-strict load")
        elif strict and self.config.ema.enable:
            raise ValueError("checkpoint missing EMA state but trainer.ema.enable=true")

        # A resumed trainer must push the restored driver weights before the
        # next Ray rollout. The policy version and worker state are runtime
        # concerns, not persisted as an initialized rollout flag.
        self._rollout_weights_initialized = False
        self._replay_parity_passed = False
        self._precision_drift_guard_pending = True
        self.rollout_schedule.reset()

    def _optimizer_parameter_manifest(self) -> list[dict[str, Any]]:
        """Stable named identity for positional optimizer checkpoint slots."""

        return [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
            }
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]


def _validate_ema_state_shapes(
    ema_state: dict[str, Any],
    model: nn.Module,
    *,
    strict: bool,
) -> None:
    ema_parameters = ema_state.get("ema_parameters") if isinstance(ema_state, dict) else None
    if not isinstance(ema_parameters, list):
        if strict:
            raise ValueError("checkpoint EMA state missing ema_parameters")
        return
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(ema_parameters) != len(trainable):
        if strict:
            raise ValueError(
                "checkpoint EMA parameter count mismatch: "
                f"checkpoint={len(ema_parameters)} current={len(trainable)}",
            )
        return
    for idx, (ema_param, param) in enumerate(zip(ema_parameters, trainable, strict=True)):
        if not isinstance(ema_param, torch.Tensor):
            if strict:
                raise ValueError(f"checkpoint EMA parameter {idx} is not a tensor")
            return
        if tuple(ema_param.shape) != tuple(param.shape):
            if strict:
                raise ValueError(
                    "checkpoint EMA parameter shape mismatch at index "
                    f"{idx}: checkpoint={tuple(ema_param.shape)} "
                    f"current={tuple(param.shape)}",
                )
            return
