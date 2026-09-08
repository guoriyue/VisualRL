"""Continuous-action GRPO for diffusion / flow-matching policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.algorithms.advantages import GroupAdvantageEstimator
from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.logprob_mismatch import (
    PrecisionCorrectionConfig,
    apply_rejection_sample_mask,
    apply_truncated_importance_weight,
    combine_keep_masks,
)
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics


@dataclass(slots=True)
class GroupAdvantageConfig:
    """Hyper-parameters shared only by group-relative advantage algorithms."""

    eps: float = 1e-4
    adv_clip_max: float = 5.0
    global_std: bool = False
    # How multiple reward components are combined into one advantage.
    # "weighted_sum_raw" (default, legacy): sum weighted raw rewards then
    # normalize once — a high-variance component dominates. "normalized_sum"
    # (DanceGRPO-style): normalize each component to a per-group advantage first,
    # then weighted-sum, so no reward dominates by scale/variance. New
    # multi-objective strategies plug into vrl.algorithms.advantages.
    advantage_combine: str = GroupAdvantageEstimator.DEFAULT_STRATEGY

    def __post_init__(self) -> None:
        GroupAdvantageEstimator.validate_strategy(self.advantage_combine)

    def build_estimator(
        self,
        *,
        component_weights: Mapping[str, float] | None = None,
    ) -> GroupAdvantageEstimator:
        """Build the runtime estimator from this algorithm configuration."""

        return GroupAdvantageEstimator(
            eps=self.eps,
            adv_clip_max=self.adv_clip_max,
            global_std=self.global_std,
            strategy=self.advantage_combine,
            component_weights=component_weights,
        )


@dataclass(slots=True)
class ClippedPolicyConfig(GroupAdvantageConfig):
    """Policy-ratio clipping and reference-KL knobs with real loss consumers."""

    clip_ratio: float = 0.2
    kl_coef: float = 0.0


@dataclass(slots=True)
class GRPOConfig(ClippedPolicyConfig):
    """Hyper-parameters for continuous GRPO."""

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        supports_step_kl_reward=True,
        sft_source="latents",
    )

    flow_kl_use_dt: bool = False
    # Diffusion-loss regularizer weight (Cosmos-Predict2.5 paper 4.2.2's
    # anti-reward-hacking term): adds sft_weight * MSE(model_pred,
    # pretraining_target) on CLEAN fine-tuning latents (data.sft_latents,
    # produced by vrl/scripts/denoise/encode_targets.py). The term is
    # computed by the trainer (it needs a model forward, which algorithm
    # losses never do); this knob rides the algorithm config so recipes tune
    # it next to kl_coef.
    sft_weight: float = 0.0


class GRPO:
    """Group Relative Policy Optimization for continuous rollout signals.

    Advantages are normalised within each prompt group:
        a_i = (r_i - mean(r)) / max(std(r), eps)

    Loss is the clipped surrogate objective (PPO-style) applied to
    per-sample log-probabilities produced by the evaluator.
    """

    uses_evaluator = True
    tolerates_off_policy_staleness = True

    # Signal-branch contract (AlgorithmAdapter.validate_inputs): the clipped
    # surrogate reads these from the evaluator replay. ref_log_prob is
    # conditional on kl_coef>0, so it is NOT a hard requirement here — the
    # KL branch validates it with its own detailed diagnostic.
    required_signal_keys = ("log_prob", "old_log_prob")
    required_data_keys: tuple[str, ...] = ()

    # Trust-region subclasses (Flow-DPPO / GRPO-Guard) flip this True so the
    # trainer requests the SDE KL intermediates (dt) even when kl_coef == 0.
    needs_kl_intermediates = False

    # Plain GRPO's clip is a safety rail, not the objective: at ppo_epochs=1 it
    # is honest REINFORCE-with-group-baseline (the ratio is ~1 and the clip is
    # simply inactive). Trust-region subclasses whose loss IS the ratio term flip
    # this True so the trainer rejects the strict + ppo_epochs=1 no-op config.
    requires_active_trust_region = False

    def __init__(
        self,
        config: GRPOConfig | None = None,
        *,
        advantage_estimator: GroupAdvantageEstimator | None = None,
    ) -> None:
        self.config = config or GRPOConfig()
        self._initialize_precision_correction()
        self._initialize_advantage_estimator(advantage_estimator)

    def _initialize_precision_correction(self) -> None:
        """Install the trainer-injected rollout/replay correction capability."""

        # Rollout->replay precision correction (TIS). Off by default; the trainer
        # injects trainer.precision_correction here at construction so the knobs
        # live at the trainer level, not in the algorithm's hyperparameters.
        self.precision_correction = PrecisionCorrectionConfig()

    def _initialize_advantage_estimator(
        self,
        advantage_estimator: GroupAdvantageEstimator | None,
    ) -> None:
        """Bind one resolved advantage strategy to this algorithm instance."""

        if advantage_estimator is None:
            advantage_estimator = self.config.build_estimator()
        self.advantage_estimator = advantage_estimator

    def compute_advantages_from_tensors(
        self,
        rewards: Any,
        group_ids: Any,
    ) -> Any:
        """Per-group advantage normalization on tensors.

        Groups are identified by ``group_ids``: samples sharing the same
        group_id are normalized together (GRPO per-prompt normalization).
        """
        return self.advantage_estimator.compute(
            rewards,
            group_ids,
        )

    def compute_advantages_from_components(
        self,
        rewards: Any,
        component_rewards: dict[str, Any],
        group_ids: Any,
    ) -> Any:
        """Compute advantages with optional raw reward-component observations.

        The default strategy consumes ``rewards``, the authoritative weighted
        total from the reward runtime. Component-aware strategies consume the
        raw observations using the weights bound to ``advantage_estimator``.
        """
        return self.advantage_estimator.compute(
            rewards,
            group_ids,
            component_rewards=component_rewards,
        )

    def compute_loss(
        self,
        inputs: AlgorithmInput,
    ) -> tuple[Any, TrainStepMetrics]:
        """Clipped surrogate loss from trajectory-native evaluator signals.

        Handles both flow-matching latent-space KL and generic log-prob KL.
        """
        import torch

        from vrl.math.denoise.flow_matching import compute_kl_divergence

        cfg = self.config
        # Presence of signals + required_signal_keys is enforced upstream by
        # AlgorithmAdapter.validate_inputs (one declarative gate).
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for GRPO")
        signals = inputs.signals.primary
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)
        old_log_probs = signals.old_log_prob

        raw_ratio = torch.exp(signals.log_prob - old_log_probs)
        # Truncated importance sampling on the rollout->replay weight before the PPO
        # clip, so quantized-rollout (FP8/NVFP4) drift on a few samples cannot dominate
        # the gradient via the unclipped negative-advantage branch.
        pc = self.precision_correction
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        # RS rejects whole samples whose rollout->replay log-ratio drift is out of
        # band — orthogonal to TIS (which clamps the per-element weight). Both feed
        # the masked-mean denominator below (true off-policy rejection, not a
        # gradient-magnitude dilution).
        rs_keep = apply_rejection_sample_mask(
            signals.log_prob - old_log_probs,
            pc,
            mask=signals.mask,
        )
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
        # Optional per-sample positive weight (Flash-GRPO's temporal gradient
        # rectification). Applied AFTER the max: for w > 0,
        # max(w*a, w*b) == w*max(a, b), so this is exactly the weighted clipped
        # surrogate without duplicating the loss body in the subclass.
        weight = self._loss_weight(signals)
        if weight is not None:
            per_sample_loss = per_sample_loss * self._broadcast_sample_values(
                weight,
                per_sample_loss,
            )
        # Unlike the historical scalar-per-denoise-step path, grouped causal
        # replay has a real transition mask. Fold it into the same denominator
        # as TIS/RS so deterministic cache-finalization passes never become
        # policy actions. Existing diffusion masks are all ones, preserving
        # their numerical behavior.
        keep = combine_keep_masks(signals.mask.to(ratio.dtype), tis_keep, rs_keep)
        if keep is not None:
            policy_loss = (per_sample_loss * keep).sum() / keep.sum().clamp_min(1.0)
            active_clip_fraction = (
                ((clipped_loss > unclipped_loss).to(keep.dtype) * keep).sum()
                / keep.sum().clamp_min(1.0)
            ).item()
        else:
            policy_loss = torch.mean(per_sample_loss)
            active_clip_fraction = (clipped_loss > unclipped_loss).float().mean().item()
        if tis_keep is not None:
            tis_clip_fraction = (1.0 - tis_keep.mean()).item()
        else:
            tis_clip_fraction = (
                0.0 if pc.tis_mode == "off" else (ratio != raw_ratio).float().mean().item()
            )
        rs_seq_masked_fraction = 0.0 if rs_keep is None else (1.0 - rs_keep.mean()).item()

        if cfg.kl_coef > 0:
            if signals.ref_log_prob is None:
                raise RuntimeError(
                    f"GRPOConfig.kl_coef={cfg.kl_coef} > 0 but "
                    "signals.ref_log_prob is None. Check: (1) ref_model "
                    "passed to OnlineTrainer, (2) SignalRequest(need_ref=True) "
                    "in the evaluator call."
                )
            if (
                signals.distribution == "flow_matching"
                and signals.prev_sample_mean is not None
                and signals.ref_prev_sample_mean is not None
            ):
                kl = compute_kl_divergence(
                    signals.prev_sample_mean,
                    signals.ref_prev_sample_mean,
                    signals.std_dev_t,
                    sqrt_neg_dt=signals.dt if cfg.flow_kl_use_dt else None,
                )
                kl_loss = torch.mean(kl)
            else:
                kl_loss = torch.mean(signals.log_prob - signals.ref_log_prob)
            kl_term = cfg.kl_coef * kl_loss
            loss = policy_loss + kl_term
        else:
            kl_loss = torch.tensor(0.0, device=signals.log_prob.device)
            kl_term = torch.tensor(0.0, device=signals.log_prob.device)
            loss = policy_loss

        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_ratio).float()).item()
        approx_kl = 0.5 * torch.mean((signals.log_prob - old_log_probs) ** 2).item()

        metrics = TrainStepMetrics(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_loss.item(),
            weighted_kl_loss=kl_term.item(),
            update=PolicyUpdateStats(
                clip_fraction=clip_fraction,
                active_clip_fraction=active_clip_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )

        return loss, metrics

    def _loss_weight(self, signals: Any) -> Any | None:
        """Per-sample positive weight on the clipped surrogate; None = unweighted.

        The Flash-GRPO subclass overrides this with its temporal gradient
        rectification factor. Base GRPO (and the trust-region subclasses, which
        own their loss bodies) return None.
        """

        return None

    @staticmethod
    def _broadcast_sample_values(values: Any, target: Any) -> Any:
        """Expand one value per sample across grouped policy-action axes."""

        value_shape = getattr(values, "shape", None)
        target_shape = getattr(target, "shape", None)
        if value_shape is None or target_shape is None:
            return values
        if not value_shape or not target_shape or int(value_shape[0]) != int(target_shape[0]):
            raise ValueError(
                "sample values and policy signals must share their leading batch axis: "
                f"{tuple(value_shape)} vs {tuple(target_shape)}",
            )
        while values.ndim < target.ndim:
            values = values.unsqueeze(-1)
        return values


@dataclass(slots=True)
class FlashGRPOConfig(GRPOConfig):
    """Hyper-parameters for Flash-GRPO (one-step policy optimization).

    Inherits every GRPO knob; the default ``clip_ratio`` drops to the paper's
    1e-3. At ppo_epochs=1 the policy does not move within an update, so the
    ratio only deviates from 1 through rollout-vs-replay numeric drift — the
    tight clip is a drift rail, not a trust region.
    """

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        supports_step_kl_reward=True,
        sft_source="unsupported",
    )

    clip_ratio: float = 1e-3


class FlashGRPO(GRPO):
    """Flash-GRPO: GRPO with per-timestep gradient rectification.

    arXiv:2605.15980 (ICML 2026). The full recipe is three mechanisms; this
    class owns the loss-side one, the other two are rollout/trainer config:

    1. Single stochastic step (rollout): ``rollout.sde.window_size=1`` with a
       ``window_range`` over the noisy early steps — one SDE step per
       trajectory, ODE elsewhere. One action per sample -> one transition to
       replay and train.
    2. Iso-temporal grouping (rollout): the window is drawn once per generation
       request, so every sample of a prompt group shares the same timestep and
       the group advantage is never confounded by timestep difficulty.
    3. Temporal gradient rectification (THIS class): under the flow-matching
       SDE with mean
       ``mu = x*(1 + std^2/(2s)*dt) + v*(1 + std^2*(1-s)/(2s))*dt`` and noise
       scale ``std*sqrt(-dt)``, the log-prob gradient w.r.t. the velocity
       scales as

           c(t) = sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma)

       which varies ~2x across the trained window (verified: 1/c reproduces
       the reference implementation's hardcoded per-timestep table
       {999: 7.4770 ... 785: 3.7754} on the Wan 20-step shift-3 schedule to
       0.1%). The loss weight is ``w_i = (1/c_i) / mean(1/c)`` with the mean
       reduced across ranks, so per-timestep gradient magnitudes equalize
       while the effective learning rate is untouched (batch-mean weight = 1).

    The cross-rank mean matches the reference's gradient_accumulation==1
    branch (per-microbatch all-reduced mean); the reference's accumulation
    branch normalizes over the whole update window instead — same expectation,
    slightly different variance.
    """

    # Rectification reads std_dev_t / dt from the SDE intermediates, exactly
    # like the trust-region subclasses.
    needs_kl_intermediates = True

    def _loss_weight(self, signals: Any) -> Any:
        import torch

        _require_rectification_signals(signals, "FlashGRPO")

        def _per_sample(value: Any) -> Any:
            tensor = torch.as_tensor(value)
            if tensor.ndim > 1:
                return tensor.reshape(tensor.shape[0], -1).mean(dim=1)
            return tensor

        std = _per_sample(signals.std_dev_t).float()
        sqrt_neg_dt = _per_sample(signals.dt).float()
        sigma = _per_sample(signals.sigma).float().clamp_min(1e-6)
        grad_scale = sqrt_neg_dt / std + std * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
        coe = 1.0 / grad_scale.clamp_min(1e-12)
        return coe / _cross_rank_mean(coe).clamp_min(1e-12)


def _require_rectification_signals(signals: Any, algorithm: str) -> None:
    """Fail fast when the SDE intermediates the rectification needs are absent.

    A missing input would otherwise silently degrade to unweighted GRPO —
    quietly changing the objective, the same failure mode
    ``_require_trust_region_signals`` guards for the trust-region losses.
    """

    if signals.std_dev_t is None or signals.dt is None or signals.sigma is None:
        raise RuntimeError(
            f"{algorithm} requires flow-matching SDE signals "
            "(std_dev_t / dt / sigma). dt comes from the evaluator's KL "
            "intermediates (needs_kl_intermediates=True drives "
            "SignalRequest(need_kl_intermediates=True)); sigma is produced only "
            "by the flow-matching SDE path — it is None on DDIM/token replays, "
            "which Flash-GRPO does not support.",
        )


def _cross_rank_mean(values: Any) -> Any:
    """Mean of ``values`` over all training ranks.

    Called once per microbatch by every rank in lockstep (the trainer's
    unanimous-work gate keeps microbatch counts balanced), so the collective
    cannot deadlock — the same argument as ``_population_std_across_ranks``.
    An empty local tensor must NOT short-circuit before the collective:
    emptiness is rank-local, so an empty rank contributes zeros instead.
    """

    from vrl.algorithms.advantages import all_reduce_sufficient_stats

    g_sum, _g_sumsq, g_count = all_reduce_sufficient_stats(values)
    return (g_sum / g_count.clamp_min(1.0)).to(values.device)


def _require_trust_region_signals(signals: Any, algorithm: str) -> Any:
    """Validate the SDE signals trust-region losses need; return the rollout mean.

    Both Flow-DPPO and GRPO-Guard read the rollout proposal mean plus the per-step
    diffusion intermediates (std_dev_t, sqrt_dt). dt is a hard requirement, not an
    optional input: a missing dt would silently drop the diffusion coefficient
    (Flow-DPPO) or collapse the step-scale to 1 (GRPO-Guard), quietly changing the
    objective — so fail fast instead.
    """

    value = signals.old_prev_sample_mean
    if value is None:
        raise RuntimeError(
            f"{algorithm} needs signals.old_prev_sample_mean (the rollout-time "
            "reverse-SDE proposal mean), but it is None. Set "
            "sampling.return_prev_sample_mean=true so generation stores it into "
            "the trajectory.",
        )
    if signals.prev_sample_mean is None or signals.std_dev_t is None or signals.dt is None:
        raise RuntimeError(
            f"{algorithm} requires flow-matching SDE signals "
            "(prev_sample_mean / std_dev_t / dt). dt comes from the evaluator's "
            "KL intermediates; needs_kl_intermediates=True must drive "
            "SignalRequest(need_kl_intermediates=True).",
        )
    return value


@dataclass(slots=True)
class FlowDPPOConfig(GroupAdvantageConfig):
    """Flow-DPPO: exact-Gaussian-KL trust region instead of the PPO ratio clip."""

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        supports_step_kl_reward=True,
        sft_source="unsupported",
    )

    # Per-sample latent KL above which an update that *widens* the gap from the
    # rollout policy is dropped (the trust-region boundary).
    kl_mask_threshold: float = 1.0
    # Fold the per-step diffusion coefficient sqrt(-dt) into the KL sigma
    # (sigma_t = std_dev_t * sqrt_dt) rather than std_dev_t alone.
    add_kl_coefficient: bool = True


class FlowDPPO(GRPO):
    """Trust-region GRPO: mask high-KL, gap-widening samples (no ratio clip).

    Asymmetric by construction — only updates that *increase* the divergence from
    the rollout policy are dropped (positive advantage pushing ratio up, or
    negative advantage pushing ratio down). Updates that pull back toward the old
    policy are always kept. This is the key difference from PPO's symmetric clip.
    """

    needs_kl_intermediates = True
    # The KL mask is the objective: at strict + ppo_epochs=1 the rollout and
    # current proposal means coincide, KL==0, nothing is masked, and the loss
    # collapses to -advantages * 1 (plain REINFORCE). Require a moving policy.
    requires_active_trust_region = True

    def __init__(
        self,
        config: FlowDPPOConfig | None = None,
        *,
        advantage_estimator: GroupAdvantageEstimator | None = None,
    ) -> None:
        cfg = config or FlowDPPOConfig()
        self.config: FlowDPPOConfig = cfg
        self._initialize_precision_correction()
        self._initialize_advantage_estimator(advantage_estimator)

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        import torch

        from vrl.math.denoise.flow_matching import compute_kl_divergence

        cfg = self.config
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for FlowDPPO")
        signals = inputs.signals.primary
        old_prev_sample_mean = _require_trust_region_signals(signals, "FlowDPPO")
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)

        raw_ratio = torch.exp(signals.log_prob - signals.old_log_prob)
        # Bound rollout->replay precision drift (FP8/NVFP4 rollout) before it
        # enters the trust-region loss — the same TIS/RS the base GRPO applies.
        # Without it a quantized rollout's logprob drift flows unclipped into the
        # negative-advantage branch; the trust-region KL mask below only catches
        # *policy* drift, not *precision* drift. No-op when precision is not split
        # (tis_mode/rs_mode default to "off").
        pc = self.precision_correction
        ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
        rs_keep = apply_rejection_sample_mask(
            signals.log_prob - signals.old_log_prob,
            pc,
            mask=signals.mask,
        )
        # Gaussian KL between the current and rollout proposal means (the
        # current-vs-rollout drift). With add_kl_coefficient the sigma folds in the
        # per-step diffusion coefficient (sigma_t = std_dev_t * sqrt_dt, the closed
        # form compute_kl_divergence uses); without it the trust region is
        # unit-variance — mean_diff_sq / 2, with NO std_dev_t in the denominator
        # (matches verl-omni's add_kl_coefficient=False branch).
        if cfg.add_kl_coefficient:
            kl_per_sample = compute_kl_divergence(
                signals.prev_sample_mean,
                old_prev_sample_mean,
                signals.std_dev_t,
                sqrt_neg_dt=signals.dt,
            )
        else:
            non_batch = tuple(range(1, signals.prev_sample_mean.ndim))
            kl_per_sample = (signals.prev_sample_mean - old_prev_sample_mean).pow(2).mean(
                dim=non_batch
            ) / 2.0
        high_kl = self._broadcast_sample_values(
            kl_per_sample >= cfg.kl_mask_threshold,
            ratio,
        )
        pos_rm = high_kl & (ratio > 1.0) & (advantages > 0)
        neg_rm = high_kl & (ratio < 1.0) & (advantages < 0)
        trust_keep = (~(pos_rm | neg_rm)).detach()
        # Intersect the trust-region keep with the TIS/RS precision keeps; when
        # precision is not split both are None and ``keep`` collapses to
        # ``trust_keep`` (exact legacy behavior).
        keep = combine_keep_masks(
            signals.mask.to(ratio.dtype),
            trust_keep.to(ratio.dtype),
            tis_keep,
            rs_keep,
        )
        unclipped_loss = -advantages * ratio
        # Masked mean, matching GRPO/GRPOGuard/TokenGRPO: the denominator is the
        # KEPT count, not the batch size. Dividing by the batch size would scale
        # the gradient by the keep fraction, so the effective learning rate would
        # shrink exactly as the trust region engages — a gradient-magnitude
        # dilution, not the true off-policy rejection the mask is meant to be.
        policy_loss = (unclipped_loss * keep).sum() / keep.sum().clamp_min(1.0)

        masked_fraction = (1.0 - keep.mean()).item()
        tis_clip_fraction = (1.0 - tis_keep.mean()).item() if tis_keep is not None else 0.0
        rs_seq_masked_fraction = (1.0 - rs_keep.mean()).item() if rs_keep is not None else 0.0
        approx_kl = (
            0.5
            * torch.mean(
                (signals.log_prob - signals.old_log_prob) ** 2,
            ).item()
        )
        metrics = TrainStepMetrics(
            loss=policy_loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=kl_per_sample.mean().item(),
            update=PolicyUpdateStats(
                clip_fraction=masked_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )
        return policy_loss, metrics


@dataclass(slots=True)
class GRPOGuardConfig(GroupAdvantageConfig):
    """GRPO-Guard: ratio-mean-bias correction + per-step magnitude normalization.

    The guard terms are derived from the per-step diffusion scale.
    """

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=True,
        supports_step_kl_reward=True,
        sft_source="unsupported",
    )

    clip_ratio: float = 0.2


class GRPOGuard(GRPO):
    """FlowGRPO with an additive ratio-mean-bias and 1/sqrt_dt**2 step-scale norm.

    Unlike Flow-DPPO (which *drops* high-KL samples), GRPO-Guard keeps every
    sample but folds the current-vs-rollout mean drift into the ratio exponent
    (a soft correction) and normalizes the loss magnitude across denoise steps so
    early and late timesteps contribute comparably.
    """

    needs_kl_intermediates = True
    # The ratio-mean-bias / step-scale guard is the objective: at strict +
    # ppo_epochs=1 the current-vs-rollout drift is 0, the guard correction
    # vanishes, and the loss collapses to plain GRPO. Require a moving policy.
    requires_active_trust_region = True

    def __init__(
        self,
        config: GRPOGuardConfig | None = None,
        *,
        advantage_estimator: GroupAdvantageEstimator | None = None,
    ) -> None:
        cfg = config or GRPOGuardConfig()
        self.config: GRPOGuardConfig = cfg
        self._initialize_precision_correction()
        self._initialize_advantage_estimator(advantage_estimator)

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        import torch

        cfg = self.config
        if inputs.advantages is None:
            raise RuntimeError("AlgorithmInput.advantages is required for GRPOGuard")
        signals = inputs.signals.primary
        old_prev_sample_mean = _require_trust_region_signals(signals, "GRPOGuard")
        advantages = self._broadcast_sample_values(inputs.advantages, signals.log_prob)

        log_ratio = signals.log_prob - signals.old_log_prob
        # Bound rollout->replay precision drift (FP8/NVFP4 rollout). GRPO-Guard keeps
        # every sample by design, so TIS-*truncate* on the raw weight does not touch
        # the soft-corrected guard ratio; RS (whole-sample band rejection on the raw
        # rollout->replay log-ratio) is the effective precision guard here, plus
        # TIS-*mask* when configured. No-op when precision is not split.
        pc = self.precision_correction
        _, tis_keep = apply_truncated_importance_weight(torch.exp(log_ratio), pc)
        rs_keep = apply_rejection_sample_mask(log_ratio, pc, mask=signals.mask)
        # dt is guaranteed present by _require_trust_region_signals (no silent
        # fallback-to-1, which would erase the per-step scale normalization).
        sqrt_dt_mean = signals.dt.mean()
        scale = sqrt_dt_mean * signals.std_dev_t.mean()
        non_batch = tuple(range(1, signals.prev_sample_mean.ndim))
        mean_diff_sq = (signals.prev_sample_mean - old_prev_sample_mean).pow(2).mean(dim=non_batch)
        ratio_mean_bias = mean_diff_sq / (2 * scale.pow(2))
        # Project the mean drift onto the log-ratio scale, then exponentiate.
        ratio = torch.exp((log_ratio + ratio_mean_bias) * scale)
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        per_sample_loss = torch.maximum(unclipped_loss, clipped_loss)
        # Reject out-of-band precision-drift samples; collapses to the plain mean
        # when no precision keep is active.
        keep = combine_keep_masks(signals.mask.to(ratio.dtype), tis_keep, rs_keep)
        if keep is not None:
            reduced = (per_sample_loss * keep).sum() / keep.sum().clamp_min(1.0)
            active_clip_fraction = (
                ((clipped_loss > unclipped_loss).to(keep.dtype) * keep).sum()
                / keep.sum().clamp_min(1.0)
            ).item()
        else:
            reduced = per_sample_loss.mean()
            active_clip_fraction = (clipped_loss > unclipped_loss).float().mean().item()
        # Per-step magnitude normalization (cross-timestep consistent gradients).
        policy_loss = reduced / sqrt_dt_mean.pow(2).clamp_min(1e-12)

        clip_fraction = torch.mean(
            (torch.abs(ratio - 1.0) > cfg.clip_ratio).float(),
        ).item()
        tis_clip_fraction = (1.0 - tis_keep.mean()).item() if tis_keep is not None else 0.0
        rs_seq_masked_fraction = (1.0 - rs_keep.mean()).item() if rs_keep is not None else 0.0
        approx_kl = 0.5 * torch.mean(log_ratio**2).item()
        metrics = TrainStepMetrics(
            loss=policy_loss.item(),
            policy_loss=policy_loss.item(),
            kl_penalty=ratio_mean_bias.mean().item(),
            update=PolicyUpdateStats(
                clip_fraction=clip_fraction,
                active_clip_fraction=active_clip_fraction,
                approx_kl=approx_kl,
                tis_clip_fraction=tis_clip_fraction,
                rs_seq_masked_fraction=rs_seq_masked_fraction,
            ),
        )
        return policy_loss, metrics
