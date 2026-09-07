"""Tests for continuous GRPO advantages + loss."""

from __future__ import annotations

import math

import pytest
import torch

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

# ---------------------------------------------------------------------------
# Regression: single-sample GRPO advantage must NOT be NaN
# ---------------------------------------------------------------------------


class TestGRPOSingleSampleNaN:
    """Groups tests for grposingle sample na n."""

    def test_single_sample_returns_zero_not_nan(self) -> None:
        """Single sample per group → advantage = 0.0, NOT NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([5.0])
        group_ids = torch.tensor([0])
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any(), f"Got NaN advantages: {advantages}"
        assert advantages[0].item() == pytest.approx(0.0)

    def test_multiple_single_sample_groups(self) -> None:
        """Multiple groups each with 1 sample → all advantages = 0."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 5.0, 10.0])
        group_ids = torch.tensor([0, 1, 2])  # each prompt has 1 sample
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        assert torch.allclose(advantages, torch.zeros(3))

    def test_group_with_multiple_samples_works(self) -> None:
        """Group with multiple samples → proper normalization, no NaN."""
        import torch

        grpo = GRPO()
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        group_ids = torch.tensor([0, 0, 0, 0])  # all same prompt
        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)
        assert not torch.isnan(advantages).any()
        # Mean=4, should be negative for 1,3 and positive for 5,7
        assert advantages[0] < 0
        assert advantages[3] > 0

    @pytest.mark.parametrize(
        ("dtype", "global_std"),
        [(torch.bfloat16, False), (torch.float32, True)],
    )
    def test_constant_non_binary_reward_returns_exact_zero(
        self,
        dtype: torch.dtype,
        global_std: bool,
    ) -> None:
        """A constant decimal reward must not create a rounding-error gradient."""

        grpo = GRPO(GRPOConfig(eps=1e-8, global_std=global_std))
        rewards = torch.full((8,), 0.9, dtype=dtype)
        group_ids = torch.zeros(8, dtype=torch.long)

        advantages = grpo.compute_advantages_from_tensors(rewards, group_ids)

        assert torch.isfinite(advantages).all()
        assert torch.equal(advantages, torch.zeros_like(advantages))


class TestGRPOFlowMatchingKL:
    """Groups tests for grpoflow matching KL."""

    def test_flow_kl_ignores_dt_by_default_to_match_flow_grpo(self) -> None:
        """Flow KL defaults to ``0.5 * ||mean - ref_mean||^2 / std^2`` without the ``dt`` factor,
        matching Flow-GRPO (kl 0.5 -> loss 0.02 at coef 0.04).
        """
        grpo = GRPO(GRPOConfig(kl_coef=0.04))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
        )

        loss, metrics = grpo.compute_loss(
            AlgorithmInput(
                signals=signals,
                advantages=torch.zeros(2),
            ),
        )

        assert loss.item() == pytest.approx(0.02)
        assert metrics.kl_penalty == pytest.approx(0.5)
        assert metrics.weighted_kl_loss == pytest.approx(0.02)

    def test_flow_kl_can_use_dt_when_explicitly_configured(self) -> None:
        """``flow_kl_use_dt=True`` scales the flow KL by the step size (kl 0.5 -> 50 at dt=0.1)."""
        grpo = GRPO(GRPOConfig(kl_coef=1.0, flow_kl_use_dt=True))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
            ref_log_prob=torch.zeros(2),
            prev_sample_mean=torch.zeros(2, 1, 1, 1),
            ref_prev_sample_mean=torch.ones(2, 1, 1, 1),
            std_dev_t=torch.ones(2, 1, 1, 1),
            dt=torch.full((2, 1, 1, 1), 0.1),
        )

        loss, metrics = grpo.compute_loss(
            AlgorithmInput(
                signals=signals,
                advantages=torch.zeros(2),
            ),
        )

        assert loss.item() == pytest.approx(50.0)
        assert metrics.kl_penalty == pytest.approx(50.0)


class TestGRPOClippedSurrogate:
    """Numeric checks on the PPO-clipped surrogate (continuous.py:102-141).

    These guard the classic sign trap in ``torch.maximum(unclipped, clipped)``
    on the negative-advantage side, plus the clip_fraction / approx_kl metrics.
    """

    def test_ratio_one_gives_negative_mean_advantage(self) -> None:
        """log_prob == old_log_prob → ratio 1 → loss == -mean(advantage)."""
        grpo = GRPO(GRPOConfig(kl_coef=0.0))
        signals = _flow_signals(
            log_prob=torch.zeros(2),
            old_log_prob=torch.zeros(2),
        )
        advantages = torch.tensor([2.0, -4.0])
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=advantages),
        )
        assert loss.item() == pytest.approx(-advantages.mean().item())
        assert metrics.policy_loss == pytest.approx(-advantages.mean().item())
        # No drift from old policy → no clipping, zero approx-KL.
        assert metrics.update.clip_fraction == pytest.approx(0.0)
        assert metrics.update.approx_kl == pytest.approx(0.0)

    def test_positive_advantage_uses_clipped_ratio_when_ratio_high(self) -> None:
        """Positive advantage + ratio above 1+clip_ratio → loss uses clipped ratio.

        ``maximum(-adv*ratio, -adv*clipped)`` with adv>0 picks the larger
        (less negative) term, i.e. the clipped one. So the surrogate is
        ``-adv*(1+clip_ratio)``, not ``-adv*ratio``.
        """
        clip_ratio = 0.2
        grpo = GRPO(GRPOConfig(kl_coef=0.0, clip_ratio=clip_ratio))
        log_ratio = 1.0  # ratio = e ≈ 2.718, well above 1+clip_ratio
        signals = _flow_signals(
            log_prob=torch.full((1,), log_ratio),
            old_log_prob=torch.zeros(1),
        )
        adv = torch.tensor([3.0])
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=adv),
        )
        assert loss.item() == pytest.approx(-adv.item() * (1.0 + clip_ratio))
        assert metrics.update.clip_fraction == pytest.approx(1.0)
        assert metrics.update.active_clip_fraction == pytest.approx(1.0)

    def test_negative_advantage_uses_unclipped_ratio_when_ratio_high(self) -> None:
        """Negative advantage + ratio above 1+clip_ratio → unclipped term wins.

        With adv<0, ``-adv*ratio`` is positive and larger than
        ``-adv*clipped``; ``maximum`` selects the unclipped surrogate. This is
        the side where a naive ``min``/``max`` mix-up silently flips training.
        """
        clip_ratio = 0.2
        grpo = GRPO(GRPOConfig(kl_coef=0.0, clip_ratio=clip_ratio))
        log_ratio = 1.0
        signals = _flow_signals(
            log_prob=torch.full((1,), log_ratio),
            old_log_prob=torch.zeros(1),
        )
        adv = torch.tensor([-3.0])
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=adv),
        )
        ratio = torch.exp(torch.tensor(log_ratio)).item()
        assert loss.item() == pytest.approx(-adv.item() * ratio)
        # The ratio is outside the band, but this direction pulls the policy
        # back toward the old policy, so the differentiable surrogate remains.
        assert metrics.update.clip_fraction == pytest.approx(1.0)
        assert metrics.update.active_clip_fraction == pytest.approx(0.0)

    def test_clip_fraction_and_approx_kl_match_formula(self) -> None:
        """clip_fraction = mean(|ratio-1|>clip_ratio); approx_kl = 0.5*mean(d^2)."""
        clip_ratio = 0.2
        grpo = GRPO(GRPOConfig(kl_coef=0.0, clip_ratio=clip_ratio))
        # One sample drifts hard (clipped), one stays put (not clipped).
        log_prob = torch.tensor([1.0, 0.0])
        old_log_prob = torch.zeros(2)
        signals = _flow_signals(log_prob=log_prob, old_log_prob=old_log_prob)
        _, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.ones(2)),
        )
        # Independent literal anchors, not the metric's own formula recomputed:
        # ratio = exp([1, 0]) -> |e-1| > 0.2 clipped, |1-1| < 0.2 not -> 0.5 clipped;
        # approx_kl = 0.5 * mean([1, 0] ** 2) = 0.25.
        assert metrics.update.clip_fraction == pytest.approx(0.5)
        assert metrics.update.active_clip_fraction == pytest.approx(0.5)
        assert metrics.update.approx_kl == pytest.approx(0.25)

    def test_objective_leaves_logprob_mismatch_to_replay_boundary(self) -> None:
        """The objective reports update math; the trainer owns backend parity."""
        grpo = GRPO(GRPOConfig(kl_coef=0.0))
        signals = _flow_signals(
            log_prob=torch.full((2,), 0.1),  # replay logprob
            old_log_prob=torch.zeros(2),  # rollout behavior logprob
        )
        _, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.ones(2)),
        )
        assert metrics.logprob_mismatch.logprob_abs_diff_mean == 0.0
        assert metrics.logprob_mismatch.ratio_abs_dev_mean == 0.0
        assert metrics.logprob_mismatch.finite is True

    def test_mismatch_metrics_zero_when_on_policy(self) -> None:
        """fresh == old → all mismatch metrics are zero."""
        grpo = GRPO(GRPOConfig(kl_coef=0.0))
        signals = _flow_signals(log_prob=torch.zeros(3), old_log_prob=torch.zeros(3))
        _, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.ones(3)),
        )
        assert metrics.logprob_mismatch.logprob_abs_diff_mean == 0.0
        assert metrics.logprob_mismatch.ratio_abs_dev_max == 0.0
        assert metrics.logprob_mismatch.mismatch_k3_kl == 0.0


class TestGRPOTruncatedImportanceSampling:
    """TIS bounds the rollout->replay importance weight (continuous.py).

    These guard the FP8/NVFP4 rollout correction path: under quantized rollout the
    weight exp(replay - rollout) inflates on a few samples and, on negative
    advantages, drives a large *unclipped* gradient. TIS caps/masks it. The TIS
    knobs live on the algorithm's injected ``precision_correction`` (the trainer
    sets it from ``trainer.precision_correction``), not in GRPOConfig.
    """

    @staticmethod
    def _grpo(**pc_kwargs) -> GRPO:
        grpo = GRPO(GRPOConfig(kl_coef=0.0))
        if pc_kwargs:
            grpo.precision_correction = PrecisionCorrectionConfig(**pc_kwargs)
        return grpo

    def test_off_mode_matches_legacy_and_reports_zero_clip(self) -> None:
        """The default (off) precision correction is a pure no-op vs the legacy loss."""
        adv = torch.tensor([2.0, -3.0, 1.0])

        def _signals():
            return _flow_signals(
                log_prob=torch.tensor([1.0, 0.0, -0.5]),
                old_log_prob=torch.zeros(3),
            )

        base, _ = self._grpo().compute_loss(AlgorithmInput(signals=_signals(), advantages=adv))
        off, off_m = self._grpo(tis_mode="off").compute_loss(
            AlgorithmInput(signals=_signals(), advantages=adv),
        )
        assert off.item() == pytest.approx(base.item())
        assert off_m.update.tis_clip_fraction == 0.0

    def test_truncate_caps_unclipped_negative_advantage_gradient(self) -> None:
        """truncate caps the ratio, so the dangerous adv<0 unclipped branch is bounded.

        Without TIS, adv=-3 at ratio=e gives loss -adv*e ≈ 8.15. With cap=2.0 the
        ratio is truncated to 2.0 → loss -adv*2.0 = 6.0.
        """
        cap = 2.0
        grpo = self._grpo(tis_mode="truncate", tis_imp_weight_cap=cap)
        signals = _flow_signals(log_prob=torch.full((1,), 1.0), old_log_prob=torch.zeros(1))
        adv = torch.tensor([-3.0])
        loss, metrics = grpo.compute_loss(AlgorithmInput(signals=signals, advantages=adv))
        assert loss.item() == pytest.approx(-adv.item() * cap)
        assert metrics.update.tis_clip_fraction == pytest.approx(1.0)

    def test_truncate_leaves_in_range_samples_untouched(self) -> None:
        """A ratio below the cap is unchanged; clip fraction reflects only altered ones."""
        grpo = self._grpo(tis_mode="truncate", tis_imp_weight_cap=2.0)
        # ratio: e (>cap, truncated) and 1.0 (untouched)
        signals = _flow_signals(log_prob=torch.tensor([1.0, 0.0]), old_log_prob=torch.zeros(2))
        _, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.tensor([-3.0, 1.0])),
        )
        assert metrics.update.tis_clip_fraction == pytest.approx(0.5)

    def test_mask_drops_out_of_range_samples_from_mean(self) -> None:
        """mask mode rejects catastrophic-drift samples entirely (masked mean)."""
        grpo = self._grpo(tis_mode="mask", tis_imp_weight_cap=2.0)
        # sample0 ratio=e>cap rejected; sample1 ratio=1 kept → loss = -adv1*1 = -1.0
        signals = _flow_signals(log_prob=torch.tensor([1.0, 0.0]), old_log_prob=torch.zeros(2))
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.tensor([5.0, 1.0])),
        )
        assert loss.item() == pytest.approx(-1.0)
        assert metrics.update.tis_clip_fraction == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tis_mode": "nope"},
            {"tis_mode": "clip", "tis_clip_low": 2.0, "tis_imp_weight_cap": 2.0},
        ],
    )
    def test_invalid_precision_correction_config_raises(self, kwargs) -> None:
        with pytest.raises(ValueError):
            PrecisionCorrectionConfig(**kwargs)


class TestGRPORejectSampling:
    """RS rejects whole samples on log-ratio drift, orthogonal to TIS (continuous).

    For continuous GRPO the log-prob is per-sample, so RS judges each sample's
    trajectory log-ratio against the band and drops rejected ones from the mean
    denominator (true off-policy rejection, not gradient dilution).
    """

    @staticmethod
    def _grpo(**pc_kwargs) -> GRPO:
        grpo = GRPO(GRPOConfig(kl_coef=0.0))
        if pc_kwargs:
            grpo.precision_correction = PrecisionCorrectionConfig(**pc_kwargs)
        return grpo

    def test_off_mode_is_noop(self) -> None:
        adv = torch.tensor([2.0, -3.0, 1.0])

        def _sig():
            return _flow_signals(
                log_prob=torch.tensor([1.0, 0.0, -0.5]),
                old_log_prob=torch.zeros(3),
            )

        base, _ = self._grpo().compute_loss(AlgorithmInput(signals=_sig(), advantages=adv))
        off, off_m = self._grpo(rs_mode="off").compute_loss(
            AlgorithmInput(signals=_sig(), advantages=adv),
        )
        assert off.item() == pytest.approx(base.item())
        assert off_m.update.rs_seq_masked_fraction == 0.0

    def test_out_of_band_sample_dropped_from_mean(self) -> None:
        # sample0 log-ratio=ln3 (ratio 3) above high=ln2 → rejected; sample1 ratio 1 kept.
        grpo = self._grpo(rs_mode="seq_mean_k1")
        signals = _flow_signals(
            log_prob=torch.tensor([math.log(3.0), 0.0]),
            old_log_prob=torch.zeros(2),
        )
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.tensor([5.0, 1.0])),
        )
        assert loss.item() == pytest.approx(-1.0)  # only kept sample1: -adv1*1
        assert metrics.update.rs_seq_masked_fraction == pytest.approx(0.5)

    def test_rs_and_tis_combine_in_denominator(self) -> None:
        # ratios [3, 1.5, 1]; TIS(mask,cap=2) drops s0; RS(band) also drops s0.
        # kept = {s1, s2}; loss = mean over the two kept per-sample losses.
        grpo = self._grpo(tis_mode="mask", tis_imp_weight_cap=2.0, rs_mode="seq_mean_k1")
        signals = _flow_signals(
            log_prob=torch.tensor([math.log(3.0), math.log(1.5), 0.0]),
            old_log_prob=torch.zeros(3),
        )
        loss, metrics = grpo.compute_loss(
            AlgorithmInput(signals=signals, advantages=torch.tensor([5.0, 2.0, 1.0])),
        )
        # s1: ratio 1.5 clipped to 1.2 at clip_ratio 0.2 → max(-2*1.5, -2*1.2) = -2.4
        # s2: ratio 1 → -1.0 ; mean over the 2 kept = -1.7
        assert loss.item() == pytest.approx(-1.7)
        assert metrics.update.tis_clip_fraction == pytest.approx(1 / 3)
        assert metrics.update.rs_seq_masked_fraction == pytest.approx(1 / 3)


def _flow_signals(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor | None = None,
    prev_sample_mean: torch.Tensor | None = None,
    ref_prev_sample_mean: torch.Tensor | None = None,
    std_dev_t: torch.Tensor | None = None,
    dt: torch.Tensor | None = None,
) -> TrajectorySignalBatch:
    return TrajectorySignalBatch(
        segments={
            "denoise": SegmentSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=torch.ones_like(log_prob),
                ref_log_prob=ref_log_prob,
                prev_sample_mean=prev_sample_mean,
                ref_prev_sample_mean=ref_prev_sample_mean,
                std_dev_t=std_dev_t,
                dt=dt,
            ),
        },
        group_ids=torch.arange(log_prob.shape[0], device=log_prob.device),
        primary_segment="denoise",
    )
