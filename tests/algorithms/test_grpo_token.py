"""Tests for vrl.algorithms.grpo.token.TokenGRPO.

Validates:
  * Inherits group-relative advantage from GRPO unchanged.
  * Per-token PPO clipped surrogate broadcasts [B] advantages → [B, L].
  * Mask zero → token excluded from loss.
  * KL k1, k2, and k3 estimators.
  * Loss == 0 when ratio==1 and KL==0.
"""

from __future__ import annotations

import math

import pytest
import torch

from vrl.algorithms.grpo.token import TokenGRPO, TokenGRPOConfig
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

# ---------------------------------------------------------------------------
# Advantage path — should be byte-identical to GRPO
# ---------------------------------------------------------------------------


class TestAdvantageInheritance:
    """Groups tests for advantage inheritance."""

    def test_matches_diffusion_grpo(self) -> None:
        """Token GRPO's group-relative advantages equal diffusion GRPO's for the same rewards and
        groups.
        """
        from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig

        torch.manual_seed(0)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        groups = torch.tensor([0, 0, 0, 1, 1, 1])

        diff_adv = GRPO(GRPOConfig(global_std=False)).compute_advantages_from_tensors(
            rewards,
            groups,
        )
        tok_adv = TokenGRPO(TokenGRPOConfig(global_std=False)).compute_advantages_from_tensors(
            rewards,
            groups,
        )
        assert torch.allclose(diff_adv, tok_adv)


# ---------------------------------------------------------------------------
# Per-token loss
# ---------------------------------------------------------------------------


def _inputs(
    new_lp: torch.Tensor,
    old_lp: torch.Tensor,
    adv: torch.Tensor,
    ref_lp: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> AlgorithmInput:
    if mask is None:
        mask = torch.ones_like(new_lp)
    return AlgorithmInput(
        signals=TrajectorySignalBatch(
            segments={
                "image_tokens": SegmentSignal(
                    name="image_tokens",
                    distribution="categorical",
                    log_prob=new_lp,
                    old_log_prob=old_lp,
                    mask=mask,
                    ref_log_prob=ref_lp,
                ),
            },
            group_ids=torch.arange(new_lp.shape[0], device=new_lp.device),
            primary_segment="image_tokens",
        ),
        advantages=adv,
    )


class TestPerTokenLoss:
    """Groups tests for per token loss."""

    def test_zero_loss_when_ratio_one_no_kl(self) -> None:
        """new_lp == old_lp → ratio == 1; positive advantage → loss = -adv."""
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp.clone()
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0))
        loss, _ = algo.compute_loss(_inputs(new_lp, old_lp, adv))
        assert loss.abs().item() < 1e-6

    def test_positive_advantage_decreases_loss_when_logprob_up(self) -> None:
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp + 0.1  # log-prob went up
        adv = torch.ones(2)  # positive advantage
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=1.0))  # disable clip
        loss, _ = algo.compute_loss(_inputs(new_lp, old_lp, adv))
        # ratio = exp(0.1) > 1; -adv * ratio < -1
        assert loss.item() < -1.0

    def test_clip_activates(self) -> None:
        old_lp = torch.zeros(1, 4)
        new_lp = old_lp + 5.0  # huge ratio
        adv = torch.ones(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=0.2))
        _loss, metrics = algo.compute_loss(_inputs(new_lp, old_lp, adv))
        assert metrics.update.clip_fraction == 1.0  # all 4 tokens clipped
        assert metrics.update.active_clip_fraction == 1.0


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


class TestMask:
    """Groups tests for mask."""

    def test_zero_mask_zeros_loss(self) -> None:
        old_lp = torch.zeros(2, 4)
        new_lp = old_lp + 1.0
        adv = torch.ones(2)
        mask = torch.zeros_like(new_lp)  # mask everything out
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0))
        loss, _ = algo.compute_loss(_inputs(new_lp, old_lp, adv, mask=mask))
        # mask sum is clamped to 1.0 to avoid NaN, but per_token_loss * 0 = 0
        assert loss.abs().item() < 1e-6

    def test_partial_mask_changes_loss(self) -> None:
        old_lp = torch.zeros(1, 4)
        new_lp = torch.tensor([[0.0, 0.5, 0.5, 0.0]])
        adv = torch.ones(1)
        mask_full = torch.ones_like(new_lp)
        mask_half = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=10.0))
        l_full, _ = algo.compute_loss(_inputs(new_lp, old_lp, adv, mask=mask_full))
        l_half, _ = algo.compute_loss(_inputs(new_lp, old_lp, adv, mask=mask_half))
        # half mask only counts the +0.5 tokens → mean is more negative
        assert l_half.item() < l_full.item()


class TestTokenTIS:
    """TIS folds into the per-token mask for the token GRPO path.

    The TIS knobs live on the injected ``precision_correction`` slot, not in
    TokenGRPOConfig (the trainer sets it from ``trainer.precision_correction``).
    """

    def test_mask_mode_rejects_high_drift_tokens(self) -> None:
        """Tokens whose rollout->replay ratio exceeds the cap are dropped from the mean."""
        # token0 ratio = e^2 (>cap), token1 ratio = 1 (kept)
        old_lp = torch.zeros(1, 2)
        new_lp = torch.tensor([[2.0, 0.0]])
        adv = torch.tensor([5.0])
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0))
        algo.precision_correction = PrecisionCorrectionConfig(
            tis_mode="mask",
            tis_imp_weight_cap=2.0,
        )
        loss, metrics = algo.compute_loss(_inputs(new_lp, old_lp, adv))
        # only token1 (ratio 1, adv 5) survives → -adv*1 = -5
        assert loss.item() == pytest.approx(-5.0)
        assert metrics.update.tis_clip_fraction == pytest.approx(0.5)

    def test_off_mode_is_noop(self) -> None:
        old_lp = torch.zeros(1, 4)
        new_lp = old_lp + 0.3
        adv = torch.ones(1)
        base, _ = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=1.0)).compute_loss(
            _inputs(new_lp, old_lp, adv),
        )
        off_algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0, clip_ratio=1.0))
        off_algo.precision_correction = PrecisionCorrectionConfig(tis_mode="off")
        off, m = off_algo.compute_loss(_inputs(new_lp, old_lp, adv))
        assert off.item() == pytest.approx(base.item())
        assert m.update.tis_clip_fraction == 0.0


class TestTokenRejectSampling:
    """RS rejects whole sequences on log-ratio drift (token GRPO, (B,L) path).

    seq_max rejects a sequence if any single step is out of band; seq_mean judges
    the per-sequence mean. RS is a per-sequence (B,1) keep mask that broadcasts
    over the token axis and folds into eff_mask alongside the token mask + TIS.
    """

    @staticmethod
    def _algo(**pc_kwargs) -> TokenGRPO:
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0))
        algo.precision_correction = PrecisionCorrectionConfig(**pc_kwargs)
        return algo

    def test_seq_max_single_outlier_rejects_whole_sequence(self) -> None:
        # token0 log-ratio=ln3 (out of band), token1=0; seq_max rejects the seq.
        new_lp = torch.tensor([[math.log(3.0), 0.0]])
        old_lp = torch.zeros(1, 2)
        algo = self._algo(rs_mode="seq_max_k1")
        loss, m = algo.compute_loss(_inputs(new_lp, old_lp, torch.tensor([5.0])))
        assert loss.item() == pytest.approx(0.0)  # whole sequence masked out
        assert m.update.rs_seq_masked_fraction == pytest.approx(1.0)

    def test_seq_mean_keeps_offsetting_steps(self) -> None:
        # steps ln3 and ln(1/3): mean 0 in band → kept under seq_mean (rejected under max).
        new_lp = torch.tensor([[math.log(3.0), math.log(1 / 3)]])
        old_lp = torch.zeros(1, 2)
        _, m = self._algo(rs_mode="seq_mean_k1").compute_loss(
            _inputs(new_lp, old_lp, torch.tensor([5.0])),
        )
        assert m.update.rs_seq_masked_fraction == pytest.approx(0.0)

    def test_rs_and_tis_fold_into_eff_mask(self) -> None:
        # token0 ratio e^ln3=3 → TIS(mask,cap2) drops token0; seq mean=ln3/2<ln2 keeps seq.
        new_lp = torch.tensor([[math.log(3.0), 0.0]])
        old_lp = torch.zeros(1, 2)
        algo = self._algo(tis_mode="mask", tis_imp_weight_cap=2.0, rs_mode="seq_mean_k1")
        loss, m = algo.compute_loss(_inputs(new_lp, old_lp, torch.tensor([5.0])))
        # only token1 survives (TIS dropped token0; RS kept the sequence) → -adv*1 = -5
        assert loss.item() == pytest.approx(-5.0)
        assert m.update.tis_clip_fraction == pytest.approx(0.5)
        assert m.update.rs_seq_masked_fraction == pytest.approx(0.0)

    def test_rs_reduction_respects_token_mask(self) -> None:
        # padded token (mask 0) carries a huge log-ratio; seq_mean must ignore it.
        new_lp = torch.tensor([[0.0, 0.0, math.log(9.0)]])
        old_lp = torch.zeros(1, 3)
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        _, m = self._algo(rs_mode="seq_mean_k1").compute_loss(
            _inputs(new_lp, old_lp, torch.tensor([5.0]), mask=valid),
        )
        assert m.update.rs_seq_masked_fraction == pytest.approx(0.0)  # padding excluded → kept


# ---------------------------------------------------------------------------
# KL estimators
# ---------------------------------------------------------------------------


class TestKL:
    """Groups tests for KL."""

    def test_kl_enabled_requires_ref_logprob(self) -> None:
        new_lp = torch.zeros(1, 2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=1.0))

        with pytest.raises(RuntimeError, match="ref_log_prob is None"):
            algo.compute_loss(_inputs(new_lp, new_lp, torch.zeros(1), ref_lp=None))

    def test_k1_signed(self) -> None:
        new_lp = torch.zeros(1, 4) + 0.5
        ref_lp = torch.zeros(1, 4)
        old_lp = torch.zeros(1, 4)
        adv = torch.zeros(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.25, kl_estimator="k1"))
        _, m = algo.compute_loss(_inputs(new_lp, old_lp, adv, ref_lp=ref_lp))
        # k1 = log_ratio = 0.5
        assert m.kl_penalty == pytest.approx(0.5, abs=1e-5)
        assert m.weighted_kl_loss == pytest.approx(0.125, abs=1e-5)

    def test_k3_nonnegative(self) -> None:
        """``k3`` is non-negative for any log-ratio."""
        torch.manual_seed(1)
        new_lp = torch.randn(2, 4)
        ref_lp = torch.randn(2, 4)
        old_lp = new_lp.clone()
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=1.0, kl_estimator="k3"))
        _, m = algo.compute_loss(_inputs(new_lp, old_lp, adv, ref_lp=ref_lp))
        assert m.kl_penalty >= 0.0

    def test_k2_half_squared_log_ratio(self) -> None:
        """``k2`` is half the squared log-ratio averaged over tokens."""
        new_lp = torch.tensor([[1.0, -1.0, 0.0, 2.0]])
        ref_lp = torch.zeros(1, 4)
        old_lp = torch.zeros(1, 4)
        adv = torch.zeros(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=1.0, kl_estimator="k2"))
        _, m = algo.compute_loss(_inputs(new_lp, old_lp, adv, ref_lp=ref_lp))

        expected = 0.5 * (new_lp - ref_lp).square().mean()
        assert m.kl_penalty == pytest.approx(float(expected), abs=1e-6)

    def test_k2_does_not_exponentiate_large_log_ratio(self) -> None:
        """``k2`` stays finite on a log-ratio of 10 (no exponentiation): 0.5 * 100 = 50."""
        new_lp = torch.full((1, 2), 10.0)
        ref_lp = torch.zeros(1, 2)
        old_lp = torch.zeros(1, 2)
        adv = torch.zeros(1)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=1.0, kl_estimator="k2"))
        _, m = algo.compute_loss(_inputs(new_lp, old_lp, adv, ref_lp=ref_lp))

        assert m.kl_penalty == pytest.approx(50.0, abs=1e-6)

    def test_unknown_estimator_raises(self) -> None:
        new_lp = torch.zeros(1, 2)
        ref_lp = torch.zeros(1, 2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=1.0, kl_estimator="bogus"))
        with pytest.raises(ValueError, match="kl_estimator"):
            algo.compute_loss(_inputs(new_lp, new_lp, torch.zeros(1), ref_lp=ref_lp))


# ---------------------------------------------------------------------------
# Shape errors
# ---------------------------------------------------------------------------


class TestShapeValidation:
    """Groups tests for shape validation."""

    def test_mismatched_logprob_shape(self) -> None:
        new_lp = torch.zeros(2, 4)
        old_lp = torch.zeros(2, 5)
        adv = torch.zeros(2)
        algo = TokenGRPO(TokenGRPOConfig(kl_coef=0.0))
        with pytest.raises(ValueError, match="log_prob/old_log_prob"):
            algo.compute_loss(_inputs(new_lp, old_lp, adv))
