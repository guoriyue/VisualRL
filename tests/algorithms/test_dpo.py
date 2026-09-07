"""Tests for vrl.algorithms.dpo — Diffusion-DPO loss correctness.

Validates that ``diffusion_dpo_loss`` matches the reference formula in
SalesforceAIResearch/DiffusionDPO ``train.py:1119-1145``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vrl.algorithms.dpo import (
    diffusion_dpo_loss,
    diffusion_sft_loss,
)


def _reference_dpo_loss(
    model_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inline copy of the reference repo's loss for parity tests."""
    model_losses = (model_pred - target).pow(2).mean(dim=tuple(range(1, model_pred.ndim)))
    model_losses_w, model_losses_l = model_losses.chunk(2)
    model_diff = model_losses_w - model_losses_l

    ref_losses = (ref_pred - target).pow(2).mean(dim=tuple(range(1, ref_pred.ndim)))
    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
    ref_diff = ref_losses_w - ref_losses_l

    inside = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside).mean()
    acc = (inside > 0).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Reference parity
# ---------------------------------------------------------------------------


class TestParityWithReference:
    """Groups tests for parity with reference."""

    def test_matches_reference_4d_image(self) -> None:
        """SD-style image latents [2B, 4, 64, 64]."""
        torch.manual_seed(0)
        B, C, H, W = 3, 4, 16, 16
        model_pred = torch.randn(2 * B, C, H, W)
        ref_pred = torch.randn(2 * B, C, H, W)
        target = torch.randn(2 * B, C, H, W)
        beta = 5000.0

        ours = diffusion_dpo_loss(model_pred, ref_pred, target, beta=beta)
        ref_loss, ref_acc = _reference_dpo_loss(model_pred, ref_pred, target, beta)

        assert torch.allclose(ours["loss"], ref_loss, atol=1e-6)
        assert torch.allclose(ours["implicit_acc"], ref_acc)

    def test_matches_reference_5d_video(self) -> None:
        """Wan video latents [2B, 16, 8, 30, 52]."""
        torch.manual_seed(1)
        B, C, T, H, W = 2, 16, 8, 8, 12
        model_pred = torch.randn(2 * B, C, T, H, W)
        ref_pred = torch.randn(2 * B, C, T, H, W)
        target = torch.randn(2 * B, C, T, H, W)
        beta = 2000.0

        ours = diffusion_dpo_loss(model_pred, ref_pred, target, beta=beta)
        ref_loss, ref_acc = _reference_dpo_loss(model_pred, ref_pred, target, beta)

        assert torch.allclose(ours["loss"], ref_loss, atol=1e-6)
        assert torch.allclose(ours["implicit_acc"], ref_acc)


# ---------------------------------------------------------------------------
# Behavioural sanity
# ---------------------------------------------------------------------------


class TestBehavior:
    """Groups tests for behavior."""

    def test_zero_when_policy_equals_reference(self) -> None:
        """Identical policy & reference → inside_term = 0 → loss = log(2)."""
        torch.manual_seed(2)
        B = 4
        model_pred = torch.randn(2 * B, 4, 8, 8)
        ref_pred = model_pred.clone()
        target = torch.randn(2 * B, 4, 8, 8)

        out = diffusion_dpo_loss(model_pred, ref_pred, target, beta=5000.0)
        assert out["loss"].item() == pytest.approx(math.log(2.0), abs=1e-5)
        # 50% accuracy because all inside_terms are exactly zero (we use > 0)
        assert out["implicit_acc"].item() == pytest.approx(0.0)

    def test_lower_loss_when_policy_prefers_winner(self) -> None:
        """Constructed: policy MSE for winner < loser, reference is uniform.

        Should produce loss < log(2).
        """
        torch.manual_seed(3)
        B = 4
        target = torch.zeros(2 * B, 4, 8, 8)

        # Winner predictions = small noise (low MSE), loser = large (high MSE).
        winner = 0.01 * torch.randn(B, 4, 8, 8)
        loser = 1.0 * torch.randn(B, 4, 8, 8)
        model_pred = torch.cat([winner, loser], dim=0)
        ref_pred = torch.randn(2 * B, 4, 8, 8) * 0.5  # neutral

        out = diffusion_dpo_loss(model_pred, ref_pred, target, beta=100.0)
        assert out["loss"].item() < math.log(2.0)
        assert out["model_diff"].item() < 0.0  # winner MSE < loser MSE
        assert out["implicit_acc"].item() > 0.5

    def test_higher_loss_when_policy_prefers_loser(self) -> None:
        """Inverted: policy MSE for winner > loser. Loss should exceed log(2)."""
        torch.manual_seed(4)
        B = 4
        target = torch.zeros(2 * B, 4, 8, 8)
        winner = 1.0 * torch.randn(B, 4, 8, 8)  # bad on winner
        loser = 0.01 * torch.randn(B, 4, 8, 8)  # good on loser
        model_pred = torch.cat([winner, loser], dim=0)
        ref_pred = torch.randn(2 * B, 4, 8, 8) * 0.5

        out = diffusion_dpo_loss(model_pred, ref_pred, target, beta=100.0)
        assert out["loss"].item() > math.log(2.0)
        assert out["model_diff"].item() > 0.0
        assert out["implicit_acc"].item() < 0.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Groups tests for validation."""

    def test_rejects_odd_batch(self) -> None:
        x = torch.randn(3, 4, 4, 4)
        with pytest.raises(ValueError, match="2\\*B"):
            diffusion_dpo_loss(x, x, x, beta=1.0)

    def test_rejects_shape_mismatch(self) -> None:
        a = torch.randn(4, 4, 4, 4)
        b = torch.randn(4, 4, 4, 5)
        with pytest.raises(ValueError, match="shape mismatch"):
            diffusion_dpo_loss(a, b, a, beta=1.0)


# ---------------------------------------------------------------------------
# Auxiliary SFT loss
# ---------------------------------------------------------------------------


def test_sft_loss_is_winner_mse() -> None:
    """``diffusion_sft_loss`` is plain MSE between the winner prediction and its target."""
    torch.manual_seed(5)
    pred_w = torch.randn(3, 4, 8, 8)
    target_w = torch.randn(3, 4, 8, 8)
    expected = F.mse_loss(pred_w, target_w)
    got = diffusion_sft_loss(pred_w, target_w)
    assert torch.allclose(got, expected)
