"""Tests for vrl.rollouts.types (ExperienceBatch stacking)."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Regression: Bug 4 — stack_batches helper for group collection
# ---------------------------------------------------------------------------

class TestStackBatches:
    def test_stacks_two_batches(self) -> None:
        """stack_batches concatenates tensor fields along batch dim."""
        import torch
        from vrl.rollouts.types import ExperienceBatch, stack_batches

        b1 = ExperienceBatch(
            observations=torch.randn(1, 3, 4),
            actions=torch.randn(1, 3, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            prompts=["hello"],
        )
        b2 = ExperienceBatch(
            observations=torch.randn(1, 3, 4),
            actions=torch.randn(1, 3, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            prompts=["world"],
        )
        combined = stack_batches([b1, b2])
        assert combined.observations.shape[0] == 2
        assert combined.rewards.shape[0] == 2
        assert combined.group_ids.tolist() == [0, 1]
        assert combined.prompts == ["hello", "world"]

    def test_stacks_tensor_extras(self) -> None:
        """Tensor extras are concatenated; non-tensor extras kept from first."""
        import torch
        from vrl.rollouts.types import ExperienceBatch, stack_batches

        b1 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "log_probs": torch.tensor([[0.1, 0.2]]),
                "scheduler": "shared_object",
            },
        )
        b2 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "log_probs": torch.tensor([[0.3, 0.4]]),
                "scheduler": "shared_object",
            },
        )
        combined = stack_batches([b1, b2])
        assert combined.extras["log_probs"].shape == (2, 2)
        assert combined.extras["scheduler"] == "shared_object"

    def test_single_batch_passthrough(self) -> None:
        """Single batch → returned as-is (no copy overhead)."""
        import torch
        from vrl.rollouts.types import ExperienceBatch, stack_batches

        b = ExperienceBatch(
            observations=torch.randn(2, 3, 4),
            actions=torch.randn(2, 3, 4),
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.tensor([True, True]),
            group_ids=torch.tensor([0, 0]),
        )
        result = stack_batches([b])
        assert result is b  # same object

    def test_context_taken_from_first_batch(self) -> None:
        """context dict is taken from the first batch (not stacked)."""
        import torch
        from vrl.rollouts.types import ExperienceBatch, stack_batches

        ctx = {"guidance_scale": 7.0, "model_family": "cosmos"}
        b1 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            context=ctx,
        )
        b2 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            context={"guidance_scale": 7.0, "model_family": "cosmos"},
        )
        combined = stack_batches([b1, b2])
        assert combined.context["guidance_scale"] == 7.0
        assert combined.context["model_family"] == "cosmos"

    def test_context_defaults_to_empty(self) -> None:
        """Without context, field defaults to empty dict."""
        from vrl.rollouts.types import ExperienceBatch

        import torch
        b = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
        )
        assert b.context == {}

    def test_only_tensor_extras_stacked(self) -> None:
        """Only tensor values in extras get stacked; non-tensors kept from first."""
        import torch
        from vrl.rollouts.types import ExperienceBatch, stack_batches

        b1 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([0]),
            extras={
                "log_probs": torch.tensor([[0.1, 0.2]]),
                "label": "first",
            },
            context={"cfg": True},
        )
        b2 = ExperienceBatch(
            observations=torch.randn(1, 2, 4),
            actions=torch.randn(1, 2, 4),
            rewards=torch.tensor([2.0]),
            dones=torch.tensor([True]),
            group_ids=torch.tensor([1]),
            extras={
                "log_probs": torch.tensor([[0.3, 0.4]]),
                "label": "second",
            },
            context={"cfg": True},
        )
        combined = stack_batches([b1, b2])
        # Tensor extras stacked
        assert combined.extras["log_probs"].shape == (2, 2)
        # Non-tensor extras kept from first
        assert combined.extras["label"] == "first"
        # Context preserved
        assert combined.context["cfg"] is True
