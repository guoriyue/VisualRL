"""Tests for trajectory tensor byte counters."""

from __future__ import annotations

import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.trajectory import build_ar_discrete_trajectory, trajectory_tensor_bytes


def test_byte_counter_counts_trajectory_tensor_leaves() -> None:
    """``trajectory_tensor_bytes`` sums ``numel * element_size`` over every tensor leaf of the
    trajectory.
    """
    token_ids = torch.tensor([[1, 2]], dtype=torch.long)
    old_log_prob = torch.zeros(1, 2, dtype=torch.float32)
    token_mask = torch.ones(1, 2, dtype=torch.float32)
    prompt_input_ids = torch.ones(1, 3, dtype=torch.long)
    prompt_attention_mask = torch.ones(1, 3, dtype=torch.long)
    uncond_input_ids = torch.zeros(1, 3, dtype=torch.long)
    uncond_attention_mask = torch.ones(1, 3, dtype=torch.long)
    trajectory = build_ar_discrete_trajectory(
        request=GenerationRequest(
            request_id="req",
            family="janus_pro",
            task="ar_t2i",
            inputs=["p"],
            samples_per_prompt=1,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0,
                sample_index=0,
                prompt="p",
                sample_id="s0",
            )
        ],
        token_ids=token_ids,
        token_log_probs=old_log_prob,
        token_mask=token_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        uncond_input_ids=uncond_input_ids,
        uncond_attention_mask=uncond_attention_mask,
        context={},
    )

    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            token_ids,
            old_log_prob,
            token_mask,
            prompt_input_ids,
            prompt_attention_mask,
            uncond_input_ids,
            uncond_attention_mask,
        )
    )

    assert trajectory_tensor_bytes(trajectory) == expected


def test_byte_counter_deduplicates_shared_references() -> None:
    """The same tensor referenced twice is counted once."""
    tensor = torch.zeros(4, dtype=torch.float32)

    assert trajectory_tensor_bytes({"a": tensor, "b": tensor}) == 16
