"""Tests for the bounded continuous rollout ready queue container."""

from __future__ import annotations

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration.continuous.queue import ContinuousRolloutQueue
from vrl.rollouts.orchestration.continuous.types import (
    ContinuousRolloutItem,
    estimate_batch_bytes,
)
from vrl.trajectory import build_ar_discrete_trajectory, trajectory_tensor_bytes


def _item(
    group_slot: int,
    version: int | None,
    *,
    samples: int = 2,
    nbytes: int = 0,
    batch_id: int = 0,
) -> ContinuousRolloutItem:
    batch = RolloutBatch(
        rewards=torch.zeros(samples),
        group_ids=torch.zeros(samples, dtype=torch.long),
    )
    return ContinuousRolloutItem(
        batch_id=batch_id,
        group_slot=group_slot,
        rollout_policy_version=version,
        attempt=1,
        batch=batch,
        nbytes=nbytes,
    )


def test_rejects_negative_byte_limit() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        ContinuousRolloutQueue(max_items=1, max_bytes=-1)


def test_item_limit_changes_only_between_prompt_batches() -> None:
    queue = ContinuousRolloutQueue(max_items=1)
    queue.set_item_limit(2)
    assert queue.max_items == 2

    queue.put(_item(group_slot=0, version=1))
    with pytest.raises(RuntimeError, match="only between batches"):
        queue.set_item_limit(3)
    assert queue.max_items == 2


def test_snapshot_and_remove_are_pure_container_ops() -> None:
    """snapshot() reads FIFO order; remove() drops by identity and fixes bytes."""
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_slot=0, version=1, nbytes=4))
    queue.put(_item(group_slot=1, version=1, nbytes=6))
    snap = queue.snapshot()
    assert [item.group_slot for item in snap] == [0, 1]

    queue.remove([snap[0]])
    assert queue.size() == 1
    assert queue.stats()["ready_bytes"] == 6.0


def test_item_count_overflow_fails_before_mutation() -> None:
    queue = ContinuousRolloutQueue(max_items=2)
    queue.put(_item(group_slot=0, version=1))
    queue.put(_item(group_slot=1, version=1))

    with pytest.raises(ValueError, match="item limit"):
        queue.put(_item(group_slot=2, version=1))

    assert queue.size() == 2
    assert [item.group_slot for item in queue.snapshot()] == [0, 1]


def test_byte_overflow_fails_before_mutation() -> None:
    queue = ContinuousRolloutQueue(max_items=100, max_bytes=10)
    queue.put(_item(group_slot=0, version=1, nbytes=6))

    with pytest.raises(ValueError, match="byte limit"):
        queue.put(_item(group_slot=1, version=1, nbytes=6))

    assert queue.stats()["ready_bytes"] == 6.0
    assert [item.group_slot for item in queue.snapshot()] == [0]


def test_batch_byte_estimate_counts_nested_extras_tensors() -> None:
    """extras payloads count even when nested (the production shape is
    extras["reward_components"] = {name: tensor})."""

    batch = _item(group_slot=0, version=1).batch
    batch.extras["component"] = torch.zeros(3, dtype=torch.float64)
    batch.extras["reward_components"] = {"aesthetic": torch.zeros(2)}

    expected = sum(
        tensor.element_size() * tensor.nelement()
        for tensor in (
            batch.rewards,
            batch.group_ids,
            batch.extras["component"],
            batch.extras["reward_components"]["aesthetic"],
        )
    )

    assert estimate_batch_bytes(batch) == expected


def test_batch_byte_estimate_counts_trajectory_without_flat_aliases_twice() -> None:
    token_ids = torch.tensor([[1, 2]], dtype=torch.long)
    prompt_input_ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["p"],
        samples_per_prompt=1,
    )
    trajectory = build_ar_discrete_trajectory(
        request=request,
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0,
                sample_index=0,
                prompt="p",
                sample_id="s0",
            )
        ],
        token_ids=token_ids,
        token_log_probs=torch.zeros(1, 2),
        token_mask=torch.ones(1, 2),
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(1, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(1, 3, dtype=torch.long),
        context={},
    )
    rewards = torch.zeros(1)
    group_ids = torch.zeros(1, dtype=torch.long)
    component = torch.zeros(3, dtype=torch.float64)
    batch = RolloutBatch(
        rewards=rewards,
        group_ids=group_ids,
        extras={"component": component},
        trajectory=trajectory,
    )

    expected = trajectory_tensor_bytes(trajectory) + sum(
        tensor.numel() * tensor.element_size() for tensor in (rewards, group_ids, component)
    )

    assert estimate_batch_bytes(batch) == expected


def test_stats_shape() -> None:
    """``stats()`` reports ready items and ready bytes under exactly those keys (the metrics
    contract), bytes summed from the items' ``nbytes``.
    """
    queue = ContinuousRolloutQueue(max_items=8)
    queue.put(_item(group_slot=0, version=1, nbytes=4))
    stats = queue.stats()
    assert stats["ready_items"] == 1.0
    assert stats["ready_bytes"] == 4.0
