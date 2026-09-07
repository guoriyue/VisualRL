"""Tests for trajectory runtime storage policy."""

from __future__ import annotations

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.trajectory import (
    TrajectoryStoragePolicy,
    apply_trajectory_storage_policy,
    build_ar_discrete_trajectory,
    trajectory_storage_policy_from_cfg,
)


def test_default_storage_policy_returns_original_batch() -> None:
    """The default policy is the identity: the same trajectory object, the same tensor leaves,
    axes untouched.
    """
    trajectory = _trajectory()
    token_ids = trajectory.segments["image_tokens"].tensors["token_ids"].value

    result = apply_trajectory_storage_policy(trajectory, TrajectoryStoragePolicy())

    assert result is trajectory
    assert result.segments["image_tokens"].tensors["token_ids"].value is token_ids
    assert result.axes["token"].kind == "discrete_token"


def test_dtype_policy_only_casts_floating_tensors() -> None:
    """A dtype policy casts floating tensors only; token ids and prompt ids keep ``torch.long``
    and axis lengths are preserved.
    """
    trajectory = _trajectory()

    result = apply_trajectory_storage_policy(
        trajectory,
        TrajectoryStoragePolicy(dtype="float16"),
    )

    segment = result.segments["image_tokens"]
    assert segment.tensors["token_ids"].value.dtype == torch.long
    assert segment.tensors["prompt_input_ids"].value.dtype == torch.long
    assert segment.tensors["old_log_prob"].value.dtype == torch.float16
    assert segment.tensors["token_mask"].value.dtype == torch.float16
    assert torch.allclose(
        segment.tensors["old_log_prob"].value.float(),
        torch.tensor([[0.125, -0.25], [0.5, -0.75]]),
        atol=1e-3,
    )
    axis_lengths = {n: a.length for n, a in result.axes.items() if a.length is not None}
    assert axis_lengths == {"sample": 2, "token": 2}


def test_cpu_storage_policy_moves_tensor_leaves_to_cpu() -> None:
    """A device policy moves every tensor leaf of every segment to that device."""
    trajectory = _trajectory()

    result = apply_trajectory_storage_policy(
        trajectory,
        TrajectoryStoragePolicy(device="cpu"),
    )

    for tensor in result.segments["image_tokens"].tensors.values():
        assert str(tensor.value.device) == "cpu"


def test_storage_policy_parser_rejects_unknown_values() -> None:
    """``None`` parses to the default policy and a mapping to its fields; an unknown dtype fails
    naming it, and a bare scalar is a misconfiguration that fails instead of degrading to a
    default.
    """
    assert trajectory_storage_policy_from_cfg(None) == TrajectoryStoragePolicy()
    assert trajectory_storage_policy_from_cfg({"device": "cpu"}).device == "cpu"

    with pytest.raises(ValueError, match="trajectory storage dtype"):
        trajectory_storage_policy_from_cfg({"dtype": "int8"})

    # A bare non-mapping scalar (e.g. ``trajectory_storage: cpu``) is a
    # misconfiguration and must fail loudly rather than degrade to a default.
    with pytest.raises(TypeError, match="must be a mapping"):
        trajectory_storage_policy_from_cfg("cpu")


def _trajectory():
    request = GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["p0"],
        samples_per_prompt=2,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="p0",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
    return build_ar_discrete_trajectory(
        request=request,
        sample_rows=sample_rows,
        token_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        token_log_probs=torch.tensor([[0.125, -0.25], [0.5, -0.75]]),
        token_mask=torch.ones(2, 2),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
