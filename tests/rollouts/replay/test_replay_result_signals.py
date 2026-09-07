"""Tests for ReplayResult lookups and signal source ownership."""

from __future__ import annotations

import pytest
import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.trajectory import TrajectorySignalBuilder
from vrl.trajectory import build_ar_discrete_trajectory


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    request = _request()
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]


def _discrete_batch() -> tuple[RolloutBatch, torch.Tensor, torch.Tensor]:
    token_ids = torch.tensor([[1, 2], [2, 3]])
    old_log_prob = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    token_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=old_log_prob,
        token_mask=token_mask,
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro"},
    )
    batch = RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )
    return batch, old_log_prob, token_mask


def test_replay_result_supports_single_segment_ar_lookup() -> None:
    """``require_segment`` / ``require_value`` hand back the stored tensor object itself for a
    single AR segment.
    """
    logits = torch.zeros(2, 2, 8)
    output = ReplayResult(
        segments={
            "image_tokens": ReplaySegmentResult(
                segment="image_tokens",
                values={"logits": logits},
            ),
        },
    )

    segment = output.require_segment("image_tokens")

    assert segment.require_value("logits") is logits


def test_replay_result_supports_single_segment_diffusion_lookup() -> None:
    """The same lookup on a single diffusion ``denoise`` segment returns the stored ``noise_pred``
    object.
    """
    noise_pred = torch.ones(2, 4, 8, 8)
    output = ReplayResult(
        segments={
            "denoise": ReplaySegmentResult(
                segment="denoise",
                values={"noise_pred": noise_pred},
            ),
        },
    )

    segment = output.require_segment("denoise")

    assert segment.require_value("noise_pred") is noise_pred


def test_replay_result_supports_multi_segment_r1_lookup() -> None:
    """With several segments present ``require_segment`` returns the one named, not the first."""
    output = ReplayResult(
        segments={
            "selfcheck_text": ReplaySegmentResult(
                segment="selfcheck_text",
                values={"logits": torch.zeros(2, 3, 10)},
            ),
            "final_image": ReplaySegmentResult(
                segment="final_image",
                values={"logits": torch.zeros(2, 4, 10)},
            ),
        },
    )

    segment = output.require_segment("final_image")

    assert segment.segment == "final_image"


def test_replay_result_fails_fast_for_missing_ref_segment() -> None:
    output = ReplayResult(
        segments={
            "selfcheck_text": ReplaySegmentResult(
                segment="selfcheck_text",
                values={"logits": torch.zeros(2, 3, 10)},
            ),
        },
    )

    with pytest.raises(KeyError, match="missing segment 'final_image'"):
        output.require_segment("final_image")


def test_replay_segment_result_fails_fast_with_available_keys() -> None:
    segment = ReplaySegmentResult(
        segment="image_tokens",
        values={"image_logits": torch.zeros(2, 2, 8), "token_ids": torch.ones(2, 2)},
    )

    with pytest.raises(KeyError, match=r"missing required key 'logits'.*image_logits"):
        segment.require_value("logits")


def test_segment_signal_reads_old_logprob_mask_and_distribution_from_trajectory() -> None:
    """The trajectory, not the replay values, is the source of ``old_log_prob`` / ``mask`` /
    ``distribution``: tensors a model stuffs into the replay segment under those names are
    ignored.
    """
    batch, old_log_prob, token_mask = _discrete_batch()
    output = ReplayResult(
        segments={
            "image_tokens": ReplaySegmentResult(
                segment="image_tokens",
                values={
                    "logits": torch.zeros(2, 2, 8),
                    "old_log_prob": torch.full((2, 2), 99.0),
                    "mask": torch.zeros(2, 2),
                },
            ),
        },
    )
    segment = output.require_segment("image_tokens")

    signal = TrajectorySignalBuilder(batch).segment_signal(
        segment_name=segment.segment,
        log_prob=torch.full((2, 2), -1.0),
    )

    assert signal.distribution == "categorical"
    assert torch.equal(signal.old_log_prob, old_log_prob)
    assert torch.equal(signal.mask, token_mask)


def test_signal_builder_requires_a_canonical_trajectory() -> None:
    batch = RolloutBatch(
        rewards=torch.zeros(1),
        group_ids=torch.zeros(1, dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match=r"batch\.trajectory.*TrajectoryBatch"):
        TrajectorySignalBuilder(batch)


def test_signal_builder_rejects_unknown_segment() -> None:
    batch, _, _ = _discrete_batch()

    with pytest.raises(RuntimeError, match="unknown trajectory segment 'missing'"):
        TrajectorySignalBuilder(batch).segment_signal(
            segment_name="missing",
            log_prob=torch.zeros(2, 2),
        )


def test_signal_builder_rejects_mismatched_trajectory_mask() -> None:
    batch, _, _ = _discrete_batch()
    assert batch.trajectory is not None
    batch.trajectory.segments["image_tokens"].tensors["token_mask"].value = torch.ones(2, 1)

    with pytest.raises(ValueError, match="log_prob/mask shape mismatch"):
        TrajectorySignalBuilder(batch).single_segment(
            segment_name="image_tokens",
            log_prob=torch.zeros(2, 2),
        )
