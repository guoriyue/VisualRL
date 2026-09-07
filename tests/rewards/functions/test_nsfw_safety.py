"""Tests for the NSFW safety penalty reward."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from vrl.rewards.functions.nsfw_safety import NSFWSafetyReward
from vrl.rewards.models.nsfw_safety import NSFWSafetyRewardModel
from vrl.rewards.types import RewardSample


def _sample(output: object, *, sample_id: str = "sample-0") -> RewardSample:
    return RewardSample(
        prompt="anime portrait",
        output=output,
        sample_id=sample_id,
    )


@pytest.mark.asyncio
async def test_nsfw_safety_reward_only_penalizes_scores_above_threshold() -> None:
    """Below the threshold the penalty is 0; above it the penalty is ``-(p - threshold) *
    penalty_scale`` capped by ``max_penalty`` (0.75 -> -1.333 at scale 2), and no score is ever
    positive.
    """
    reward = NSFWSafetyReward(
        model_name="test",
        threshold=0.25,
        penalty_scale=2.0,
        max_penalty=1.5,
        scorer=lambda images: [0.10, 0.75],
    )
    image = Image.new("RGB", (8, 8), color=(128, 128, 128))

    output = await reward.score_batch(
        [_sample(image), _sample(image, sample_id="sample-1")],
    )

    assert output.scores[0] == pytest.approx(0.0)
    assert output.scores[1] == pytest.approx(-1.3333333333)
    assert all(score <= 0.0 for score in output.scores)


@pytest.mark.asyncio
async def test_nsfw_safety_reward_uses_max_probability_for_image_batches() -> None:
    """For an image batch the reward takes the maximum NSFW probability across the sampled images
    (0.9 -> -0.8 at threshold 0.5).
    """
    reward = NSFWSafetyReward(
        model_name="test",
        threshold=0.50,
        penalty_scale=1.0,
        image_sample_count=2,
        scorer=lambda images: [0.20, 0.90],
    )
    output = torch.rand(2, 3, 8, 8)

    score = await reward.score(_sample(output))

    assert score == pytest.approx(-0.8)


def test_nsfw_classifier_result_parsing_prefers_nsfw_labels() -> None:
    """The classifier-result parser reads the NSFW-labelled entry's score ('unsafe'), not the top-
    scoring label.
    """
    model = NSFWSafetyRewardModel({"model_name": "test", "threshold": 0.35})

    probability = model._probability_from_classifier_result(
        [
            {"label": "normal", "score": 0.80},
            {"label": "unsafe", "score": 0.20},
        ],
    )

    assert probability == pytest.approx(0.20)


def test_nsfw_safety_reward_rejects_invalid_reverse_like_config() -> None:
    with pytest.raises(ValueError, match="penalty_scale"):
        NSFWSafetyReward(model_name="test", penalty_scale=-1.0)
