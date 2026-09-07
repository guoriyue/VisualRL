"""The production contract a reward declares (``RewardFunction.production``).

``production.<reward>.enabled`` turns on one gate whose entire content is the
contract the reward declares. These pin what it refuses and, just as important,
what it deliberately leaves to running the reward (the preflight entry).
"""

from __future__ import annotations

import pytest

from vrl.rewards.base import ProductionContract, RewardFunction
from vrl.rewards.functions.kling_video_reward import KlingVideoReward

_CONTRACT = ProductionContract(task_types=frozenset({"text_to_video", "video2world"}))
_VALID = {
    "media_type": "video",
    "artifact_format": "mp4",
    "reward_name": "org/model@main",
    "worker_config": {},
}


def test_a_valid_production_config_passes() -> None:
    _CONTRACT.require("kling_video_reward", _VALID, task_type="text_to_video")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"media_type": "image"}, "media_type=video"),
        ({"artifact_format": "tensor"}, "artifact_format=mp4"),
        ({"reward_name": "  "}, "reward_name"),
        ({"worker_config": {"model_factory": "fake:factory"}}, "remove extra loader fields"),
    ],
)
def test_each_broken_field_is_named(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _CONTRACT.require("kling_video_reward", {**_VALID, **override}, task_type="text_to_video")


def test_an_unvalidated_task_type_lists_the_validated_ones() -> None:
    with pytest.raises(ValueError, match="text_to_video, video2world"):
        _CONTRACT.require("kling_video_reward", _VALID, task_type="text_to_image")


def test_only_live_loader_keys_are_locked() -> None:
    """``import_path`` is a GenEval constructor kwarg, never read out of
    ``worker_config``; locking it there would protect nothing."""

    assert {"model_factory"} == ProductionContract.LOCKED_WORKER_CONFIG_KEYS
    _CONTRACT.require(
        "kling_video_reward",
        {**_VALID, "worker_config": {"import_path": "fake:thing"}},
        task_type="text_to_video",
    )


def test_a_reward_declares_its_contract_and_most_rewards_have_none() -> None:
    """The gate's whole per-reward vocabulary is this one declaration; a reward
    without one has no production gate (enabling it is a config error)."""

    assert KlingVideoReward.production is not None
    assert KlingVideoReward.production.task_types == {
        "text_to_video",
        "image_to_video",
        "video2world",
    }
    assert KlingVideoReward.production.media_type == KlingVideoReward.default_media_type
    assert KlingVideoReward.production.artifact_format == KlingVideoReward.default_artifact_format
    assert RewardFunction.production is None


def test_the_contract_is_frozen() -> None:
    with pytest.raises(AttributeError):
        _CONTRACT.task_types = frozenset({"anything"})  # type: ignore[misc]
