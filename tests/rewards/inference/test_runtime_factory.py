"""Tests for the in-process reward runtime: worker-config validation and model-factory loading."""

from __future__ import annotations

import pytest

from vrl.rewards.functions.kling_video_reward import (
    KlingVideoReward,
)
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
)
from vrl.rewards.runtime import InProcessRewardScorer


class _FakeRewardModel:
    def __init__(self, worker_config):
        self.worker_config = worker_config

    def __call__(self, artifact):
        assert artifact.prompt == "prompt"
        assert self.worker_config["reward_model_name"] == "KlingTeam/VideoReward@main"
        return {"overall_reward": 3.0, "motion_quality": 1.0}


def build_fake_reward_model(worker_config) -> _FakeRewardModel:
    """Module-level factory so the in-process runtime can import it by path."""
    return _FakeRewardModel(worker_config)


def test_runtime_requires_model_factory() -> None:
    """Checks the runtime rejects a worker config without a model factory."""
    import asyncio

    runtime = InProcessRewardScorer(
        {"reward_model_name": "KlingTeam/VideoReward@main"},
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                sample_id="sample-0",
                path="/tmp/a0.mp4",
            ),
        ),
    )
    with pytest.raises(ValueError, match="model_factory"):
        asyncio.run(runtime.score_batch(request))


def test_video_reward_derives_internal_model_factory_from_reward_name(tmp_path) -> None:
    """A hub-style ``reward_name`` becomes both ``reward_model_name`` and ``reward_model_version``
    in the scorer's worker config, next to the class's own ``model_factory``.
    """
    reward = KlingVideoReward(
        reward_name="KlingTeam/VideoReward@main",
        score_key="overall_reward",
        artifact_dir=str(tmp_path),
        worker_config={"model_path": "", "dtype": "bfloat16"},
    )

    assert isinstance(reward.scorer, InProcessRewardScorer)
    assert reward.scorer._worker_config == {
        "model_path": "",
        "dtype": "bfloat16",
        "model_factory": KlingVideoReward.model_factory,
        "reward_model_name": "KlingTeam/VideoReward@main",
        "reward_model_version": "KlingTeam/VideoReward@main",
    }


@pytest.mark.asyncio
async def test_runtime_loads_reward_model_via_factory() -> None:
    """Checks the runtime loads the reward model via the configured factory."""
    runtime = InProcessRewardScorer(
        {
            "model_factory": (
                "tests.rewards.inference.test_runtime_factory:build_fake_reward_model"
            ),
            "reward_model_name": "KlingTeam/VideoReward@main",
            "reward_model_version": "KlingTeam/VideoReward@main",
        },
    )
    request = RewardInferenceRequest(
        request_id="req",
        artifacts=(
            RewardInferenceArtifact(
                artifact_id="a0",
                sample_id="sample-0",
                path="/tmp/a0.mp4",
                prompt="prompt",
            ),
        ),
    )

    results = await runtime.score_batch(request)

    assert results[0].scores["overall_reward"] == pytest.approx(3.0)
    assert results[0].reward_model_version == "KlingTeam/VideoReward@main"
