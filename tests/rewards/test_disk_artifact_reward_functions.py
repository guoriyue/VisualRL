"""Disk-artifact reward functions as thin inference-runtime adapters.

HPSv3 and VideoScore2 wrap ``DiskArtifactRewardFunction`` identically: they
materialize the sample, hand it to the scorer, select ``score_key`` from the
returned axes, and log requests/results. One parametrized module pins that
contract for both instead of two byte-identical files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import RewardConfig
from vrl.rewards.functions.hpsv3 import HPSv3Reward
from vrl.rewards.functions.videoscore2 import VideoScore2Reward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.types import RewardSample


@dataclass(frozen=True)
class _Case:
    reward_cls: type
    reward_name: str
    reward_model_name: str
    default_score_key: str
    alternate_score_key: str
    fake_scores: dict[str, float]


_CASES = [
    _Case(
        HPSv3Reward,
        "hpsv3",
        "MizzenAI/HPSv3@main",
        "top_frame_mean",
        "frame_mean",
        {"top_frame_mean": 9.5, "frame_mean": 7.25, "frame_min": 1.5},
    ),
    _Case(
        VideoScore2Reward,
        "videoscore2",
        "TIGER-Lab/VideoScore2@main",
        "physical_common_sense",
        "visual_quality",
        {
            "visual_quality": 4.0,
            "text_alignment": 2.5,
            "physical_common_sense": 3.25,
            "overall": 3.25,
        },
    ),
]
_CASE_IDS = [case.reward_name for case in _CASES]


class _FakeRuntime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = dict(scores)
        self.requests: list[Any] = []

    async def score_batch(self, request):
        self.requests.append(request)
        return [
            RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores=dict(self.scores),
                reward_model_version="fake-test",
                timing_ms={"inference_ms": 1.0},
            )
            for artifact in request.artifacts
        ]

    async def shutdown(self) -> None:
        return None


def _sample() -> RewardSample:
    return RewardSample(
        prompt="a red fox curled on mossy stones",
        output=torch.ones(1, 2, 2, 2),
        sample_id="sample-a",
        metadata={"policy_version": 7},
    )


def _build_reward(case: _Case, tmp_path: Path, *, score_key: str):
    return case.reward_cls(
        reward_name=case.reward_name,
        score_key=score_key,
        media_type="video",
        # tensor avoids the imageio mp4 writer; this checks materialization wiring.
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "artifacts"),
        debug_dir=str(tmp_path / "debug"),
        retain_artifacts=True,
        scorer=_FakeRuntime(case.fake_scores),
    )


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_materializes_artifacts_and_selects_the_score_key(
    case: _Case, tmp_path: Path
) -> None:
    """Default and alternate score keys select their axis; debug logs every public key."""
    reward = _build_reward(case, tmp_path, score_key=case.default_score_key)
    output = await reward.score_batch([_sample()])

    assert output.scores == pytest.approx([case.fake_scores[case.default_score_key]])
    request = reward.scorer.requests[0]
    assert len(request.artifacts) == 1
    assert Path(request.artifacts[0].path).exists()
    assert (tmp_path / "debug" / f"{case.reward_name}_requests.jsonl").exists()
    body = (tmp_path / "debug" / f"{case.reward_name}_results.jsonl").read_text(encoding="utf-8")
    assert all(key in body for key in case.fake_scores)

    alternate = _build_reward(case, tmp_path / "alt", score_key=case.alternate_score_key)
    output = await alternate.score_batch([_sample()])
    assert output.scores == pytest.approx([case.fake_scores[case.alternate_score_key]])


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
@pytest.mark.asyncio
async def test_missing_score_key_fails_fast(case: _Case, tmp_path: Path) -> None:
    """An unknown score_key raises rather than silently scoring zero."""
    reward = _build_reward(case, tmp_path, score_key="not_a_real_axis")
    with pytest.raises(KeyError, match="missing score keys"):
        await reward.score_batch([_sample()])


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_config_accepts_the_shipped_component_shape(case: _Case) -> None:
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {case.reward_name: 1.0},
                "kwargs": {
                    case.reward_name: {
                        "reward_name": case.reward_name,
                        "score_key": case.default_score_key,
                        "worker_config": {"reward_model_name": case.reward_model_name},
                    },
                },
            },
        },
    )
    RewardConfig.from_cfg(cfg)
