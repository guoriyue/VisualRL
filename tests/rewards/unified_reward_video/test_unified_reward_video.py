"""UnifiedReward-2.0 video reward: parsing, rubric loading, and facade wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.schema import RewardConfig
from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
from vrl.rewards.inference import RewardInferenceResult
from vrl.rewards.models.unified_reward_video import _load_rubric, _parse_axis_scores
from vrl.rewards.types import RewardSample

_FAKE_SCORES = {"alignment": 4.0, "physics": 2.0, "style": 3.0, "overall": 3.0}


class _FakeRuntime:
    scoring_is_nonblocking = False
    external_accelerator_isolation_verified = False

    def __init__(self) -> None:
        self.requests = []

    async def score_batch(self, request):
        self.requests.append(request)
        return [
            RewardInferenceResult(
                artifact_id=artifact.artifact_id,
                scores=dict(_FAKE_SCORES),
                reward_model_version="fake",
                timing_ms={"inference_ms": 1.0},
            )
            for artifact in request.artifacts
        ]

    async def shutdown(self) -> None:
        return None


def _sample() -> RewardSample:
    return RewardSample(
        prompt="a spinning dancer",
        output=torch.ones(1, 2, 2, 2),
        sample_id="sample-0",
    )


def test_parse_axis_scores_reads_floats() -> None:
    text = "Alignment Score (1-5): 2.4036\nPhysics Score (1-5): 3.0987\nStyle Score (1-5): 3.3889"
    scores = _parse_axis_scores(text)
    assert scores["alignment"] == pytest.approx(2.4036)
    assert scores["physics"] == pytest.approx(3.0987)
    assert scores["style"] == pytest.approx(3.3889)
    assert scores["overall"] == pytest.approx((2.4036 + 3.0987 + 3.3889) / 3.0)


def test_parse_axis_scores_normalizes_model_declared_scale() -> None:
    text = (
        "Alignment Score (1-5): 2.7907\n"
        "Physics Score (1.0-10.0): 4.8017\n"
        "Style Score (1-5): 2.7633"
    )
    scores = _parse_axis_scores(text)
    assert scores["alignment"] == pytest.approx(2.7907)
    assert scores["physics"] == pytest.approx(1.0 + 4.0 * (4.8017 - 1.0) / 9.0)
    assert scores["style"] == pytest.approx(2.7633)


@pytest.mark.parametrize(
    ("physics_line", "message"),
    [
        ("Physics Score (5-1): 3", "invalid 'physics' score range"),
        ("Physics Score (1-5): 7", "out-of-range 'physics' score"),
    ],
)
def test_parse_axis_scores_rejects_invalid_declared_scale(
    physics_line: str,
    message: str,
) -> None:
    text = f"Alignment Score (1-5): 4\n{physics_line}\nStyle Score (1-5): 3"
    with pytest.raises(ValueError, match=message):
        _parse_axis_scores(text)


def test_parse_axis_scores_missing_axis_raises() -> None:
    with pytest.raises(ValueError, match="missing 'style'"):
        _parse_axis_scores("Alignment Score (1-5): 4\nPhysics Score (1-5): 3")


def test_load_rubric_default_and_override(tmp_path: Path) -> None:
    assert "{prompt}" in _load_rubric("")
    rubric = tmp_path / "r.yaml"
    rubric.write_text(
        "problem_template: 'judge {prompt} on Style Score (1-5): Z'\n", encoding="utf-8"
    )
    assert "{prompt}" in _load_rubric(str(rubric))


def test_load_rubric_rejects_missing_prompt_slot(tmp_path: Path) -> None:
    rubric = tmp_path / "bad.yaml"
    rubric.write_text("problem_template: 'no slot here'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a"):
        _load_rubric(str(rubric))


@pytest.mark.asyncio
async def test_facade_selects_physics_and_fails_fast(tmp_path: Path) -> None:
    reward = UnifiedRewardVideoReward(
        reward_name="unified_reward_video",
        score_key="physics",
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "a"),
        scorer=_FakeRuntime(),
    )
    assert (await reward.score_batch([_sample()])).scores == pytest.approx([2.0])

    bad = UnifiedRewardVideoReward(
        reward_name="unified_reward_video",
        score_key="identity_consistency",
        artifact_format="tensor",
        artifact_dir=str(tmp_path / "b"),
        scorer=_FakeRuntime(),
    )
    with pytest.raises(KeyError, match="missing score keys"):
        await bad.score_batch([_sample()])


def test_config_validates() -> None:
    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"unified_reward_video": 1.0},
                "kwargs": {
                    "unified_reward_video": {
                        "reward_name": "unified_reward_video",
                        "score_key": "overall",
                        "worker_config": {
                            "reward_model_name": "CodeGoat24/UnifiedReward-2.0-qwen-7b@main"
                        },
                    },
                },
            },
        },
    )
    RewardConfig.from_cfg(cfg)
