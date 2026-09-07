"""Tests for model-backed RewardFunction over the in-process transport."""

from __future__ import annotations

import pytest
import torch

from vrl.rewards.artifacts import InMemoryRewardArtifactStore
from vrl.rewards.base import (
    InferenceRewardFunction,
    RewardCleanupError,
)
from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.runtime import InProcessRewardScorer
from vrl.rewards.types import RewardSample
from vrl.utils.config import import_from_path


class _FakeTorchReward(TorchRewardModel):
    """Toy torch reward model: score = mean of the in-memory media tensor."""

    def _load_module(self) -> torch.nn.Module:
        self.loaded_marker = True
        return torch.nn.Identity()

    def score_media(self, *, media, prompt):
        return {"fake": float(media.float().mean().item())}


def _sample(
    output: torch.Tensor,
    *,
    sample_id: str = "sample-0",
    policy_version: int = 2,
) -> RewardSample:
    return RewardSample(
        prompt="p",
        output=output,
        sample_id=sample_id,
        metadata={"policy_version": policy_version},
    )


def _reward_function_in_process() -> InferenceRewardFunction:
    return InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        scorer=InProcessRewardScorer(
            model=_FakeTorchReward({"device": "cpu"}),
        ),
    )


@pytest.mark.asyncio
async def test_reward_function_in_process_scores_without_disk() -> None:
    """Checks reward function scoring in-process without disk artifacts."""
    reward = _reward_function_in_process()
    report = await reward.score_batch(
        [
            _sample(torch.full((1, 3, 2, 2), 0.5)),
            _sample(torch.ones(1, 3, 2, 2), sample_id="sample-1"),
        ],
    )

    assert report.scores == pytest.approx([0.5, 1.0])
    await reward.shutdown()


@pytest.mark.asyncio
async def test_reward_function_in_process_single_score() -> None:
    """A single in-process ``score`` on an all-zero image returns the model's score for it (0.0
    for the test model).
    """
    reward = _reward_function_in_process()
    assert await reward.score(_sample(torch.zeros(1, 3, 2, 2))) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("factory_path", "expected_type"),
    [
        (
            "vrl.rewards.models.aesthetic:AestheticRewardModel",
            "AestheticRewardModel",
        ),
        (
            "vrl.rewards.models.pickscore:PickScoreRewardModel",
            "PickScoreRewardModel",
        ),
    ],
)
def test_reward_model_factory_class_paths_construct(
    factory_path: str,
    expected_type: str,
) -> None:
    model = import_from_path(factory_path)({"device": "cpu"})

    assert type(model).__name__ == expected_type


def test_lazy_reward_models_defer_module_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights land in the runtime's CuMem build frame, never in ``__init__``."""
    from vrl.rewards.models.aesthetic import AestheticRewardModel
    from vrl.rewards.models.motion_dynamics import MotionDynamicsModel
    from vrl.rewards.models.pickscore import PickScoreRewardModel
    from vrl.rewards.models.target_dino_similarity import TargetDinoSimilarityModel

    models = [
        AestheticRewardModel({"device": "cuda:7"}),
        PickScoreRewardModel({"device": "cuda:7"}),
        MotionDynamicsModel({"device": "cuda:7"}),
        TargetDinoSimilarityModel({"device": "cuda:7"}),
    ]
    for model in models:
        built_on: list[str] = []
        module = object()

        def build(*, owner=model, result=module, devices=built_on):
            devices.append(owner.device)
            return result

        monkeypatch.setattr(model, "_load_module", build)
        assert model._module is None  # construction allocated nothing

        model.prepare_for_inference()

        assert built_on == ["cuda:7"]
        assert model._module is module

        model.prepare_for_inference()
        assert built_on == ["cuda:7"]  # one build per runtime lifetime


def test_inference_reward_defaults_to_inmemory_artifact_store() -> None:
    class _Runtime:
        scoring_is_nonblocking = False
        external_accelerator_isolation_verified = False

        async def score_batch(self, request):
            return []

        async def shutdown(self) -> None:
            return None

    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        scorer=_Runtime(),
    )
    assert isinstance(reward.artifact_store, InMemoryRewardArtifactStore)


def test_inference_reward_rejects_empty_score_key_component() -> None:
    class _Runtime:
        scoring_is_nonblocking = False
        external_accelerator_isolation_verified = False

        async def score_batch(self, request):
            return []

        async def shutdown(self) -> None:
            return None

    with pytest.raises(ValueError, match="empty component"):
        InferenceRewardFunction(
            reward_name="fake",
            score_key="a++b",
            scorer=_Runtime(),
        )


@pytest.mark.asyncio
async def test_reward_reports_operation_and_artifact_cleanup_failures() -> None:
    class _FailingRuntime:
        scoring_is_nonblocking = False
        external_accelerator_isolation_verified = False

        async def score_batch(self, request):
            raise RuntimeError("score failed")

        async def shutdown(self) -> None:
            return None

    class _FailingCleanupStore(InMemoryRewardArtifactStore):
        def release(self, artifacts) -> None:
            assert artifacts
            raise OSError("cleanup failed")

    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        scorer=_FailingRuntime(),
        artifact_store=_FailingCleanupStore(),
    )

    with pytest.raises(RewardCleanupError) as error:
        await reward.score(_sample(torch.zeros(1, 3, 2, 2)))

    assert [str(item) for item in error.value.errors] == [
        "score failed",
        "cleanup failed",
    ]


@pytest.mark.asyncio
async def test_reward_retains_artifacts_when_remote_state_is_ambiguous() -> None:
    class _AmbiguousRuntime:
        scoring_is_nonblocking = False
        external_accelerator_isolation_verified = False

        async def score_batch(self, request):
            error = RuntimeError("remote state unknown")
            error.retain_reward_artifacts = True
            raise error

        async def shutdown(self) -> None:
            return None

    class _RecordingStore(InMemoryRewardArtifactStore):
        released = False
        retained = False

        def release(self, artifacts) -> None:
            self.released = True

        def retain(self, artifacts) -> None:
            assert artifacts
            self.retained = True

    store = _RecordingStore()
    reward = InferenceRewardFunction(
        reward_name="fake",
        score_key="fake",
        scorer=_AmbiguousRuntime(),
        artifact_store=store,
    )

    with pytest.raises(RuntimeError, match="remote state unknown"):
        await reward.score(_sample(torch.zeros(1, 3, 2, 2)))

    assert store.retained is True
    assert store.released is False
