"""Tests for video reward artifact materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from vrl.rewards.artifacts import DiskRewardArtifactStore
from vrl.rewards.types import RewardSample


def _sample(output: torch.Tensor) -> RewardSample:
    return RewardSample(
        prompt="prompt",
        output=output,
        sample_id="sample-x",
        metadata={
            "policy_version": 4,
            "fps": 8,
            "task_type": "video2world",
            "reference_image": "/tmp/reference.png",
            "source_repo": "lerobot/droid_100",
            "source_episode": "000001",
        },
    )


def test_video_artifact_store_writes_tensor_with_provenance(tmp_path: Path) -> None:
    """Checks video artifact store writes the tensor and provenance metadata."""
    store = DiskRewardArtifactStore(tmp_path, media_type="video")

    artifacts = store.materialize([_sample(torch.ones(1, 2, 2, 2))])

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_id.startswith("sample-x:")
    assert artifact.sample_id == "sample-x"
    assert Path(artifact.path).is_absolute()
    assert Path(artifact.path).exists()
    assert artifact.size_bytes == Path(artifact.path).stat().st_size
    assert artifact.sha256 == hashlib.sha256(Path(artifact.path).read_bytes()).hexdigest()
    assert torch.load(artifact.path).shape == (1, 2, 2, 2)
    assert artifact.metadata["reference_image"] == "/tmp/reference.png"
    assert artifact.metadata["source_repo"] == "lerobot/droid_100"


def test_video_artifact_store_writes_mp4_for_reward_models(tmp_path: Path) -> None:
    """``artifact_format="mp4"`` materializes a video sample as an absolute, existing ``.mp4``
    path.
    """
    store = DiskRewardArtifactStore(tmp_path, media_type="video", artifact_format="mp4")

    artifacts = store.materialize([_sample(torch.ones(3, 2, 4, 4))])

    artifact = artifacts[0]
    assert artifact.path.endswith(".mp4")
    assert Path(artifact.path).is_absolute()
    assert Path(artifact.path).exists()


def test_video_artifact_store_rejects_mp4_for_non_video_media_type(tmp_path: Path) -> None:
    """Checks mp4 artifacts require explicit video media type."""

    with pytest.raises(ValueError, match="artifact_format=mp4 requires media_type=video"):
        DiskRewardArtifactStore(tmp_path, media_type="image", artifact_format="mp4")


def test_video_artifact_store_rejects_bad_shape(tmp_path: Path) -> None:
    store = DiskRewardArtifactStore(tmp_path, media_type="video")

    with pytest.raises(ValueError, match="video reward artifact expects"):
        store.materialize([_sample(torch.ones(2, 2))])


def test_video_artifact_store_rejects_unknown_artifact_format(tmp_path: Path) -> None:
    """An unknown artifact_format is rejected against the Literal-derived allow-list."""

    with pytest.raises(ValueError, match="artifact_format must be one of"):
        DiskRewardArtifactStore(tmp_path, media_type="video", artifact_format="webm")


def test_video_artifact_store_never_overwrites_and_releases_owned_paths(
    tmp_path: Path,
) -> None:
    store = DiskRewardArtifactStore(tmp_path, media_type="video")
    first = store.materialize([_sample(torch.zeros(1, 2, 2, 2))])
    second = store.materialize([_sample(torch.ones(1, 2, 2, 2))])

    assert first[0].artifact_id != second[0].artifact_id
    assert first[0].path != second[0].path
    assert torch.count_nonzero(torch.load(first[0].path, weights_only=True)) == 0
    assert torch.all(torch.load(second[0].path, weights_only=True))

    store.release(first + second)
    store.release(first + second)

    assert not Path(first[0].path).exists()
    assert not Path(second[0].path).exists()
