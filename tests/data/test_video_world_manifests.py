from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.trainers.data.artifacts import (
    ArtifactManifestReport,
)
from vrl.utils.artifacts import ArtifactManifestError


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _source_backed_metadata(episode: str) -> dict:
    return {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": episode,
        "source_video": "videos/camera/batch-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_first_frame",
        "conditioning": "first_frame",
    }


def test_video_world_manifests_validate_reference_artifacts(tmp_path: Path) -> None:
    """A train / eval pair with resolvable references and distinct source episodes reports one
    row, no overlap and no warnings.
    """
    data_root = tmp_path / "external"
    _write_ppm(data_root / "video_world" / "references" / "ref.ppm")
    train_manifest = tmp_path / "v2w_train.jsonl"
    eval_manifest = tmp_path / "v2w_eval.jsonl"
    _write_jsonl(
        train_manifest,
        {
            "prompt": "The robot arm moves toward the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": {"source_episode": "episode_train"},
        },
    )
    _write_jsonl(
        eval_manifest,
        {
            "prompt": "The robot arm moves away from the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": {"source_episode": "episode_eval"},
        },
    )

    report = ArtifactManifestReport.from_manifest(
        train_manifest,
        eval_manifest=eval_manifest,
        data_root=data_root,
        artifact_fields=("reference_image",),
        required_artifact_fields=("reference_image",),
    )

    assert report.row_count == 1
    assert report.eval_manifest_path is not None
    assert report.source_episode_overlap == ()
    assert report.warnings == ()


def test_source_backed_video_world_manifest_requires_provenance(tmp_path: Path) -> None:
    """The source-backed loader accepts rows whose metadata carries full provenance and reports no
    episode overlap.
    """
    data_root = tmp_path / "external"
    _write_ppm(data_root / "video_world" / "references" / "ref.ppm")
    train_manifest = tmp_path / "v2w_train.jsonl"
    eval_manifest = tmp_path / "v2w_eval.jsonl"
    _write_jsonl(
        train_manifest,
        {
            "prompt": "The robot arm moves toward the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": _source_backed_metadata("episode_train"),
        },
    )
    _write_jsonl(
        eval_manifest,
        {
            "prompt": "The robot arm moves away from the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": _source_backed_metadata("episode_eval"),
        },
    )

    report = ArtifactManifestReport.from_video_world_manifest(
        train_manifest,
        eval_manifest=eval_manifest,
        data_root=data_root,
    )

    assert report.row_count == 1
    assert report.source_episode_overlap == ()


def test_source_backed_video_world_manifest_can_require_target_video(tmp_path: Path) -> None:
    """Checks target-aware Video2World manifests require target clips."""
    data_root = tmp_path / "external"
    _write_ppm(data_root / "video_world" / "references" / "ref.ppm")
    target = data_root / "video_world" / "targets" / "target.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-video")
    train_manifest = tmp_path / "v2w_train.jsonl"
    eval_manifest = tmp_path / "v2w_eval.jsonl"
    _write_jsonl(
        train_manifest,
        {
            "prompt": "The robot arm moves toward the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "target_video": "video_world/targets/target.mp4",
            "metadata": _source_backed_metadata("episode_train"),
        },
    )
    _write_jsonl(
        eval_manifest,
        {
            "prompt": "The robot arm moves away from the cup.",
            "reference_image": "video_world/references/ref.ppm",
            "target_video": "video_world/targets/target.mp4",
            "metadata": _source_backed_metadata("episode_eval"),
        },
    )

    report = ArtifactManifestReport.from_video_world_manifest(
        train_manifest,
        eval_manifest=eval_manifest,
        data_root=data_root,
        require_target_video=True,
    )

    assert report.artifact_count == 2


def test_source_backed_video_world_manifest_rejects_missing_required_target_video(
    tmp_path: Path,
) -> None:
    """Checks target-aware Video2World validation rejects missing target clips."""
    data_root = tmp_path / "external"
    _write_ppm(data_root / "video_world" / "references" / "ref.ppm")
    train_manifest = tmp_path / "v2w_train.jsonl"
    eval_manifest = tmp_path / "v2w_eval.jsonl"
    row = {
        "prompt": "The robot arm moves toward the cup.",
        "reference_image": "video_world/references/ref.ppm",
        "metadata": _source_backed_metadata("episode_train"),
    }
    _write_jsonl(train_manifest, row)
    _write_jsonl(eval_manifest, row)

    with pytest.raises(ArtifactManifestError, match="missing required field target_video"):
        ArtifactManifestReport.from_video_world_manifest(
            train_manifest,
            eval_manifest=eval_manifest,
            data_root=data_root,
            require_target_video=True,
        )


def test_source_backed_video_world_manifest_rejects_placeholder_rows(tmp_path: Path) -> None:
    """Source-backed rows without ``metadata.source_repo`` are placeholders and are rejected."""
    data_root = tmp_path / "external"
    _write_ppm(data_root / "video_world" / "references" / "ref.ppm")
    train_manifest = tmp_path / "v2w_train.jsonl"
    eval_manifest = tmp_path / "v2w_eval.jsonl"
    row = {
        "prompt": "The robot arm moves toward the cup.",
        "reference_image": "video_world/references/ref.ppm",
        "metadata": {"source": "droid", "source_episode": "episode_train"},
    }
    _write_jsonl(train_manifest, row)
    _write_jsonl(eval_manifest, row)

    with pytest.raises(ArtifactManifestError, match=r"metadata\.source_repo is required"):
        ArtifactManifestReport.from_video_world_manifest(
            train_manifest,
            eval_manifest=eval_manifest,
            data_root=data_root,
        )


def test_video_world_source_episode_overlap_is_reported(tmp_path: Path) -> None:
    """The same ``source_episode`` in train and eval is reported as overlap and surfaced as a
    warning.
    """
    reference = tmp_path / "video_world" / "references" / "ref.ppm"
    _write_ppm(reference)
    train = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    row = {
        "prompt": "same episode",
        "reference_image": "video_world/references/ref.ppm",
        "metadata": {"source_episode": "episode_001"},
    }
    _write_jsonl(train, row)
    _write_jsonl(eval_manifest, row)

    report = ArtifactManifestReport.from_manifest(
        train,
        eval_manifest=eval_manifest,
        data_root=tmp_path,
        required_artifact_fields=("reference_image",),
    )

    assert report.source_episode_overlap == ("episode_001",)
    assert report.warnings == ("train/eval source_episode overlap: episode_001",)


def test_video_world_v2w_manifest_requires_reference_image(tmp_path: Path) -> None:
    manifest = tmp_path / "missing_reference.jsonl"
    manifest.write_text(json.dumps({"prompt": "no reference"}) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="missing required field reference_image"):
        ArtifactManifestReport.from_manifest(
            manifest,
            data_root=tmp_path,
            required_artifact_fields=("reference_image",),
        )
