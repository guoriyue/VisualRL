from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.scripts.data import setup as setup_cli
from vrl.trainers.data import PromptExample, load_prompt_manifest
from vrl.trainers.data.artifacts import (
    ArtifactManifestReport,
    resolve_prompt_example_references,
)
from vrl.utils.artifacts import ArtifactManifestError, resolve_artifact_path


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")


def _write_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_artifact_manifest_resolves_relative_references_via_data_root(tmp_path: Path) -> None:
    """A relative ``reference_image`` stays relative on the loaded example while the report
    resolves it under ``data_root`` and counts it as one artifact.
    """
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    manifest = tmp_path / "v2w_train.jsonl"
    _write_ppm(reference)
    _write_manifest(
        manifest,
        {
            "prompt": "reference prompt",
            "reference_image": "video_world/references/ref.ppm",
        },
    )

    examples = load_prompt_manifest(manifest)
    report = ArtifactManifestReport.from_manifest(
        manifest,
        data_root=data_root,
        required_artifact_fields=("reference_image",),
    )

    assert examples[0].reference_image == "video_world/references/ref.ppm"
    assert report.row_count == 1
    assert report.artifact_count == 1
    assert report.resolved_artifacts[0].resolved_path == reference.resolve()


def test_target_artifacts_are_prompt_fields_and_validate(tmp_path: Path) -> None:
    """Checks target media fields are first-class artifact fields."""
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    target = data_root / "video_world" / "targets" / "target.ppm"
    manifest = tmp_path / "home_train.jsonl"
    _write_ppm(reference)
    _write_ppm(target)
    _write_manifest(
        manifest,
        {
            "prompt": "open the drawer",
            "reference_image": "video_world/references/ref.ppm",
            "target_image": "video_world/targets/target.ppm",
        },
    )

    examples = load_prompt_manifest(manifest)
    report = ArtifactManifestReport.from_manifest(
        manifest,
        data_root=data_root,
        required_artifact_fields=("reference_image", "target_image"),
    )

    assert examples[0].target_image == "video_world/targets/target.ppm"
    assert report.artifact_count == 2
    assert {item.field for item in report.resolved_artifacts} == {
        "reference_image",
        "target_image",
    }


def test_reference_resolution_preserves_target_identity_fields(tmp_path: Path) -> None:
    """Reference consumers get absolute paths without changing target lookup keys."""

    example = PromptExample(
        prompt="open the drawer",
        reference_image="references/frame.ppm",
        reference_video="references/context.mp4",
        target_image="targets/result.ppm",
        target_video="targets/result.mp4",
        references=["references/alternate.ppm"],
    )

    resolved = resolve_prompt_example_references(example, data_root=tmp_path)

    assert resolved.reference_image == str((tmp_path / "references/frame.ppm").resolve())
    assert resolved.reference_video == str((tmp_path / "references/context.mp4").resolve())
    assert resolved.references == [
        str((tmp_path / "references/alternate.ppm").resolve()),
    ]
    assert resolved.target_image == "targets/result.ppm"
    assert resolved.target_video == "targets/result.mp4"
    assert example.reference_image == "references/frame.ppm"

    blank = resolve_prompt_example_references(
        PromptExample(prompt="no reference", reference_image="  "),
        data_root=tmp_path,
    )
    assert blank.reference_image is None


def test_missing_reference_image_fails_with_manifest_row(tmp_path: Path) -> None:
    manifest = tmp_path / "missing.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "missing ref",
                "reference_image": "video_world/references/missing.png",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactManifestError, match=r"row 0 reference_image does not exist"):
        ArtifactManifestReport.from_manifest(
            manifest,
            data_root=tmp_path,
            required_artifact_fields=("reference_image",),
        )


def test_absolute_artifact_paths_are_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "ref.ppm"
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="absolute artifact paths"):
        resolve_artifact_path(path)


def test_production_metadata_domain_is_rejected(tmp_path: Path) -> None:
    """Manifest ``metadata.domain`` is production-owned and rejected in a user manifest."""
    reference = tmp_path / "video_world" / "references" / "ref.ppm"
    _write_ppm(reference)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        {
            "prompt": "bad metadata",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": {"domain": "robotics"},
        },
    )

    with pytest.raises(ArtifactManifestError, match=r"metadata\.domain"):
        ArtifactManifestReport.from_manifest(
            manifest,
            data_root=tmp_path,
            required_artifact_fields=("reference_image",),
        )


def test_setup_cli_creates_ignored_external_dirs(
    tmp_path: Path,
) -> None:
    """``init-dirs video-world`` creates the git-ignored ``references`` and ``targets``
    directories under the data root.
    """
    data_root = tmp_path / "external"

    assert (
        setup_cli.main(
            [
                "init-dirs",
                "video-world",
                "--data-root",
                str(data_root),
            ],
        )
        is None
    )

    assert (data_root / "video_world" / "references").is_dir()
    assert (data_root / "video_world" / "targets").is_dir()
