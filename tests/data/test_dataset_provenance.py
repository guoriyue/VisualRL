"""Source-backed dataset provenance (``DatasetProvenance.from_config``).

One check for every production task type: the manifests load through the
loader the config declares, their artifacts resolve and their rows carry the
task type's provenance vocabulary, and the builder's ``report.json`` agrees
with them. The image-to-video rows go through the real image-caption loader,
so the check sees exactly the ``PromptExample`` rows training will read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.trainers.data.provenance import PROVENANCE_SPECS, DatasetProvenance, SourceReport


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


_I2V_METADATA = {
    "source_repo": "videophysics/videophy_test_public",
    "source_video_url": "https://example.invalid/clip.mp4",
    "source_frame_index": 0,
    "decode_method": "pyav_first_frame",
    "conditioning": "reference_image",
}


def _i2v_dataset(tmp_path: Path, *, eval_rows: int = 1, report_eval_rows: int | None = None):
    data_root = tmp_path / "external"
    _write_ppm(data_root / "i2v" / "ref.ppm")
    row = {"image": "i2v/ref.ppm", "caption": "a cat", "metadata": dict(_I2V_METADATA)}
    train, eval_manifest, report = (
        tmp_path / "train.jsonl",
        tmp_path / "eval.jsonl",
        tmp_path / "report.json",
    )
    _write_jsonl(train, [row, row])
    _write_jsonl(eval_manifest, [row] * eval_rows)
    _write_report(
        report,
        {
            "dataset": "videophy_i2v",
            "source_repo": "videophysics/videophy_test_public",
            "source_csv": "videophy_test.csv",
            "source_split": "test",
            "decode_method": "pyav_first_frame",
            "train_rows": 2,
            "eval_rows": eval_rows if report_eval_rows is None else report_eval_rows,
            "train_manifest": str(train),
            "eval_manifest": str(eval_manifest),
            "reference_dir": str(data_root / "i2v"),
        },
    )
    return parse_config(
        OmegaConf.create(
            {
                "data": {
                    "loader": "prompt_image_manifest",
                    "manifest": str(train),
                    "eval_manifest": str(eval_manifest),
                    "source_report": str(report),
                    "artifact_data_root": str(data_root),
                    "task_type": "image_to_video",
                    "preprocessing": {
                        "format": "image_caption_jsonl",
                        "image_field": "image",
                        "caption_field": "caption",
                        "conditioning": "reference_image",
                    },
                    "sampler": {"type": "random_without_replacement"},
                }
            }
        )
    ).data


def _v2w_dataset(tmp_path: Path, *, with_target: bool):
    data_root = tmp_path / "external"
    _write_ppm(data_root / "v2w" / "ref.ppm")
    _write_ppm(data_root / "v2w" / "target.ppm")
    row = {
        "prompt": "open the drawer",
        "reference_image": "v2w/ref.ppm",
        "metadata": {
            "source": "bridge",
            "source_repo": "org/bridge",
            "source_split": "main",
            "source_episode": "ep-1",
            "source_video": "ep-1.mp4",
            "source_frame_index": 0,
            "decode_method": "pyav_http_first_frame",
            "conditioning": "reference_image",
        },
    }
    if with_target:
        row["target_video"] = "v2w/target.ppm"
    train, eval_manifest, report = (
        tmp_path / "train.jsonl",
        tmp_path / "eval.jsonl",
        tmp_path / "report.json",
    )
    _write_jsonl(train, [row])
    _write_jsonl(
        eval_manifest, [{**row, "metadata": {**row["metadata"], "source_episode": "ep-2"}}]
    )
    _write_report(
        report,
        {
            "dataset": "video_world_bridge",
            "source": "bridge",
            "repo_id": "org/bridge",
            "source_split": "main",
            "decode_method": "pyav_http_first_frame",
            "train_rows": 1,
            "eval_rows": 1,
            "train_manifest": str(train),
            "eval_manifest": str(eval_manifest),
            "reference_dir": str(data_root / "v2w"),
            "validation_summary": {"row_count": 1},
        },
    )
    return parse_config(
        OmegaConf.create(
            {
                "data": {
                    "loader": "prompt_manifest",
                    "manifest": str(train),
                    "eval_manifest": str(eval_manifest),
                    "source_report": str(report),
                    "artifact_data_root": str(data_root),
                    "task_type": "video2world",
                    "preprocessing": {},
                    "sampler": {"type": "random_without_replacement"},
                }
            }
        )
    ).data


def test_image_to_video_provenance_loads_rows_through_the_image_caption_loader(tmp_path) -> None:
    provenance = DatasetProvenance.from_config(_i2v_dataset(tmp_path))

    assert provenance.spec is PROVENANCE_SPECS["image_to_video"]
    assert provenance.train.row_count == 2
    assert {item.field for item in provenance.train.resolved_artifacts} == {"reference_image"}
    assert provenance.report is not None and provenance.report.dataset == "videophy_i2v"


def test_image_to_video_report_rows_must_match_the_manifests(tmp_path) -> None:
    data = _i2v_dataset(tmp_path, eval_rows=1, report_eval_rows=3)
    with pytest.raises(ValueError, match="eval_rows does not match"):
        DatasetProvenance.from_config(data)


def test_image_to_video_rows_need_their_provenance_metadata(tmp_path) -> None:
    data = _i2v_dataset(tmp_path)
    manifest = Path(data.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    del rows[0]["metadata"]["source_video_url"]
    _write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match=r"metadata\.source_video_url is required"):
        DatasetProvenance.from_config(data)


def test_video_world_provenance_passes_and_reward_artifacts_become_requirements(tmp_path) -> None:
    """A reward that reads ``target_video`` turns it into a hard row requirement."""
    without_target = _v2w_dataset(tmp_path, with_target=False)

    assert DatasetProvenance.from_config(without_target).spec is not None
    with pytest.raises(ValueError, match="missing required field target_video"):
        DatasetProvenance.from_config(without_target, extra_artifact_fields=("target_video",))

    with_target = _v2w_dataset(tmp_path / "targets", with_target=True)
    provenance = DatasetProvenance.from_config(
        with_target, extra_artifact_fields=("target_video",)
    )
    assert {item.field for item in provenance.train.resolved_artifacts} == {
        "reference_image",
        "target_video",
    }


def test_video_world_report_needs_its_validation_summary(tmp_path) -> None:
    data = _v2w_dataset(tmp_path, with_target=False)
    report = Path(data.source_report)
    payload = json.loads(report.read_text())
    payload["validation_summary"] = {}
    _write_report(report, payload)
    with pytest.raises(
        ValueError, match="video2world provenance fields: \\['validation_summary'\\]"
    ):
        DatasetProvenance.from_config(data)


def test_plain_text_task_types_only_need_the_three_files(tmp_path) -> None:
    for name in ("train.jsonl", "eval.jsonl"):
        _write_jsonl(tmp_path / name, [{"prompt": "a cat"}])
    _write_report(tmp_path / "report.json", {})
    data = parse_config(
        OmegaConf.create(
            {
                "data": {
                    "loader": "prompt_manifest",
                    "manifest": str(tmp_path / "train.jsonl"),
                    "eval_manifest": str(tmp_path / "eval.jsonl"),
                    "source_report": str(tmp_path / "report.json"),
                    "task_type": "text_to_video",
                    "preprocessing": {},
                    "sampler": {"type": "random_without_replacement"},
                }
            }
        )
    ).data

    assert DatasetProvenance.from_config(data).spec is None
    data_missing = data.model_copy(update={"source_report": str(tmp_path / "nope.json")})
    with pytest.raises(ValueError, match=r"data\.source_report does not exist"):
        DatasetProvenance.from_config(data_missing)


def test_source_report_rejects_empty_row_counts(tmp_path) -> None:
    report = tmp_path / "report.json"
    _write_report(
        report,
        {key: 0 if key.endswith("_rows") else "x" for key in SourceReport.REQUIRED_KEYS},
    )
    with pytest.raises(ValueError, match="non-empty train and eval rows"):
        SourceReport.from_json(report)
