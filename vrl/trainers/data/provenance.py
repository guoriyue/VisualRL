"""Source-backed dataset provenance: what a production run requires of its data.

Every dataset builder under ``vrl/scripts/data`` writes the same two things
next to its manifests: prompt rows whose artifacts resolve under the data root
and carry provenance metadata, and a ``report.json`` describing the dump
(``SourceReport``). What differs per prompt task type is the vocabulary — which
artifact fields, which metadata keys, which report keys — and that is a
``DatasetProvenanceSpec`` in ``PROVENANCE_SPECS``, keyed by ``data.task_type``.

``DatasetProvenance.from_config`` is the one check, in the repo's
constructor-validates shape (like ``ArtifactManifestReport.from_manifest`` and
``PrecisionPolicy.from_section``): it loads both manifests with the loader the
config declares (so the rows are exactly what training will read), runs the
artifact / metadata checks through ``ArtifactManifestReport``, and cross-checks
the source report against the loaded rows. It reads files, so it is reached
from a launch gate (``vrl/config/validation.py``), never a schema rule.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vrl.trainers.data.artifacts import ArtifactManifestReport
from vrl.trainers.data.prompts import (
    PromptExample,
    load_prompt_image_manifest,
    load_prompt_manifest,
)
from vrl.utils.artifacts import SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS

if TYPE_CHECKING:
    from vrl.config.schema import DataConfig


@dataclass(frozen=True, slots=True)
class SourceReport:
    """The ``report.json`` a dataset builder writes beside its manifests."""

    # Keys every builder writes (vrl/scripts/data/videophy_i2v.py, video_world/cli.py).
    REQUIRED_KEYS = (
        "dataset",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
    )

    path: Path
    dataset: str
    train_rows: int
    eval_rows: int
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> SourceReport:
        report_path = Path(path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"data.source_report must be a JSON object: {report_path}")
        missing = sorted(key for key in cls.REQUIRED_KEYS if key not in payload)
        if missing:
            raise ValueError(f"data.source_report is missing provenance fields: {missing}")
        train_rows = int(payload.get("train_rows") or 0)
        eval_rows = int(payload.get("eval_rows") or 0)
        if train_rows <= 0 or eval_rows <= 0:
            raise ValueError("data.source_report must record non-empty train and eval rows")
        return cls(
            path=report_path,
            dataset=str(payload["dataset"]),
            train_rows=train_rows,
            eval_rows=eval_rows,
            payload=payload,
        )

    def require_fields(self, fields: Sequence[str], *, task_type: str) -> None:
        """Task-specific report keys, each present and non-empty."""

        missing = sorted(
            field
            for field in fields
            if field not in self.payload
            or self.payload[field] is None
            or self.payload[field] == ""
            or self.payload[field] == {}
        )
        if missing:
            raise ValueError(
                f"data.source_report is missing {task_type} provenance fields: {missing}",
            )


@dataclass(frozen=True, slots=True)
class DatasetProvenanceSpec:
    """The provenance vocabulary of one prompt task type."""

    task_type: str
    # PromptExample fields that must resolve to readable files under the data root.
    artifact_fields: tuple[str, ...]
    # Per-row ``metadata`` keys that must be present and non-empty.
    required_metadata_fields: tuple[str, ...]
    # Report keys beyond ``SourceReport.REQUIRED_KEYS``, each non-empty.
    report_fields: tuple[str, ...] = ()
    # Accepted ``report.dataset`` names; ``None`` accepts any.
    dataset_names: tuple[str, ...] | None = None
    # Whether the report's row counts must equal the loaded manifests' row counts
    # (a dump that is consumed whole) or merely be positive (a dump whose
    # manifests may be re-split downstream).
    rows_must_match: bool = False
    # Whether ``data.artifact_data_root`` must be declared explicitly.
    requires_data_root: bool = False

    @staticmethod
    def load_manifest(data: DataConfig, path: str) -> list[PromptExample]:
        """Load one manifest the way the parsed ``data`` section declares."""

        if data.loader == "prompt_image_manifest":
            preprocessing = data.preprocessing
            return load_prompt_image_manifest(
                path,
                image_field=str((preprocessing.image_field if preprocessing else None) or "image"),
                caption_field=str(
                    (preprocessing.caption_field if preprocessing else None) or "caption"
                ),
                default_task_type=str(data.task_type or "image_to_video"),
            )
        if data.loader == "prompt_manifest":
            return load_prompt_manifest(path)
        raise ValueError(f"dataset provenance has no loader for data.loader={data.loader!r}")


PROVENANCE_SPECS: dict[str, DatasetProvenanceSpec] = {
    "video2world": DatasetProvenanceSpec(
        task_type="video2world",
        artifact_fields=("reference_image",),
        required_metadata_fields=SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS,
        report_fields=("source", "repo_id", "validation_summary"),
    ),
    "image_to_video": DatasetProvenanceSpec(
        task_type="image_to_video",
        artifact_fields=("reference_image",),
        # Shares the {source_repo, source_frame_index, decode_method,
        # conditioning} sub-vocabulary with the video-world fields; the rows
        # name a URL (``source_video_url``) rather than a local ``source_video``.
        required_metadata_fields=(
            "source_repo",
            "source_video_url",
            "source_frame_index",
            "decode_method",
            "conditioning",
        ),
        report_fields=("source_repo", "source_csv"),
        dataset_names=("videophy_i2v",),
        rows_must_match=True,
        requires_data_root=True,
    ),
}


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """What a source-backed dataset established at launch.

    ``spec`` / ``train`` / ``report`` are ``None`` for a task type without a
    provenance spec (plain text prompts): only the three files' existence is
    required then.
    """

    spec: DatasetProvenanceSpec | None
    train: ArtifactManifestReport | None
    report: SourceReport | None

    @classmethod
    def from_config(cls, data: DataConfig) -> DatasetProvenance:
        """Check the manifests and source report a source-backed dataset ships.

        Whether the configured rewards can read these rows (a target clip, a
        reference) is not decided here: ``python -m vrl.scripts.rewards.preflight``
        runs the real reward over the rows before training does.
        """

        for name in ("manifest", "eval_manifest", "source_report"):
            value = str(getattr(data, name) or "").strip()
            if not value:
                raise ValueError(f"config missing required field: data.{name}")
            if not Path(value).exists():
                raise ValueError(f"data.{name} does not exist: {value}")
        if not isinstance(data.manifest, str):
            raise ValueError("dataset provenance validates one data.manifest path, not a mixture")

        spec = PROVENANCE_SPECS.get(str(data.task_type or ""))
        if spec is None:
            return cls(spec=None, train=None, report=None)
        data_root = str(data.artifact_data_root or "").strip()
        if spec.requires_data_root and not data_root:
            raise ValueError("config missing required field: data.artifact_data_root")

        artifact_fields = spec.artifact_fields
        train_examples = spec.load_manifest(data, data.manifest)
        eval_examples = spec.load_manifest(data, str(data.eval_manifest))
        train = ArtifactManifestReport.from_examples(
            train_examples,
            manifest_path=data.manifest,
            eval_examples=eval_examples,
            eval_manifest_path=str(data.eval_manifest),
            data_root=data_root or None,
            artifact_fields=artifact_fields,
            required_artifact_fields=artifact_fields,
            required_metadata_fields=spec.required_metadata_fields,
        )

        report = SourceReport.from_json(str(data.source_report))
        report.require_fields(spec.report_fields, task_type=spec.task_type)
        if spec.dataset_names is not None and report.dataset not in spec.dataset_names:
            expected = ", ".join(spec.dataset_names)
            raise ValueError(
                f"data.source_report dataset must be {expected}, got {report.dataset!r}"
            )
        if spec.rows_must_match:
            if report.train_rows != len(train_examples):
                raise ValueError("data.source_report train_rows does not match data.manifest")
            if report.eval_rows != len(eval_examples):
                raise ValueError("data.source_report eval_rows does not match data.eval_manifest")
        return cls(spec=spec, train=train, report=report)


__all__ = [
    "PROVENANCE_SPECS",
    "DatasetProvenance",
    "DatasetProvenanceSpec",
    "SourceReport",
]
