"""Production Kling VideoReward gate (a launch gate, validation tier 3).

Enabled by ``production.kling_video_reward.enabled``. Two parts:

- ``validate_production_reward_contract`` — the structural contract the
  reward kwargs and the task type must meet. Per-reward config knowledge
  deliberately does not live in the schema (rewards own their contracts at
  construction); this exists because a production misconfiguration is
  unrecoverable mid-run.
- ``validate_production_data`` — the data provenance the contract relies on:
  the manifests and the source report must exist, and for the two
  conditioned tasks their rows and report fields are read and cross-checked.
  Filesystem reads are why this is a launch gate and not a schema rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

_PRODUCTION_TASK_TYPES = frozenset({"text_to_video", "image_to_video", "video2world"})


def validate_production_kling_video_reward_config(root: RootConfig) -> None:
    """The whole gate: structural contract, then data provenance."""

    validate_production_reward_contract(root)
    validate_production_data(root)


def validate_production_reward_contract(root: RootConfig) -> None:
    """Structural production contract for the Kling VideoReward."""

    reward = root.reward
    vr_kwargs = (reward.kwargs.get("kling_video_reward") if reward is not None else None) or {}
    if str(vr_kwargs.get("media_type", "")) != "video":
        raise ValueError(
            "production.kling_video_reward requires "
            "reward.kwargs.kling_video_reward.media_type=video"
        )
    if str(vr_kwargs.get("artifact_format", "")) != "mp4":
        raise ValueError("production.kling_video_reward requires artifact_format=mp4")
    if not str(vr_kwargs.get("reward_name", "")).strip():
        raise ValueError(
            "production.kling_video_reward requires reward.kwargs.kling_video_reward.reward_name"
        )
    worker_config = vr_kwargs.get("worker_config") or {}
    from vrl.rewards.functions.kling_video_reward import (
        PRODUCTION_LOCKED_WORKER_CONFIG_KEYS,
    )

    forbidden = sorted(k for k in PRODUCTION_LOCKED_WORKER_CONFIG_KEYS if k in worker_config)
    if forbidden:
        raise ValueError(
            "production.kling_video_reward worker_config should name the reward "
            "model directly; "
            f"remove extra loader fields: {', '.join(forbidden)}",
        )
    task_type = str((root.data.task_type if root.data is not None else None) or "")
    if task_type not in _PRODUCTION_TASK_TYPES:
        raise ValueError(
            "production.kling_video_reward requires "
            "data.task_type=text_to_video, image_to_video, or video2world"
        )


def validate_production_data(root: RootConfig) -> None:
    """Manifests and source report exist; conditioned tasks get their rows checked."""

    data = root.data
    for name in ("manifest", "eval_manifest", "source_report"):
        value = str((getattr(data, name) if data is not None else None) or "").strip()
        if not value:
            raise ValueError(f"config missing required field: data.{name}")
        if not Path(value).exists():
            raise ValueError(f"data.{name} does not exist: {value}")
    assert data is not None
    task_type = str(data.task_type or "")
    if task_type == "video2world":
        _validate_video_world_production_data(root)
    if task_type == "image_to_video":
        _validate_image_to_video_production_data(root)


# ---- video2world ------------------------------------------------------------


def _validate_video_world_production_data(root: RootConfig) -> None:
    from vrl.trainers.data.artifacts import ArtifactManifestReport

    data = root.data
    assert data is not None
    data_root = str(data.artifact_data_root or "").strip()
    kwargs = {"data_root": data_root} if data_root else {}
    reward_components = root.reward.components if root.reward is not None else {}
    # The target-clip-reading reward is target_dino_similarity (successor to the deleted
    # pixel-L1 target_video_similarity); it consumes metadata['target_video'], so its
    # presence is what makes target clips a hard manifest requirement.
    require_target_video = "target_dino_similarity" in reward_components
    ArtifactManifestReport.from_video_world_manifest(
        str(data.manifest),
        eval_manifest=str(data.eval_manifest),
        require_target_video=require_target_video,
        **kwargs,
    )
    _validate_video_world_source_report(Path(str(data.source_report)))


def _validate_video_world_source_report(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "dataset",
        "source",
        "repo_id",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
        "validation_summary",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Video2World provenance fields: {missing}",
        )
    if int(payload.get("train_rows") or 0) <= 0 or int(payload.get("eval_rows") or 0) <= 0:
        raise ValueError("data.source_report must record non-empty train and eval rows")
    validation_summary = payload.get("validation_summary")
    if not isinstance(validation_summary, dict) or not validation_summary:
        raise ValueError("data.source_report must include a non-empty validation_summary")


# ---- image_to_video ---------------------------------------------------------


def _validate_image_to_video_production_data(root: RootConfig) -> None:
    data = root.data
    assert data is not None
    data_root = str(data.artifact_data_root or "").strip()
    if not data_root:
        raise ValueError("config missing required field: data.artifact_data_root")
    preprocessing = data.preprocessing
    image_field = str((preprocessing.image_field if preprocessing else None) or "image")
    caption_field = str((preprocessing.caption_field if preprocessing else None) or "caption")
    train_count = _validate_image_to_video_manifest(
        Path(str(data.manifest)),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    eval_count = _validate_image_to_video_manifest(
        Path(str(data.eval_manifest)),
        data_root=Path(data_root),
        image_field=image_field,
        caption_field=caption_field,
    )
    _validate_image_to_video_source_report(
        Path(str(data.source_report)),
        train_count=train_count,
        eval_count=eval_count,
    )


def _validate_image_to_video_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    image_field: str,
    caption_field: str,
) -> int:
    from vrl.utils.artifacts import resolve_artifact_path

    # Shares the {source_repo, source_frame_index, decode_method, conditioning}
    # provenance sub-vocabulary with
    # vrl/utils/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS —
    # keep in sync. This is a separate schema (Image2Video manifest rows use
    # source_video_url, not source_video), so the two are not unified.
    required_metadata = {
        "source_repo",
        "source_video_url",
        "source_frame_index",
        "decode_method",
        "conditioning",
    }
    row_count = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}: row {row_index} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}: row {row_index} must be an object")
            image = str(row.get(image_field, "")).strip()
            caption = str(row.get(caption_field, "")).strip()
            if not image:
                raise ValueError(f"{manifest_path}: row {row_index} missing {image_field}")
            if not caption:
                raise ValueError(f"{manifest_path}: row {row_index} missing {caption_field}")
            resolved_image = resolve_artifact_path(image, data_root=data_root)
            if not resolved_image.exists():
                raise ValueError(
                    f"{manifest_path}: row {row_index} image does not exist: {resolved_image}",
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{manifest_path}: row {row_index} metadata is required")
            missing = sorted(
                field
                for field in required_metadata
                if metadata.get(field) is None or str(metadata.get(field)).strip() == ""
            )
            if missing:
                raise ValueError(
                    f"{manifest_path}: row {row_index} missing source metadata: {missing}",
                )
    if row_count == 0:
        raise ValueError(f"{manifest_path} must contain at least one image-to-video row")
    return row_count


def _validate_image_to_video_source_report(
    path: Path,
    *,
    train_count: int,
    eval_count: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Report-level (dataset-wide) schema — distinct from the per-row provenance
    # vocabulary in _validate_image_to_video_manifest and
    # vrl/utils/artifacts.py SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS.
    # Do NOT fold this into those: these keys describe the whole source dump
    # (source_csv, train_rows, reference_dir), not a single manifest row.
    required_keys = {
        "dataset",
        "source_repo",
        "source_csv",
        "source_split",
        "decode_method",
        "train_rows",
        "eval_rows",
        "train_manifest",
        "eval_manifest",
        "reference_dir",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ValueError(
            f"data.source_report is missing Image2Video provenance fields: {missing}",
        )
    if payload.get("dataset") != "videophy_i2v":
        raise ValueError("data.source_report dataset must be videophy_i2v")
    if int(payload.get("train_rows") or 0) != train_count:
        raise ValueError("data.source_report train_rows does not match data.manifest")
    if int(payload.get("eval_rows") or 0) != eval_count:
        raise ValueError("data.source_report eval_rows does not match data.eval_manifest")


__all__ = [
    "validate_production_data",
    "validate_production_kling_video_reward_config",
    "validate_production_reward_contract",
]
