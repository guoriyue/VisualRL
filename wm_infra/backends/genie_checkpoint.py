"""Checkpoint delta helpers for stage-oriented Genie runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class GenieCheckpointDelta:
    """Replayable checkpoint delta for a bounded generated frame window."""

    rollout_id: str
    sample_id: str
    start_frame: int
    end_frame: int
    parent_state_handle_id: str | None
    bytes_size: int
    frames: int
    artifact_id: str | None = None
    path: str | None = None
    tokens_delta: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def should_checkpoint(
    *,
    completed_frame: int,
    total_frames: int,
    checkpoint_every_n_frames: int,
) -> bool:
    """Return True when the current frame boundary should emit checkpoint work."""

    if completed_frame >= total_frames:
        return True
    if checkpoint_every_n_frames <= 0:
        return False
    return completed_frame % checkpoint_every_n_frames == 0


def checkpoint_due(frame_end: int, total_frames: int, checkpoint_every_n_frames: int) -> bool:
    if frame_end >= total_frames:
        return False
    return should_checkpoint(
        completed_frame=frame_end,
        total_frames=total_frames,
        checkpoint_every_n_frames=checkpoint_every_n_frames,
    )


def build_checkpoint_delta(
    *,
    rollout_id: str,
    sample_id: str,
    parent_state_handle_id: str | None,
    all_tokens: np.ndarray,
    start_frame: int,
    end_frame: int,
    checkpoint_every_n_frames: int,
    runner_mode: str,
) -> GenieCheckpointDelta:
    """Build a checkpoint delta from a generated token frame range."""

    safe_start = max(start_frame, 0)
    safe_end = min(end_frame, int(all_tokens.shape[0]))
    total_frames = int(all_tokens.shape[0])
    delta = np.asarray(all_tokens[safe_start:safe_end], dtype=np.uint32).copy()
    return GenieCheckpointDelta(
        rollout_id=rollout_id,
        sample_id=sample_id,
        start_frame=safe_start,
        end_frame=safe_end,
        parent_state_handle_id=parent_state_handle_id,
        bytes_size=int(delta.nbytes),
        frames=max(safe_end - safe_start, 0),
        artifact_id=f"{sample_id}:checkpoint-delta:{safe_end:04d}",
        tokens_delta=delta,
        metadata={
            "checkpoint_every_n_frames": checkpoint_every_n_frames,
            "runner_mode": runner_mode,
            "frame_range": [safe_start, safe_end],
            "dtype": "uint32",
            "frame_start": safe_start,
            "frame_end": safe_end,
            "frame_count": max(safe_end - safe_start, 0),
            "total_frames": total_frames,
            "bytes": int(delta.nbytes),
        },
    )


def persist_checkpoint_delta(sample_dir: Path, delta: GenieCheckpointDelta) -> str:
    delta_dir = sample_dir / "checkpoint_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)
    path = delta_dir / f"{delta.end_frame:04d}.json"
    payload = {**delta.metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    delta.path = str(path)
    return delta.path
