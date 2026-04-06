"""Persistence helpers for stage-oriented Genie runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GeniePersistPaths:
    """Filesystem paths used by a Genie sample execution."""

    sample_dir: Path
    request_path: Path
    log_path: Path
    runtime_path: Path
    checkpoint_path: Path
    recovery_path: Path


def build_persist_paths(sample_dir: Path) -> GeniePersistPaths:
    """Build the canonical per-sample file paths."""

    return GeniePersistPaths(
        sample_dir=sample_dir,
        request_path=sample_dir / "request.json",
        log_path=sample_dir / "runner.log",
        runtime_path=sample_dir / "runtime.json",
        checkpoint_path=sample_dir / "checkpoint.json",
        recovery_path=sample_dir / "recovery.json",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_log(path: Path, lines: list[str]) -> None:
    """Persist a plain-text log."""

    path.write_text("\n".join(lines) + "\n")


def build_runtime_status_history(*, started_at: float, completed_at: float, terminal_status: str) -> list[dict[str, Any]]:
    """Build the canonical queued -> running -> terminal status history."""

    return [
        {"status": "queued", "timestamp": started_at},
        {"status": "running", "timestamp": started_at},
        {"status": terminal_status, "timestamp": completed_at},
    ]
