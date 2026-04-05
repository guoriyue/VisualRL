"""Benchmark/profiling helpers for temporal and video workloads.

These helpers are intentionally backend-agnostic. They standardize run summaries,
serialize benchmark artifacts, and enforce honest comparability rules across
systems such as wm-infra, vLLM, and sglang.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ComparableAxis:
    name: str
    value: Any


_COMPARISON_METRICS = [
    ("submit_mean_ms", ("summary", "latency", "submit", "mean_ms")),
    ("submit_p95_ms", ("summary", "latency", "submit", "p95_ms")),
    ("terminal_mean_ms", ("summary", "latency", "terminal", "mean_ms")),
    ("terminal_p95_ms", ("summary", "latency", "terminal", "p95_ms")),
    ("success_rate", ("summary", "success_rate")),
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    if pct < 0 or pct > 100:
        raise ValueError("pct must be in [0, 100]")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_latency_ms(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    values = [float(v) for v in latencies_ms]
    return {
        "count": float(len(values)),
        "mean_ms": mean(values),
        "min_ms": min(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def canonical_workload_key(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must match for honest comparisons.

    We compare only like-for-like workloads. System-specific knobs are excluded.
    """
    keys = [
        "workload_kind",
        "task_type",
        "backend_family",
        "model",
        "prompt_shape",
        "num_prompts",
        "frame_count",
        "width",
        "height",
        "num_steps",
        "fps",
        "input_modality",
        "tokenizer_kind",
        "prompt_frames",
    ]
    return {key: workload.get(key) for key in keys if workload.get(key) is not None}


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def capture_runtime_context(cwd: str | Path | None = None) -> dict[str, Any]:
    root = Path(cwd) if cwd is not None else Path.cwd()
    git_commit = _run_git(["rev-parse", "HEAD"], root)
    git_branch = _run_git(["branch", "--show-current"], root)
    git_remote = _run_git(["remote", "get-url", "origin"], root)
    git_dirty = _run_git(["status", "--short"], root)
    return {
        "captured_at": utc_timestamp(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "remote": git_remote,
            "dirty": bool(git_dirty),
        },
    }


def comparable_run_pair(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, list[str]]:
    left_key = canonical_workload_key(left.get("workload", {}))
    right_key = canonical_workload_key(right.get("workload", {}))
    mismatches: list[str] = []
    for key in sorted(set(left_key) | set(right_key)):
        if left_key.get(key) != right_key.get(key):
            mismatches.append(f"{key}: {left_key.get(key)!r} != {right_key.get(key)!r}")
    return len(mismatches) == 0, mismatches


def comparison_report(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    ok, mismatches = comparable_run_pair(baseline, current)
    report: dict[str, Any] = {
        "comparable": ok,
        "mismatches": mismatches,
        "baseline_workload": canonical_workload_key(baseline.get("workload", {})),
        "current_workload": canonical_workload_key(current.get("workload", {})),
        "metrics": {},
    }
    if not ok:
        return report

    metrics: dict[str, Any] = {}
    for metric_name, path in _COMPARISON_METRICS:
        base_value = _dig(baseline, path)
        current_value = _dig(current, path)
        if base_value is None or current_value is None:
            continue
        base_f = float(base_value)
        current_f = float(current_value)
        delta = current_f - base_f
        metrics[metric_name] = {
            "baseline": base_f,
            "current": current_f,
            "delta": delta,
            "delta_pct": (delta / base_f * 100.0) if base_f else None,
        }
    report["metrics"] = metrics
    return report


def run_summary_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    succeeded = sum(1 for item in samples if item.get("status") == "succeeded")
    failed = sum(1 for item in samples if item.get("status") == "failed")
    queued = sum(1 for item in samples if item.get("status") == "queued")
    running = sum(1 for item in samples if item.get("status") == "running")
    terminal_latencies = [
        float(item["metrics"]["terminal_latency_ms"])
        for item in samples
        if "metrics" in item and "terminal_latency_ms" in item["metrics"]
    ]
    submit_latencies = [
        float(item["metrics"]["submit_latency_ms"])
        for item in samples
        if "metrics" in item and "submit_latency_ms" in item["metrics"]
    ]
    return {
        "counts": {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "queued": queued,
            "running": running,
        },
        "latency": {
            "submit": summarize_latency_ms(submit_latencies),
            "terminal": summarize_latency_ms(terminal_latencies),
        },
        "success_rate": (succeeded / total) if total else 0.0,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
