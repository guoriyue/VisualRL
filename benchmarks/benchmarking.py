"""Benchmark/profiling helpers for temporal and video workloads.

These helpers are intentionally backend-agnostic. They standardize run summaries,
serialize benchmark artifacts, and enforce honest comparability rules across
systems such as wm-infra, vLLM, and sglang.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class ComparableAxis:
    name: str
    value: Any


@dataclass
class GpuSampler:
    """Best-effort GPU profiler based on `nvidia-smi` polling."""

    poll_interval_s: float = 0.1
    device_index: int = 0
    enabled: bool = True
    available: bool = True
    source: str = "nvidia-smi"
    error: str | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop_event: threading.Event | None = None
    _thread: threading.Thread | None = None
    _started_at_s: float | None = None
    _stopped_at_s: float | None = None

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self.available = self.enabled and shutil.which("nvidia-smi") is not None
        if not self.available:
            self.enabled = False

    @classmethod
    def disabled(cls) -> "GpuSampler":
        return cls(enabled=False, available=False)

    def start(self) -> "GpuSampler":
        if not self.enabled or not self.available or self._thread is not None:
            return self

        self._started_at_s = time.perf_counter()

        def _run() -> None:
            while self._stop_event is not None and not self._stop_event.is_set():
                sample = _sample_gpu_snapshot(self.device_index, self.source)
                if sample is not None:
                    self.samples.append(sample)
                if self._stop_event.wait(self.poll_interval_s):
                    break

        self._thread = threading.Thread(target=_run, name="gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._stopped_at_s = time.perf_counter()
        return self.summary()

    def __enter__(self) -> "GpuSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def summary(self) -> dict[str, Any]:
        return summarize_gpu_samples(
            self.samples,
            enabled=self.enabled,
            available=self.available,
            source=self.source,
            device_index=self.device_index,
            poll_interval_s=self.poll_interval_s,
            error=self.error,
            started_at_s=self._started_at_s,
            stopped_at_s=self._stopped_at_s,
        )


def _summarize_numeric_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    numbers = [float(value) for value in values]
    return {
        "count": float(len(numbers)),
        "mean": mean(numbers),
        "min": min(numbers),
        "p50": percentile(numbers, 50),
        "p95": percentile(numbers, 95),
        "p99": percentile(numbers, 99),
        "max": max(numbers),
    }


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "N/A", "NA", "Unknown"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _sample_gpu_snapshot(device_index: int, source: str = "nvidia-smi") -> dict[str, Any] | None:
    if source != "nvidia-smi":
        raise ValueError(f"Unsupported GPU sampler source: {source}")
    if shutil.which("nvidia-smi") is None:
        return None

    query = "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        cmd.append(f"--id={device_index}")

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=2.0)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if completed.returncode != 0:
        return None

    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 7:
        return None

    return {
        "captured_at_s": time.perf_counter(),
        "timestamp": parts[0],
        "gpu_index": parts[1],
        "gpu_name": parts[2],
        "utilization_gpu_pct": _coerce_optional_float(parts[3]),
        "utilization_memory_pct": _coerce_optional_float(parts[4]),
        "memory_used_mib": _coerce_optional_float(parts[5]),
        "memory_total_mib": _coerce_optional_float(parts[6]),
    }


def summarize_gpu_samples(
    samples: list[dict[str, Any]],
    *,
    enabled: bool = True,
    available: bool = True,
    source: str = "nvidia-smi",
    device_index: int | None = None,
    poll_interval_s: float | None = None,
    error: str | None = None,
    started_at_s: float | None = None,
    stopped_at_s: float | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": enabled,
        "available": available,
        "source": source,
        "device_index": device_index,
        "poll_interval_s": poll_interval_s,
        "sample_count": len(samples),
        "error": error,
    }
    if started_at_s is not None and stopped_at_s is not None:
        summary["duration_s"] = max(stopped_at_s - started_at_s, 0.0)

    if not samples:
        summary["series"] = {}
        return summary

    numeric_series = {
        "utilization_gpu_pct": [value for value in (_coerce_optional_float(item.get("utilization_gpu_pct")) for item in samples) if value is not None],
        "utilization_memory_pct": [value for value in (_coerce_optional_float(item.get("utilization_memory_pct")) for item in samples) if value is not None],
        "memory_used_mib": [value for value in (_coerce_optional_float(item.get("memory_used_mib")) for item in samples) if value is not None],
        "memory_total_mib": [value for value in (_coerce_optional_float(item.get("memory_total_mib")) for item in samples) if value is not None],
    }

    summary["series"] = {name: _summarize_numeric_series(values) for name, values in numeric_series.items() if values}
    summary["peak"] = {
        name: series.get("max")
        for name, series in summary["series"].items()
        if series
    }
    summary["first_sample"] = {key: samples[0].get(key) for key in ("timestamp", "gpu_index", "gpu_name")}
    summary["last_sample"] = {key: samples[-1].get(key) for key in ("timestamp", "gpu_index", "gpu_name")}
    return summary


def format_gpu_summary(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "GPU profiling: unavailable"
    if not summary.get("available"):
        return "GPU profiling: unavailable"

    series = summary.get("series", {})
    gpu = series.get("utilization_gpu_pct", {})
    mem = series.get("memory_used_mib", {})
    gpu_mean = gpu.get("mean")
    gpu_max = gpu.get("max")
    mem_max = mem.get("max")
    sample_count = summary.get("sample_count", 0)
    parts = [f"samples={sample_count}"]
    if gpu_mean is not None:
        parts.append(f"gpu_mean={gpu_mean:.1f}%")
    if gpu_max is not None:
        parts.append(f"gpu_peak={gpu_max:.1f}%")
    if mem_max is not None:
        parts.append(f"mem_peak={mem_max:.0f} MiB")
    if summary.get("duration_s") is not None:
        parts.append(f"window={summary['duration_s']:.3f}s")
    return "GPU profiling: " + " | ".join(parts)


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
    summary = _summarize_numeric_series(latencies_ms)
    return {("count" if name == "count" else f"{name}_ms"): value for name, value in summary.items()}


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
        "execution_device",
        "runtime_execution_mode",
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


def benchmark_gate_report(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_terminal_mean_ratio: float,
    max_terminal_p95_ratio: float,
    required_success_rate: float = 1.0,
) -> dict[str, Any]:
    """Validate one benchmark artifact against an explicit latency/success gate."""

    report = comparison_report(current, baseline)
    gate: dict[str, Any] = {
        "comparable": report["comparable"],
        "mismatches": report["mismatches"],
        "required_success_rate": required_success_rate,
        "max_terminal_mean_ratio": max_terminal_mean_ratio,
        "max_terminal_p95_ratio": max_terminal_p95_ratio,
        "pass": False,
        "reasons": [],
    }
    if not report["comparable"]:
        gate["reasons"].append("benchmark artifacts are not comparable")
        return gate

    metrics = report.get("metrics", {})
    terminal_mean = metrics.get("terminal_mean_ms")
    terminal_p95 = metrics.get("terminal_p95_ms")
    success_rate = metrics.get("success_rate")
    if terminal_mean is None or terminal_p95 is None or success_rate is None:
        gate["reasons"].append("benchmark artifact is missing terminal latency or success metrics")
        return gate

    terminal_mean_ratio = float(terminal_mean["current"]) / max(float(terminal_mean["baseline"]), 1e-12)
    terminal_p95_ratio = float(terminal_p95["current"]) / max(float(terminal_p95["baseline"]), 1e-12)
    current_success_rate = float(success_rate["current"])

    gate.update(
        {
            "success_rate": current_success_rate,
            "success_rate_pass": current_success_rate >= required_success_rate,
            "terminal_mean_ratio": terminal_mean_ratio,
            "terminal_mean_pass": terminal_mean_ratio <= max_terminal_mean_ratio,
            "terminal_p95_ratio": terminal_p95_ratio,
            "terminal_p95_pass": terminal_p95_ratio <= max_terminal_p95_ratio,
        }
    )
    if not gate["success_rate_pass"]:
        gate["reasons"].append(
            f"success_rate {current_success_rate:.4f} is below required {required_success_rate:.4f}"
        )
    if not gate["terminal_mean_pass"]:
        gate["reasons"].append(
            f"terminal_mean_ratio {terminal_mean_ratio:.4f} exceeds {max_terminal_mean_ratio:.4f}"
        )
    if not gate["terminal_p95_pass"]:
        gate["reasons"].append(
            f"terminal_p95_ratio {terminal_p95_ratio:.4f} exceeds {max_terminal_p95_ratio:.4f}"
        )
    gate["pass"] = (
        gate["success_rate_pass"]
        and gate["terminal_mean_pass"]
        and gate["terminal_p95_pass"]
    )
    return gate


def genie_cleanup_gate_report(
    *,
    default_baseline: dict[str, Any],
    default_batched: dict[str, Any],
    heavy_off: dict[str, Any],
    heavy_on: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the benchmark-backed cleanup gate for Genie ECS."""

    default_gate = benchmark_gate_report(
        default_batched,
        default_baseline,
        max_terminal_mean_ratio=1.05,
        max_terminal_p95_ratio=1.10,
        required_success_rate=1.0,
    )
    heavy_gate = benchmark_gate_report(
        heavy_on,
        heavy_off,
        max_terminal_mean_ratio=1.05,
        max_terminal_p95_ratio=1.10,
        required_success_rate=1.0,
    )
    return {
        "default": default_gate,
        "heavy": heavy_gate,
        "overall_pass": default_gate["pass"] and heavy_gate["pass"],
    }


def run_summary_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    succeeded = sum(1 for item in samples if item.get("status") == "succeeded")
    accepted = sum(1 for item in samples if item.get("status") == "accepted")
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
            "accepted": accepted,
            "failed": failed,
            "queued": queued,
            "running": running,
        },
        "latency": {
            "submit": summarize_latency_ms(submit_latencies),
            "terminal": summarize_latency_ms(terminal_latencies),
        },
        "success_rate": ((succeeded + accepted) / total) if total else 0.0,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
