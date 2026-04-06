import subprocess

from wm_infra import benchmarking as bench
from wm_infra.benchmarking import capture_runtime_context, comparable_run_pair, comparison_report, format_gpu_summary, percentile, run_summary_from_samples, summarize_gpu_samples, summarize_latency_ms


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([10, 20, 30, 40], 0) == 10.0
    assert percentile([10, 20, 30, 40], 100) == 40.0


def test_summarize_latency_ms():
    summary = summarize_latency_ms([10, 20, 30])
    assert summary["count"] == 3.0
    assert summary["mean_ms"] == 20
    assert summary["p50_ms"] == 20
    assert summary["max_ms"] == 30


def test_run_summary_from_samples():
    summary = run_summary_from_samples([
        {"status": "succeeded", "metrics": {"submit_latency_ms": 10, "terminal_latency_ms": 100}},
        {"status": "accepted", "metrics": {"submit_latency_ms": 15, "terminal_latency_ms": 150}},
        {"status": "failed", "metrics": {"submit_latency_ms": 20, "terminal_latency_ms": 200}},
        {"status": "queued", "metrics": {"submit_latency_ms": 30}},
    ])
    assert summary["counts"]["total"] == 4
    assert summary["counts"]["succeeded"] == 1
    assert summary["counts"]["accepted"] == 1
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["queued"] == 1
    assert summary["latency"]["submit"]["p95_ms"] > 0
    assert summary["latency"]["terminal"]["count"] == 3.0
    assert summary["success_rate"] == 2 / 4


def test_comparable_run_pair_rejects_mismatched_workloads():
    left = {
        "workload": {
            "workload_kind": "sample_api",
            "task_type": "text_to_video",
            "backend_family": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "frame_count": 9,
            "width": 832,
            "height": 480,
            "num_steps": 4,
            "runtime_execution_mode": "chunked",
        }
    }
    right = {
        "workload": {
            "workload_kind": "sample_api",
            "task_type": "text_to_video",
            "backend_family": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "frame_count": 17,
            "width": 832,
            "height": 480,
            "num_steps": 4,
            "runtime_execution_mode": "legacy",
        }
    }
    ok, mismatches = comparable_run_pair(left, right)
    assert ok is False
    assert any("runtime_execution_mode" in item for item in mismatches)


def test_capture_runtime_context_includes_reproducibility_metadata():
    context = capture_runtime_context()
    assert "captured_at" in context
    assert context["python"]["version"]
    assert context["python"]["implementation"]
    assert context["platform"]["system"]
    assert set(context["git"]) >= {"commit", "branch", "remote", "dirty"}


def test_comparison_report_embeds_metric_deltas():
    baseline = {
        "workload": {
            "workload_kind": "sample_api",
            "task_type": "text_to_video",
            "backend_family": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "frame_count": 9,
            "width": 832,
            "height": 480,
            "num_steps": 4,
        },
        "summary": {
            "latency": {
                "submit": {"mean_ms": 10.0, "p95_ms": 12.0},
                "terminal": {"mean_ms": 100.0, "p95_ms": 120.0},
            },
            "success_rate": 0.5,
        },
    }
    current = {
        "workload": {
            "workload_kind": "sample_api",
            "task_type": "text_to_video",
            "backend_family": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "frame_count": 9,
            "width": 832,
            "height": 480,
            "num_steps": 4,
        },
        "summary": {
            "latency": {
                "submit": {"mean_ms": 14.0, "p95_ms": 16.0},
                "terminal": {"mean_ms": 90.0, "p95_ms": 110.0},
            },
            "success_rate": 1.0,
        },
    }

    report = comparison_report(current, baseline)
    assert report["comparable"] is True
    assert report["metrics"]["submit_mean_ms"]["delta"] == 4.0
    assert report["metrics"]["terminal_mean_ms"]["delta"] == -10.0
    assert report["metrics"]["success_rate"]["delta"] == 0.5


def test_sample_gpu_snapshot_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="2026/04/05 10:00:00.000, 0, NVIDIA GeForce RTX 5090, 73, 28, 1200, 32607\n",
            stderr="",
        ),
    )

    sample = bench._sample_gpu_snapshot(0)
    assert sample is not None
    assert sample["gpu_index"] == "0"
    assert sample["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert sample["utilization_gpu_pct"] == 73.0
    assert sample["memory_used_mib"] == 1200.0


def test_summarize_gpu_samples_and_format():
    summary = summarize_gpu_samples(
        [
            {
                "timestamp": "2026/04/05 10:00:00.000",
                "gpu_index": "0",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "utilization_gpu_pct": 50.0,
                "utilization_memory_pct": 20.0,
                "memory_used_mib": 1000.0,
                "memory_total_mib": 32607.0,
            },
            {
                "timestamp": "2026/04/05 10:00:00.100",
                "gpu_index": "0",
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "utilization_gpu_pct": 80.0,
                "utilization_memory_pct": 30.0,
                "memory_used_mib": 1400.0,
                "memory_total_mib": 32607.0,
            },
        ],
        started_at_s=1.0,
        stopped_at_s=2.0,
        poll_interval_s=0.1,
    )

    assert summary["sample_count"] == 2
    assert summary["duration_s"] == 1.0
    assert summary["series"]["utilization_gpu_pct"]["mean"] == 65.0
    assert summary["series"]["memory_used_mib"]["max"] == 1400.0
    assert "gpu_mean=65.0%" in format_gpu_summary(summary)
