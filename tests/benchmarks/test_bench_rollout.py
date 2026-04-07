import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_rollout.py"
    spec = importlib.util.spec_from_file_location("bench_rollout_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_fork_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_state_fork.py"
    spec = importlib.util.spec_from_file_location("bench_state_fork_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_resets_execution_stats_after_warmup():
    bench = _load_benchmark_module()

    result = bench.run_benchmark(
        device="cpu",
        num_steps=2,
        batch_size=2,
        execution_mode="chunked",
        hidden_dim=64,
        num_layers=2,
        num_tokens=16,
        latent_dim=6,
        action_dim=8,
        warmup_runs=1,
        benchmark_runs=1,
        gpu_sample_interval_s=0.01,
    )

    assert result["execution_stats"]["transition_entities"] == 4
    assert result["execution_stats"]["transition_chunks"] == 2
    assert result["transfer_profile"]["input_h2d_bytes_total"] == 896
    assert result["residency_profile"]["final"]["logical_bytes"] == result["state_snapshot"]["logical_bytes"]
    assert result["residency_profile"]["final"]["reuse_hit_rate"] == result["state_snapshot"]["reuse_hit_rate"]
    assert result["state_snapshot"]["logical_bytes"] >= result["warmup_state_snapshot"]["logical_bytes"]


def test_state_fork_benchmark_exposes_memory_and_transfer_profiles():
    bench = _load_state_fork_module()

    args = SimpleNamespace(
        device="cpu",
        fork_mode="copy_on_write",
        history_steps=2,
        branches=3,
        tokens=8,
        latent_dim=4,
        action_dim=2,
        append_steps_per_branch=1,
        max_memory_gb=1.0,
        output="ignored.json",
    )

    result = bench.run_benchmark(args)

    assert result["summary"]["bytes_saved_via_sharing"] > 0
    assert result["profiling"]["residency"]["logical_bytes"] == result["state_stats"]["logical_bytes"]
    assert result["profiling"]["residency"]["physical_bytes"] == result["state_stats"]["physical_bytes"]
    assert result["profiling"]["transfer"]["total_history_bytes"] == 680
