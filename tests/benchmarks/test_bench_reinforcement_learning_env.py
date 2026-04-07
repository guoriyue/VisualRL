import importlib.util
from pathlib import Path


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_rl_env.py"
    spec = importlib.util.spec_from_file_location("bench_rl_env_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_emits_required_reinforcement_learning_metrics(tmp_path):
    bench = _load_benchmark_module()

    result = bench.run_benchmark(
        updates=8,
        num_envs=8,
        horizon=4,
        eval_num_envs=4,
        eval_episodes=4,
        eval_interval=4,
        replay_dir=str(tmp_path / "replay"),
        temporal_root=str(tmp_path / "temporal"),
        output=str(tmp_path / "rl_benchmark.json"),
    )

    summary = result["summary"]
    assert summary["env_steps_per_sec"]["mean"] > 0.0
    assert summary["step_latency_ms"]["mean"] > 0.0
    assert summary["reward_stage_latency_ms"]["mean"] >= 0.0
    assert summary["trajectory_persist_latency_ms"]["mean"] >= 0.0
    assert summary["chunk_count"]["mean"] >= 1.0
    assert summary["max_chunk_size"]["max"] >= 1.0
    assert summary["state_locality_hit_rate"]["mean"] >= 0.0
    assert summary["auto_reset_count"]["mean"] >= 0.0
    assert Path(tmp_path / "rl_benchmark.json").exists()


def test_run_benchmark_supports_genie_env_smoke(tmp_path):
    bench = _load_benchmark_module()

    result = bench.run_benchmark(
        updates=2,
        num_envs=4,
        horizon=2,
        eval_num_envs=2,
        eval_episodes=2,
        eval_interval=1,
        replay_dir=str(tmp_path / "replay_genie"),
        temporal_root=str(tmp_path / "temporal_genie"),
        output=str(tmp_path / "rl_genie_benchmark.json"),
        experiment_name="genie-token-bench",
        env_name="genie-token-grid-v0",
        train_task_id="genie-token-train",
        eval_task_id="genie-token-eval",
    )

    assert result["workload"]["backend_family"] == "genie-rollout"
    assert result["workload"]["backend_mode"] in {"stub", "real"}
    assert result["workload"]["env_name"] == "genie-token-grid-v0"
    assert result["summary"]["env_steps_per_sec"]["mean"] > 0.0
    assert result["summary"]["auto_reset_count"]["mean"] >= 0.0
    assert Path(tmp_path / "rl_genie_benchmark.json").exists()
