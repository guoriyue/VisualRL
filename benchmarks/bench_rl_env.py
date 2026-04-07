#!/usr/bin/env python3
"""Benchmark wm-infra RL env/runtime throughput and latency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.benchmarking import capture_runtime_context, load_json, utc_timestamp, write_json
from wm_infra.workloads.rl.training import ExperimentSpec, run_local_experiment


_BACKEND_BY_ENV = {
    "toy-line-v0": "toy-line-world-model",
    "genie-token-grid-v0": "genie-rollout",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary_from_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", [])
    env_steps_per_sec = [float(item.get("env_steps_per_sec", 0.0)) for item in metrics]
    step_latency_ms = [float(item.get("step_latency_ms", 0.0)) for item in metrics]
    reward_stage_latency_ms = [float(item.get("reward_stage_latency_ms", 0.0)) for item in metrics]
    trajectory_persist_latency_ms = [float(item.get("trajectory_persist_latency_ms", 0.0)) for item in metrics]
    chunk_count = [float(item.get("chunk_count", 0.0)) for item in metrics]
    max_chunk_size = [float(item.get("max_chunk_size", 0.0)) for item in metrics]
    avg_chunk_size = [float(item.get("avg_chunk_size", 0.0)) for item in metrics]
    state_locality = [float(item.get("state_locality_hit_rate", 0.0)) for item in metrics]
    auto_reset_count = [float(item.get("auto_reset_count", 0.0)) for item in metrics]

    return {
        "env_steps_per_sec": {
            "mean": _mean(env_steps_per_sec),
            "min": min(env_steps_per_sec) if env_steps_per_sec else 0.0,
            "max": max(env_steps_per_sec) if env_steps_per_sec else 0.0,
        },
        "step_latency_ms": {
            "mean": _mean(step_latency_ms),
            "min": min(step_latency_ms) if step_latency_ms else 0.0,
            "max": max(step_latency_ms) if step_latency_ms else 0.0,
        },
        "reward_stage_latency_ms": {
            "mean": _mean(reward_stage_latency_ms),
            "min": min(reward_stage_latency_ms) if reward_stage_latency_ms else 0.0,
            "max": max(reward_stage_latency_ms) if reward_stage_latency_ms else 0.0,
        },
        "trajectory_persist_latency_ms": {
            "mean": _mean(trajectory_persist_latency_ms),
            "min": min(trajectory_persist_latency_ms) if trajectory_persist_latency_ms else 0.0,
            "max": max(trajectory_persist_latency_ms) if trajectory_persist_latency_ms else 0.0,
        },
        "chunk_count": {
            "mean": _mean(chunk_count),
            "max": max(chunk_count) if chunk_count else 0.0,
        },
        "avg_chunk_size": {
            "mean": _mean(avg_chunk_size),
            "max": max(avg_chunk_size) if avg_chunk_size else 0.0,
        },
        "max_chunk_size": {
            "mean": _mean(max_chunk_size),
            "max": max(max_chunk_size) if max_chunk_size else 0.0,
        },
        "state_locality_hit_rate": {
            "mean": _mean(state_locality),
            "min": min(state_locality) if state_locality else 0.0,
        },
        "auto_reset_count": {
            "mean": _mean(auto_reset_count),
            "max": max(auto_reset_count) if auto_reset_count else 0.0,
        },
        "final_success_rate": float(result.get("final_success_rate", 0.0)),
        "best_mean_return": float(result.get("best_mean_return", 0.0)),
        "final_mean_return": float(result.get("final_mean_return", 0.0)),
        "last_evaluation": result.get("last_evaluation"),
        "replay_shard": result.get("replay_shard"),
    }


def run_benchmark(
    *,
    updates: int = 40,
    num_envs: int = 32,
    horizon: int = 8,
    eval_num_envs: int = 8,
    eval_episodes: int = 8,
    eval_interval: int = 10,
    replay_dir: str = "benchmarks/results/rl_toy_replay",
    temporal_root: str = "/tmp/wm_infra_rl_benchmark",
    output: str = "benchmarks/results/rl_toy_env_benchmark.json",
    baseline_file: str | None = None,
    experiment_name: str = "toy-line-actor-critic",
    env_name: str = "toy-line-v0",
    train_task_id: str = "toy-line-train",
    eval_task_id: str = "toy-line-eval",
) -> dict[str, Any]:
    spec = ExperimentSpec(
        experiment_name=experiment_name,
        updates=updates,
        num_envs=num_envs,
        horizon=horizon,
        eval_num_envs=eval_num_envs,
        eval_episodes=eval_episodes,
        eval_interval=eval_interval,
        train_env_name=env_name,
        train_task_id=train_task_id,
        eval_task_id=eval_task_id,
        replay_dir=replay_dir,
        temporal_root=temporal_root,
    )
    result = run_local_experiment(spec)
    artifact = {
        "schema_version": 1,
        "recorded_at": utc_timestamp(),
        "run_context": capture_runtime_context(),
        "system": {
            "name": "wm-infra",
            "runner": "rl-env-benchmark",
        },
        "workload": {
            "workload_kind": "rl_env",
            "backend_family": _BACKEND_BY_ENV.get(spec.train_env_name, "unknown"),
            "backend_mode": result.get("backend_runtime", {}).get("runner_mode", "unknown"),
            "env_name": spec.train_env_name,
            "task_id": spec.train_task_id,
            "eval_task_id": spec.eval_task_id,
            "num_envs": spec.num_envs,
            "horizon": spec.horizon,
            "updates": spec.updates,
            "max_episode_steps": spec.max_episode_steps,
            "runtime_execution_mode": "chunked_env_step",
        },
        "summary": _summary_from_result(result),
        "result": result,
    }
    if baseline_file:
        artifact["baseline"] = load_json(baseline_file)

    write_json(output, artifact)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wm-infra RL env/runtime")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--replay-dir", default="benchmarks/results/rl_toy_replay")
    parser.add_argument("--temporal-root", default="/tmp/wm_infra_rl_benchmark")
    parser.add_argument("--output", default="benchmarks/results/rl_toy_env_benchmark.json")
    parser.add_argument("--baseline-file")
    parser.add_argument("--experiment-name", default="toy-line-actor-critic")
    parser.add_argument("--env-name", default="toy-line-v0")
    parser.add_argument("--train-task-id", default="toy-line-train")
    parser.add_argument("--eval-task-id", default="toy-line-eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = run_benchmark(
        updates=args.updates,
        num_envs=args.num_envs,
        horizon=args.horizon,
        eval_num_envs=args.eval_num_envs,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        replay_dir=args.replay_dir,
        temporal_root=args.temporal_root,
        output=args.output,
        baseline_file=args.baseline_file,
        experiment_name=args.experiment_name,
        env_name=args.env_name,
        train_task_id=args.train_task_id,
        eval_task_id=args.eval_task_id,
    )
    summary = artifact["summary"]
    print(
        "\n".join(
            [
                f"env_steps_per_sec={summary['env_steps_per_sec']['mean']:.3f}",
                f"step_latency_ms={summary['step_latency_ms']['mean']:.3f}",
                f"reward_stage_latency_ms={summary['reward_stage_latency_ms']['mean']:.6f}",
                f"trajectory_persist_latency_ms={summary['trajectory_persist_latency_ms']['mean']:.6f}",
                f"chunk_count={summary['chunk_count']['mean']:.3f}",
                f"max_chunk_size={summary['max_chunk_size']['max']:.0f}",
                f"state_locality_hit_rate={summary['state_locality_hit_rate']['mean']:.3f}",
                f"auto_reset_count={summary['auto_reset_count']['mean']:.3f}",
                f"final_success_rate={summary['final_success_rate']:.3f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
