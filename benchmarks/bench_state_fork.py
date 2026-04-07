#!/usr/bin/env python3
"""Benchmark branch/fork state reuse for temporal workloads.

This benchmark compares deep-copy fork behavior against copy-on-write sharing
inside ``LatentStateManager``. It is meant to approximate branch-heavy
world-model workflows such as RL environment branching and rollout fan-out.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.benchmarking import capture_runtime_context, utc_timestamp, write_json
from wm_infra.core.state import LatentStateManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark state fork/reuse behavior")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fork-mode", choices=["deep_copy", "copy_on_write"], default="copy_on_write")
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--branches", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--append-steps-per-branch", type=int, default=1)
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--output", default="benchmarks/results/state_fork_benchmark.json")
    return parser.parse_args()


def _device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda, but torch.cuda.is_available() is false")
    return device


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel * element_size)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    manager = LatentStateManager(
        max_concurrent=max(args.branches + 2, 4),
        max_memory_gb=args.max_memory_gb,
        device=device,
        fork_mode=args.fork_mode,
    )

    initial_state = torch.randn(args.tokens, args.latent_dim, device=device, dtype=dtype)
    action = torch.randn(args.action_dim, device=device, dtype=dtype)
    predicted_state = torch.randn(args.tokens, args.latent_dim, device=device, dtype=dtype)

    source = manager.create("source", initial_state, max_steps=args.history_steps + args.append_steps_per_branch)
    for step_idx in range(args.history_steps):
        manager.append_step(
            "source",
            action + step_idx * 0.01,
            predicted_state + step_idx * 0.01,
        )

    fork_start = time.perf_counter()
    branch_ids = []
    for branch_idx in range(args.branches):
        branch_id = f"branch_{branch_idx}"
        manager.fork("source", branch_id, max_steps=source.max_steps)
        branch_ids.append(branch_id)
    fork_elapsed_ms = round((time.perf_counter() - fork_start) * 1000.0, 3)

    append_start = time.perf_counter()
    for step_idx in range(args.append_steps_per_branch):
        for branch_id in branch_ids:
            manager.append_step(
                branch_id,
                action + step_idx * 0.02,
                predicted_state + step_idx * 0.02,
            )
    append_elapsed_ms = round((time.perf_counter() - append_start) * 1000.0, 3)

    stats = manager.stats_snapshot()
    logical_bytes = int(stats["logical_bytes"])
    physical_bytes = int(stats["physical_bytes"])
    logical_to_physical_ratio = (logical_bytes / physical_bytes) if physical_bytes else 0.0
    initial_state_bytes = _tensor_nbytes((args.tokens, args.latent_dim), dtype)
    action_bytes = _tensor_nbytes((args.action_dim,), dtype)
    predicted_state_bytes = _tensor_nbytes((args.tokens, args.latent_dim), dtype)
    source_history_bytes = args.history_steps * (action_bytes + predicted_state_bytes)
    branch_history_bytes = args.branches * args.append_steps_per_branch * (action_bytes + predicted_state_bytes)

    result = {
        "schema_version": 1,
        "recorded_at": utc_timestamp(),
        "run_context": capture_runtime_context(),
        "system": {"name": "wm-infra", "runner": "state-fork-benchmark"},
        "workload": {
            "workload_kind": "state_fork",
            "fork_mode": args.fork_mode,
            "device": device,
            "history_steps": args.history_steps,
            "branches": args.branches,
            "tokens": args.tokens,
            "latent_dim": args.latent_dim,
            "action_dim": args.action_dim,
            "append_steps_per_branch": args.append_steps_per_branch,
        },
        "summary": {
            "fork_elapsed_ms": fork_elapsed_ms,
            "append_elapsed_ms": append_elapsed_ms,
            "logical_to_physical_ratio": logical_to_physical_ratio,
            "bytes_saved_via_sharing": int(stats["bytes_saved_via_sharing"]),
            "reuse_hit_rate": float(stats["reuse_hit_rate"]),
            "num_active": int(stats["num_active"]),
        },
        "profiling": {
            "transfer": {
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "initial_state_bytes": initial_state_bytes,
                "action_bytes_per_step": action_bytes,
                "predicted_state_bytes_per_step": predicted_state_bytes,
                "source_history_bytes": source_history_bytes,
                "branch_history_bytes": branch_history_bytes,
                "total_history_bytes": source_history_bytes + branch_history_bytes,
            },
            "residency": {
                "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes,
                "logical_to_physical_ratio": logical_to_physical_ratio,
                "bytes_saved_via_sharing": int(stats["bytes_saved_via_sharing"]),
                "reuse_hit_rate": float(stats["reuse_hit_rate"]),
                "num_active": int(stats["num_active"]),
            },
            "state_stats": stats,
        },
        "state_stats": stats,
    }
    return result


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    write_json(args.output, result)
    print(f"Wrote benchmark artifact to {args.output}")
    print(result["summary"])


if __name__ == "__main__":
    main()
