"""Benchmark world model rollout throughput and latency.

Usage:
    python benchmarks/bench_rollout.py [--device cuda] [--steps 16] [--batch 4]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

from benchmarks.benchmarking import GpuSampler, format_gpu_summary
from wm_infra.config import EngineConfig, DynamicsConfig
from wm_infra.core.engine import WorldModelEngine, RolloutJob
from wm_infra.models.dynamics import LatentDynamicsModel


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel * element_size)


def _residency_profile(snapshot: dict[str, Any]) -> dict[str, Any]:
    logical_bytes = int(snapshot.get("logical_bytes") or 0)
    physical_bytes = int(snapshot.get("physical_bytes") or 0)
    return {
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "bytes_saved_via_sharing": int(snapshot.get("bytes_saved_via_sharing") or 0),
        "reuse_hit_rate": float(snapshot.get("reuse_hit_rate") or 0.0),
        "num_active": int(snapshot.get("num_active") or 0),
        "tensor_materializations": int(snapshot.get("tensor_materializations") or 0),
        "tensor_reuse_hits": int(snapshot.get("tensor_reuse_hits") or 0),
        "fork_count": int(snapshot.get("fork_count") or 0),
        "memory_used_bytes": int(snapshot.get("memory_used_bytes") or physical_bytes),
        "logical_to_physical_ratio": (logical_bytes / physical_bytes) if physical_bytes else 0.0,
    }


def run_benchmark(
    device: str = "cpu",
    num_steps: int = 16,
    batch_size: int = 4,
    execution_mode: str = "chunked",
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_tokens: int = 64,
    latent_dim: int = 16,
    action_dim: int = 32,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    gpu_sample_interval_s: float = 0.1,
):
    config = EngineConfig(
        device=device,
        dtype="float16" if device == "cuda" else "float32",
        dynamics=DynamicsConfig(
            hidden_dim=hidden_dim,
            num_heads=max(4, hidden_dim // 64),
            num_layers=num_layers,
            action_dim=action_dim,
            latent_token_dim=latent_dim,
            max_rollout_steps=num_steps * 2,
        ),
    )

    dynamics = LatentDynamicsModel(config.dynamics)
    engine = WorldModelEngine(config, dynamics, tokenizer=None, execution_mode=execution_mode)

    total_params = sum(p.numel() for p in dynamics.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"Device: {device}, dtype: {config.dtype}")
    print(f"Config: hidden={hidden_dim}, layers={num_layers}, tokens={num_tokens}")
    print(f"Benchmark: {batch_size} jobs x {num_steps} steps")
    print(f"Execution mode: {execution_mode}")
    print("-" * 60)

    # Warmup
    for warmup_idx in range(warmup_runs):
        for i in range(batch_size):
            engine.submit_job(RolloutJob(
                job_id=f"warmup_{warmup_idx}_{i}",
                initial_latent=torch.randn(num_tokens, latent_dim),
                actions=torch.randn(num_steps, action_dim),
                num_steps=num_steps,
                return_frames=False,
                return_latents=True,
            ))
        engine.run_until_done()
    warmup_state_snapshot = engine.state_manager.stats_snapshot()
    engine.reset_execution_stats()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with GpuSampler(poll_interval_s=gpu_sample_interval_s) as gpu_sampler:
        for run in range(benchmark_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            for i in range(batch_size):
                engine.submit_job(RolloutJob(
                    job_id=f"bench_{run}_{i}",
                    initial_latent=torch.randn(num_tokens, latent_dim),
                    actions=torch.randn(num_steps, action_dim),
                    num_steps=num_steps,
                    return_frames=False,
                    return_latents=True,
                ))
            engine.run_until_done()

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            total_steps = batch_size * num_steps
            steps_per_sec = total_steps / elapsed
            avg_step_ms = elapsed * 1000 / total_steps

            print(f"Run {run + 1}: {elapsed:.3f}s | {steps_per_sec:.1f} steps/s | {avg_step_ms:.2f} ms/step")

    print("-" * 60)
    avg_latency = sum(latencies) / len(latencies)
    total_steps = batch_size * num_steps
    print(f"Average: {avg_latency:.3f}s | {total_steps / avg_latency:.1f} steps/s | {avg_latency * 1000 / total_steps:.2f} ms/step")
    execution_stats = engine.execution_stats_snapshot()
    print(f"Execution stats: {execution_stats}")

    final_state_snapshot = engine.state_manager.stats_snapshot()
    latent_bytes = _tensor_nbytes((num_tokens, latent_dim), engine.dtype)
    action_bytes = _tensor_nbytes((action_dim,), engine.dtype)
    transfer_profile = {
        "device": device,
        "dtype": str(engine.dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "num_steps": num_steps,
        "latent_bytes_per_job": latent_bytes,
        "action_bytes_per_step": action_bytes,
        "input_h2d_bytes_per_job": latent_bytes + (num_steps * action_bytes),
        "input_h2d_bytes_total": batch_size * (latent_bytes + (num_steps * action_bytes)),
        "expected_state_bytes_per_job": latent_bytes + (num_steps * (latent_bytes + action_bytes)),
        "expected_state_bytes_total": batch_size * (latent_bytes + (num_steps * (latent_bytes + action_bytes))),
    }
    residency_profile = {
        "warmup": _residency_profile(warmup_state_snapshot),
        "final": _residency_profile(final_state_snapshot),
    }

    gpu_summary = gpu_sampler.summary()
    print(format_gpu_summary(gpu_summary))

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved_mem = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"Peak GPU memory: {peak_mem:.2f} GB")
        print(f"Peak GPU reserved: {peak_reserved_mem:.2f} GB")
    else:
        peak_mem = None
        peak_reserved_mem = None

    return {
        "avg_latency_s": avg_latency,
        "avg_steps_per_sec": total_steps / avg_latency,
        "avg_step_ms": avg_latency * 1000 / total_steps,
        "execution_stats": execution_stats,
        "state_snapshot": final_state_snapshot,
        "warmup_state_snapshot": warmup_state_snapshot,
        "transfer_profile": transfer_profile,
        "residency_profile": residency_profile,
        "gpu_summary": gpu_summary,
        "peak_mem_gb": peak_mem,
        "peak_reserved_mem_gb": peak_reserved_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark world model rollout")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--execution-mode", choices=["chunked"], default="chunked")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--gpu-sample-interval-ms", type=float, default=100.0)
    args = parser.parse_args()

    run_benchmark(
        device=args.device,
        num_steps=args.steps,
        batch_size=args.batch,
        execution_mode=args.execution_mode,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_tokens=args.tokens,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        gpu_sample_interval_s=args.gpu_sample_interval_ms / 1000.0,
    )


if __name__ == "__main__":
    main()
