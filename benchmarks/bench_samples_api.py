#!/usr/bin/env python3
"""Benchmark wm-infra sample production APIs for temporal/video workloads.

This harness measures:
- submit latency for POST /v1/samples
- time-to-terminal-state for async backends (for example wan-video)
- per-sample outcome and queue polling behavior

It can run against a live server (`--base-url`) or an in-process app (`--in-process`).
Results are written as a structured JSON artifact for later comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import torch
from asgi_lifespan import LifespanManager

from wm_infra.api.server import create_app
from wm_infra.benchmarking import GpuSampler, capture_runtime_context, comparison_report, format_gpu_summary, load_json, run_summary_from_samples, utc_timestamp, write_json
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore


POLL_INTERVAL_S = 0.05


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda, but torch.cuda.is_available() is false")
    return device


def _test_config(tmp_root: Path, device: str) -> EngineConfig:
    return EngineConfig(
        device=device,
        dtype="float16" if device == "cuda" else "float32",
        dynamics=DynamicsConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            action_dim=8,
            latent_token_dim=6,
            max_rollout_steps=16,
        ),
        tokenizer=TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        ),
        state_cache=StateCacheConfig(
            max_batch_size=8,
            max_rollout_steps=16,
            latent_dim=6,
            num_latent_tokens=16,
            pool_size_gb=0.1,
        ),
        controlplane=ControlPlaneConfig(
            manifest_store_root=str(tmp_root / "manifests"),
            wan_output_root=str(tmp_root / "wan"),
            genie_output_root=str(tmp_root / "genie"),
        ),
    )


async def _wait_for_terminal_sample(client: httpx.AsyncClient, sample_id: str, timeout_s: float) -> tuple[dict[str, Any], int, float]:
    deadline = time.perf_counter() + timeout_s
    polls = 0
    while time.perf_counter() < deadline:
        polls += 1
        resp = await client.get(f"/v1/samples/{sample_id}")
        resp.raise_for_status()
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed", "rejected", "accepted"}:
            return payload, polls, time.perf_counter()
        await asyncio.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Sample {sample_id} did not reach terminal state within {timeout_s}s")


async def _run_iteration(client: httpx.AsyncClient, request_payload: dict[str, Any], timeout_s: float, iteration: int) -> dict[str, Any]:
    submit_start = time.perf_counter()
    response = await client.post("/v1/samples", json=request_payload)
    submit_end = time.perf_counter()
    response.raise_for_status()
    created = response.json()

    result = {
        "iteration": iteration,
        "sample_id": created["sample_id"],
        "initial_status": created["status"],
        "status": created["status"],
        "metrics": {
            "submit_latency_ms": round((submit_end - submit_start) * 1000.0, 3),
        },
    }

    if created["status"] in {"queued", "running"}:
        terminal_payload, polls, terminal_ts = await _wait_for_terminal_sample(client, created["sample_id"], timeout_s)
        result["status"] = terminal_payload["status"]
        result["terminal_payload"] = terminal_payload
        result["metrics"]["poll_count"] = float(polls)
        result["metrics"]["terminal_latency_ms"] = round((terminal_ts - submit_start) * 1000.0, 3)
    else:
        result["terminal_payload"] = created
        result["metrics"]["poll_count"] = 0.0
        result["metrics"]["terminal_latency_ms"] = round((submit_end - submit_start) * 1000.0, 3)

    return result


async def _prepare_request_payload(client: httpx.AsyncClient, request_payload: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(request_payload)
    if payload.get("backend") != "genie-rollout":
        return payload

    temporal = payload.get("temporal") or {}
    if temporal.get("episode_id") and temporal.get("branch_id") and temporal.get("state_handle_id"):
        return payload

    episode_resp = await client.post("/v1/episodes", json={"title": "Benchmark Genie Episode"})
    episode_resp.raise_for_status()
    episode = episode_resp.json()

    branch_resp = await client.post("/v1/branches", json={"episode_id": episode["episode_id"], "name": "main"})
    branch_resp.raise_for_status()
    branch = branch_resp.json()

    state_resp = await client.post(
        "/v1/state-handles",
        json={
            "episode_id": episode["episode_id"],
            "branch_id": branch["branch_id"],
            "kind": "latent",
            "dtype": "float16",
            "shape": [16, 6],
        },
    )
    state_resp.raise_for_status()
    state = state_resp.json()

    payload["temporal"] = {
        "episode_id": episode["episode_id"],
        "branch_id": branch["branch_id"],
        "state_handle_id": state["state_handle_id"],
    }
    return payload


async def _run_client(client: httpx.AsyncClient, request_payload: dict[str, Any], iterations: int, timeout_s: float, concurrency: int) -> list[dict[str, Any]]:
    request_payload = await _prepare_request_payload(client, request_payload)
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(i: int) -> dict[str, Any]:
        async with semaphore:
            return await _run_iteration(client, request_payload, timeout_s, i)

    return await asyncio.gather(*[_wrapped(i) for i in range(iterations)])


async def _run_live(base_url: str, request_payload: dict[str, Any], iterations: int, timeout_s: float, concurrency: int, gpu_poll_interval_s: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with GpuSampler(poll_interval_s=gpu_poll_interval_s) as sampler:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s + 5.0) as client:
            samples = await _run_client(client, request_payload, iterations, timeout_s, concurrency)
    return samples, sampler.summary()


async def _run_in_process(request_payload: dict[str, Any], iterations: int, timeout_s: float, concurrency: int, device: str, gpu_poll_interval_s: float, execution_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with TemporaryDirectory(prefix="wm_infra_bench_") as tmp:
        tmp_root = Path(tmp)
        config = _test_config(tmp_root, device)
        app = create_app(config, sample_store=SampleManifestStore(tmp_root / "manifests"), execution_mode=execution_mode)
        with GpuSampler(poll_interval_s=gpu_poll_interval_s) as sampler:
            async with LifespanManager(app) as manager:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url="http://bench", timeout=timeout_s + 5.0) as client:
                    samples = await _run_client(client, request_payload, iterations, timeout_s, concurrency)
        return samples, sampler.summary()


DEFAULT_WAN_PAYLOAD = {
    "task_type": "text_to_video",
    "backend": "wan-video",
    "model": "wan2.2-t2v-A14B",
    "sample_spec": {"prompt": "a corgi running through a neon lab"},
    "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480, "memory_profile": "low_vram"},
}

DEFAULT_ROLLOUT_PAYLOAD = {
    "task_type": "world_model_rollout",
    "backend": "rollout-engine",
    "model": "latent_dynamics",
    "sample_spec": {"prompt": "predict the next robot movement", "width": 256, "height": 256},
    "task_config": {"num_steps": 4, "frame_count": 9, "width": 256, "height": 256},
}

DEFAULT_GENIE_PAYLOAD = {
    "task_type": "genie_rollout",
    "backend": "genie-rollout",
    "model": "genie-local",
    "sample_spec": {"prompt": "roll forward in time", "width": 256, "height": 256},
    "task_config": {"num_steps": 4, "width": 256, "height": 256},
    "genie_config": {"num_frames": 9, "num_prompt_frames": 4, "maskgit_steps": 3, "temperature": 0.0},
    "return_artifacts": ["metadata"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wm-infra sample APIs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-url", help="Live wm-infra server base URL, e.g. http://127.0.0.1:8000")
    mode.add_argument("--in-process", action="store_true", help="Run against an in-process ASGI app")
    parser.add_argument("--workload", choices=["wan", "rollout", "genie"], default="wan")
    parser.add_argument("--payload-file", help="Optional JSON payload for POST /v1/samples")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--execution-mode", choices=["legacy", "chunked"], default="chunked")
    parser.add_argument("--gpu-sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--system-name", default="wm-infra")
    parser.add_argument("--runner-name", default="unknown")
    parser.add_argument("--output", default="benchmarks/results/sample_api_benchmark.json")
    parser.add_argument("--baseline-file", help="Optional benchmark JSON file to compare against and embed in the output artifact")
    return parser.parse_args()


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file:
        import json
        return json.loads(Path(args.payload_file).read_text())
    if args.workload == "wan":
        return DEFAULT_WAN_PAYLOAD
    if args.workload == "genie":
        return DEFAULT_GENIE_PAYLOAD
    return DEFAULT_ROLLOUT_PAYLOAD


async def _main_async() -> None:
    args = parse_args()
    device = args.device
    if args.in_process:
        device = _resolve_device(device)
    payload = _load_payload(args)
    if args.in_process:
        samples, gpu_profile = await _run_in_process(payload, args.iterations, args.timeout_s, args.concurrency, device, args.gpu_sample_interval_ms / 1000.0, args.execution_mode)
        execution_mode = "in_process"
    else:
        samples, gpu_profile = await _run_live(args.base_url, payload, args.iterations, args.timeout_s, args.concurrency, args.gpu_sample_interval_ms / 1000.0)
        execution_mode = "remote"

    task_cfg = payload.get("wan_config") or payload.get("task_config") or {}
    genie_cfg = payload.get("genie_config") or {}
    summary = run_summary_from_samples(samples)
    result = {
        "schema_version": 1,
        "recorded_at": utc_timestamp(),
        "run_context": capture_runtime_context(),
        "system": {
            "name": args.system_name,
            "runner": args.runner_name,
            "execution_mode": execution_mode,
        },
        "execution": {
            "mode": execution_mode,
            "iterations": args.iterations,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout_s,
            "device": device,
            "execution_mode": args.execution_mode,
            "gpu_sample_interval_ms": args.gpu_sample_interval_ms,
            "base_url": args.base_url,
            "in_process": args.in_process,
            "workload": args.workload,
            "payload_file": args.payload_file,
        },
        "workload": {
            "workload_kind": "sample_api",
            "task_type": payload.get("task_type"),
            "backend_family": payload.get("backend"),
            "model": payload.get("model"),
            "num_prompts": 1,
            "prompt_shape": "single",
            "frame_count": task_cfg.get("frame_count") or genie_cfg.get("num_frames"),
            "width": task_cfg.get("width") or payload.get("sample_spec", {}).get("width"),
            "height": task_cfg.get("height") or payload.get("sample_spec", {}).get("height"),
            "num_steps": task_cfg.get("num_steps"),
            "input_modality": payload.get("task_type"),
            "execution_device": device,
            "runtime_execution_mode": args.execution_mode,
        },
        "request_payload": payload,
        "summary": summary,
        "gpu_profile": gpu_profile,
        "samples": samples,
        "notes": [
            "This harness measures wm-infra API and queue behavior, not model quality.",
            "Comparisons against vLLM or sglang are only valid when the workload definition matches and the external system can actually execute that workload.",
        ],
    }
    if args.baseline_file:
        baseline = load_json(args.baseline_file)
        result["baseline"] = {
            "path": args.baseline_file,
            "recorded_at": baseline.get("recorded_at"),
            "system": baseline.get("system"),
            "comparison": comparison_report(result, baseline),
    }
    write_json(args.output, result)
    print(f"Wrote benchmark artifact to {args.output}")
    print(format_gpu_summary(gpu_profile))
    print(summary)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
