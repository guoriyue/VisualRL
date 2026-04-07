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
from collections import Counter
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import torch
from asgi_lifespan import LifespanManager

from wm_infra.api.server import create_app
from benchmarks.benchmarking import GpuSampler, capture_runtime_context, comparison_report, format_gpu_summary, load_json, run_summary_from_samples, utc_timestamp, write_json
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore


POLL_INTERVAL_S = 0.05


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda, but torch.cuda.is_available() is false")
    return device


def _test_config(
    tmp_root: Path,
    device: str,
    *,
    genie_max_concurrent_jobs: int = 1,
    genie_max_batch_size: int = 1,
    genie_batch_wait_ms: float = 0.0,
) -> EngineConfig:
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
            genie_device=device,
            genie_max_concurrent_jobs=genie_max_concurrent_jobs,
            genie_max_batch_size=genie_max_batch_size,
            genie_batch_wait_ms=genie_batch_wait_ms,
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
        result["accounting"] = _runtime_accounting(terminal_payload)
        result["metrics"]["poll_count"] = float(polls)
        result["metrics"]["terminal_latency_ms"] = round((terminal_ts - submit_start) * 1000.0, 3)
    else:
        result["terminal_payload"] = created
        result["accounting"] = _runtime_accounting(created)
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


async def _run_in_process(
    request_payload: dict[str, Any],
    iterations: int,
    timeout_s: float,
    concurrency: int,
    device: str,
    gpu_poll_interval_s: float,
    execution_mode: str,
    *,
    persist_root: str | None,
    genie_max_concurrent_jobs: int,
    genie_cross_request_batching: str,
    genie_transition_batch_wait_ms: float,
    genie_transition_max_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    async def _run_with_root(tmp_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        config = _test_config(
            tmp_root,
            device,
            genie_max_concurrent_jobs=genie_max_concurrent_jobs,
            genie_max_batch_size=1 if genie_cross_request_batching == "off" else genie_transition_max_batch_size,
            genie_batch_wait_ms=genie_transition_batch_wait_ms if genie_cross_request_batching == "on" else 0.0,
        )
        app = create_app(config, sample_store=SampleManifestStore(tmp_root / "manifests"), execution_mode=execution_mode)
        with GpuSampler(poll_interval_s=gpu_poll_interval_s) as sampler:
            async with LifespanManager(app) as manager:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url="http://bench", timeout=timeout_s + 5.0) as client:
                    samples = await _run_client(client, request_payload, iterations, timeout_s, concurrency)
        return samples, sampler.summary()

    if persist_root:
        tmp_root = Path(persist_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        return await _run_with_root(tmp_root)

    with TemporaryDirectory(prefix="wm_infra_bench_") as tmp:
        tmp_root = Path(tmp)
        return await _run_with_root(tmp_root)


DEFAULT_WAN_PAYLOAD = {
    "task_type": "text_to_video",
    "backend": "wan-video",
    "model": "wan2.2-t2v-A14B",
    "sample_spec": {"prompt": "a corgi running through a neon lab"},
    "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480, "memory_profile": "low_vram"},
}

DEFAULT_ROLLOUT_PAYLOAD = {
    "task_type": "temporal_rollout",
    "backend": "rollout-engine",
    "model": "latent_dynamics",
    "sample_spec": {"prompt": "predict the next robot movement", "width": 256, "height": 256},
    "task_config": {"num_steps": 4, "frame_count": 9, "width": 256, "height": 256},
}

DEFAULT_GENIE_PAYLOAD = {
    "task_type": "temporal_rollout",
    "backend": "genie-rollout",
    "model": "genie-local",
    "sample_spec": {"prompt": "roll forward in time", "width": 256, "height": 256},
    "task_config": {"num_steps": 4, "width": 256, "height": 256},
    "genie_config": {"num_frames": 9, "num_prompt_frames": 4, "maskgit_steps": 3, "temperature": 0.0},
    "return_artifacts": ["metadata"],
}

DEFAULT_COSMOS_PAYLOAD = {
    "task_type": "text_to_video",
    "backend": "cosmos-predict",
    "model": "cosmos-predict1-7b-text2world",
    "sample_spec": {"prompt": "A warehouse robot driving past stacked pallets.", "width": 1024, "height": 640},
    "task_config": {"num_steps": 35, "frame_count": 16, "width": 1024, "height": 640},
    "cosmos_config": {"variant": "predict1_text2world", "model_size": "7B", "frames_per_second": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark wm-infra sample APIs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-url", help="Live wm-infra server base URL, e.g. http://127.0.0.1:8000")
    mode.add_argument("--in-process", action="store_true", help="Run against an in-process ASGI app")
    parser.add_argument("--workload", choices=["wan", "rollout", "genie", "cosmos"], default="wan")
    parser.add_argument("--payload-file", help="Optional JSON payload for POST /v1/samples")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--execution-mode", choices=["chunked"], default="chunked")
    parser.add_argument("--gpu-sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--persist-root", help="Optional root directory to keep in-process sample manifests and backend outputs")
    parser.add_argument("--genie-max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--genie-cross-request-batching", choices=["on", "off"], default="on")
    parser.add_argument("--genie-transition-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--genie-transition-max-batch-size", type=int, default=8)
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
    if args.workload == "cosmos":
        return DEFAULT_COSMOS_PAYLOAD
    return DEFAULT_ROLLOUT_PAYLOAD


def _observed_runtime_fields(*, in_process: bool, resolved_device: str, requested_device: str, requested_execution_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_fields: dict[str, Any] = {
        "requested_device": requested_device,
        "requested_runtime_execution_mode": requested_execution_mode,
    }
    workload_fields: dict[str, Any] = {}
    if in_process:
        execution_fields["device"] = resolved_device
        execution_fields["runtime_execution_mode"] = requested_execution_mode
        workload_fields["execution_device"] = resolved_device
        workload_fields["runtime_execution_mode"] = requested_execution_mode
    return execution_fields, workload_fields


def _flatten_residency_records(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    records = records or []
    tier_counts: Counter[str] = Counter()
    bytes_by_tier: Counter[str] = Counter()
    total_bytes = 0
    for record in records:
        tier = str(record.get("tier") or "unknown")
        bytes_size = int(record.get("bytes_size") or 0)
        tier_counts[tier] += 1
        bytes_by_tier[tier] += bytes_size
        total_bytes += bytes_size
    return {
        "count": len(records),
        "total_bytes": total_bytes,
        "tier_counts": dict(tier_counts),
        "bytes_by_tier": dict(bytes_by_tier),
    }


def _runtime_accounting(sample_payload: dict[str, Any]) -> dict[str, Any]:
    runtime = sample_payload.get("runtime") or {}
    compiled_graph_pool = runtime.get("compiled_graph_pool") or {}
    runtime_state = runtime.get("runtime_state") or {}
    execution_family = runtime.get("execution_family") or compiled_graph_pool.get("execution_family") or {}
    transfer_plan = runtime.get("transfer_plan") or runtime_state.get("transfer_plan") or {}
    residency = runtime.get("residency") or runtime_state.get("residency") or []
    residency_summary = _flatten_residency_records(residency)

    compile_info = {
        "profile_id": compiled_graph_pool.get("profile_id"),
        "compile_state": compiled_graph_pool.get("compile_state"),
        "warm_profile_hit": compiled_graph_pool.get("warm_profile_hit"),
        "compiled_batch_size_hit": compiled_graph_pool.get("compiled_batch_size_hit"),
        "compiled_batch_sizes": compiled_graph_pool.get("compiled_batch_sizes") or [],
        "reuse_count": compiled_graph_pool.get("reuse_count"),
        "prewarmed": compiled_graph_pool.get("prewarmed"),
        "graph_key": (compiled_graph_pool.get("compiled_profile") or {}).get("graph_key")
        or compiled_graph_pool.get("graph_family_key")
        or compiled_graph_pool.get("graph_key"),
        "execution_family": execution_family,
    }
    cache_info = {
        "prompt_reuse_hit": runtime_state.get("prompt_reuse_hit"),
        "resident_tier": runtime_state.get("resident_tier"),
        "reuse_hits": runtime_state.get("reuse_hits"),
        "reuse_misses": runtime_state.get("reuse_misses"),
        "source_cache_key": runtime_state.get("source_cache_key"),
        "checkpoint_delta_ref": runtime_state.get("checkpoint_delta_ref"),
        "page_size_tokens": runtime_state.get("page_size_tokens"),
        "page_count": runtime_state.get("page_count"),
    }
    transfer_info = {
        "h2d_bytes": int(transfer_plan.get("h2d_bytes") or 0),
        "d2h_bytes": int(transfer_plan.get("d2h_bytes") or 0),
        "device_to_device_bytes": int(transfer_plan.get("device_to_device_bytes") or 0),
        "artifact_io_bytes": int(transfer_plan.get("artifact_io_bytes") or 0),
        "staging_bytes": int(transfer_plan.get("staging_bytes") or 0),
        "overlap_h2d_with_compute": bool(transfer_plan.get("overlap_h2d_with_compute", False)),
        "overlap_d2h_with_io": bool(transfer_plan.get("overlap_d2h_with_io", False)),
        "staging_tier": transfer_plan.get("staging_tier"),
    }
    transfer_info["total_bytes"] = int(
        transfer_plan.get("total_bytes")
        or transfer_info["h2d_bytes"]
        + transfer_info["d2h_bytes"]
        + transfer_info["device_to_device_bytes"]
        + transfer_info["artifact_io_bytes"]
    )
    return {
        "backend": sample_payload.get("backend"),
        "compile": compile_info,
        "cache": cache_info,
        "transfer": transfer_info,
        "residency": residency_summary,
    }


def _profile_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    compile_states: Counter[str] = Counter()
    backend_hits: Counter[str] = Counter()
    residency_tiers: Counter[str] = Counter()
    total_transfer_bytes = 0
    total_h2d_bytes = 0
    total_d2h_bytes = 0
    total_artifact_io_bytes = 0
    total_residency_bytes = 0
    warm_profile_hits = 0
    prompt_cache_hits = 0
    page_counts: list[int] = []

    for sample in samples:
        accounting = sample.get("accounting") or {}
        backend = str(accounting.get("backend") or sample.get("backend") or "")
        if backend:
            backend_hits[backend] += 1
        compile_info = accounting.get("compile") or {}
        cache_info = accounting.get("cache") or {}
        transfer_info = accounting.get("transfer") or {}
        residency_info = accounting.get("residency") or {}

        compile_state = compile_info.get("compile_state")
        if compile_state:
            compile_states[str(compile_state)] += 1
        if compile_info.get("warm_profile_hit"):
            warm_profile_hits += 1
        if cache_info.get("prompt_reuse_hit"):
            prompt_cache_hits += 1
        total_transfer_bytes += int(transfer_info.get("total_bytes") or 0)
        total_h2d_bytes += int(transfer_info.get("h2d_bytes") or 0)
        total_d2h_bytes += int(transfer_info.get("d2h_bytes") or 0)
        total_artifact_io_bytes += int(transfer_info.get("artifact_io_bytes") or 0)
        total_residency_bytes += int(residency_info.get("total_bytes") or 0)
        for tier, count in (residency_info.get("tier_counts") or {}).items():
            residency_tiers[str(tier)] += int(count)
        page_count = cache_info.get("page_count")
        if page_count is not None:
            page_counts.append(int(page_count))

    total_samples = len(samples)
    return {
        "compile": {
            "warm_profile_hits": warm_profile_hits,
            "warm_profile_hit_rate": (warm_profile_hits / total_samples) if total_samples else 0.0,
            "compile_states": dict(compile_states),
            "backend_hits": dict(backend_hits),
        },
        "cache": {
            "prompt_cache_hits": prompt_cache_hits,
            "prompt_cache_hit_rate": (prompt_cache_hits / total_samples) if total_samples else 0.0,
            "mean_page_count": (sum(page_counts) / len(page_counts)) if page_counts else 0.0,
        },
        "transfer": {
            "total_bytes": total_transfer_bytes,
            "h2d_bytes": total_h2d_bytes,
            "d2h_bytes": total_d2h_bytes,
            "artifact_io_bytes": total_artifact_io_bytes,
        },
        "residency": {
            "total_bytes": total_residency_bytes,
            "tier_counts": dict(residency_tiers),
        },
    }


async def _main_async() -> None:
    args = parse_args()
    device = args.device
    if args.in_process:
        device = _resolve_device(device)
    payload = _load_payload(args)
    if args.in_process:
        samples, gpu_profile = await _run_in_process(
            payload,
            args.iterations,
            args.timeout_s,
            args.concurrency,
            device,
            args.gpu_sample_interval_ms / 1000.0,
            args.execution_mode,
            persist_root=args.persist_root,
            genie_max_concurrent_jobs=args.genie_max_concurrent_jobs,
            genie_cross_request_batching=args.genie_cross_request_batching,
            genie_transition_batch_wait_ms=args.genie_transition_batch_wait_ms,
            genie_transition_max_batch_size=args.genie_transition_max_batch_size,
        )
        execution_mode = "in_process"
    else:
        samples, gpu_profile = await _run_live(args.base_url, payload, args.iterations, args.timeout_s, args.concurrency, args.gpu_sample_interval_ms / 1000.0)
        execution_mode = "remote"

    task_cfg = payload.get("wan_config") or payload.get("task_config") or {}
    genie_cfg = payload.get("genie_config") or {}
    summary = run_summary_from_samples(samples)
    profiling = _profile_samples(samples)
    execution_fields, workload_runtime_fields = _observed_runtime_fields(
        in_process=args.in_process,
        resolved_device=device,
        requested_device=args.device,
        requested_execution_mode=args.execution_mode,
    )
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
            "gpu_sample_interval_ms": args.gpu_sample_interval_ms,
            "persist_root": args.persist_root,
            "genie_max_concurrent_jobs": args.genie_max_concurrent_jobs,
            "genie_cross_request_batching": args.genie_cross_request_batching,
            "genie_transition_batch_wait_ms": args.genie_transition_batch_wait_ms,
            "genie_transition_max_batch_size": args.genie_transition_max_batch_size,
            "base_url": args.base_url,
            "in_process": args.in_process,
            "workload": args.workload,
            "payload_file": args.payload_file,
            **execution_fields,
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
            **workload_runtime_fields,
        },
        "request_payload": payload,
        "summary": summary,
        "profiling": profiling,
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
