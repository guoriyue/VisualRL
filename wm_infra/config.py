"""Configuration for wm-infra engine and gateway.

Supports layered config loading: defaults → YAML file → env vars → CLI args.
Use ``load_config()`` to merge all layers into a single ``EngineConfig``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class SchedulerPolicy(str, Enum):
    FCFS = "fcfs"  # first-come first-serve
    SJF = "sjf"  # shortest-job-first (fewest remaining steps)
    DEADLINE = "deadline"  # earliest-deadline-first
    MEMORY_AWARE = "memory_aware"  # prefer lighter frame/resolution jobs first


@dataclass
class SchedulerConfig:
    """Rollout scheduler configuration."""

    max_batch_size: int = 32
    max_waiting_time_ms: float = 50.0
    policy: SchedulerPolicy = SchedulerPolicy.SJF
    max_concurrent_rollouts: int = 64
    max_batch_resource_units: float | None = None


@dataclass
class ServerConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8400
    max_concurrent_requests: int = 128
    stream_chunk_interval_ms: float = 33.0  # ~30fps streaming
    api_key: str | None = None


@dataclass
class IPCConfig:
    """ZMQ IPC configuration for gateway <-> engine communication."""

    enabled: bool = False
    socket_path: str = "/tmp/wm-engine.sock"
    artifact_root: str = "/dev/shm/wm-engine"
    artifact_ttl_s: float = 300.0


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    device: DeviceType = DeviceType.CUDA
    dtype: str = "float16"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    model_path: str | None = None
    seed: int = 42


# ─── Config loading ───


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _load_yaml(path: str | Path) -> dict:
    """Load YAML config file. Returns empty dict if PyYAML not installed."""
    try:
        import yaml
    except ImportError as err:
        raise ImportError("PyYAML required for YAML config files: pip install pyyaml") from err
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _env_overrides() -> dict:
    """Read WM_* environment variables into a nested dict.

    Mapping (all prefixed with WM_):
        WM_DEVICE=cpu                  → {"device": "cpu"}
        WM_DTYPE=bfloat16              → {"dtype": "bfloat16"}
        WM_MODEL_PATH=/path            → {"model_path": "/path"}
        WM_SEED=123                    → {"seed": 123}
        WM_PORT=9000                   → {"server": {"port": 9000}}
        WM_HOST=127.0.0.1              → {"server": {"host": "127.0.0.1"}}
        WM_API_KEY=secret              → {"server": {"api_key": "secret"}}
        WM_MAX_BATCH_SIZE=16           → {"scheduler": {"max_batch_size": 16}}
        WM_IPC_ENABLED=true            → {"ipc": {"enabled": True}}
        WM_IPC_SOCKET_PATH=/tmp/x.sock → {"ipc": {"socket_path": "/tmp/x.sock"}}
        WM_IPC_ARTIFACT_ROOT=/dev/shm  → {"ipc": {"artifact_root": "/dev/shm"}}
        WM_IPC_ARTIFACT_TTL_S=300      → {"ipc": {"artifact_ttl_s": 300.0}}
    """
    overrides: dict[str, Any] = {}
    env_map: dict[str, tuple[list[str], type]] = {
        "WM_DEVICE": (["device"], str),
        "WM_DTYPE": (["dtype"], str),
        "WM_MODEL_PATH": (["model_path"], str),
        "WM_SEED": (["seed"], int),
        "WM_PORT": (["server", "port"], int),
        "WM_HOST": (["server", "host"], str),
        "WM_API_KEY": (["server", "api_key"], str),
        "WM_MAX_BATCH_SIZE": (["scheduler", "max_batch_size"], int),
        "WM_MAX_CONCURRENT_ROLLOUTS": (["scheduler", "max_concurrent_rollouts"], int),
        "WM_IPC_ENABLED": (
            ["ipc", "enabled"],
            lambda value: value.lower() in {"1", "true", "yes", "on"},
        ),
        "WM_IPC_SOCKET_PATH": (["ipc", "socket_path"], str),
        "WM_IPC_ARTIFACT_ROOT": (["ipc", "artifact_root"], str),
        "WM_IPC_ARTIFACT_TTL_S": (["ipc", "artifact_ttl_s"], float),
    }

    for env_key, (path, typ) in env_map.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        converted = typ(val)
        d = overrides
        for part in path[:-1]:
            d = d.setdefault(part, {})
        d[path[-1]] = converted

    return overrides


def _dict_to_config(d: dict) -> EngineConfig:
    """Convert a flat/nested dict into an EngineConfig dataclass."""
    if "device" in d and isinstance(d["device"], str):
        d["device"] = DeviceType(d["device"])

    d.pop("tokenizer", None)  # Legacy field, ignored
    d.pop("dynamics", None)  # Legacy field, ignored
    d.pop("controlplane", None)  # Legacy field, ignored
    d.pop("state_cache", None)  # Legacy field, ignored
    sched_d = d.pop("scheduler", {})
    serv_d = d.pop("server", {})
    ipc_d = d.pop("ipc", {})

    if "policy" in sched_d and isinstance(sched_d["policy"], str):
        sched_d["policy"] = SchedulerPolicy(sched_d["policy"])

    return EngineConfig(
        scheduler=SchedulerConfig(**sched_d) if sched_d else SchedulerConfig(),
        server=ServerConfig(**serv_d) if serv_d else ServerConfig(),
        ipc=IPCConfig(**ipc_d) if ipc_d else IPCConfig(),
        **d,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for wm-serve."""
    parser = argparse.ArgumentParser(
        prog="wm-serve",
        description="Temporal model serving and control-plane server",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model weights")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "float32", "bfloat16"], default=None
    )
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8400)")
    parser.add_argument("--host", type=str, default=None, help="Server host (default: 0.0.0.0)")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key required for protected endpoints",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=None, help="Max batch size for scheduling"
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def load_config(
    cli_args: list[str] | None = None,
    config_path: str | None = None,
) -> EngineConfig:
    """Load config by merging: defaults → YAML → env vars → CLI args.

    Args:
        cli_args: CLI argument list (None = use sys.argv)
        config_path: Explicit YAML path (overrides --config CLI arg)
    """
    merged: dict[str, Any] = asdict(EngineConfig())
    merged["device"] = (
        merged["device"].value if hasattr(merged["device"], "value") else merged["device"]
    )
    merged["scheduler"]["policy"] = (
        merged["scheduler"]["policy"].value
        if hasattr(merged["scheduler"]["policy"], "value")
        else merged["scheduler"]["policy"]
    )

    parser = build_parser()
    args = parser.parse_args(cli_args if cli_args is not None else [])

    yaml_path = config_path or args.config
    if yaml_path and Path(yaml_path).exists():
        yaml_d = _load_yaml(yaml_path)
        merged = _deep_merge(merged, yaml_d)

    env_d = _env_overrides()
    if env_d:
        merged = _deep_merge(merged, env_d)

    cli_overrides: dict[str, Any] = {}
    if args.device is not None:
        cli_overrides["device"] = args.device
    if args.dtype is not None:
        cli_overrides["dtype"] = args.dtype
    if args.model_path is not None:
        cli_overrides["model_path"] = args.model_path
    if args.seed is not None:
        cli_overrides["seed"] = args.seed
    if args.port is not None:
        cli_overrides.setdefault("server", {})["port"] = args.port
    if args.host is not None:
        cli_overrides.setdefault("server", {})["host"] = args.host
    if args.api_key is not None:
        cli_overrides.setdefault("server", {})["api_key"] = args.api_key
    if args.max_batch_size is not None:
        cli_overrides.setdefault("scheduler", {})["max_batch_size"] = args.max_batch_size

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return _dict_to_config(merged)
