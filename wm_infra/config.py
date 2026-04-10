"""Configuration for wm-infra temporal serving and control-plane runtime.

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
class TokenizerConfig:
    """Video tokenizer configuration (COSMOS-style)."""

    spatial_downsample: int = 8
    temporal_downsample: int = 4
    latent_channels: int = 16
    codebook_size: int = 64  # FSQ levels per dimension
    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    causal_temporal: bool = True
    input_channels: int = 3  # RGB


@dataclass
class MoEConfig:
    """Configuration for Mixture-of-Experts feed-forward layers."""

    num_experts: int = 8
    top_k: int = 2
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    weight_dtype: str = "float16"
    max_experts_in_gpu: int | None = None
    has_shared_experts: bool = False
    shared_expert_intermediate_dim: int | None = None
    use_shared_expert_gate: bool = False
    use_expert_bias: bool = False
    expert_bias_lr: float = 0.001
    aux_loss_weight: float = 0.01
    renormalize: bool = True
    use_stream_overlap: bool = False


@dataclass
class TransformerConfig:
    """Configuration for transformer attention + FFN blocks."""

    hidden_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    moe: MoEConfig = field(default_factory=MoEConfig)
    attention_type: str = "mha"
    attention_backend: str = "auto"
    compile_attention: bool = False
    qkv_bias: bool = False
    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qk_head_dim: int | None = None
    v_head_dim: int = 128
    speculative_top_k: int | None = None

    def __post_init__(self) -> None:
        if self.num_kv_heads <= 0:
            self.num_kv_heads = self.num_heads
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.moe.hidden_dim != self.hidden_dim:
            self.moe.hidden_dim = self.hidden_dim


@dataclass
class ModelConfig:
    """Configuration for full decoder-only transformer models."""

    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    block: TransformerConfig = field(default_factory=TransformerConfig)
    moe_layer_indices: list[int] = field(default_factory=list)
    intermediate_dim_dense: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.block.hidden_dim != self.hidden_dim:
            self.block.hidden_dim = self.hidden_dim
            if self.block.moe.hidden_dim != self.hidden_dim:
                self.block.moe.hidden_dim = self.hidden_dim
        if self.intermediate_dim_dense is None:
            self.intermediate_dim_dense = self.block.moe.intermediate_dim


@dataclass
class StateCacheConfig:
    """Latent state cache configuration."""

    max_batch_size: int = 64
    max_rollout_steps: int = 128
    latent_dim: int = 16
    num_latent_tokens: int = 256  # tokens per frame after spatial tokenization
    pool_size_gb: float = 4.0  # GPU memory pool for state cache
    eviction_policy: str = "lru"


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
class ControlPlaneConfig:
    """Control-plane persistence configuration."""

    manifest_store_root: str | None = None
    wan_output_root: str | None = None
    # Legacy external-execution settings. Gateway startup now rejects them.
    wan_shell_runner: str | None = None
    wan_shell_runner_timeout_s: int | None = None
    wan_repo_dir: str | None = None
    wan_conda_env: str | None = None
    wan_ckpt_dir: str | None = None
    wan_i2v_diffusers_dir: str | None = None
    conda_sh_path: str | None = None
    wan_engine_adapter: str | None = None
    wan_max_queue_size: int = 64
    wan_max_concurrent_jobs: int = 1
    wan_max_batch_size: int = 4
    wan_batch_wait_ms: float = 2.0
    wan_warm_pool_size: int = 16
    wan_prewarm_common_signatures: bool = False
    wan_admission_max_units: float | None = None
    wan_admission_max_vram_gb: float | None = 32.0
    # Legacy Cosmos external-execution settings. Gateway startup now rejects them.
    cosmos_output_root: str | None = None
    cosmos_base_url: str | None = None
    cosmos_api_key: str | None = None
    cosmos_model_name: str | None = None
    cosmos_shell_runner: str | None = None
    cosmos_timeout_s: int = 600
    cosmos_max_queue_size: int = 64
    cosmos_max_concurrent_jobs: int = 1


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
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    state_cache: StateCacheConfig = field(default_factory=StateCacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    controlplane: ControlPlaneConfig = field(default_factory=ControlPlaneConfig)
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
        WM_PORT=9000                   → {"server": {"port": 9000}}
        WM_HOST=127.0.0.1              → {"server": {"host": "127.0.0.1"}}
        WM_API_KEY=secret              → {"server": {"api_key": "secret"}}
        WM_MAX_BATCH_SIZE=16           → {"scheduler": {"max_batch_size": 16}}
        WM_MANIFEST_STORE_ROOT=/data   → {"controlplane": {"manifest_store_root": "/data"}}
        WM_WAN_OUTPUT_ROOT=/data/wan   → {"controlplane": {"wan_output_root": "/data/wan"}}
        WM_WAN_SHELL_RUNNER=...        → {"controlplane": {"wan_shell_runner": "..."}} (legacy, rejected at startup)
        WM_WAN_SHELL_RUNNER_TIMEOUT_S=600 → {"controlplane": {"wan_shell_runner_timeout_s": 600}} (deprecated legacy knob kept only for explicit startup errors)
        WM_WAN_REPO_DIR=/path/to/Wan2.2 → {"controlplane": {"wan_repo_dir": "/path/to/Wan2.2"}}
        WM_WAN_CONDA_ENV=kosen         → {"controlplane": {"wan_conda_env": "kosen"}}
        WM_WAN_CKPT_DIR=/path/to/ckpt  → {"controlplane": {"wan_ckpt_dir": "/path/to/ckpt"}}
        WM_WAN_I2V_DIFFUSERS_DIR=/path/to/diffusers → {"controlplane": {"wan_i2v_diffusers_dir": "/path/to/diffusers"}}
        WM_CONDA_SH_PATH=/path/conda.sh → {"controlplane": {"conda_sh_path": "/path/conda.sh"}}
        WM_WAN_ENGINE_ADAPTER=module:factory → {"controlplane": {"wan_engine_adapter": "module:factory"}}
        WM_WAN_MAX_QUEUE_SIZE=32       → {"controlplane": {"wan_max_queue_size": 32}}
        WM_WAN_MAX_CONCURRENT_JOBS=1   → {"controlplane": {"wan_max_concurrent_jobs": 1}}
        WM_WAN_MAX_BATCH_SIZE=4        → {"controlplane": {"wan_max_batch_size": 4}}
        WM_WAN_BATCH_WAIT_MS=5         → {"controlplane": {"wan_batch_wait_ms": 5.0}}
        WM_WAN_WARM_POOL_SIZE=16       → {"controlplane": {"wan_warm_pool_size": 16}}
        WM_WAN_PREWARM_COMMON_SIGNATURES=true → {"controlplane": {"wan_prewarm_common_signatures": True}}
        WM_WAN_ADMISSION_MAX_UNITS=16  → {"controlplane": {"wan_admission_max_units": 16.0}}
        WM_WAN_ADMISSION_MAX_VRAM_GB=32 → {"controlplane": {"wan_admission_max_vram_gb": 32.0}}
        WM_COSMOS_BASE_URL=http://...  → {"controlplane": {"cosmos_base_url": "http://..."}} (legacy, rejected at startup)
        WM_COSMOS_API_KEY=secret       → {"controlplane": {"cosmos_api_key": "secret"}} (legacy, rejected with base URL)
        WM_COSMOS_SHELL_RUNNER=...     → {"controlplane": {"cosmos_shell_runner": "..."}} (legacy, rejected at startup)
        WM_SEED=123                    → {"seed": 123}
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
        "WM_MANIFEST_STORE_ROOT": (["controlplane", "manifest_store_root"], str),
        "WM_WAN_OUTPUT_ROOT": (["controlplane", "wan_output_root"], str),
        "WM_WAN_SHELL_RUNNER": (["controlplane", "wan_shell_runner"], str),
        "WM_WAN_SHELL_RUNNER_TIMEOUT_S": (["controlplane", "wan_shell_runner_timeout_s"], int),
        "WM_WAN_REPO_DIR": (["controlplane", "wan_repo_dir"], str),
        "WM_WAN_CONDA_ENV": (["controlplane", "wan_conda_env"], str),
        "WM_WAN_CKPT_DIR": (["controlplane", "wan_ckpt_dir"], str),
        "WM_WAN_I2V_DIFFUSERS_DIR": (["controlplane", "wan_i2v_diffusers_dir"], str),
        "WM_CONDA_SH_PATH": (["controlplane", "conda_sh_path"], str),
        "WM_WAN_ENGINE_ADAPTER": (["controlplane", "wan_engine_adapter"], str),
        "WM_WAN_MAX_QUEUE_SIZE": (["controlplane", "wan_max_queue_size"], int),
        "WM_WAN_MAX_CONCURRENT_JOBS": (["controlplane", "wan_max_concurrent_jobs"], int),
        "WM_WAN_MAX_BATCH_SIZE": (["controlplane", "wan_max_batch_size"], int),
        "WM_WAN_BATCH_WAIT_MS": (["controlplane", "wan_batch_wait_ms"], float),
        "WM_WAN_WARM_POOL_SIZE": (["controlplane", "wan_warm_pool_size"], int),
        "WM_WAN_PREWARM_COMMON_SIGNATURES": (
            ["controlplane", "wan_prewarm_common_signatures"],
            lambda value: value.lower() in {"1", "true", "yes", "on"},
        ),
        "WM_WAN_ADMISSION_MAX_UNITS": (["controlplane", "wan_admission_max_units"], float),
        "WM_WAN_ADMISSION_MAX_VRAM_GB": (["controlplane", "wan_admission_max_vram_gb"], float),
        "WM_COSMOS_OUTPUT_ROOT": (["controlplane", "cosmos_output_root"], str),
        "WM_COSMOS_BASE_URL": (["controlplane", "cosmos_base_url"], str),
        "WM_COSMOS_API_KEY": (["controlplane", "cosmos_api_key"], str),
        "WM_COSMOS_MODEL_NAME": (["controlplane", "cosmos_model_name"], str),
        "WM_COSMOS_SHELL_RUNNER": (["controlplane", "cosmos_shell_runner"], str),
        "WM_COSMOS_TIMEOUT_S": (["controlplane", "cosmos_timeout_s"], int),
        "WM_COSMOS_MAX_QUEUE_SIZE": (["controlplane", "cosmos_max_queue_size"], int),
        "WM_COSMOS_MAX_CONCURRENT_JOBS": (["controlplane", "cosmos_max_concurrent_jobs"], int),
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

    tok_d = d.pop("tokenizer", {})
    d.pop("dynamics", None)  # Legacy field, ignored
    sc_d = d.pop("state_cache", {})
    sched_d = d.pop("scheduler", {})
    serv_d = d.pop("server", {})
    ctrl_d = d.pop("controlplane", {})
    ipc_d = d.pop("ipc", {})

    if "policy" in sched_d and isinstance(sched_d["policy"], str):
        sched_d["policy"] = SchedulerPolicy(sched_d["policy"])

    return EngineConfig(
        tokenizer=TokenizerConfig(**tok_d) if tok_d else TokenizerConfig(),
        state_cache=StateCacheConfig(**sc_d) if sc_d else StateCacheConfig(),
        scheduler=SchedulerConfig(**sched_d) if sched_d else SchedulerConfig(),
        server=ServerConfig(**serv_d) if serv_d else ServerConfig(),
        controlplane=ControlPlaneConfig(**ctrl_d) if ctrl_d else ControlPlaneConfig(),
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
