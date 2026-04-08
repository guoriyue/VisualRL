import importlib.util
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from wm_infra.api.server import create_app
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_samples_api.py"
    spec = importlib.util.spec_from_file_location("bench_samples_api_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(
        device="cpu",
        dtype="float32",
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
            manifest_store_root=str(tmp_path / "manifests"),
            wan_output_root=str(tmp_path / "wan"),
            cosmos_output_root=str(tmp_path / "cosmos"),
        ),
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    config = _test_config(tmp_path)
    app = create_app(
        config,
        sample_store=SampleManifestStore(tmp_path / "manifests"),
    )

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


@pytest.mark.asyncio
async def test_prepare_request_payload_is_identity_for_matrix_payload(client):
    bench = _load_benchmark_module()

    prepared = await bench._prepare_request_payload(client, bench.DEFAULT_MATRIX_PAYLOAD)
    assert prepared == bench.DEFAULT_MATRIX_PAYLOAD


@pytest.mark.asyncio
async def test_prepare_request_payload_leaves_non_matrix_payload_unchanged(client):
    bench = _load_benchmark_module()

    prepared = await bench._prepare_request_payload(client, bench.DEFAULT_WAN_PAYLOAD)
    assert prepared == bench.DEFAULT_WAN_PAYLOAD


def test_test_config_respects_requested_device(tmp_path):
    bench = _load_benchmark_module()

    config = bench._test_config(tmp_path, "cuda")
    assert config.device == "cuda"


def test_parse_args_accepts_chunked_execution_mode(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr("sys.argv", ["bench_samples_api.py", "--in-process", "--execution-mode", "chunked"])
    args = bench.parse_args()
    assert args.execution_mode == "chunked"


def test_parse_args_rejects_legacy_execution_mode(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr("sys.argv", ["bench_samples_api.py", "--in-process", "--execution-mode", "legacy"])
    with pytest.raises(SystemExit):
        bench.parse_args()


def test_load_payload_supports_cosmos(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr("sys.argv", ["bench_samples_api.py", "--in-process", "--workload", "cosmos"])
    args = bench.parse_args()
    payload = bench._load_payload(args)
    assert payload["backend"] == "cosmos-predict"


def test_load_payload_supports_matrix(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr("sys.argv", ["bench_samples_api.py", "--in-process", "--workload", "matrix"])
    args = bench.parse_args()
    payload = bench._load_payload(args)
    assert payload["backend"] == "matrix-game"


def test_observed_runtime_fields_in_process():
    bench = _load_benchmark_module()

    execution_fields, workload_fields = bench._observed_runtime_fields(
        in_process=True,
        resolved_device="cuda",
        requested_device="cuda",
        requested_execution_mode="chunked",
    )

    assert execution_fields["device"] == "cuda"
    assert execution_fields["runtime_execution_mode"] == "chunked"
    assert execution_fields["requested_device"] == "cuda"
    assert execution_fields["requested_runtime_execution_mode"] == "chunked"
    assert workload_fields == {
        "execution_device": "cuda",
        "runtime_execution_mode": "chunked",
    }


def test_observed_runtime_fields_remote_omit_unverified_server_runtime():
    bench = _load_benchmark_module()

    execution_fields, workload_fields = bench._observed_runtime_fields(
        in_process=False,
        resolved_device="cpu",
        requested_device="cpu",
        requested_execution_mode="chunked",
    )

    assert execution_fields == {
        "requested_device": "cpu",
        "requested_runtime_execution_mode": "chunked",
    }
    assert workload_fields == {}


def test_runtime_accounting_extracts_compile_cache_transfer_and_residency():
    bench = _load_benchmark_module()

    wan_payload = {
        "backend": "wan-video",
        "runtime": {
            "execution_family": {"backend": "wan-video", "stage": "pipeline"},
            "compiled_graph_pool": {
                "profile_id": "wan-profile-1",
                "compile_state": "warm_profile_batch_hit",
                "warm_profile_hit": True,
                "compiled_batch_size_hit": True,
                "compiled_batch_sizes": [1, 2],
                "reuse_count": 3,
                "prewarmed": True,
                "compiled_profile": {"graph_key": "wan-graph-abc"},
            },
            "transfer_plan": {
                "h2d_bytes": 1024,
                "d2h_bytes": 2048,
                "device_to_device_bytes": 128,
                "artifact_io_bytes": 4096,
                "staging_bytes": 512,
                "overlap_h2d_with_compute": True,
                "overlap_d2h_with_io": True,
                "staging_tier": "cpu_pinned_warm",
            },
            "residency": [
                {"tier": "gpu_hot", "bytes_size": 1024},
                {"tier": "cpu_pinned_warm", "bytes_size": 512},
            ],
        },
    }
    wan_accounting = bench._runtime_accounting(wan_payload)
    assert wan_accounting["compile"]["graph_key"] == "wan-graph-abc"
    assert wan_accounting["compile"]["warm_profile_hit"] is True
    assert wan_accounting["transfer"]["total_bytes"] == 7296
    assert wan_accounting["residency"]["tier_counts"]["gpu_hot"] == 1

    matrix_payload = {
        "backend": "matrix-game",
        "runtime": {
            "runtime_state": {
                "prompt_reuse_hit": True,
                "resident_tier": "hot_gpu",
                "reuse_hits": 2,
                "reuse_misses": 1,
                "source_cache_key": "state_handle:1",
                "checkpoint_delta_ref": "delta:1",
                "page_size_tokens": 512,
                "page_count": 4,
                "transfer_plan": {
                    "h2d_bytes": 256,
                    "d2h_bytes": 0,
                    "device_to_device_bytes": 0,
                    "artifact_io_bytes": 128,
                    "staging_bytes": 64,
                    "overlap_h2d_with_compute": False,
                    "overlap_d2h_with_io": True,
                    "staging_tier": "cpu_pinned_warm",
                },
                "residency": [
                    {"tier": "gpu_hot", "bytes_size": 4096},
                    {"tier": "cpu_pinned_warm", "bytes_size": 2048},
                ],
            }
        },
    }
    matrix_accounting = bench._runtime_accounting(matrix_payload)
    assert matrix_accounting["cache"]["prompt_reuse_hit"] is True
    assert matrix_accounting["cache"]["page_count"] == 4
    assert matrix_accounting["transfer"]["artifact_io_bytes"] == 128


def test_profile_samples_aggregates_runtime_accounting():
    bench = _load_benchmark_module()

    samples = [
        {
            "backend": "wan-video",
            "accounting": {
                "backend": "wan-video",
                "compile": {"compile_state": "warm_profile_batch_hit", "warm_profile_hit": True},
                "cache": {"prompt_reuse_hit": False, "page_count": None},
                "transfer": {"total_bytes": 7296, "h2d_bytes": 1024, "d2h_bytes": 2048, "artifact_io_bytes": 4096},
                "residency": {"total_bytes": 1536, "tier_counts": {"gpu_hot": 1, "cpu_pinned_warm": 1}},
            },
        },
        {
            "backend": "matrix-game",
            "accounting": {
                "backend": "matrix-game",
                "compile": {"compile_state": "cold_start", "warm_profile_hit": False},
                "cache": {"prompt_reuse_hit": True, "page_count": 4},
                "transfer": {"total_bytes": 384, "h2d_bytes": 256, "d2h_bytes": 0, "artifact_io_bytes": 128},
                "residency": {"total_bytes": 6144, "tier_counts": {"gpu_hot": 1, "cpu_pinned_warm": 1}},
            },
        },
    ]

    profile = bench._profile_samples(samples)
    assert profile["compile"]["warm_profile_hits"] == 1
    assert profile["compile"]["backend_hits"] == {"wan-video": 1, "matrix-game": 1}
    assert profile["cache"]["prompt_cache_hits"] == 1
    assert profile["cache"]["mean_page_count"] == 4.0
    assert profile["transfer"]["total_bytes"] == 7680
    assert profile["residency"]["tier_counts"]["gpu_hot"] == 2
