import importlib.util
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from wm_infra.api.server import create_app
from wm_infra.backends import BackendRegistry, GenieRolloutBackend
from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_samples_api.py"
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
            genie_output_root=str(tmp_path / "genie"),
        ),
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    config = _test_config(tmp_path)
    temporal_store = TemporalStore(tmp_path / "temporal")
    registry = BackendRegistry()
    genie_runner = GenieRunner()
    genie_runner._mode = "stub"
    genie_runner.load = lambda: "stub"  # type: ignore[method-assign]
    registry.register(
        GenieRolloutBackend(
            temporal_store,
            output_root=tmp_path / "genie",
            runner=genie_runner,
        )
    )
    app = create_app(
        config,
        sample_store=SampleManifestStore(tmp_path / "manifests"),
        backend_registry=registry,
        temporal_store=temporal_store,
    )

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


@pytest.mark.asyncio
async def test_prepare_request_payload_creates_temporal_refs_for_genie(client):
    bench = _load_benchmark_module()

    prepared = await bench._prepare_request_payload(client, bench.DEFAULT_GENIE_PAYLOAD)
    assert prepared["backend"] == "genie-rollout"
    assert prepared["temporal"]["episode_id"]
    assert prepared["temporal"]["branch_id"]
    assert prepared["temporal"]["state_handle_id"]


@pytest.mark.asyncio
async def test_prepare_request_payload_leaves_non_genie_payload_unchanged(client):
    bench = _load_benchmark_module()

    prepared = await bench._prepare_request_payload(client, bench.DEFAULT_WAN_PAYLOAD)
    assert prepared == bench.DEFAULT_WAN_PAYLOAD


def test_test_config_respects_requested_device(tmp_path):
    bench = _load_benchmark_module()

    config = bench._test_config(tmp_path, "cuda")
    assert config.device == "cuda"


def test_parse_args_accepts_execution_mode(monkeypatch):
    bench = _load_benchmark_module()
    monkeypatch.setattr("sys.argv", ["bench_samples_api.py", "--in-process", "--execution-mode", "legacy"])
    args = bench.parse_args()
    assert args.execution_mode == "legacy"
