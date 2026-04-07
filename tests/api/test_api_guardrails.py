"""Tests for northbound auth and sample metrics."""

from __future__ import annotations

import httpx
import pytest

from wm_infra.api.server import create_app
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, ServerConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore


def _guarded_config() -> EngineConfig:
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
        server=ServerConfig(api_key="test-secret"),
        controlplane=ControlPlaneConfig(),
    )


@pytest.mark.asyncio
async def test_api_key_guard_protects_sample_and_queue_endpoints(tmp_path):
    from asgi_lifespan import LifespanManager

    config = _guarded_config()
    app = create_app(config, sample_store=SampleManifestStore(tmp_path))

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as client:
            health_resp = await client.get("/v1/health")
            assert health_resp.status_code == 200

            denied = await client.post("/v1/samples", json={
                "task_type": "temporal_rollout",
                "backend": "rollout-engine",
                "model": "latent_dynamics",
                "sample_spec": {"prompt": "guarded"},
                "return_artifacts": ["latent"],
            })
            assert denied.status_code == 401

            auth_metrics = await client.get("/metrics")
            assert auth_metrics.status_code == 200
            assert 'wm_api_auth_failures_total{endpoint="/v1/samples"} 1.0' in auth_metrics.text

            allowed = await client.post(
                "/v1/samples",
                headers={"X-API-Key": "test-secret"},
                json={
                    "task_type": "temporal_rollout",
                    "backend": "rollout-engine",
                    "model": "latent_dynamics",
                    "sample_spec": {"prompt": "guarded"},
                    "return_artifacts": ["latent"],
                },
            )
            assert allowed.status_code == 200
            assert allowed.json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_sample_metrics_are_exposed(tmp_path):
    from asgi_lifespan import LifespanManager

    config = _guarded_config()
    config.server.api_key = None
    app = create_app(config, sample_store=SampleManifestStore(tmp_path))

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/v1/samples", json={
                "task_type": "temporal_rollout",
                "backend": "rollout-engine",
                "model": "latent_dynamics",
                "sample_spec": {"prompt": "metrics"},
                "return_artifacts": ["latent"],
            })
            assert resp.status_code == 200

            await client.get("/v1/queue/status")
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert 'wm_sample_total{backend="rollout-engine",status="succeeded"}' in metrics.text
            assert 'wm_sample_duration_seconds_count{backend="rollout-engine",status="succeeded"}' in metrics.text
            assert "wm_queue_depth" in metrics.text
