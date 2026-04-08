from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from wm_infra.api.server import create_app
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore


def _test_config() -> EngineConfig:
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
        controlplane=ControlPlaneConfig(),
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    from asgi_lifespan import LifespanManager

    config = _test_config()
    config.controlplane.cosmos_output_root = str(tmp_path / "cosmos")
    config.controlplane.wan_output_root = str(tmp_path / "wan")
    temporal_store = TemporalStore(tmp_path / "temporal")
    app = create_app(
        config,
        sample_store=SampleManifestStore(tmp_path),
        temporal_store=temporal_store,
    )

    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


@pytest.mark.asyncio
async def test_reinforcement_learning_catalog_endpoints_expose_default_env_and_tasks(client):
    envs_resp = await client.get("/v1/env-specs")
    assert envs_resp.status_code == 200
    envs = envs_resp.json()["environment_specs"]
    assert any(item["env_name"] == "toy-line-v0" for item in envs)

    tasks_resp = await client.get("/v1/task-specs")
    assert tasks_resp.status_code == 200
    tasks = tasks_resp.json()["task_specs"]
    assert {item["task_id"] for item in tasks} >= {"toy-line-train", "toy-line-eval"}


@pytest.mark.asyncio
async def test_stateless_initialize_and_predict_transition(client):
    init_resp = await client.post(
        "/v1/transitions/initialize",
        json={"env_name": "toy-line-v0", "task_id": "toy-line-eval", "seed": 7, "policy_version": "pi-1"},
    )
    assert init_resp.status_code == 200
    initialized = init_resp.json()
    assert initialized["current_step"] == 0
    assert initialized["trajectory_id"] is not None
    assert initialized["state_handle_id"] is not None

    step_resp = await client.post(
        "/v1/transitions/predict",
        json={
            "state_handle_id": initialized["state_handle_id"],
            "trajectory_id": initialized["trajectory_id"],
            "action": [0.0, 0.0, 1.0],
            "policy_version": "pi-1",
        },
    )
    assert step_resp.status_code == 200
    stepped = step_resp.json()
    assert stepped["step_idx"] == 1
    assert stepped["reward"] == pytest.approx(-0.16)
    assert stepped["terminated"] is False
    assert stepped["truncated"] is False
    assert stepped["trajectory_id"] == initialized["trajectory_id"]
    assert stepped["state_handle_id"] != initialized["state_handle_id"]

    transitions_resp = await client.get("/v1/transitions", params={"trajectory_id": initialized["trajectory_id"]})
    assert transitions_resp.status_code == 200
    transitions = transitions_resp.json()["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["reward"] == pytest.approx(-0.16)

    trajectories_resp = await client.get("/v1/trajectories", params={"episode_id": initialized["episode_id"]})
    assert trajectories_resp.status_code == 200
    trajectories = trajectories_resp.json()["trajectories"]
    assert len(trajectories) == 1
    assert trajectories[0]["num_steps"] == 1
    assert trajectories[0]["return_value"] == pytest.approx(-0.16)


@pytest.mark.asyncio
async def test_stateless_predict_many_batches_explicit_state_handles(client):
    init_responses = []
    for seed in (3, 9):
        resp = await client.post(
            "/v1/transitions/initialize",
            json={"env_name": "toy-line-v0", "task_id": "toy-line-eval", "seed": seed},
        )
        assert resp.status_code == 200
        init_responses.append(resp.json())

    step_many_resp = await client.post(
        "/v1/transitions/predict_many",
        json={
            "items": [
                {
                    "state_handle_id": init_responses[0]["state_handle_id"],
                    "trajectory_id": init_responses[0]["trajectory_id"],
                    "action": [0.0, 0.0, 1.0],
                },
                {
                    "state_handle_id": init_responses[1]["state_handle_id"],
                    "trajectory_id": init_responses[1]["trajectory_id"],
                    "action": [1.0, 0.0, 0.0],
                },
            ],
            "policy_version": "pi-batch",
            "checkpoint": True,
        },
    )
    assert step_many_resp.status_code == 200
    payload = step_many_resp.json()
    assert len(payload["results"]) == 2
    assert payload["runtime"]["execution_path"] == "chunked_stateless_transition"
    assert payload["runtime"]["chunk_count"] >= 1
    assert payload["runtime"]["max_chunk_size"] >= 2
    assert all(item["step_idx"] == 1 for item in payload["results"])
    assert all(item["checkpoint_id"] is not None for item in payload["results"])
