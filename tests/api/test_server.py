"""Integration tests for the FastAPI server.

Uses httpx AsyncClient with the ASGI transport to test the full
request/response cycle without starting a real server process.
"""

import asyncio
import json
import time
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio

import wm_infra.api.server as server_module
from wm_infra.api.server import create_app
from wm_infra.config import ControlPlaneConfig, DynamicsConfig, EngineConfig, StateCacheConfig, TokenizerConfig
from wm_infra.controlplane import ArtifactKind, SampleManifestStore, TemporalStore


def _test_config() -> EngineConfig:
    """Small CPU config for integration testing."""
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
        controlplane=ControlPlaneConfig(wan_engine_adapter="stub"),
    )


def test_create_async_engine_uses_strict_state_loading(monkeypatch):
    calls = {}

    class DummyDynamics:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state_dict, strict=True):
            calls["dynamics_strict"] = strict
            calls["dynamics_state_dict"] = state_dict

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class DummyTokenizer:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state_dict, strict=True):
            calls["tokenizer_strict"] = strict
            calls["tokenizer_state_dict"] = state_dict

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    class DummyEngine:
        def __init__(self, config, dynamics, tokenizer, execution_mode="chunked"):
            calls["execution_mode"] = execution_mode
            calls["config"] = config
            calls["dynamics"] = dynamics
            calls["tokenizer"] = tokenizer

    monkeypatch.setattr(server_module, "LatentDynamicsModel", DummyDynamics)
    monkeypatch.setattr(server_module, "VideoTokenizer", DummyTokenizer)
    monkeypatch.setattr(server_module, "AsyncWorldModelEngine", DummyEngine)
    monkeypatch.setattr(
        server_module.torch,
        "load",
        lambda *args, **kwargs: {"dynamics": {"weights": 1}, "tokenizer": {"vocab": 2}},
    )

    config = _test_config()
    config.model_path = "dummy-model.pt"
    engine = server_module._create_async_engine(config)

    assert isinstance(engine, DummyEngine)
    assert calls["dynamics_strict"] is True
    assert calls["tokenizer_strict"] is True
    assert calls["execution_mode"] == "chunked"


@pytest_asyncio.fixture
async def client(tmp_path):
    """Create an httpx AsyncClient connected to the test app."""
    from asgi_lifespan import LifespanManager

    config = _test_config()
    config.controlplane.cosmos_output_root = str(tmp_path / "cosmos")
    config.controlplane.wan_output_root = str(tmp_path / "wan")
    config.controlplane.wan_engine_adapter = "stub"
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


async def _wait_for_terminal_sample(client: httpx.AsyncClient, sample_id: str, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        resp = await client.get(f"/v1/samples/{sample_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in {"succeeded", "failed", "accepted"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"sample {sample_id} did not reach terminal state; last={last}")


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["engine_running"] is True

    @pytest.mark.asyncio
    async def test_health_active_rollouts_zero_at_rest(self, client):
        resp = await client.get("/v1/health")
        assert resp.json()["active_rollouts"] == 0


class TestModels:
    @pytest.mark.asyncio
    async def test_list_models(self, client):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "latent_dynamics" in data["models"]


class TestBackends:
    @pytest.mark.asyncio
    async def test_list_backends(self, client):
        resp = await client.get("/v1/backends")
        assert resp.status_code == 200
        backends = {backend["name"]: backend for backend in resp.json()["backends"]}
        assert "cosmos-predict" in backends
        assert "matrix-game" in backends
        assert "rollout-engine" in backends
        assert "wan-video" in backends
        assert backends["cosmos-predict"]["world_model_kind"] == "generation"
        assert backends["matrix-game"]["world_model_kind"] == "dynamics"
        assert backends["matrix-game"]["runtime_substrate"] == "rollout-engine"
        assert backends["matrix-game"]["stateful"] is True
        assert backends["matrix-game"]["async_queue"] is False
        assert backends["cosmos-predict"]["runner_mode"] == "stub"
        assert backends["cosmos-predict"]["async_queue"] is True
        assert backends["cosmos-predict"]["operator"]["family"] == "generation"
        assert backends["cosmos-predict"]["operator"]["runtime"]["backend"] == "native"
        assert backends["wan-video"]["world_model_kind"] == "generation"
        assert backends["wan-video"]["shell_runner_configured"] is False
        assert backends["wan-video"]["runner_mode"] == "stub"
        assert backends["wan-video"]["async_queue"] is True
        assert backends["wan-video"]["execution_backend"] == "in_process_stage_scheduler"
        assert backends["wan-video"]["engine_adapter"]["name"] == "stub-wan-engine"
        assert backends["wan-video"]["operator"]["family"] == "generation"
        assert backends["wan-video"]["operator"]["runtime"]["local_scheduler"] is True
        assert backends["wan-video"]["max_batch_size"] == 4
        assert backends["wan-video"]["batch_wait_ms"] == 2.0
        assert backends["wan-video"]["prewarm_common_signatures"] is False
        assert backends["wan-video"]["warm_pool"]["prewarmed_profiles"] == 0
        assert backends["wan-video"]["admission_max_vram_gb"] == 32.0


class TestRollout:
    @pytest.mark.asyncio
    async def test_basic_rollout(self, client):
        resp = await client.post("/v1/rollout", json={
            "num_steps": 2,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_completed"] == 2
        assert data["elapsed_ms"] > 0
        assert data["latents"] is not None
        assert len(data["latents"]) == 2

    @pytest.mark.asyncio
    async def test_rollout_with_latent_input(self, client):
        N, D = 16, 6
        latent = [[0.1 * i for _ in range(D)] for i in range(N)]
        resp = await client.post("/v1/rollout", json={
            "initial_latent": latent,
            "num_steps": 3,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        assert resp.json()["steps_completed"] == 3

    @pytest.mark.asyncio
    async def test_rollout_with_actions(self, client):
        N, D, A = 16, 6, 8
        latent = [[0.0] * D for _ in range(N)]
        actions = [[0.1] * A for _ in range(2)]
        resp = await client.post("/v1/rollout", json={
            "initial_latent": latent,
            "actions": actions,
            "num_steps": 2,
            "return_latents": True,
            "return_frames": False,
        })
        assert resp.status_code == 200
        assert resp.json()["steps_completed"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_rollouts(self, client):
        async def single_request(_i):
            return await client.post("/v1/rollout", json={
                "num_steps": 2,
                "return_latents": True,
                "return_frames": False,
            })

        responses = await asyncio.gather(*[single_request(i) for i in range(4)])
        for resp in responses:
            assert resp.status_code == 200
            assert resp.json()["steps_completed"] == 2


class TestSamples:
    @pytest.mark.asyncio
    async def test_create_and_get_sample_manifest(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "temporal_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "return_artifacts": [ArtifactKind.LATENT.value],
            "task_config": {"num_steps": 2, "frame_count": 9, "width": 832, "height": 480, "memory_profile": "low_vram"},
            "sample_spec": {
                "prompt": "predict the next dog jump",
                "width": 832,
                "height": 480,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "succeeded"
        assert data["runtime"]["steps_completed"] == 2
        assert data["task_config"]["num_steps"] == 2
        assert data["task_config"]["frame_count"] == 9
        assert data["task_config"]["memory_profile"] == "low_vram"
        assert data["artifacts"][0]["kind"] == ArtifactKind.LATENT.value
        assert data["resource_estimate"]["bottleneck"] == "frame_pressure"

        sample_id = data["sample_id"]
        get_resp = await client.get(f"/v1/samples/{sample_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["sample_id"] == sample_id

    @pytest.mark.asyncio
    async def test_create_sample_does_not_hydrate_num_steps_from_sample_metadata(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "temporal_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "return_artifacts": [ArtifactKind.LATENT.value],
            "sample_spec": {
                "prompt": "predict the next dog jump",
                "metadata": {"num_steps": 2},
            },
        })
        assert resp.status_code == 200
        assert resp.json()["task_type"] == "temporal_rollout"
        assert resp.json()["task_config"] is None
        assert resp.json()["runtime"]["steps_completed"] == 1

    @pytest.mark.asyncio
    async def test_create_cosmos_sample_queues_and_completes(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "cosmos-predict",
            "model": "cosmos-predict1-7b-text2world",
            "world_model_kind": "generation",
            "sample_spec": {
                "prompt": "A warehouse robot navigating around boxes.",
                "width": 1024,
                "height": 640,
            },
            "task_config": {"num_steps": 12, "frame_count": 16, "width": 1024, "height": 640},
            "cosmos_config": {"variant": "predict1_text2world", "model_size": "7B", "frames_per_second": 16},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        terminal = await _wait_for_terminal_sample(client, data["sample_id"])
        assert terminal["status"] == "succeeded"
        assert terminal["runtime"]["runner_mode"] == "stub"
        assert terminal["world_model_kind"] == "generation"

    @pytest.mark.asyncio
    async def test_create_matrix_sample_returns_dynamics_record(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "temporal_rollout",
            "backend": "matrix-game",
            "model": "matrix-game-bringup",
            "world_model_kind": "dynamics",
            "return_artifacts": [ArtifactKind.LATENT.value],
            "task_config": {"num_steps": 3, "frame_count": 3},
            "sample_spec": {
                "prompt": "roll the world forward under player actions",
                "controls": {
                    "actions": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                },
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "succeeded"
        assert data["world_model_kind"] == "dynamics"
        assert data["runtime"]["runtime_substrate"] == "rollout-engine"
        assert data["runtime"]["action_count"] == 3
        assert any(artifact["kind"] == "latent" for artifact in data["artifacts"])

    @pytest.mark.asyncio
    async def test_create_sample_rejects_mismatched_world_model_kind(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "cosmos-predict",
            "model": "cosmos-predict1-7b-text2world",
            "world_model_kind": "dynamics",
            "sample_spec": {"prompt": "bad world model family"},
            "task_config": {"num_steps": 12, "frame_count": 16, "width": 1024, "height": 640},
            "cosmos_config": {"variant": "predict1_text2world", "model_size": "7B", "frames_per_second": 16},
        })
        assert resp.status_code == 400
        assert "world_model_kind=generation" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_sample_persists_under_experiment_directory(self, client, tmp_path):
        resp = await client.post("/v1/samples", json={
            "task_type": "temporal_rollout",
            "backend": "rollout-engine",
            "model": "latent_dynamics",
            "task_config": {"num_steps": 1},
            "experiment": {"experiment_id": "exp_server_test"},
            "sample_spec": {"prompt": "organized sample"},
        })
        assert resp.status_code == 200
        sample_id = resp.json()["sample_id"]
        assert (tmp_path / "samples" / "exp_server_test" / f"{sample_id}.json").exists()

    @pytest.mark.asyncio
    async def test_create_wan_video_sample_returns_queued(self, client):
        """Wan video POST /v1/samples now returns immediately with queued status."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "a corgi surfing through a data center"},
            "wan_config": {
                "num_steps": 4,
                "frame_count": 9,
                "width": 832,
                "height": 480,
                "memory_profile": "low_vram"
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["wan_config"]["frame_count"] == 9
        assert data["resource_estimate"]["estimated_vram_gb"] > 0
        assert data["runtime"]["runner"] == "stub"
        assert data["runtime"]["async"] is True
        assert data["runtime"]["status_history"][0]["status"] == "queued"
        assert data["runtime"]["admission"]["admitted"] is True
        assert data["runtime"]["queue_position"] >= 0
        assert data["runtime"]["scheduler"]["max_batch_size"] == 4
        assert data["metadata"]["async"] is True

    @pytest.mark.asyncio
    async def test_wan_async_job_completes_in_background(self, client):
        """Submit a Wan job async and verify it transitions to succeeded."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "a cat jumping over a fence"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        assert resp.status_code == 200
        sample_id = resp.json()["sample_id"]
        assert resp.json()["status"] == "queued"

        final = await _wait_for_terminal_sample(client, sample_id)
        assert final["status"] == "accepted"
        assert final["runtime"]["runner"] == "stub"
        assert final["runtime"]["execution_backend"] == "in_process_stage_scheduler"
        assert final["runtime"]["pipeline"]["stage_count"] >= 6
        assert final["runtime"]["stages"][0]["name"] == "text_encode"
        assert final["runtime"]["elapsed_ms"] >= 0
        kinds = {a["kind"] for a in final["artifacts"]}
        assert "video" not in kinds
        assert {"log", "metadata"}.issubset(kinds)
        assert final["metadata"]["stubbed"] is True

    @pytest.mark.asyncio
    async def test_create_wan_video_requires_prompt_for_t2v(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "   "},
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_wan_video_can_be_rejected_by_admission(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.wan_admission_max_vram_gb = 10.0
        app = create_app(config, sample_store=SampleManifestStore(tmp_path))

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                resp = await c.post("/v1/samples", json={
                    "task_type": "text_to_video",
                    "backend": "wan-video",
                    "model": "wan2.2-t2v-A14B",
                    "sample_spec": {"prompt": "too big"},
                    "wan_config": {"num_steps": 4, "frame_count": 17, "width": 832, "height": 480},
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "rejected"
                assert data["runtime"]["admission"]["admitted"] is False
                assert "estimated_vram_gb" in " ".join(data["runtime"]["admission"]["reasons"])
                assert data["runtime"]["admission"]["quality_cost_hints"]["suggested_adjustments"]
                fetched = await c.get(f"/v1/samples/{data['sample_id']}")
                assert fetched.status_code == 200
                assert fetched.json()["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_create_sample_rejects_unknown_backend(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "temporal_rollout",
            "backend": "missing-backend",
            "model": "latent_dynamics",
            "sample_spec": {"prompt": "bad backend"},
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_samples(self, client):
        create_resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "experiment": {"experiment_id": "exp_list"},
            "sample_spec": {"prompt": "list me"},
        })
        assert create_resp.status_code == 200

        resp = await client.get("/v1/samples", params={"backend": "wan-video", "experiment_id": "exp_list"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert data["samples"][0]["backend"] == "wan-video"

    @pytest.mark.asyncio
    async def test_get_missing_sample(self, client):
        resp = await client.get("/v1/samples/does-not-exist")
        assert resp.status_code == 404


class TestArtifacts:
    @pytest.mark.asyncio
    async def test_list_artifacts(self, client):
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "artifact list"},
        })
        sample_id = resp.json()["sample_id"]
        data = await _wait_for_terminal_sample(client, sample_id)
        assert data["status"] == "accepted"

        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts")
        assert art_resp.status_code == 200
        artifact_kinds = {artifact["kind"] for artifact in art_resp.json()["artifacts"]}
        assert {"log", "metadata"}.issubset(artifact_kinds)
        assert "video" not in artifact_kinds
        assert art_resp.json()["count"] == len(art_resp.json()["artifacts"])

    @pytest.mark.asyncio
    async def test_get_artifact_metadata(self, client):
        """Submit wan job, wait for completion, check artifact metadata endpoint."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "artifact test"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        sample_id = resp.json()["sample_id"]

        data = await _wait_for_terminal_sample(client, sample_id)
        assert data["status"] == "accepted"

        # Fetch log artifact metadata
        log_artifact_id = quote(f"{sample_id}:log", safe="")
        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/{log_artifact_id}")
        assert art_resp.status_code == 200
        art_data = art_resp.json()
        assert art_data["kind"] == "log"
        assert art_data["mime_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_get_artifact_content(self, client):
        """Submit wan job, wait for completion, download artifact content."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "content test"},
            "wan_config": {"num_steps": 4, "frame_count": 9, "width": 832, "height": 480}
        })
        sample_id = resp.json()["sample_id"]

        await _wait_for_terminal_sample(client, sample_id)

        # Fetch the log file content
        log_artifact_id = quote(f"{sample_id}:log", safe="")
        content_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/{log_artifact_id}/content")
        assert content_resp.status_code == 200
        assert "stub mode" in content_resp.text

    @pytest.mark.asyncio
    async def test_get_artifact_not_found(self, client):
        """Should 404 for missing artifact."""
        resp = await client.post("/v1/samples", json={
            "task_type": "text_to_video",
            "backend": "wan-video",
            "model": "wan2.2-t2v-A14B",
            "sample_spec": {"prompt": "missing art"},
        })
        sample_id = resp.json()["sample_id"]
        art_resp = await client.get(f"/v1/samples/{sample_id}/artifacts/nonexistent")
        assert art_resp.status_code == 404


class TestTemporalControlPlane:
    @pytest.mark.asyncio
    async def test_default_wan_backend_uses_controlplane_batching_config(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.manifest_store_root = str(tmp_path / "manifests")
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.cosmos_output_root = str(tmp_path / "cosmos")
        config.controlplane.wan_max_batch_size = 3
        config.controlplane.wan_batch_wait_ms = 7.5
        config.controlplane.wan_warm_pool_size = 5
        config.controlplane.wan_prewarm_common_signatures = False

        temporal_store = TemporalStore(tmp_path / "temporal")
        app = create_app(
            config,
            sample_store=SampleManifestStore(tmp_path / "manifests"),
            temporal_store=temporal_store,
        )

        async with LifespanManager(app):
            backend = app.state.backend_registry.get("wan-video")
            assert backend is not None
            assert backend.max_batch_size == 3
            assert backend.batch_wait_ms == 7.5
            assert backend.warm_pool_size == 5
            assert backend.prewarm_common_signatures is False
            assert backend.execution_backend == "in_process_stage_scheduler"
            assert app.state.wan_job_queue is not None
            snapshot = app.state.wan_job_queue.snapshot()
            assert snapshot["max_batch_size"] == 3
            assert snapshot["batch_select_enabled"] is True


class TestQueueStatus:
    @pytest.mark.asyncio
    async def test_queue_status_endpoint(self, client):
        resp = await client.get("/v1/queue/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue_enabled"] is True
        assert "pending" in data
        assert "running" in data
        assert "total_tracked" in data
        assert "queues" in data
        assert "max_queue_size" in data["queues"]["wan"]
        assert isinstance(data["queues"]["wan"]["queued_sample_ids"], list)


class TestShellRunner:
    @pytest.mark.asyncio
    async def test_wan_shell_runner_failure_is_recorded(self, tmp_path):
        from asgi_lifespan import LifespanManager

        config = _test_config()
        config.controlplane.wan_output_root = str(tmp_path / "wan")
        config.controlplane.wan_engine_adapter = "disabled"
        config.controlplane.wan_shell_runner = "python -c \"import sys; print('runner boom'); sys.exit(7)\""
        app = create_app(config, sample_store=SampleManifestStore(tmp_path))

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as shell_client:
                resp = await shell_client.post("/v1/samples", json={
                    "task_type": "text_to_video",
                    "backend": "wan-video",
                    "model": "wan2.2-t2v-A14B",
                    "sample_spec": {"prompt": "fail loudly"},
                })

                # Job is queued async — wait for background worker to complete it
                assert resp.status_code == 200
                sample_id = resp.json()["sample_id"]
                assert resp.json()["status"] == "queued"

                data = await _wait_for_terminal_sample(shell_client, sample_id)
                assert data["status"] == "failed"
                assert data["runtime"]["runner"] == "shell"
                assert data["runtime"]["returncode"] == 7
                failure_artifacts = [artifact for artifact in data["artifacts"] if artifact["metadata"].get("role") == "failure"]
                assert len(failure_artifacts) == 1
                failure_meta = await shell_client.get(f"/v1/samples/{sample_id}/artifacts/{quote(f'{sample_id}:failure', safe='')}")
                assert failure_meta.status_code == 200


class TestOfficialRunner:
    @pytest.mark.asyncio
    async def test_official_runner_mode_reflected_in_backend(self, tmp_path):
        """Verify that when wan_repo_dir is set, runner_mode is 'official'."""
        from asgi_lifespan import LifespanManager
        from wm_infra.backends import BackendRegistry, WanVideoBackend

        config = _test_config()
        wan_root = str(tmp_path / "wan")
        registry = BackendRegistry()
        wan_backend = WanVideoBackend(
            wan_root,
            wan_engine_adapter="disabled",
            wan_repo_dir="/fake/Wan2.2",
            wan_conda_env="test_env",
        )
        registry.register(wan_backend)

        app = create_app(config, sample_store=SampleManifestStore(tmp_path), backend_registry=registry)

        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app),
                base_url="http://test",
            ) as c:
                resp = await c.get("/v1/backends")
                assert resp.status_code == 200
                backends = {b["name"]: b for b in resp.json()["backends"]}
                assert backends["wan-video"]["runner_mode"] == "official"

    def test_official_runner_builds_i2v_command_with_reference(self, tmp_path):
        from wm_infra.backends import WanVideoBackend
        from wm_infra.controlplane import ProduceSampleRequest

        backend = WanVideoBackend(
            str(tmp_path / "wan"),
            wan_engine_adapter="disabled",
            wan_repo_dir="/fake/Wan2.2",
            wan_conda_env="test_env",
        )
        request = ProduceSampleRequest.model_validate({
            "task_type": "image_to_video",
            "backend": "wan-video",
            "model": "wan2.2-i2v-A14B",
            "sample_spec": {"prompt": "animate this", "references": ["/tmp/input.png"]},
        })
        cmd = backend._build_official_command(request, "sample123", backend._resolve_wan_config(request), tmp_path / "out.mp4")
        assert "--task i2v-A14B" in cmd
        assert "--image /tmp/input.png" in cmd
        assert "--save_file" in cmd

    def test_official_runner_builds_i2v_command_with_file_uri_reference(self, tmp_path):
        from wm_infra.backends import WanVideoBackend
        from wm_infra.controlplane import ProduceSampleRequest

        backend = WanVideoBackend(
            str(tmp_path / "wan"),
            wan_engine_adapter="disabled",
            wan_repo_dir="/fake/Wan2.2",
            wan_conda_env="test_env",
        )
        request = ProduceSampleRequest.model_validate({
            "task_type": "image_to_video",
            "backend": "wan-video",
            "model": "wan2.2-i2v-A14B",
            "sample_spec": {"prompt": "animate this", "references": ["file:///tmp/input.png"]},
        })
        cmd = backend._build_official_command(request, "sample123", backend._resolve_wan_config(request), tmp_path / "out.mp4")
        assert "--image /tmp/input.png" in cmd


class TestStreaming:
    @pytest.mark.asyncio
    async def test_sse_streaming(self, client):
        async with client.stream(
            "POST",
            "/v1/rollout",
            json={
                "num_steps": 3,
                "return_latents": True,
                "return_frames": False,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            events = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        events.append("[DONE]")
                    else:
                        events.append(json.loads(payload))

            assert len(events) == 4
            assert events[-1] == "[DONE]"
            assert events[0]["step"] == 0
            assert events[1]["step"] == 1
            assert events[2]["step"] == 2
            assert events[0]["latent"] is not None


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "wm_request_total" in text or "wm_batch_size" in text


class TestErrors:
    @pytest.mark.asyncio
    async def test_invalid_num_steps(self, client):
        resp = await client.post("/v1/rollout", json={
            "num_steps": 0,
            "return_frames": False,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_job(self, client):
        resp = await client.get("/v1/rollout/nonexistent-id")
        assert resp.status_code == 404
