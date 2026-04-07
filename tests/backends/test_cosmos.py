import json
from pathlib import Path

import pytest

from wm_infra.backends.cosmos import CosmosPredictBackend
from wm_infra.backends.cosmos_runner import CosmosRunner
from wm_infra.backends.job_queue import CosmosJobQueue
from wm_infra.controlplane import (
    CosmosTaskConfig,
    ProduceSampleRequest,
    RolloutTaskConfig,
    SampleManifestStore,
    SampleSpec,
    SampleStatus,
    TaskType,
)


def _cosmos_request() -> ProduceSampleRequest:
    return ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="cosmos-predict",
        model="cosmos-predict1-7b-text2world",
        sample_spec=SampleSpec(prompt="A forklift moves through a warehouse.", width=1024, height=640),
        task_config=RolloutTaskConfig(num_steps=12, frame_count=16, width=1024, height=640),
        cosmos_config=CosmosTaskConfig(variant="predict1_text2world", model_size="7B", frames_per_second=16),
    )


class TestCosmosRunner:
    def test_stub_mode_writes_deterministic_video_bytes(self, tmp_path):
        runner = CosmosRunner()
        request = _cosmos_request()
        task_config = request.task_config
        cosmos_config = request.cosmos_config
        assert task_config is not None
        assert cosmos_config is not None

        result_a = runner.run(output_dir=tmp_path / "a", request=request, task_config=task_config, cosmos_config=cosmos_config)
        result_b = runner.run(output_dir=tmp_path / "b", request=request, task_config=task_config, cosmos_config=cosmos_config)

        assert result_a.mode == "stub"
        assert Path(result_a.output_path).read_bytes() == Path(result_b.output_path).read_bytes()

    def test_nim_payload_uses_official_video_param_names(self):
        runner = CosmosRunner(base_url="http://127.0.0.1:8000")
        request = _cosmos_request()
        task_config = request.task_config
        cosmos_config = request.cosmos_config
        assert task_config is not None
        assert cosmos_config is not None

        payload = runner._nim_payload(request, task_config, cosmos_config)
        assert payload["video_params"] == {
            "width": 1024,
            "height": 640,
            "frames_count": 16,
            "frames_per_sec": 16,
        }
        assert "seed" not in payload or payload["seed"] is None


class TestCosmosBackend:
    @pytest.mark.asyncio
    async def test_produce_sample_writes_video_and_runtime_artifacts(self, tmp_path):
        backend = CosmosPredictBackend(tmp_path / "cosmos_output", runner=CosmosRunner())
        record = await backend.produce_sample(_cosmos_request())

        assert record.status == SampleStatus.SUCCEEDED
        assert record.cosmos_config is not None
        assert record.runtime["runner_mode"] == "stub"
        assert record.runtime["chunk_summary"]["max_chunk_size"] == 1
        assert "infer" in record.runtime["stage_timings_ms"]
        assert any(artifact.artifact_id.endswith(":video") for artifact in record.artifacts)

        runtime_artifact = next(artifact for artifact in record.artifacts if artifact.artifact_id.endswith(":metadata"))
        runtime_path = Path(runtime_artifact.uri[7:])
        payload = json.loads(runtime_path.read_text())
        assert payload["runner_mode"] == "stub"

    @pytest.mark.asyncio
    async def test_repeated_requests_observe_reference_reuse_hit(self, tmp_path):
        backend = CosmosPredictBackend(tmp_path / "cosmos_output", runner=CosmosRunner())

        first = await backend.produce_sample(_cosmos_request())
        second = await backend.produce_sample(_cosmos_request())

        assert first.runtime["reference_reuse_hit"] is False
        assert second.runtime["reference_reuse_hit"] is True
        assert second.runtime["cache"]["hits"] >= 1
        assert second.runtime["scheduler"]["prompt_or_reference_hot"] is True

    @pytest.mark.asyncio
    async def test_submit_async_returns_queued_record_and_worker_executes(self, tmp_path):
        store = SampleManifestStore(tmp_path / "manifests")
        backend = CosmosPredictBackend(tmp_path / "cosmos_output", runner=CosmosRunner())
        queue = CosmosJobQueue(execute_fn=backend.execute_job, store=store, queue_name="cosmos", max_queue_size=8, max_concurrent=1)
        backend._job_queue = queue
        queue.start()
        try:
            queued = backend.submit_async(_cosmos_request())
            assert queued.status == SampleStatus.QUEUED
            store.put(queued)

            for _ in range(50):
                record = store.get(queued.sample_id)
                if record is not None and record.status in {SampleStatus.SUCCEEDED, SampleStatus.FAILED}:
                    break
                await __import__("asyncio").sleep(0.05)

            record = store.get(queued.sample_id)
            assert record is not None
            assert record.status == SampleStatus.SUCCEEDED
        finally:
            await queue.stop()
