"""Focused tests for Wan runtime batching and warm profile hints."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from wm_infra.backends.wan import WanVideoBackend
from wm_infra.backends.wan_engine import (
    DiffusersWanI2VAdapter,
    OfficialWanInProcessAdapter,
    WanCompiledGraphManager,
    WanCompiledStageWorkload,
    WanExecutionContext,
    load_wan_engine_adapter,
)
from wm_infra.backends.wan_runtime import build_wan_transfer_plan
from wm_infra.controlplane import ProduceSampleRequest, SampleSpec, TaskType, WanTaskConfig


def _wan_request(
    *,
    width: int = 832,
    height: int = 480,
    frame_count: int = 9,
    num_steps: int = 4,
    guidance_scale: float = 4.0,
    high_noise_guidance_scale: float | None = None,
    sample_solver: str = "unipc",
) -> ProduceSampleRequest:
    return ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="wan runtime test"),
        wan_config=WanTaskConfig(
            width=width,
            height=height,
            frame_count=frame_count,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            high_noise_guidance_scale=high_noise_guidance_scale,
            sample_solver=sample_solver,
        ),
    )


class _FakeCapturedWorkload(WanCompiledStageWorkload):
    backend_name = "fake-captured"

    def __init__(self) -> None:
        self.captured = False
        self.replayed = 0

    def capture(self) -> tuple[bool, float, list[str]]:
        self.captured = True
        return True, 1.25, ["Captured fake stage workload."]

    def replay(self) -> bool:
        if not self.captured:
            return False
        self.replayed += 1
        return True

    def describe(self) -> dict[str, object]:
        return {"backend": self.backend_name, "captured": self.captured, "replayed": self.replayed}


def _wan_context(tmp_path: Path, *, sample_id: str = "sample") -> WanExecutionContext:
    request = _wan_request()
    return WanExecutionContext(
        sample_id=sample_id,
        request=request,
        wan_config=request.wan_config,
        sample_dir=tmp_path / sample_id,
        plan_path=tmp_path / f"{sample_id}-request.json",
        log_path=tmp_path / f"{sample_id}.log",
        video_path=tmp_path / f"{sample_id}.mp4",
        runtime_path=tmp_path / f"{sample_id}-runtime.json",
        batch_size=1,
        batch_index=0,
        batch_sample_ids=[sample_id],
        scheduler_payload={},
        engine_profile={
            "profile_id": "wan-profile-test",
            "compiled_profile": {"execution_family": {"cache_key": "wan-family"}},
        },
    )


def test_queue_batch_key_groups_same_shape_and_near_cfg(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", wan_engine_adapter="stub")

    key_a = backend.queue_batch_key(_wan_request(guidance_scale=4.1))
    key_b = backend.queue_batch_key(_wan_request(guidance_scale=4.2))
    key_c = backend.queue_batch_key(_wan_request(guidance_scale=4.8))
    key_d = backend.queue_batch_key(_wan_request(width=960))

    assert key_a == key_b
    assert key_a != key_c
    assert key_a != key_d


def test_queue_batch_key_separates_solver_and_high_noise_guidance(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", wan_engine_adapter="stub")

    key_unipc = backend.queue_batch_key(_wan_request(sample_solver="unipc", high_noise_guidance_scale=5.0))
    key_dpm = backend.queue_batch_key(_wan_request(sample_solver="dpm++", high_noise_guidance_scale=5.0))
    key_high_cfg = backend.queue_batch_key(_wan_request(sample_solver="unipc", high_noise_guidance_scale=7.0))

    assert key_unipc != key_dpm
    assert key_unipc != key_high_cfg


def test_conditioning_cache_key_includes_frame_count(tmp_path):
    request = ProduceSampleRequest(
        task_type=TaskType.IMAGE_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-i2v-A14B",
        sample_spec=SampleSpec(prompt="cache", references=["file:///tmp/ref.png"]),
        wan_config=WanTaskConfig(width=832, height=480, frame_count=5),
    )
    request_more_frames = request.model_copy(
        update={"wan_config": request.wan_config.model_copy(update={"frame_count": 9})},
        deep=True,
    )

    common_kwargs = {
        "sample_id": "sample",
        "sample_dir": tmp_path,
        "plan_path": tmp_path / "request.json",
        "log_path": tmp_path / "runner.log",
        "video_path": tmp_path / "sample.mp4",
        "runtime_path": tmp_path / "runtime.json",
        "batch_size": 1,
        "batch_index": 0,
        "batch_sample_ids": ["sample"],
        "scheduler_payload": {},
        "engine_profile": {},
    }
    first = WanExecutionContext(request=request, wan_config=request.wan_config, **common_kwargs)
    second = WanExecutionContext(
        request=request_more_frames,
        wan_config=request_more_frames.wan_config,
        **common_kwargs,
    )

    assert first.conditioning_cache_key != second.conditioning_cache_key


def test_official_adapter_rejects_incomplete_i2v_checkpoint_layout(tmp_path):
    checkpoint_dir = tmp_path / "Wan2.2-I2V-A14B"
    (checkpoint_dir / "google" / "umt5-xxl").mkdir(parents=True)
    (checkpoint_dir / "low_noise_model").mkdir()
    (checkpoint_dir / "high_noise_model").mkdir()
    (checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth").write_text("x", encoding="utf-8")
    (checkpoint_dir / "Wan2.1_VAE.pth").write_text("x", encoding="utf-8")
    (checkpoint_dir / "high_noise_model" / "diffusion_pytorch_model.safetensors.index.json").write_text("{}", encoding="utf-8")

    adapter = OfficialWanInProcessAdapter(repo_dir=tmp_path)
    config = SimpleNamespace(
        t5_checkpoint="models_t5_umt5-xxl-enc-bf16.pth",
        t5_tokenizer="google/umt5-xxl",
        vae_checkpoint="Wan2.1_VAE.pth",
        low_noise_checkpoint="low_noise_model",
        high_noise_checkpoint="high_noise_model",
    )

    with pytest.raises(FileNotFoundError, match="low-noise model weights are missing"):
        adapter._validate_checkpoint_layout("i2v-A14B", checkpoint_dir, config)


def test_real_adapters_build_compiled_stage_workloads(tmp_path):
    official = OfficialWanInProcessAdapter(repo_dir=tmp_path)
    diffusers = DiffusersWanI2VAdapter(default_model_dir=tmp_path)
    context = _wan_context(tmp_path, sample_id="sample-real")

    official_diffusion = official.build_compiled_stage_workload("diffusion", context, {})
    official_vae = official.build_compiled_stage_workload("vae_decode", context, {})
    diffusers_diffusion = diffusers.build_compiled_stage_workload("diffusion", context, {})
    diffusers_vae = diffusers.build_compiled_stage_workload("vae_decode", context, {})

    assert official_diffusion is not None
    assert official_vae is not None
    assert diffusers_diffusion is not None
    assert diffusers_vae is not None

    official_diffusion.bind_runtime_value((9, 60, 104, 16))
    diffusers_vae.bind_runtime_value((9, 480, 832, 3))
    official_profile = official_diffusion.execute()
    diffusers_profile = diffusers_vae.execute()

    assert official_profile["compute_backend"] in {"eager_stage_compute", "cuda_graph_replay"}
    assert official_profile["compute_shape"][0] == context.wan_config.frame_count
    assert official_profile["runtime_bound"] is True
    assert diffusers_profile["compute_backend"] in {"eager_stage_compute", "cuda_graph_replay"}
    assert diffusers_profile["compute_shape"][0] == context.wan_config.frame_count
    assert diffusers_profile["runtime_bound"] is True


@pytest.mark.asyncio
async def test_execute_job_batch_records_shared_scheduler_and_profile(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", max_batch_size=4, batch_wait_ms=2.0, wan_engine_adapter="stub")
    batch_records = await backend.execute_job_batch(
        [
            (_wan_request(), "sample-a"),
            (_wan_request(), "sample-b"),
        ]
    )

    assert len(batch_records) == 2
    assert batch_records[0].runtime["scheduler"]["batched_across_requests"] is True
    assert batch_records[0].runtime["scheduler"]["batch_id"] == batch_records[1].runtime["scheduler"]["batch_id"]
    assert batch_records[0].runtime["compiled_graph_pool"]["profile_id"] == batch_records[1].runtime["compiled_graph_pool"]["profile_id"]
    assert batch_records[0].runtime["compiled_graph_pool"]["compile_state"] == "cold_start"
    assert batch_records[0].runtime["execution_family"]["backend"] == "wan-video"
    assert batch_records[0].runtime["compiled_graph_pool"]["compiled_profile"]["execution_family"]["stage"] == "pipeline"
    assert batch_records[0].runtime["transfer_plan"]["staging_tier"] == "cpu_pinned_warm"
    assert batch_records[0].runtime["residency"][0]["tier"] == "gpu_hot"

    warm_record = await backend.execute_job(_wan_request(), "sample-c")
    assert warm_record.runtime["compiled_graph_pool"]["warm_profile_hit"] is True
    assert warm_record.runtime["compiled_graph_pool"]["compile_state"] in {
        "warm_profile_new_batch_size",
        "warm_profile_batch_hit",
    }


@pytest.mark.asyncio
async def test_in_process_stub_scheduler_records_pipeline_and_stage_metadata(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", wan_engine_adapter="stub")

    record = await backend.execute_job(_wan_request(), "sample-stage")

    assert record.runtime["execution_backend"] == "in_process_stage_scheduler"
    assert record.runtime["engine"]["name"] == "stub-wan-engine"
    assert record.runtime["pipeline"]["execution_backend"] == "in_process_stage_scheduler"
    assert record.runtime["pipeline"]["stage_count"] >= 6
    assert record.runtime["pipeline"]["stage_sequence"][0] == "text_encode"
    assert record.runtime["pipeline"]["stage_sequence"][-1] == "persist"
    assert record.runtime["pipeline"]["compiled_graph_lifecycle"]["graph_count"] >= 1
    assert record.runtime["stages"][0]["name"] == "text_encode"
    assert record.runtime["stages"][2]["name"] == "vae_decode" or record.runtime["stages"][2]["name"] == "diffusion"
    assert record.runtime["stage_state"]["latent_id"].startswith("latent-")
    assert record.metadata["engine_adapter"] == "stub-wan-engine"
    assert record.runtime["execution_family"]["layout_key"].startswith("latent:")
    assert record.runtime["compiled_graph_pool"]["compiled_profile"]["graph_key"]
    assert record.runtime["stages"][2]["compiled_graph"]["graph_id"].startswith("wan-graph:")
    assert record.runtime["stages"][2]["compiled_graph"]["capture_state"] in {"warmup", "capture_ready", "captured", "replayed", "reused"}
    assert "replay_count" in record.runtime["stages"][2]["compiled_graph"]
    diffusion_stage = next(stage for stage in record.runtime["stages"] if stage["name"] == "diffusion")
    vae_stage = next(stage for stage in record.runtime["stages"] if stage["name"] == "vae_decode")
    assert diffusion_stage["outputs"]["compute_profile"]["compute_backend"] in {"eager_stage_compute", "cuda_graph_replay"}
    assert vae_stage["outputs"]["compute_profile"]["compute_backend"] in {"eager_stage_compute", "cuda_graph_replay"}


@pytest.mark.asyncio
async def test_stub_scheduler_reuses_stage_graph_family_across_requests(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", wan_engine_adapter="stub")

    first = await backend.execute_job(_wan_request(), "sample-graph-a")
    second = await backend.execute_job(_wan_request(), "sample-graph-b")

    first_diffusion = next(stage for stage in first.runtime["stages"] if stage["name"] == "diffusion")
    second_diffusion = next(stage for stage in second.runtime["stages"] if stage["name"] == "diffusion")

    assert first_diffusion["compiled_graph"]["graph_key"] == second_diffusion["compiled_graph"]["graph_key"]
    assert second_diffusion["compiled_graph"]["capture_state"] in {"reused", "captured", "replayed"}
    assert second_diffusion["compiled_graph"]["reuse_count"] >= 1


@pytest.mark.asyncio
async def test_execute_job_batch_accepts_near_shape_requests_selected_by_score(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", max_batch_size=4, batch_wait_ms=2.0, wan_engine_adapter="stub")

    records = await backend.execute_job_batch(
        [
            (_wan_request(num_steps=4, guidance_scale=4.0), "sample-a"),
            (_wan_request(num_steps=5, guidance_scale=4.5), "sample-b"),
        ]
    )

    assert len(records) == 2
    assert records[0].runtime["scheduler"]["batch_size"] == 2
    assert records[1].runtime["scheduler"]["batch_size"] == 2
    assert records[0].runtime["compiled_graph_pool"]["profile_id"] == records[1].runtime["compiled_graph_pool"]["profile_id"]


def test_compiled_graph_manager_captures_and_replays_registered_workload(tmp_path):
    manager = WanCompiledGraphManager(enable_cuda_graphs=True, warmup_runs=1)
    context = _wan_context(tmp_path)
    workload = _FakeCapturedWorkload()

    first = manager.begin_stage("diffusion", "cuda:0", context)
    first_done = manager.finish_stage(first["graph_key"], workload=workload)
    assert first_done["capture_state"] == "warmup"
    assert first_done["capture_count"] == 0

    second = manager.begin_stage("diffusion", "cuda:0", context)
    assert second["capture_state"] == "capture_ready"
    second_done = manager.finish_stage(second["graph_key"], workload=workload)
    assert second_done["capture_state"] == "captured"
    assert second_done["capture_count"] == 1
    assert second_done["capture_backend"] == "fake-captured"
    assert second_done["capture_latency_ms"] == 1.25

    third = manager.begin_stage("diffusion", "cuda:0", context)
    replay = manager.maybe_replay(third["graph_key"])
    assert replay is not None
    assert replay["capture_state"] == "replayed"
    assert replay["replay_count"] == 1


def test_admission_quality_cost_hints_offer_step_and_preview_fallbacks(tmp_path):
    backend = WanVideoBackend(
        tmp_path / "wan",
        wan_engine_adapter="stub",
        wan_admission_max_vram_gb=10.0,
        wan_admission_max_units=6.0,
    )
    request = _wan_request(width=1280, height=720, frame_count=21, num_steps=12)
    wan_config = backend._resolve_wan_config(request)

    admitted, admission, _estimate = backend._admission_result(request, wan_config)

    assert admitted is False
    policies = {item["policy"] for item in admission["quality_cost_hints"]["suggested_adjustments"]}
    assert "auto_step_reduction" in policies
    assert "progressive_preview" in policies


def test_load_diffusers_i2v_adapter_alias(tmp_path):
    adapter = load_wan_engine_adapter(
        "diffusers-i2v",
        i2v_diffusers_dir=tmp_path,
    )

    assert isinstance(adapter, DiffusersWanI2VAdapter)
    assert adapter.describe()["name"] == "wan22-diffusers-i2v"


def test_transfer_plan_accepts_string_reference_uris():
    request = ProduceSampleRequest(
        task_type=TaskType.IMAGE_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-i2v-A14B",
        sample_spec=SampleSpec(
            prompt="wan runtime test",
            references=["file:///tmp/reference.png"],
        ),
        wan_config=WanTaskConfig(width=832, height=480, frame_count=5),
    )

    plan = build_wan_transfer_plan(wan_config=request.wan_config, request=request)

    assert plan.overlap_h2d_with_compute is True


def test_backend_auto_selects_hybrid_adapter_when_repo_and_diffusers_i2v_are_available(tmp_path):
    repo_dir = tmp_path / "Wan2.2"
    repo_dir.mkdir()
    diffusers_dir = tmp_path / "Wan2.2-I2V-A14B-Diffusers"
    diffusers_dir.mkdir()
    (diffusers_dir / "model_index.json").write_text("{}", encoding="utf-8")

    backend = WanVideoBackend(
        tmp_path / "wan",
        wan_repo_dir=str(repo_dir),
        wan_i2v_diffusers_dir=str(diffusers_dir),
    )

    assert backend.runner_mode == "real"
    assert backend.execution_backend == "in_process_stage_scheduler"
    assert backend.engine_adapter is not None
    description = backend.engine_adapter.describe()
    assert description["name"] == "hybrid-wan-engine"
    assert description["delegates"]["text_to_video"]["name"] == "wan22-official-python"
    assert description["delegates"]["image_to_video"]["name"] == "wan22-diffusers-i2v"


# ---------------------------------------------------------------------------
# Wan truthfulness tests (merged from test_wan_truthfulness.py)
# ---------------------------------------------------------------------------

from wm_infra.controlplane import ArtifactKind, SampleStatus


@pytest.mark.asyncio
async def test_stub_mode_does_not_emit_video_artifact(tmp_path):
    backend = WanVideoBackend(str(tmp_path / "wan"), wan_engine_adapter="stub")
    request = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="stub truthfulness"),
    )

    record = await backend.produce_sample(request)

    assert record.status == SampleStatus.ACCEPTED
    assert record.runtime["runner"] == "stub"
    assert record.metadata["stubbed"] is True
    assert not (tmp_path / "wan" / record.sample_id / "sample.mp4").exists()

    artifact_kinds = {artifact.kind for artifact in record.artifacts}
    assert ArtifactKind.VIDEO not in artifact_kinds
    assert {ArtifactKind.LOG, ArtifactKind.METADATA}.issubset(artifact_kinds)


@pytest.mark.asyncio
async def test_stub_batch_records_scheduler_and_warm_pool_state(tmp_path):
    backend = WanVideoBackend(
        str(tmp_path / "wan"),
        max_batch_size=4,
        prewarm_common_signatures=False,
        wan_engine_adapter="stub",
    )
    request = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="batched stub"),
    )

    records = await backend.execute_job_batch(
        [
            (request.model_copy(deep=True), "sample-a"),
            (request.model_copy(deep=True), "sample-b"),
        ]
    )

    assert len(records) == 2
    assert records[0].metadata["queue_batched"] is True
    assert records[0].runtime["scheduler"]["batch_size"] == 2
    assert records[0].runtime["scheduler"]["batched_across_requests"] is True
    assert records[0].runtime["compiled_graph_pool"]["warm_profile_hit"] is False
    assert records[0].runtime["engine_pool_snapshot"]["profiles"] == 1
    assert records[0].runtime["compiled_graph_pool"]["execution_family"]["batch_size_family"] == "pair"
    assert records[0].runtime["transfer_plan"]["total_bytes"] >= records[0].runtime["transfer_plan"]["artifact_io_bytes"]

    warm_followup = await backend.execute_job(request.model_copy(deep=True), "sample-c")
    assert warm_followup.runtime["compiled_graph_pool"]["warm_profile_hit"] is True


# ---------------------------------------------------------------------------
# Wan batching tests (merged from test_wan_batching.py)
# ---------------------------------------------------------------------------


def _wan_batch_request(
    *,
    prompt: str = "A corgi runs through a data center.",
    num_steps: int = 4,
    frame_count: int = 9,
    width: int = 832,
    height: int = 480,
    guidance_scale: float = 4.0,
    shift: float = 12.0,
) -> ProduceSampleRequest:
    return ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt=prompt),
        wan_config=WanTaskConfig(
            num_steps=num_steps,
            frame_count=frame_count,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            shift=shift,
        ),
    )


def test_queue_batch_score_allows_nearby_wan_shapes(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", wan_engine_adapter="stub")

    reference = _wan_batch_request(num_steps=4, guidance_scale=4.0)
    nearby = _wan_batch_request(num_steps=5, guidance_scale=4.5)

    assert backend.queue_batch_key(reference) != backend.queue_batch_key(nearby)
    assert backend.queue_batch_score(reference, nearby) is not None
    assert backend.queue_batch_score(reference, nearby) > 0


@pytest.mark.asyncio
async def test_execute_job_batch_records_scheduler_and_warm_pool_metadata(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", prewarm_common_signatures=True, wan_engine_adapter="stub")

    records = await backend.execute_job_batch(
        [
            (_wan_batch_request(prompt="batch-a"), "sample-a"),
            (_wan_batch_request(prompt="batch-b"), "sample-b"),
        ]
    )

    assert len(records) == 2
    for record in records:
        assert record.metadata["queue_batched"] is True
        assert record.runtime["scheduler"]["batch_size"] == 2
        assert record.runtime["scheduler"]["execution_mode"] == "queue_coalesced_serial"
        assert record.runtime["compiled_graph_pool"]["compile_state"] == "prewarmed"
        assert record.runtime["engine_pool_snapshot"]["prewarmed_profiles"] >= 1
        assert record.runtime["scheduler"]["batch_signature"]["width"] == 832
        assert record.runtime["scheduler"]["batch_signature"]["num_steps"] == 4
        assert record.runtime["pipeline"]["compiled_graph_lifecycle"]["graph_count"] >= 1


def test_queue_batch_size_limit_respects_backend_cap(tmp_path):
    backend = WanVideoBackend(tmp_path / "wan", max_batch_size=3, wan_engine_adapter="stub")

    assert backend.queue_batch_size_limit(1) == 1
    assert backend.queue_batch_size_limit(3) == 3
    assert backend.queue_batch_size_limit(8) == 3
