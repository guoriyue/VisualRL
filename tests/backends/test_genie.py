"""Tests for Genie runner and backend.

These tests run without the real Genie model — the runner operates in stub
mode and verifies that artifacts are correctly persisted, temporal records
are created, and the API layer reports the right info.
"""

import asyncio
import base64
import json
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import pytest_asyncio
import torch

from wm_infra.backends.genie_runner import GenieRunner, GenieRunResult, genie_available
from wm_infra.backends.genie import GenieRolloutBackend
from wm_infra.controlplane import (
    ArtifactKind,
    BranchCreate,
    CheckpointCreate,
    EpisodeCreate,
    ProduceSampleRequest,
    GenieTaskConfig,
    RolloutTaskConfig,
    SampleSpec,
    SampleStatus,
    StateHandleCreate,
    TaskType,
    TemporalRefs,
    TemporalStore,
    TokenInputSpec,
    TokenInputSource,
    TokenizerFamily,
    TokenizerKind,
)


def _force_stub_runner() -> GenieRunner:
    runner = GenieRunner()
    runner._mode = "stub"
    runner.load = lambda: "stub"  # type: ignore[method-assign]
    return runner


# ---------------------------------------------------------------------------
# GenieRunner unit tests
# ---------------------------------------------------------------------------


class TestGenieRunner:
    def test_stub_generates_tokens_npy(self, tmp_path):
        runner = _force_stub_runner()
        runner.load()
        result = runner.run(
            output_dir=tmp_path / "out",
            prompt="test prompt",
            seed=123,
            num_frames=16,
        )
        assert result.mode in ("stub", "real")
        assert result.tokens_path is not None
        assert result.state_path is not None

        # Verify tokens.npy is a valid numpy file with expected shape
        tokens = np.load(result.tokens_path)
        assert tokens.shape == (result.total_frames, result.spatial_h, result.spatial_w)
        assert tokens.dtype == np.uint32

        # Verify state.json is valid JSON
        state = json.loads(open(result.state_path).read())
        assert state["mode"] in ("stub", "real")
        assert state["seed"] == 123
        assert state["prompt"] == "test prompt"
        assert state["total_frames"] == result.total_frames
        assert state["frames_generated"] == result.frames_generated

    def test_stub_is_deterministic(self, tmp_path):
        runner = _force_stub_runner()
        runner.load()
        r1 = runner.run(output_dir=tmp_path / "a", prompt="hello", seed=42, num_frames=16)
        r2 = runner.run(output_dir=tmp_path / "b", prompt="hello", seed=42, num_frames=16)

        t1 = np.load(r1.tokens_path)
        t2 = np.load(r2.tokens_path)
        np.testing.assert_array_equal(t1, t2)

    def test_different_prompts_produce_different_tokens(self, tmp_path):
        runner = _force_stub_runner()
        runner.load()
        r1 = runner.run(output_dir=tmp_path / "a", prompt="cat", seed=42, num_frames=16)
        r2 = runner.run(output_dir=tmp_path / "b", prompt="dog", seed=42, num_frames=16)

        t1 = np.load(r1.tokens_path)
        t2 = np.load(r2.tokens_path)
        assert not np.array_equal(t1, t2)

    def test_result_fields(self, tmp_path):
        runner = _force_stub_runner()
        runner.load()
        result = runner.run(output_dir=tmp_path / "out", prompt="fields", seed=1)
        assert result.frames_generated > 0
        assert result.tokens_generated > 0
        assert result.prompt_frames >= 0
        assert result.total_frames > result.prompt_frames
        assert result.spatial_h == 16
        assert result.spatial_w == 16
        assert result.vocab_size == 262144
        assert result.elapsed_s >= 0
        assert result.error is None

    def test_run_window_batch_marks_batch_metadata_in_stub_mode(self):
        runner = _force_stub_runner()
        runner.load()
        prepared_a = runner.prepare_inputs(prompt="a", seed=1, num_frames=9, num_prompt_frames=4)
        prepared_b = runner.prepare_inputs(prompt="b", seed=2, num_frames=9, num_prompt_frames=4)

        results = runner.run_window_batch(
            [prepared_a, prepared_b],
            frame_start=4,
            frame_end=9,
        )

        assert len(results) == 2
        assert all(result.batch_size == 2 for result in results)
        assert all(result.batched is True for result in results)

    def test_run_accepts_per_request_overrides(self, tmp_path):
        runner = _force_stub_runner()
        runner.num_prompt_frames = 8
        runner.maskgit_steps = 2
        runner.temperature = 0.0
        runner.load()

        result = runner.run(
            output_dir=tmp_path / "out",
            prompt="override settings",
            seed=5,
            num_frames=12,
            num_prompt_frames=3,
            maskgit_steps=7,
            temperature=0.4,
        )

        state = json.loads((tmp_path / "out" / "state.json").read_text())
        assert result.total_frames == 12
        assert state["num_prompt_frames"] == 3
        assert state["maskgit_steps"] == 7
        assert state["temperature"] == 0.4
        assert runner.num_prompt_frames == 8
        assert runner.maskgit_steps == 2
        assert runner.temperature == 0.0

    def test_real_mode_failure_is_reported_without_stub_fallback(self, tmp_path, monkeypatch):
        runner = GenieRunner()
        runner._mode = "real"
        runner._model = object()

        def fake_run_real(output_dir, prompt, seed, num_frames, input_tokens, num_prompt_frames, maskgit_steps, temperature):
            return GenieRunResult(
                mode="real",
                tokens_generated=0,
                frames_generated=0,
                prompt_frames=num_prompt_frames,
                total_frames=num_frames,
                spatial_h=16,
                spatial_w=16,
                vocab_size=262144,
                elapsed_s=0.1,
                model_name="genie-local",
                device="cuda",
                error="unsupported device",
            )

        monkeypatch.setattr(runner, "_run_real", fake_run_real)

        result = runner.run(output_dir=tmp_path / "out", prompt="fallback", seed=7, num_frames=16)
        assert result.mode == "real"
        assert result.error == "unsupported device"
        assert result.extra == {}


# ---------------------------------------------------------------------------
# GenieRolloutBackend unit tests
# ---------------------------------------------------------------------------


class TestGenieRolloutBackend:
    def _make_backend(self, tmp_path):
        temporal_store = TemporalStore(tmp_path / "temporal")
        runner = _force_stub_runner()
        runner.load()
        backend = GenieRolloutBackend(
            temporal_store,
            output_root=tmp_path / "genie_output",
            runner=runner,
        )
        return backend, temporal_store

    def _make_real_mode_backend(self, tmp_path):
        temporal_store = TemporalStore(tmp_path / "temporal")
        runner = GenieRunner()
        runner._mode = "real"
        runner._model = SimpleNamespace(
            config=SimpleNamespace(T=16, image_vocab_size=1001),
            h=20,
            w=20,
        )
        runner.load = lambda: "real"  # type: ignore[method-assign]
        backend = GenieRolloutBackend(
            temporal_store,
            output_root=tmp_path / "genie_output",
            runner=runner,
        )
        return backend, temporal_store

    def _make_episode_and_state(self, temporal_store):
        episode = temporal_store.create_episode(EpisodeCreate(title="test ep"))
        branch = temporal_store.create_branch(
            BranchCreate(episode_id=episode.episode_id, name="main")
        )
        state = temporal_store.create_state_handle(
            StateHandleCreate(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                kind="latent",
                dtype="float16",
                shape=[16, 6],
            )
        )
        return episode, branch, state

    @pytest.mark.asyncio
    async def test_produce_sample_creates_artifacts_on_disk(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="test rollout"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            task_config=RolloutTaskConfig(num_steps=3),
        )

        record = await backend.produce_sample(request)
        assert record.status == SampleStatus.SUCCEEDED
        assert record.runtime["runner_mode"] in ("stub", "real")
        assert record.runtime["frames_generated"] > 0
        assert record.runtime["tokens_generated"] > 0
        assert record.metadata["runner_mode"] in ("stub", "real")
        assert "stage_timings_ms" in record.runtime
        assert record.runtime["stage_timings_ms"]["runner_load_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["state_token_prep_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["runner_exec_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["artifact_persist_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["temporal_persist_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["total_elapsed_ms"] >= record.runtime["stage_timings_ms"]["runner_exec_ms"]
        assert record.metadata["stage_timings_ms"] == record.runtime["stage_timings_ms"]
        assert record.runtime["stage_graph"] == [
            "admission",
            "state_materialize",
            "prompt_prepare",
            "transition",
            "checkpoint",
            "artifact_persist",
            "controlplane_commit",
        ]
        assert record.runtime["runtime_state"]["paged_state_key"].endswith(":paged-state")
        assert record.runtime["runtime_state"]["paged_state"]["page_count"] >= 1
        assert record.runtime["runtime_state"]["paged_state"]["page_pool"]["physical_bytes"] >= record.runtime["runtime_state"]["paged_state"]["page_pool"]["logical_bytes"]
        assert record.runtime["runtime_state"]["paged_state"]["page_pool"]["host_pool"]["page_count"] >= 1
        assert record.runtime["runtime_state"]["transfer_fast_path"]["page_window"]["pages_touched"] >= 1
        assert record.runtime["runtime_state"]["transfer_fast_path"]["pool_path"] in {"host_only", "host_to_gpu"}
        assert record.runtime["runtime_state"]["transfer_plan"]["staging_tier"] == "cpu_pinned_warm"
        assert record.runtime["runtime_state"]["transfer_plan"]["staging_bytes"] >= 0
        assert record.runtime["runtime_state"]["transfer_plan"]["d2h_bytes"] > 0

    def test_queue_batch_key_skips_multi_window_rollout(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="checkpoint-heavy rollout"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            task_config=RolloutTaskConfig(num_steps=4),
            genie_config=GenieTaskConfig(
                num_frames=16,
                num_prompt_frames=4,
                checkpoint_every_n_frames=4,
            ),
        )

        assert backend.queue_batch_key(request) is None

    def test_queue_batch_key_keeps_single_window_rollout(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="single window rollout"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            task_config=RolloutTaskConfig(num_steps=4),
            genie_config=GenieTaskConfig(
                num_frames=9,
                num_prompt_frames=4,
                checkpoint_every_n_frames=0,
            ),
        )

        assert backend.queue_batch_key(request) is not None

    def test_transition_batcher_uses_backend_config(self, tmp_path):
        temporal_store = TemporalStore(tmp_path / "temporal")
        runner = _force_stub_runner()
        runner.load()
        backend = GenieRolloutBackend(
            temporal_store,
            output_root=tmp_path / "genie_output",
            runner=runner,
            transition_max_batch_size=2,
            transition_batch_wait_ms=7.5,
        )

        assert backend._transition_batcher.max_batch_size == 2
        assert backend._transition_batcher.batch_wait_ms == 7.5

    def test_queue_batch_size_limit_caps_whole_request_batches(self, tmp_path):
        backend, _temporal_store = self._make_backend(tmp_path)

        assert backend.queue_batch_size_limit(1) == 1
        assert backend.queue_batch_size_limit(2) == 2
        assert backend.queue_batch_size_limit(8) == 2

    @pytest.mark.asyncio
    async def test_produce_sample_creates_temporal_records(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="temporal test"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            task_config=RolloutTaskConfig(num_steps=2),
        )

        record = await backend.produce_sample(request)

        # Verify rollout record
        rollout = temporal_store.rollouts.get(record.temporal.rollout_id)
        assert rollout is not None
        assert rollout.status.value == "succeeded"
        assert rollout.metrics["frames_generated"] > 0
        assert rollout.metrics["runner_load_ms"] >= 0
        assert rollout.metrics["state_token_prep_ms"] >= 0
        assert rollout.metrics["runner_exec_ms"] >= 0
        assert rollout.metrics["artifact_persist_ms"] >= 0
        assert rollout.metrics["temporal_persist_ms"] >= 0
        assert rollout.metrics["total_elapsed_ms"] >= rollout.metrics["runner_exec_ms"]
        assert rollout.metadata["stage_timings_ms"] == record.runtime["stage_timings_ms"]
        assert rollout.metadata["stage_graph"] == record.runtime["stage_graph"]
        assert rollout.metadata["stage_profile"] == record.runtime["stage_profile"]
        assert rollout.metadata["benchmark_profile"] == record.runtime["benchmark_profile"]
        assert f"{record.sample_id}:tokens" in rollout.artifact_ids

        # Verify checkpoint
        cp = temporal_store.checkpoints.get(record.temporal.checkpoint_id)
        assert cp is not None
        assert cp.metadata["runner_mode"] in ("stub", "real")

        # Verify checkpoint attached to rollout
        assert record.temporal.checkpoint_id in rollout.checkpoint_ids

    @pytest.mark.asyncio
    async def test_execute_job_batch_emits_consistent_runtime_profile(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        requests = [
            ProduceSampleRequest(
                task_type=TaskType.TEMPORAL_ROLLOUT,
                backend="genie-rollout",
                model="genie-local",
                sample_spec=SampleSpec(prompt=f"batched-{index}"),
                temporal=TemporalRefs(
                    episode_id=episode.episode_id,
                    branch_id=branch.branch_id,
                    state_handle_id=state.state_handle_id,
                ),
                task_config=RolloutTaskConfig(num_steps=2),
                genie_config=GenieTaskConfig(
                    num_frames=9,
                    num_prompt_frames=4,
                    checkpoint_every_n_frames=0,
                ),
            )
            for index in range(2)
        ]

        records = await backend.execute_job_batch(
            [(request, f"sample-batch-{index}") for index, request in enumerate(requests)]
        )

        assert len(records) == 2
        for record in records:
            assert record.runtime["scheduler"]["execution_path"] == "runner_window_batch"
            assert record.runtime["scheduler"]["transition_entities"] == 1
            assert record.runtime["scheduler"]["batched_across_requests"] is True
            assert record.runtime["scheduler"]["max_observed_batch_size"] == 2
            assert record.runtime["scheduler"]["cross_request_batcher"] is None
            assert record.runtime["benchmark_profile"]["max_observed_batch_size"] == 2
            assert record.runtime["benchmark_profile"]["chunk_count"] == 1
            assert record.runtime["stage_profile"]["completed_stages"] == [
                "admission",
                "state_materialize",
                "prompt_prepare",
                "transition",
                "artifact_persist",
                "controlplane_commit",
            ]
            assert record.runtime["stage_profile"]["stages"]["transition"]["count"] == 1
            assert record.runtime["stage_profile"]["stages"]["checkpoint"]["count"] == 0
            assert record.runtime["runtime_state"]["source_cache_key"] is not None
            rollout = temporal_store.rollouts.get(record.temporal.rollout_id)
            assert rollout is not None
            assert rollout.metadata["stage_profile"] == record.runtime["stage_profile"]
            assert rollout.metadata["benchmark_profile"] == record.runtime["benchmark_profile"]

    @pytest.mark.asyncio
    async def test_single_and_batched_execution_preserve_semantics(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        def make_request(prompt: str) -> ProduceSampleRequest:
            return ProduceSampleRequest(
                task_type=TaskType.TEMPORAL_ROLLOUT,
                backend="genie-rollout",
                model="genie-local",
                sample_spec=SampleSpec(prompt=prompt),
                temporal=TemporalRefs(
                    episode_id=episode.episode_id,
                    branch_id=branch.branch_id,
                    state_handle_id=state.state_handle_id,
                ),
                task_config=RolloutTaskConfig(num_steps=4),
                genie_config=GenieTaskConfig(
                    num_frames=16,
                    num_prompt_frames=4,
                    checkpoint_every_n_frames=4,
                ),
            )

        single = await backend.execute_job(make_request("single"), "sample-single")
        batched = await backend.execute_job_batch(
            [
                (make_request("batched-a"), "sample-batched-a"),
                (make_request("batched-b"), "sample-batched-b"),
            ]
        )

        assert len(batched) == 2
        reference = batched[0]
        assert single.status == reference.status == SampleStatus.SUCCEEDED
        assert single.runtime["stage_graph"] == reference.runtime["stage_graph"]
        assert single.runtime["benchmark_profile"].keys() == reference.runtime["benchmark_profile"].keys()
        assert set(a.kind for a in single.artifacts) == set(a.kind for a in reference.artifacts)
        assert {a.artifact_id.rsplit(":", 1)[-1] for a in single.artifacts} == {
            a.artifact_id.rsplit(":", 1)[-1] for a in reference.artifacts
        }
        assert len(single.runtime["checkpoint_deltas"]) == len(reference.runtime["checkpoint_deltas"])
        assert single.temporal.episode_id == reference.temporal.episode_id == episode.episode_id
        assert single.temporal.branch_id == reference.temporal.branch_id == branch.branch_id
        assert single.temporal.parent_state_handle_id == reference.temporal.parent_state_handle_id == state.state_handle_id
        assert single.temporal.rollout_id is not None
        assert reference.temporal.rollout_id is not None
        assert single.temporal.checkpoint_id is not None
        assert reference.temporal.checkpoint_id is not None
        assert single.temporal.state_handle_id is not None
        assert reference.temporal.state_handle_id is not None

        for record in [single, *batched]:
            rollout = temporal_store.rollouts.get(record.temporal.rollout_id)
            assert rollout is not None
            assert rollout.output_state_handle_id == record.temporal.state_handle_id
            assert record.temporal.checkpoint_id in rollout.checkpoint_ids
            assert any(artifact.artifact_id.endswith(":tokens") for artifact in record.artifacts)
            assert any(artifact.artifact_id.endswith(":checkpoint") for artifact in record.artifacts)
            assert any(artifact.artifact_id.endswith(":recovery") for artifact in record.artifacts)

    @pytest.mark.asyncio
    async def test_genie_config_drives_execution_and_round_trips(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="config roundtrip"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            task_config=RolloutTaskConfig(num_steps=2),
            genie_config=GenieTaskConfig(
                num_frames=10,
                num_prompt_frames=4,
                maskgit_steps=6,
                temperature=0.25,
            ),
        )

        record = await backend.produce_sample(request)
        assert record.genie_config is not None
        assert record.genie_config.num_frames == 10
        assert record.genie_config.num_prompt_frames == 4
        assert record.genie_config.maskgit_steps == 6
        assert record.genie_config.temperature == 0.25
        assert record.task_config is not None
        assert record.task_config.frame_count == 10
        assert record.runtime["genie_config"]["num_frames"] == 10
        assert record.runtime["genie_config"]["num_prompt_frames"] == 4

        state_payload = json.loads(Path(record.runtime["state_path"]).read_text())
        assert state_payload["total_frames"] == 10
        assert state_payload["num_prompt_frames"] == 4
        assert state_payload["maskgit_steps"] == 6
        assert state_payload["temperature"] == 0.25

    @pytest.mark.asyncio
    async def test_genie_config_accepts_base64_input_tokens(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        raw_tokens = np.arange(4 * 16 * 16, dtype=np.uint32).reshape(4, 16, 16)
        buffer = BytesIO()
        np.save(buffer, raw_tokens)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="b64 tokens"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(
                num_frames=8,
                num_prompt_frames=2,
                input_tokens_b64=encoded,
            ),
        )

        record = await backend.produce_sample(request)
        assert record.runtime["token_input"]["token_input_mode"] == "genie_config_b64"
        input_artifacts = [artifact for artifact in record.artifacts if artifact.artifact_id.endswith(":input-tokens")]
        assert len(input_artifacts) == 1
        output_tokens = np.load(record.runtime["tokens_path"])
        np.testing.assert_array_equal(output_tokens[:2], raw_tokens[:2])

    @pytest.mark.asyncio
    async def test_checkpoint_cadence_emits_intermediate_deltas(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="checkpoint cadence"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(num_frames=12, num_prompt_frames=4, checkpoint_every_n_frames=4),
        )

        record = await backend.produce_sample(request)
        assert len(record.runtime["checkpoint_deltas"]) == 1
        delta = record.runtime["checkpoint_deltas"][0]
        assert delta["frame_end"] == 8
        assert any(artifact.artifact_id.endswith(":checkpoint-delta:0008") for artifact in record.artifacts)

    @pytest.mark.asyncio
    async def test_real_mode_rejects_non_genie_tokenizer_kind(self, tmp_path):
        backend, temporal_store = self._make_real_mode_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="bad tokenizer"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(
                num_frames=8,
                num_prompt_frames=2,
                tokenizer_kind=TokenizerKind.MAGVIT2,
            ),
        )

        with pytest.raises(ValueError, match="tokenizer_kind=genie_stmaskgit"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_real_mode_rejects_magvit2_scaffold_tokens(self, tmp_path):
        backend, temporal_store = self._make_real_mode_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="bad scaffold"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(num_frames=8, num_prompt_frames=2),
            token_input=TokenInputSpec(
                source=TokenInputSource.INLINE,
                tokenizer_family=TokenizerFamily.MAGVIT2,
                layout="flat",
                shape=[4, 16, 16],
                inline_tokens=list(range(4 * 16 * 16)),
                dtype="uint32",
            ),
        )

        with pytest.raises(ValueError, match="tokenizer_family=raw"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_real_mode_rejects_incompatible_token_grid(self, tmp_path):
        backend, temporal_store = self._make_real_mode_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="bad shape"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(num_frames=8, num_prompt_frames=2),
            token_input=TokenInputSpec(
                source=TokenInputSource.INLINE,
                tokenizer_family=TokenizerFamily.RAW,
                layout="flat",
                shape=[4, 16, 16],
                inline_tokens=list(range(4 * 16 * 16)),
                dtype="uint32",
            ),
        )

        with pytest.raises(ValueError, match=r"spatial shape \[T,20,20\]"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_real_mode_rejects_too_many_frames(self, tmp_path):
        backend, temporal_store = self._make_real_mode_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="too long"),
            temporal=TemporalRefs(
                episode_id=episode.episode_id,
                branch_id=branch.branch_id,
                state_handle_id=state.state_handle_id,
            ),
            genie_config=GenieTaskConfig(num_frames=17, num_prompt_frames=4),
        )

        with pytest.raises(ValueError, match="num_frames <= 16"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_rejects_wrong_task_type(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.TEXT_TO_VIDEO,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="wrong type"),
            temporal=TemporalRefs(episode_id=episode.episode_id),
        )

        with pytest.raises(ValueError, match="temporal_rollout"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_rejects_missing_episode(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="no episode"),
            temporal=TemporalRefs(episode_id="nonexistent"),
        )

        with pytest.raises(ValueError, match="Unknown episode_id"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_rejects_missing_temporal(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)

        request = ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="genie-rollout",
            model="genie-local",
            sample_spec=SampleSpec(prompt="no temporal"),
        )

        with pytest.raises(ValueError, match="temporal.episode_id"):
            await backend.produce_sample(request)

    def test_runner_mode_property(self, tmp_path):
        backend, _ = self._make_backend(tmp_path)
        backend.ensure_runner_loaded()
        assert backend.runner_mode in ("stub", "real")


# ---------------------------------------------------------------------------
# Transition batcher tests (merged from test_genie_batcher.py)
# ---------------------------------------------------------------------------

from wm_infra.backends.genie_batcher import GenieTransitionBatcher
from wm_infra.backends.genie_runtime import GenieExecutionChunk, make_stage_signature


def _force_stub_runner() -> GenieRunner:
    runner = GenieRunner()
    runner._mode = "stub"
    runner.load = lambda: "stub"  # type: ignore[method-assign]
    return runner


def _transition_chunk() -> GenieExecutionChunk:
    signature = make_stage_signature(
        backend="genie-rollout",
        model_name="genie-local",
        stage="transition",
        device="cpu",
        dtype="uint32",
        tokenizer_kind="genie_stmaskgit",
        spatial_h=16,
        spatial_w=16,
        window_num_frames=4,
        num_prompt_frames=4,
        maskgit_steps=1,
        temperature=0.0,
        checkpoint_every_n_frames=4,
        runner_mode="stub",
        needs_persist=False,
    )
    return GenieExecutionChunk(
        chunk_id="transition:checkpoint_heavy:4:1",
        signature=signature,
        entity_ids=["sample-a:0008"],
        runnable_stage="transition",
        frame_range=(4, 8),
        estimated_vram_bytes=4 * 16 * 16 * 4,
        estimated_transfer_bytes=0,
        estimated_flops=4 * 16 * 16,
        queue_lane="checkpoint_heavy",
        expected_occupancy=0.5,
    )


@pytest.mark.asyncio
async def test_transition_batcher_batches_compatible_requests():
    runner = _force_stub_runner()
    runner.load()
    batcher = GenieTransitionBatcher(runner, max_batch_size=8, batch_wait_ms=10.0)

    prepared_a = runner.prepare_inputs(prompt="alpha", seed=1, num_frames=9, num_prompt_frames=4, maskgit_steps=1, temperature=0.0)
    prepared_b = runner.prepare_inputs(prompt="beta", seed=2, num_frames=9, num_prompt_frames=4, maskgit_steps=1, temperature=0.0)
    chunk = _transition_chunk()

    result_a, result_b = await asyncio.gather(
        batcher.run_transition(sample_id="sample-a", prepared=prepared_a, chunk=chunk),
        batcher.run_transition(sample_id="sample-b", prepared=prepared_b, chunk=chunk),
    )

    assert result_a.batch_size == 2
    assert result_b.batch_size == 2
    assert result_a.batch_id == result_b.batch_id
    assert set(result_a.sample_ids) == {"sample-a", "sample-b"}
    assert prepared_a.generated_until == 8
    assert prepared_b.generated_until == 8
    assert batcher.snapshot()["max_observed_batch_size"] == 2
