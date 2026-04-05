"""Tests for Genie runner and backend.

These tests run without the real Genie model — the runner operates in stub
mode and verifies that artifacts are correctly persisted, temporal records
are created, and the API layer reports the right info.
"""

import json

import numpy as np
import pytest
import pytest_asyncio

from wm_infra.backends.genie_runner import GenieRunner, GenieRunResult, genie_available
from wm_infra.backends.genie import GenieRolloutBackend
from wm_infra.controlplane import (
    ArtifactKind,
    BranchCreate,
    CheckpointCreate,
    EpisodeCreate,
    ProduceSampleRequest,
    RolloutTaskConfig,
    SampleSpec,
    SampleStatus,
    StateHandleCreate,
    TaskType,
    TemporalRefs,
    TemporalStore,
)


# ---------------------------------------------------------------------------
# GenieRunner unit tests
# ---------------------------------------------------------------------------


class TestGenieRunner:
    def test_stub_mode_when_genie_unavailable(self):
        runner = GenieRunner()
        mode = runner.load()
        # In CI without genie deps, this must be "stub"
        assert mode in ("stub", "real")

    def test_stub_generates_tokens_npy(self, tmp_path):
        runner = GenieRunner()
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
        runner = GenieRunner()
        runner.load()
        r1 = runner.run(output_dir=tmp_path / "a", prompt="hello", seed=42, num_frames=16)
        r2 = runner.run(output_dir=tmp_path / "b", prompt="hello", seed=42, num_frames=16)

        t1 = np.load(r1.tokens_path)
        t2 = np.load(r2.tokens_path)
        np.testing.assert_array_equal(t1, t2)

    def test_different_prompts_produce_different_tokens(self, tmp_path):
        runner = GenieRunner()
        runner.load()
        r1 = runner.run(output_dir=tmp_path / "a", prompt="cat", seed=42, num_frames=16)
        r2 = runner.run(output_dir=tmp_path / "b", prompt="dog", seed=42, num_frames=16)

        t1 = np.load(r1.tokens_path)
        t2 = np.load(r2.tokens_path)
        assert not np.array_equal(t1, t2)

    def test_result_fields(self, tmp_path):
        runner = GenieRunner()
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

    def test_real_mode_failure_falls_back_to_stub(self, tmp_path, monkeypatch):
        runner = GenieRunner()
        runner._mode = "real"
        runner._model = object()

        def fake_run_real(output_dir, prompt, seed, num_frames, input_tokens):
            return GenieRunResult(
                mode="real",
                tokens_generated=0,
                frames_generated=0,
                prompt_frames=8,
                total_frames=16,
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
        assert result.mode == "stub"
        assert result.error is None
        assert result.extra["fallback_from"] == "real"
        assert result.extra["fallback_error"] == "unsupported device"
        tokens = np.load(result.tokens_path)
        assert tokens.shape == (result.total_frames, result.spatial_h, result.spatial_w)


# ---------------------------------------------------------------------------
# GenieRolloutBackend unit tests
# ---------------------------------------------------------------------------


class TestGenieRolloutBackend:
    def _make_backend(self, tmp_path):
        temporal_store = TemporalStore(tmp_path / "temporal")
        runner = GenieRunner()
        runner.load()
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
            task_type=TaskType.GENIE_ROLLOUT,
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

        # Verify artifacts
        artifact_kinds = {a.kind for a in record.artifacts}
        assert ArtifactKind.LOG in artifact_kinds
        assert ArtifactKind.METADATA in artifact_kinds
        assert ArtifactKind.LATENT in artifact_kinds  # tokens
        assert any(a.artifact_id.endswith(":checkpoint") for a in record.artifacts)
        assert any(a.artifact_id.endswith(":recovery") for a in record.artifacts)

        # Verify token artifact points to real file
        token_arts = [a for a in record.artifacts if a.artifact_id.endswith(":tokens")]
        assert len(token_arts) == 1
        assert token_arts[0].uri.startswith("file://")
        token_path = token_arts[0].uri[7:]
        tokens = np.load(token_path)
        assert tokens.shape[0] > 0  # has frames
        assert tokens.dtype == np.uint32

        # Verify state handle has URI
        sh = temporal_store.state_handles.get(record.temporal.state_handle_id)
        assert sh is not None
        assert sh.uri is not None
        assert sh.uri.startswith("file://")
        assert sh.dtype == "uint32"
        assert sh.checkpoint_id == record.temporal.checkpoint_id
        assert f"{record.sample_id}:checkpoint" in sh.artifact_ids

    @pytest.mark.asyncio
    async def test_produce_sample_creates_temporal_records(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.GENIE_ROLLOUT,
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
        assert f"{record.sample_id}:tokens" in rollout.artifact_ids

        # Verify checkpoint
        cp = temporal_store.checkpoints.get(record.temporal.checkpoint_id)
        assert cp is not None
        assert cp.metadata["runner_mode"] in ("stub", "real")

        # Verify checkpoint attached to rollout
        assert record.temporal.checkpoint_id in rollout.checkpoint_ids

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

        with pytest.raises(ValueError, match="rollout-style"):
            await backend.produce_sample(request)

    @pytest.mark.asyncio
    async def test_rejects_missing_episode(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)

        request = ProduceSampleRequest(
            task_type=TaskType.GENIE_ROLLOUT,
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
            task_type=TaskType.GENIE_ROLLOUT,
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
