"""Tests for Genie runner and backend.

These tests run without the real Genie model — the runner operates in stub
mode and verifies that artifacts are correctly persisted, temporal records
are created, and the API layer reports the right info.
"""

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
    def test_stub_mode_when_genie_unavailable(self):
        runner = GenieRunner()
        mode = runner.load()
        # In CI without genie deps, this must be "stub"
        assert mode in ("stub", "real")

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

    @pytest.mark.skipif(not genie_available() or not torch.cuda.is_available(), reason="Real Genie CUDA path unavailable")
    def test_real_mode_supports_shorter_requested_windows(self, tmp_path):
        runner = GenieRunner(maskgit_steps=1)
        if runner.load() != "real":
            pytest.skip("Genie model did not load in real mode")

        result = runner.run(
            output_dir=tmp_path / "real",
            prompt="short real rollout",
            seed=3,
            num_frames=9,
            num_prompt_frames=8,
            maskgit_steps=1,
            temperature=0.0,
        )

        assert result.mode == "real"
        assert result.error is None
        assert result.total_frames == 9
        tokens = np.load(result.tokens_path)
        assert tokens.shape == (9, result.spatial_h, result.spatial_w)


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
        assert "stage_timings_ms" in record.runtime
        assert record.runtime["stage_timings_ms"]["runner_load_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["state_token_prep_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["runner_exec_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["artifact_persist_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["temporal_persist_ms"] >= 0
        assert record.runtime["stage_timings_ms"]["total_elapsed_ms"] >= record.runtime["stage_timings_ms"]["runner_exec_ms"]
        assert record.metadata["stage_timings_ms"] == record.runtime["stage_timings_ms"]

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
        assert rollout.metrics["runner_load_ms"] >= 0
        assert rollout.metrics["state_token_prep_ms"] >= 0
        assert rollout.metrics["runner_exec_ms"] >= 0
        assert rollout.metrics["artifact_persist_ms"] >= 0
        assert rollout.metrics["temporal_persist_ms"] >= 0
        assert rollout.metrics["total_elapsed_ms"] >= rollout.metrics["runner_exec_ms"]
        assert rollout.metadata["stage_timings_ms"] == record.runtime["stage_timings_ms"]
        assert f"{record.sample_id}:tokens" in rollout.artifact_ids

        # Verify checkpoint
        cp = temporal_store.checkpoints.get(record.temporal.checkpoint_id)
        assert cp is not None
        assert cp.metadata["runner_mode"] in ("stub", "real")

        # Verify checkpoint attached to rollout
        assert record.temporal.checkpoint_id in rollout.checkpoint_ids

    @pytest.mark.asyncio
    async def test_genie_config_drives_execution_and_round_trips(self, tmp_path):
        backend, temporal_store = self._make_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.GENIE_ROLLOUT,
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
            task_type=TaskType.GENIE_ROLLOUT,
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
    async def test_real_mode_rejects_non_genie_tokenizer_kind(self, tmp_path):
        backend, temporal_store = self._make_real_mode_backend(tmp_path)
        episode, branch, state = self._make_episode_and_state(temporal_store)

        request = ProduceSampleRequest(
            task_type=TaskType.GENIE_ROLLOUT,
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
            task_type=TaskType.GENIE_ROLLOUT,
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
            task_type=TaskType.GENIE_ROLLOUT,
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
            task_type=TaskType.GENIE_ROLLOUT,
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
