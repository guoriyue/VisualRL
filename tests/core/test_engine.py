"""Tests for the world model engine end-to-end."""

import asyncio

import pytest
import torch

from wm_infra.config import EngineConfig, DynamicsConfig, TokenizerConfig, StateCacheConfig
from wm_infra.rollout_engine import AsyncWorldModelEngine, LatentStateManager, RolloutJob, RolloutRequest, RolloutScheduler, WorldModelEngine
from wm_infra.models.dynamics import LatentDynamicsModel
from wm_infra.models.base import RolloutInput
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer, FSQQuantizer


# ─── Fixtures ───


def _small_config() -> EngineConfig:
    """Minimal config for fast CPU testing."""
    return EngineConfig(
        device="cpu" if not torch.cuda.is_available() else "cuda",
        dtype="float32",
        dynamics=DynamicsConfig(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            action_dim=8,
            latent_token_dim=6,  # matches FSQ dim
            max_rollout_steps=16,
        ),
        tokenizer=TokenizerConfig(
            spatial_downsample=2,  # small for testing
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        ),
        state_cache=StateCacheConfig(
            max_batch_size=4,
            max_rollout_steps=16,
            latent_dim=6,
            num_latent_tokens=16,
            pool_size_gb=0.1,
        ),
    )


# ─── Unit Tests ───


class TestFSQQuantizer:
    def test_roundtrip(self):
        levels = [4, 4, 4, 3, 3, 3]
        fsq = FSQQuantizer(levels)
        z = torch.randn(2, 10, len(levels))
        z_q, indices = fsq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (2, 10)

        # Decode indices should match quantized values
        z_decoded = fsq.decode_indices(indices)
        # After quantization + decode, values should be close (within grid)
        assert z_decoded.shape == z_q.shape

    def test_codebook_size(self):
        levels = [8, 8, 8, 5, 5, 5]
        fsq = FSQQuantizer(levels)
        assert fsq.codebook_size == 8 * 8 * 8 * 5 * 5 * 5

    def test_gradient_flows(self):
        fsq = FSQQuantizer([4, 4, 4])
        z = torch.randn(1, 5, 3, requires_grad=True)
        z_q, _ = fsq(z)
        loss = z_q.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape


class TestVideoTokenizer:
    def test_encode_decode_shape(self):
        config = TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=16,
            fsq_levels=[4, 4, 4, 3, 3, 3],
        )
        tok = VideoTokenizer(config)

        video = torch.randn(1, 2, 3, 8, 8)  # [B, T, C, H, W]
        z_q, indices = tok.encode(video)

        assert z_q.ndim == 4  # [B, T', N, D]
        assert indices.ndim == 3  # [B, T', N]

    def test_single_frame(self):
        config = TokenizerConfig(
            spatial_downsample=2,
            temporal_downsample=1,
            latent_channels=8,
            fsq_levels=[4, 4, 4],
        )
        tok = VideoTokenizer(config)
        frame = torch.randn(1, 3, 8, 8)
        z_q, indices = tok.encode_frame(frame)
        assert z_q.ndim == 4


class TestLatentDynamicsModel:
    def test_predict_next_shape(self):
        config = DynamicsConfig(
            hidden_dim=64, num_heads=4, num_layers=2,
            action_dim=8, latent_token_dim=6,
        )
        model = LatentDynamicsModel(config)
        state = torch.randn(1, 16, 6)  # [B, N, D]
        action = torch.randn(1, 8)  # [B, A]

        next_state = model.predict_next(state, action)
        assert next_state.shape == state.shape

    def test_rollout(self):
        config = DynamicsConfig(
            hidden_dim=64, num_heads=4, num_layers=2,
            action_dim=8, latent_token_dim=6,
        )
        model = LatentDynamicsModel(config)
        model.eval()

        inp = RolloutInput(
            latent_state=torch.randn(1, 16, 6),
            actions=torch.randn(1, 4, 8),
            num_steps=4,
        )
        with torch.inference_mode():
            out = model.rollout(inp)

        assert out.predicted_states.shape == (1, 4, 16, 6)


class TestLatentStateManager:
    def test_create_and_append(self):
        mgr = LatentStateManager(max_concurrent=4, max_memory_gb=0.01, device="cpu")
        state = mgr.create("r1", torch.randn(16, 6), max_steps=5)
        assert state.current_step == 0
        assert mgr.num_active == 1

        mgr.append_step("r1", torch.randn(8), torch.randn(16, 6))
        state = mgr.get("r1")
        assert state.current_step == 1
        assert len(state.latent_states) == 2

    def test_fork(self):
        mgr = LatentStateManager(max_concurrent=4, device="cpu")
        mgr.create("r1", torch.randn(16, 6), max_steps=10)
        mgr.append_step("r1", torch.randn(8), torch.randn(16, 6))
        memory_before_fork = mgr._current_memory

        forked = mgr.fork("r1", "r1_fork")
        assert forked.current_step == 1
        assert len(forked.latent_states) == 2
        assert mgr.num_active == 2
        assert forked.latent_states[0] is mgr.get("r1").latent_states[0]
        assert mgr._current_memory == memory_before_fork
        assert mgr.stats_snapshot()["tensor_reuse_hits"] > 0

    def test_deep_copy_fork_mode_allocates_new_memory(self):
        mgr = LatentStateManager(max_concurrent=4, device="cpu", fork_mode="deep_copy")
        mgr.create("r1", torch.randn(16, 6), max_steps=10)
        mgr.append_step("r1", torch.randn(8), torch.randn(16, 6))
        memory_before_fork = mgr._current_memory

        forked = mgr.fork("r1", "r1_fork")

        assert forked.latent_states[0] is not mgr.get("r1").latent_states[0]
        assert mgr._current_memory > memory_before_fork
        assert mgr.stats_snapshot()["tensor_reuse_hits"] == 0

    def test_eviction(self):
        mgr = LatentStateManager(max_concurrent=2, device="cpu")
        mgr.create("r1", torch.randn(4, 4), max_steps=5)
        mgr.create("r2", torch.randn(4, 4), max_steps=5)
        # Creating a 3rd should evict r1 (oldest)
        mgr.create("r3", torch.randn(4, 4), max_steps=5)
        assert mgr.num_active == 2
        assert "r1" not in mgr._states

    def test_memory_budget_evicts_oldest_state(self):
        mgr = LatentStateManager(max_concurrent=4, max_memory_gb=0.0000001, device="cpu")
        mgr.create("r1", torch.zeros(4, 4), max_steps=5)
        mgr.create("r2", torch.zeros(4, 4), max_steps=5)

        assert mgr.num_active == 1
        assert "r1" not in mgr._states
        assert "r2" in mgr._states

    def test_memory_budget_raises_when_no_state_can_be_evicted(self):
        mgr = LatentStateManager(max_concurrent=4, max_memory_gb=0.00000009, device="cpu")
        mgr.create("r1", torch.zeros(4, 4), max_steps=5)

        with pytest.raises(MemoryError, match="Latent state budget exceeded"):
            mgr.append_step("r1", torch.zeros(8), torch.zeros(4, 4))


class TestRolloutScheduler:
    def test_submit_and_schedule(self):
        from wm_infra.config import SchedulerConfig
        scheduler = RolloutScheduler(SchedulerConfig(max_batch_size=2))

        scheduler.submit(RolloutRequest(request_id="r1", num_steps=3))
        scheduler.submit(RolloutRequest(request_id="r2", num_steps=5))

        batch = scheduler.schedule_batch()
        assert batch.size == 2
        assert "r1" in batch.request_ids
        assert "r2" in batch.request_ids

    def test_sjf_ordering(self):
        from wm_infra.config import SchedulerConfig, SchedulerPolicy
        scheduler = RolloutScheduler(SchedulerConfig(max_batch_size=1, policy=SchedulerPolicy.SJF))

        scheduler.submit(RolloutRequest(request_id="long", num_steps=100))
        scheduler.submit(RolloutRequest(request_id="short", num_steps=2))

        batch = scheduler.schedule_batch()
        assert batch.request_ids[0] == "short"

    def test_memory_aware_policy_prefers_lighter_video_requests(self):
        from wm_infra.config import SchedulerConfig, SchedulerPolicy
        scheduler = RolloutScheduler(SchedulerConfig(max_batch_size=1, policy=SchedulerPolicy.MEMORY_AWARE))

        scheduler.submit(RolloutRequest(request_id="heavy", num_steps=8, frame_count=33, width=832, height=480))
        scheduler.submit(RolloutRequest(request_id="light", num_steps=4, frame_count=9, width=832, height=480))

        batch = scheduler.schedule_batch()
        assert batch.request_ids[0] == "light"

    def test_batch_resource_budget_skips_overcommit(self):
        from wm_infra.config import SchedulerConfig, SchedulerPolicy
        scheduler = RolloutScheduler(SchedulerConfig(max_batch_size=4, max_batch_resource_units=20.0, policy=SchedulerPolicy.MEMORY_AWARE))

        scheduler.submit(RolloutRequest(request_id="small", num_steps=2, frame_count=9, width=320, height=240))
        scheduler.submit(RolloutRequest(request_id="medium", num_steps=4, frame_count=9, width=832, height=480))
        scheduler.submit(RolloutRequest(request_id="large", num_steps=8, frame_count=33, width=832, height=480))

        batch = scheduler.schedule_batch()
        assert "small" in batch.request_ids
        assert "large" not in batch.request_ids


class TestWorldModelEngine:
    def test_state_capacity_uses_rollout_concurrency_not_batch_size(self):
        config = _small_config()
        config.device = "cpu"
        config.state_cache.max_batch_size = 1
        config.scheduler.max_concurrent_rollouts = 2
        dynamics = LatentDynamicsModel(config.dynamics)
        engine = WorldModelEngine(config, dynamics, tokenizer=None, execution_mode="chunked")

        for i in range(2):
            engine.submit_job(RolloutJob(
                job_id=f"concurrency_job{i}",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(1, 8),
                num_steps=1,
                return_frames=False,
                return_latents=True,
            ))

        results = engine.run_until_done()

        assert len(results) == 2
        assert {result.job_id for result in results} == {"concurrency_job0", "concurrency_job1"}

    def test_batch_peers_are_not_evicted_under_memory_pressure(self):
        config = _small_config()
        config.device = "cpu"
        config.scheduler.max_concurrent_rollouts = 3
        config.state_cache.max_batch_size = 3
        config.state_cache.pool_size_gb = 0.0000016
        dynamics = LatentDynamicsModel(config.dynamics)
        engine = WorldModelEngine(config, dynamics, tokenizer=None, execution_mode="chunked")

        for i in range(3):
            engine.submit_job(RolloutJob(
                job_id=f"peer_job{i}",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(2, 8),
                num_steps=2,
                return_frames=False,
                return_latents=True,
            ))

        with pytest.raises(MemoryError, match="Latent state budget exceeded"):
            engine.step()

        assert set(engine.state_manager._states) == {"peer_job0", "peer_job1", "peer_job2"}

    def test_chunked_execution_forms_real_multi_entity_chunks(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)
        engine = WorldModelEngine(config, dynamics, tokenizer=None, execution_mode="chunked")

        for i in range(3):
            engine.submit_job(RolloutJob(
                job_id=f"chunk_job{i}",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(2, 8),
                num_steps=2,
                return_frames=False,
                return_latents=True,
            ))

        results = engine.run_until_done()
        stats = engine.execution_stats_snapshot()

        assert len(results) == 3
        assert stats["mode"] == "chunked"
        assert stats["max_transition_chunk_size"] >= 3
        assert stats["avg_transition_chunk_size"] >= 3.0

    def test_chunked_mode_rejects_non_chunked_execution_mode(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        with pytest.raises(ValueError, match="Unsupported execution_mode: legacy"):
            WorldModelEngine(config, dynamics, tokenizer=None, execution_mode="legacy")

    def test_end_to_end_with_latent(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        engine = WorldModelEngine(config, dynamics, tokenizer=None)

        job = RolloutJob(
            job_id="test1",
            initial_latent=torch.randn(16, 6),
            actions=torch.randn(3, 8),
            num_steps=3,
            return_frames=False,
            return_latents=True,
        )

        engine.submit_job(job)
        results = engine.run_until_done()

        assert len(results) == 1
        result = results[0]
        assert result.steps_completed == 3
        assert result.predicted_latents is not None
        assert result.elapsed_ms > 0

    def test_multiple_concurrent_jobs(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)
        engine = WorldModelEngine(config, dynamics, tokenizer=None)

        for i in range(3):
            engine.submit_job(RolloutJob(
                job_id=f"job{i}",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(2, 8),
                num_steps=2,
                return_frames=False,
                return_latents=True,
            ))

        results = engine.run_until_done()
        assert len(results) == 3
        for r in results:
            assert r.steps_completed == 2


class TestAsyncEngine:
    @pytest.mark.asyncio
    async def test_single_job(self):
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        async_engine = AsyncWorldModelEngine(config, dynamics, tokenizer=None)
        async_engine.start()

        try:
            job = RolloutJob(
                job_id="async_test1",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(3, 8),
                num_steps=3,
                return_frames=False,
                return_latents=True,
            )
            result = await async_engine.submit(job)
            assert result.steps_completed == 3
            assert result.predicted_latents is not None
            assert result.job_id == "async_test1"
        finally:
            await async_engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_jobs(self):
        """Multiple concurrent submits should all complete."""
        config = _small_config()
        config.device = "cpu"
        config.state_cache.max_batch_size = 8
        config.scheduler.max_concurrent_rollouts = 8
        dynamics = LatentDynamicsModel(config.dynamics)

        async_engine = AsyncWorldModelEngine(config, dynamics, tokenizer=None)
        async_engine.start()

        try:
            jobs = []
            for i in range(5):
                job = RolloutJob(
                    job_id=f"concurrent_{i}",
                    initial_latent=torch.randn(16, 6),
                    actions=torch.randn(2, 8),
                    num_steps=2,
                    return_frames=False,
                    return_latents=True,
                )
                jobs.append(job)

            # Submit all concurrently
            results = await asyncio.gather(
                *[async_engine.submit(job) for job in jobs]
            )

            assert len(results) == 5
            job_ids = {r.job_id for r in results}
            assert job_ids == {f"concurrent_{i}" for i in range(5)}
            for r in results:
                assert r.steps_completed == 2
                assert r.predicted_latents is not None
        finally:
            await async_engine.stop()

    @pytest.mark.asyncio
    async def test_engine_lifecycle(self):
        """Engine can be started and stopped cleanly."""
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        async_engine = AsyncWorldModelEngine(config, dynamics, tokenizer=None)
        assert not async_engine.is_running

        async_engine.start()
        assert async_engine.is_running

        await async_engine.stop()
        assert not async_engine.is_running

    @pytest.mark.asyncio
    async def test_step_callback(self):
        """Per-step callback fires for each prediction step."""
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        async_engine = AsyncWorldModelEngine(config, dynamics, tokenizer=None)
        async_engine.start()

        try:
            callback_log = []

            def on_step(job_id: str, step_idx: int, latent: torch.Tensor):
                callback_log.append((job_id, step_idx))

            job = RolloutJob(
                job_id="callback_test",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(4, 8),
                num_steps=4,
                return_frames=False,
                return_latents=True,
                step_callback=on_step,
            )
            result = await async_engine.submit(job)
            assert result.steps_completed == 4
            assert len(callback_log) == 4
            assert callback_log[0] == ("callback_test", 0)
            assert callback_log[3] == ("callback_test", 3)
        finally:
            await async_engine.stop()

    @pytest.mark.asyncio
    async def test_submit_stream(self):
        """submit_stream yields each step as it completes."""
        config = _small_config()
        config.device = "cpu"
        dynamics = LatentDynamicsModel(config.dynamics)

        async_engine = AsyncWorldModelEngine(config, dynamics, tokenizer=None)
        async_engine.start()

        try:
            job = RolloutJob(
                job_id="stream_test",
                initial_latent=torch.randn(16, 6),
                actions=torch.randn(3, 8),
                num_steps=3,
                return_frames=False,
                return_latents=True,
            )

            streamed_steps = []
            async for step_idx, latent in async_engine.submit_stream(job):
                streamed_steps.append((step_idx, latent.shape))

            assert len(streamed_steps) == 3
            assert streamed_steps[0][0] == 0
            assert streamed_steps[1][0] == 1
            assert streamed_steps[2][0] == 2
            # Each latent should be [N, D] = [16, 6]
            assert streamed_steps[0][1] == torch.Size([16, 6])
        finally:
            await async_engine.stop()
