import asyncio

import pytest

from wm_infra.backends.genie_batcher import GenieTransitionBatcher
from wm_infra.backends.genie_runner import GenieRunner
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
