import numpy as np

from wm_infra.backends.genie_checkpoint import build_checkpoint_delta, checkpoint_due
from wm_infra.backends.genie_runtime import (
    GenieBatchSignature,
    GenieExecutionEntity,
    GenieRuntimeState,
    build_transition_entities,
    default_window_size,
    temperature_bucket,
)
from wm_infra.backends.genie_scheduler import GenieScheduler


def _root_entity() -> GenieExecutionEntity:
    signature = GenieBatchSignature(
        backend="genie-rollout",
        model_name="genie-local",
        stage="transition",
        device="cpu",
        dtype="uint32",
        tokenizer_kind="genie_stmaskgit",
        spatial_h=16,
        spatial_w=16,
        window_num_frames=12,
        num_prompt_frames=4,
        maskgit_steps=2,
        temperature_bucket=temperature_bucket(0.0),
        checkpoint_every_n_frames=4,
        runner_mode="stub",
        needs_persist=False,
    )
    return GenieExecutionEntity(
        entity_id="sample:root",
        rollout_id="rollout-1",
        episode_id="episode-1",
        branch_id="branch-1",
        sample_id="sample",
        input_state_handle_id="state-1",
        current_stage="transition",
        next_stage="artifact_persist",
        window_start_frame=4,
        window_num_frames=12,
        total_frames=16,
        num_prompt_frames=4,
        checkpoint_every_n_frames=4,
        priority=1.0,
        deadline_s=None,
        batch_signature=signature,
        queue_lane="hot_continuation",
    )


def test_default_window_size_respects_checkpoint_cadence():
    assert default_window_size(total_frames=16, prompt_frames=4, checkpoint_every_n_frames=0) == 12
    assert default_window_size(total_frames=16, prompt_frames=4, checkpoint_every_n_frames=4) == 4


def test_build_transition_entities_splits_by_checkpoint_window():
    entities = build_transition_entities(_root_entity())
    assert len(entities) == 3
    assert entities[0].window_start_frame == 4
    assert entities[0].window_end_frame == 8
    assert entities[1].window_start_frame == 8
    assert entities[2].window_end_frame == 16


def test_scheduler_prefers_hot_continuation_lane():
    runtime_state = GenieRuntimeState(
        rollout_id="rollout-1",
        prompt_tokens_ref=None,
        generated_tokens_ref=None,
        last_completed_frame=4,
        resident_tier="hot_gpu",
        ancestor_state_ref="state-1",
        checkpoint_delta_ref=None,
        materialized_bytes=4096,
        dirty_since_checkpoint=False,
        reuse_hits=1,
        reuse_misses=0,
    )
    scheduler = GenieScheduler(max_chunk_size=8)
    chunks = scheduler.build_chunks(build_transition_entities(_root_entity()), runtime_state)
    assert len(chunks) == 3
    assert chunks[0].queue_lane == "checkpoint_heavy"
    assert chunks[0].expected_occupancy > 0


def test_checkpoint_due_and_delta_metadata():
    assert checkpoint_due(frame_end=8, total_frames=16, checkpoint_every_n_frames=4) is True
    assert checkpoint_due(frame_end=16, total_frames=16, checkpoint_every_n_frames=4) is False

    token_window = np.zeros((4, 16, 16), dtype=np.uint32)
    delta = build_checkpoint_delta(
        rollout_id="rollout-1",
        sample_id="sample",
        parent_state_handle_id="state-1",
        all_tokens=np.concatenate([np.zeros((4, 16, 16), dtype=np.uint32), token_window], axis=0),
        start_frame=4,
        end_frame=8,
        checkpoint_every_n_frames=4,
        runner_mode="stub",
    )
    assert delta.artifact_id == "sample:checkpoint-delta:0008"
    assert delta.bytes_size == token_window.nbytes
    assert delta.metadata["frame_count"] == 4
