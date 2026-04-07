"""Tests for control-plane schemas and storage."""

from wm_infra.controlplane import (
    ArtifactKind,
    BranchCreate,
    CheckpointCreate,
    EvaluationRecord,
    EvaluationStatus,
    ExecutionStateRef,
    EpisodeCreate,
    ExperimentRef,
    FailureTag,
    ProduceSampleRequest,
    RolloutCreate,
    RolloutTaskConfig,
    SampleManifestStore,
    SampleRecord,
    SampleSpec,
    SampleStatus,
    StateLineageRef,
    StateHandleCreate,
    StateResidency,
    TaskType,
    TemporalRefs,
    TemporalStore,
    VideoMemoryProfile,
    WanTaskConfig,
    estimate_wan_request,
)


def test_produce_sample_request_defaults():
    req = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-runtime",
        model="wan2.2-i2v",
        sample_spec=SampleSpec(prompt="a corgi walking in neon tokyo"),
    )

    assert req.task_type == TaskType.TEXT_TO_VIDEO
    assert req.return_artifacts == [ArtifactKind.VIDEO]
    assert req.sample_spec.prompt == "a corgi walking in neon tokyo"


def test_produce_sample_request_accepts_typed_rollout_task_config():
    req = ProduceSampleRequest(
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        sample_spec=SampleSpec(prompt="roll forward"),
        task_config=RolloutTaskConfig(num_steps=3, frame_count=9, width=832, height=480),
    )

    assert req.task_config is not None
    assert req.task_config.num_steps == 3
    assert req.task_config.frame_count == 9
    assert req.task_config.width == 832
    assert req.task_config.height == 480


def test_produce_sample_request_backfills_legacy_rollout_fields_from_sample_metadata():
    req = ProduceSampleRequest(
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        sample_spec=SampleSpec(
            prompt="legacy rollout",
            metadata={
                "num_steps": 4,
                "frame_num": 9,
                "offload_model": True,
                "convert_model_dtype": True,
                "t5_cpu": True,
                "memory_profile": "low_vram",
            },
        ),
    )

    assert req.task_config is not None
    assert req.task_config.num_steps == 4
    assert req.task_config.frame_count == 9
    assert req.task_config.offload_model is True
    assert req.task_config.convert_model_dtype is True
    assert req.task_config.t5_cpu is True
    assert req.task_config.memory_profile == VideoMemoryProfile.LOW_VRAM


def test_sample_record_can_capture_lineage_evaluation_and_resource_estimate():
    record = SampleRecord(
        sample_id="sample_001",
        task_type=TaskType.IMAGE_TO_VIDEO,
        backend="diffusers-runtime",
        model="wan2.2-live2d",
        status=SampleStatus.ACCEPTED,
        experiment=ExperimentRef(experiment_id="exp_live2d", run_id="run_01"),
        sample_spec=SampleSpec(
            prompt="subtle live2d breathing motion",
            references=["asset://character/front.png"],
            width=512,
            height=512,
            fps=12,
        ),
        lineage_parent_ids=["asset_prepare_001"],
        resource_estimate={"estimated_units": 12.5, "bottleneck": "frame_pressure"},
        evaluations=[
            EvaluationRecord(
                evaluator="auto_qc_v1",
                status=EvaluationStatus.HUMAN_REVIEW_REQUIRED,
                score=0.81,
                failure_tags=[FailureTag.LOW_MOTION_QUALITY],
            )
        ],
    )

    assert record.status == SampleStatus.ACCEPTED
    assert record.experiment is not None
    assert record.experiment.experiment_id == "exp_live2d"
    assert record.evaluations[0].failure_tags == [FailureTag.LOW_MOTION_QUALITY]
    assert record.resource_estimate is not None
    assert record.resource_estimate.estimated_units == 12.5


def test_sample_manifest_store_round_trip(tmp_path):
    store = SampleManifestStore(tmp_path)
    record = SampleRecord(
        sample_id="sample_roundtrip",
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.SUCCEEDED,
        experiment=ExperimentRef(experiment_id="exp_roundtrip"),
        sample_spec=SampleSpec(prompt="rollout a latent dog dream"),
        runtime={"steps_completed": 2},
        resource_estimate={"estimated_units": 2.0, "bottleneck": "frame_pressure"},
    )

    store.put(record)
    loaded = store.get("sample_roundtrip")

    assert loaded is not None
    assert loaded.sample_id == record.sample_id
    assert loaded.runtime["steps_completed"] == 2
    assert loaded.resource_estimate is not None
    assert loaded.resource_estimate.estimated_units == 2.0
    assert len(store.list()) == 1
    assert (tmp_path / "samples" / "exp_roundtrip" / "sample_roundtrip.json").exists()


def test_wan_request_accepts_first_class_wan_config_and_estimates_resources():
    req = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(prompt="a corgi sprinting through cyberpunk rain"),
        wan_config=WanTaskConfig(frame_count=9, width=832, height=480, num_steps=4),
    )

    assert req.wan_config is not None
    assert req.wan_config.frame_count == 9
    estimate = estimate_wan_request(req.wan_config)
    assert estimate.bottleneck == "frame_pressure"
    assert estimate.estimated_vram_gb is not None
    assert estimate.estimated_vram_gb > 0


def test_wan_request_can_hydrate_legacy_metadata_into_wan_config():
    req = ProduceSampleRequest(
        task_type=TaskType.TEXT_TO_VIDEO,
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        sample_spec=SampleSpec(
            prompt="legacy wan metadata",
            width=832,
            height=480,
            metadata={
                "frame_num": 9,
                "sample_steps": 4,
                "guidance_scale": 5.0,
                "shift": 8.0,
                "memory_profile": "low_vram",
            },
        ),
    )

    assert req.wan_config is not None
    assert req.wan_config.frame_count == 9
    assert req.wan_config.num_steps == 4
    assert req.wan_config.guidance_scale == 5.0
    assert req.wan_config.shift == 8.0
    assert req.wan_config.width == 832
    assert req.wan_config.height == 480


def test_produce_sample_request_can_capture_temporal_refs():
    req = ProduceSampleRequest(
        task_type=TaskType.GENIE_ROLLOUT,
        backend="genie-rollout",
        model="genie-local",
        sample_spec=SampleSpec(prompt="temporal step"),
        temporal=TemporalRefs(
            episode_id="ep_1",
            branch_id="br_1",
            state_handle_id="state_1",
        ),
        task_config=RolloutTaskConfig(num_steps=3),
    )

    assert req.temporal is not None
    assert req.temporal.episode_id == "ep_1"
    assert req.temporal.state_handle_id == "state_1"


def test_temporal_store_round_trip(tmp_path):
    store = TemporalStore(tmp_path)
    episode = store.create_episode(EpisodeCreate(title="dog world"))
    branch = store.create_branch(BranchCreate(episode_id=episode.episode_id, name="main"))
    rollout = store.create_rollout(
        RolloutCreate(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            backend="genie-rollout",
            model="genie-local",
            step_count=4,
        )
    )
    state = store.create_state_handle(
        StateHandleCreate(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            rollout_id=rollout.rollout_id,
            kind="video_latent",
        )
    )
    checkpoint = store.create_checkpoint(
        CheckpointCreate(
            episode_id=episode.episode_id,
            rollout_id=rollout.rollout_id,
            branch_id=branch.branch_id,
            state_handle_id=state.state_handle_id,
            step_index=3,
            tag="terminal",
        )
    )
    store.attach_checkpoint_to_rollout(rollout.rollout_id, checkpoint.checkpoint_id)

    loaded_rollout = store.rollouts.get(rollout.rollout_id)
    assert loaded_rollout is not None
    assert loaded_rollout.checkpoint_ids == [checkpoint.checkpoint_id]
    assert store.episodes.get(episode.episode_id) is not None
    assert store.branches.get(branch.branch_id) is not None
    assert store.state_handles.get(state.state_handle_id) is not None
    assert store.checkpoints.get(checkpoint.checkpoint_id) is not None


def test_state_handle_can_separate_execution_state_from_lineage(tmp_path):
    store = TemporalStore(tmp_path)
    episode = store.create_episode(EpisodeCreate(title="runtime split"))
    branch = store.create_branch(BranchCreate(episode_id=episode.episode_id, name="main"))
    handle = store.create_state_handle(
        StateHandleCreate(
            episode_id=episode.episode_id,
            branch_id=branch.branch_id,
            kind="latent",
            execution_state_ref=ExecutionStateRef(
                residency=StateResidency.INLINE,
                storage_backend="state_handle_metadata",
                state_key="latent_state",
                goal_key="goal_state",
                step_key="step_idx",
                device="cpu",
            ),
            lineage_ref=StateLineageRef(
                env_name="toy-line-v0",
                task_id="toy-line-train",
                trajectory_id="traj-1",
                step_idx=3,
                parent_state_handle_id="parent-1",
            ),
            metadata={
                "latent_state": [[0.0]],
                "goal_state": [[0.5]],
                "step_idx": 3,
            },
        )
    )

    loaded = store.state_handles.get(handle.state_handle_id)

    assert loaded is not None
    assert loaded.execution_state_ref is not None
    assert loaded.execution_state_ref.residency == StateResidency.INLINE
    assert loaded.lineage_ref is not None
    assert loaded.lineage_ref.trajectory_id == "traj-1"
    assert loaded.lineage_ref.parent_state_handle_id == "parent-1"
