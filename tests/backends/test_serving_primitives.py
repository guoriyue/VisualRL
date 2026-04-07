from wm_infra.backends.serving_primitives import (
    CompiledProfile,
    ExecutionFamily,
    ResidencyRecord,
    ResidencyTier,
    TransferPlan,
)


def test_execution_family_cache_key_is_stable():
    family = ExecutionFamily(
        backend="wan-video",
        model="wan2.2-t2v-A14B",
        stage="pipeline",
        device="cuda",
        dtype="float16",
        runner_mode="official",
        batch_size_family="pair",
        width=832,
        height=480,
        frame_count=9,
        num_steps=4,
        memory_mode="offload_model+t5_cpu",
    )

    left = family.as_dict()
    right = family.as_dict()

    assert left["cache_key"] == right["cache_key"]


def test_transfer_plan_tracks_total_bytes():
    plan = TransferPlan(overlap_h2d_with_compute=True, overlap_d2h_with_io=True)
    plan.add_h2d(1024)
    plan.add_d2h(2048)
    plan.add_artifact_io(512)

    assert plan.total_bytes == 3584
    assert plan.as_dict()["total_bytes"] == 3584


def test_compiled_profile_embeds_execution_family():
    family = ExecutionFamily(
        backend="genie-rollout",
        model="genie-local",
        stage="transition",
        device="cuda",
        dtype="uint32",
        runner_mode="real",
        batch_size_family="small",
        width=16,
        height=16,
        frame_count=4,
        num_steps=2,
        prompt_frames=4,
        tokenizer_kind="genie_stmaskgit",
    )
    profile = CompiledProfile(
        profile_id="profile-1",
        execution_family=family,
        graph_key="graph-1",
        compile_state="warm_profile_batch_hit",
        warm_profile_hit=True,
        compiled_batch_size_hit=True,
        compiled_batch_sizes=[1, 2],
        reuse_count=3,
    )
    payload = profile.as_dict()

    assert payload["execution_family"]["backend"] == "genie-rollout"
    assert payload["compiled_batch_sizes"] == [1, 2]


def test_residency_record_serializes_tier():
    record = ResidencyRecord(
        object_id="rollout-1:state-pages",
        tier=ResidencyTier.CPU_PINNED_WARM,
        bytes_size=4096,
        pinned=True,
    )

    assert record.as_dict()["tier"] == "cpu_pinned_warm"
