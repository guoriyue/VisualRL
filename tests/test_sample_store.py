"""Focused tests for the sample manifest store durability behavior."""

from wm_infra.controlplane import ExperimentRef, SampleManifestStore, SampleRecord, SampleSpec, SampleStatus, TaskType


def test_sample_manifest_store_put_writes_clean_final_file(tmp_path):
    store = SampleManifestStore(tmp_path)
    record = SampleRecord(
        sample_id="sample_atomic",
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.SUCCEEDED,
        experiment=ExperimentRef(experiment_id="exp_atomic"),
        sample_spec=SampleSpec(prompt="atomic write"),
    )

    store.put(record)

    final_path = tmp_path / "samples" / "exp_atomic" / "sample_atomic.json"
    assert final_path.exists()
    assert store.get("sample_atomic") is not None
    assert [path.name for path in final_path.parent.iterdir()] == ["sample_atomic.json"]


def test_sample_manifest_store_skips_corrupt_manifest_files(tmp_path):
    store = SampleManifestStore(tmp_path)
    record = SampleRecord(
        sample_id="sample_corrupt",
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.SUCCEEDED,
        experiment=ExperimentRef(experiment_id="exp_clean"),
        sample_spec=SampleSpec(prompt="keep the clean record"),
    )

    store.put(record)

    corrupt_legacy_path = tmp_path / "samples" / "sample_corrupt.json"
    corrupt_legacy_path.write_text("{not valid json")
    corrupt_other_path = tmp_path / "samples" / "other_bucket" / "broken.json"
    corrupt_other_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_other_path.write_text("{still not valid json")

    loaded = store.get("sample_corrupt")
    assert loaded is not None
    assert loaded.sample_id == "sample_corrupt"

    records = store.list()
    assert len(records) == 1
    assert records[0].sample_id == "sample_corrupt"


def test_sample_manifest_store_returns_none_for_corrupt_only_sample(tmp_path):
    store = SampleManifestStore(tmp_path)
    corrupt_path = tmp_path / "samples" / "sample_missing.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{broken")

    assert store.get("sample_missing") is None
    assert store.list() == []
