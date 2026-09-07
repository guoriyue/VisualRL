from __future__ import annotations

from pathlib import Path

from vrl.models.families.registry import get_model_family_entry
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.trainers.data import load_prompt_manifest
from vrl.trainers.data.artifacts import (
    resolve_prompt_example_artifacts,
    resolve_prompt_example_references,
    validate_reference_images,
)


def _write_reference_manifest(root: Path) -> Path:
    reference = root / "video_world" / "references" / "ref.ppm"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    manifest = root / "video_world" / "v2w_train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        (
            '{"prompt":"The robot arm moves toward the cup.",'
            '"reference_image":"video_world/references/ref.ppm",'
            '"metadata":{"source_episode":"episode_train"}}\n'
        ),
        encoding="utf-8",
    )
    return manifest


def test_resolved_reference_image_flows_to_collector_metadata(tmp_path: Path) -> None:
    """A manifest row's reference image, resolved against the data root, reaches both the request
    input and the collector metadata.
    """
    manifest = _write_reference_manifest(tmp_path)
    example = resolve_prompt_example_artifacts(
        load_prompt_manifest(manifest)[0],
        data_root=tmp_path,
    )
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    collector_request = builder.build(
        [example.generation_input()],
        1,
        metadata=example.reward_metadata(),
    )

    assert collector_request.metadata["reference_image"].endswith("ref.ppm")
    assert collector_request.request.inputs[0].reference_image.endswith("ref.ppm")


def test_cosmos_per_sample_reference_uses_vrl_data_root(monkeypatch, tmp_path: Path) -> None:
    """Checks Cosmos per sample reference resolves at load and passes the hook."""
    manifest = _write_reference_manifest(tmp_path)
    monkeypatch.setenv("VRL_DATA_ROOT", str(tmp_path))
    # Production sequence: run_online_recipe resolves reference paths at load
    # time, then the family hook only validates.
    examples = [
        resolve_prompt_example_references(example, allow_absolute=True)
        for example in load_prompt_manifest(manifest)
    ]
    validate_reference_images(
        examples,
        manifest_path=manifest,
    )

    assert examples[0].reference_image == str(
        (tmp_path / "video_world" / "references" / "ref.ppm").resolve(),
    )
    assert "reference_image" not in examples[0].metadata


def test_cosmos_per_sample_reference_uses_artifact_data_root(tmp_path: Path) -> None:
    """Checks cosmos per_sample references resolve under data.artifact_data_root."""
    manifest = _write_reference_manifest(tmp_path)
    examples = [
        resolve_prompt_example_references(
            example,
            data_root=tmp_path,
            allow_absolute=True,
        )
        for example in load_prompt_manifest(manifest)
    ]
    validate_reference_images(
        examples,
        manifest_path=manifest,
    )

    assert examples[0].reference_image == str(
        (tmp_path / "video_world" / "references" / "ref.ppm").resolve(),
    )
    assert "reference_image" not in examples[0].metadata
