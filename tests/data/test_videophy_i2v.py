from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vrl.scripts.data import bootstrap
from vrl.scripts.data.videophy_i2v import (
    DECODE_METHOD,
    VideoPhyVideoRow,
    prepare_videophy_i2v_dataset,
    select_videos_for_prompts,
)


def test_select_videophy_videos_prefers_aligned_physical_candidate() -> None:
    """Among candidates for one caption the selection prefers the one with both semantic alignment
    and physical commonsense (sa=1, pc=1), matching captions after whitespace trimming.
    """
    candidates = [
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/weak.mp4",
            row_index=0,
            majority_sa=1,
            majority_pc=0,
        ),
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/best.mp4",
            row_index=1,
            majority_sa=1,
            majority_pc=1,
        ),
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/misaligned.mp4",
            row_index=2,
            majority_sa=0,
            majority_pc=1,
        ),
    ]

    selected = select_videos_for_prompts(
        [" A wheel steadily rolls across a cement path. "],
        candidates,
        split="train",
    )

    assert len(selected) == 1
    assert selected[0].source_row.video_url == "https://example.test/best.mp4"


def test_prepare_videophy_i2v_dataset_writes_source_backed_manifest(tmp_path: Path) -> None:
    """Preparation fetches each split's video, extracts its first frame resized to 832x480, writes
    source-backed manifest rows (URL, decode method, frame sizes) plus a report, and does not
    keep the videos.
    """
    csv_path = tmp_path / "videophy_test_public.csv"
    csv_path.write_text(
        "\n".join(
            [
                "video_url,caption,states_of_matter,source,complexity,majority_sa,majority_pc",
                "https://example.test/train.mp4,A wheel rolls.,solid_solid,lavie,0,1,1",
                "https://example.test/eval.mp4,Honey diffuses.,fluid_fluid,lavie,1,1,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    train_prompts = tmp_path / "train.txt"
    eval_prompts = tmp_path / "eval.txt"
    train_prompts.write_text("A wheel rolls.\n", encoding="utf-8")
    eval_prompts.write_text("Honey diffuses.\n", encoding="utf-8")
    data_root = tmp_path / "external"

    fetched: list[tuple[str, Path]] = []

    def fake_fetch(url: str, target: Path) -> None:
        fetched.append((url, target))
        target.write_bytes(b"fake-video")

    def fake_extract(video_path: Path, image_path: Path) -> None:
        assert video_path.exists()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (10, 20, 30)).save(image_path)

    report = prepare_videophy_i2v_dataset(
        csv_path=csv_path,
        train_prompts=train_prompts,
        eval_prompts=eval_prompts,
        data_root=data_root,
        repo_id="videophysics/videophy_test_public",
        csv_file="videophy_test_public.csv",
        fetch_video=fake_fetch,
        extract_first_frame=fake_extract,
    )

    dataset_root = data_root / "videophy_i2v"
    train_manifest = dataset_root / "manifests" / "train.jsonl"
    eval_manifest = dataset_root / "manifests" / "eval.jsonl"
    train_rows = [json.loads(line) for line in train_manifest.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in eval_manifest.read_text().splitlines()]

    assert report["train_rows"] == 1
    assert report["eval_rows"] == 1
    assert fetched[0][0] == "https://example.test/train.mp4"
    assert train_rows[0]["image"] == "images/train/000.png"
    assert train_rows[0]["caption"] == "A wheel rolls."
    assert train_rows[0]["task_type"] == "image_to_video"
    assert train_rows[0]["metadata"]["source_video_url"] == "https://example.test/train.mp4"
    assert train_rows[0]["metadata"]["decode_method"] == DECODE_METHOD
    assert train_rows[0]["metadata"]["source_frame_size"] == {"width": 4, "height": 4}
    assert train_rows[0]["metadata"]["image_size"] == {"width": 832, "height": 480}
    assert eval_rows[0]["image"] == "images/eval/000.png"
    assert (dataset_root / train_rows[0]["image"]).exists()
    assert Image.open(dataset_root / train_rows[0]["image"]).size == (832, 480)
    assert not (dataset_root / "videos" / "train" / "000.mp4").exists()
    assert (dataset_root / "report.json").exists()


def test_for_experiment_flags_videophy_i2v_population_command(tmp_path: Path) -> None:
    """Missing VideoPhy I2V manifests leave the plan not ready and point every step at the
    ``videophy-i2v`` command.
    """
    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_image_manifest",
            "manifest": "data/external/videophy_i2v/manifests/train.jsonl",
            "eval_manifest": "data/external/videophy_i2v/manifests/eval.jsonl",
        },
        repo_root=tmp_path,
    )

    assert plan["ready"] is False
    assert all("videophy-i2v" in step["get"] for step in plan["steps"])


def test_for_experiment_rejects_partial_videophy_i2v_smoke_data(tmp_path: Path) -> None:
    """A manifest with fewer rows than its prompt file is incomplete: the plan stays not ready
    with the shortfall reported, while the complete eval split passes.
    """
    split_root = tmp_path / "datasets" / "videophy"
    split_root.mkdir(parents=True)
    (split_root / "train.txt").write_text("prompt a\nprompt b\n", encoding="utf-8")
    (split_root / "eval.txt").write_text("prompt c\n", encoding="utf-8")
    manifest_root = tmp_path / "data" / "external" / "videophy_i2v" / "manifests"
    manifest_root.mkdir(parents=True)
    (manifest_root / "train.jsonl").write_text('{"caption":"prompt a"}\n', encoding="utf-8")
    (manifest_root / "eval.jsonl").write_text('{"caption":"prompt c"}\n', encoding="utf-8")

    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_image_manifest",
            "manifest": "data/external/videophy_i2v/manifests/train.jsonl",
            "eval_manifest": "data/external/videophy_i2v/manifests/eval.jsonl",
        },
        repo_root=tmp_path,
    )

    train_step = next(step for step in plan["steps"] if step["role"] == "manifest")
    eval_step = next(step for step in plan["steps"] if step["role"] == "eval_manifest")
    assert plan["ready"] is False
    assert train_step["rows"] == 1
    assert train_step["expected_rows"] == 2
    assert train_step["complete"] is False
    assert "videophy-i2v" in train_step["get"]
    assert eval_step["complete"] is True
