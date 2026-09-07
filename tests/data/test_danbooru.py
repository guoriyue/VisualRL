from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from vrl.scripts.data.danbooru.anatomy import build_prompt_rows, split_prompt_rows
from vrl.scripts.data.danbooru.assets import (
    build_positive_images,
    download_danbooru_images,
    hand_crop_rows,
    positive_image_rows,
)
from vrl.scripts.data.danbooru.config import DOMAIN, TEMPLATE_ID
from vrl.scripts.data.danbooru.safety import (
    build_danbooru_safety_prompt_rows,
    build_safety_prompts,
    split_safety_prompt_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_prompt_rows_tracks_provenance_and_buckets(tmp_path: Path) -> None:
    """Prompt rows carry domain / template / prompt-style provenance and a bucket, mix tag-style
    and language-style prompts, and split into train and eval by the requested limits.
    """
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 1,
                "score": 10,
                "tag_string_general": (
                    "1girl solo full_body standing boots simple_background hands long_hair"
                ),
            },
            {
                "id": 2,
                "score": 12,
                "tags": ["1boy", "solo", "full_body", "running", "sportswear", "feet"],
            },
            {
                "id": 3,
                "score": 20,
                "tag_string_general": "1girl solo portrait standing hands",
            },
        ],
    )

    rows = build_prompt_rows(metadata, min_score=0, seed=7)
    train_rows, eval_rows = split_prompt_rows(rows, train_limit=1, eval_limit=1)

    assert len(rows) == 2
    assert len(train_rows) == 1
    assert len(eval_rows) == 1
    assert {row["metadata"]["domain"] for row in rows} == {DOMAIN}
    assert {row["metadata"]["template_id"] for row in rows} == {TEMPLATE_ID}
    assert {row["metadata"]["prompt_style"] for row in rows} == {"tag", "language"}
    assert any(row["prompt"].startswith("1girl, solo") for row in rows)
    assert any(row["prompt"].startswith("single anime boy") for row in rows)
    assert any("both hands visible" in row["prompt"] for row in rows)
    assert any("long hair" in row["prompt"] for row in rows)
    assert any(row["metadata"]["bucket"] == "running" for row in rows)


def test_build_prompt_rows_reads_danbooru_json_and_tar_gz(tmp_path: Path) -> None:
    """Metadata is read from a JSON-lines file or from a ``.tar.gz`` holding one, with tags taken
    from ``tag_string``.
    """
    rows = [
        {
            "id": 1,
            "score": 10,
            "tag_string": (
                "1girl solo full_body standing hands boots simple_background long_hair"
            ),
        },
        {
            "id": 2,
            "score": 12,
            "tag_string": "1boy solo full_body walking shoes city_street backpack",
        },
    ]
    metadata_json = tmp_path / "posts.json"
    _write_jsonl(metadata_json, rows)
    assert len(build_prompt_rows(metadata_json, min_score=0, seed=1)) == 2

    archive_path = tmp_path / "posts.tar.gz"
    inner = tmp_path / "posts_inner.json"
    _write_jsonl(inner, rows)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(inner, arcname="posts.json")

    archive_rows = build_prompt_rows(archive_path, min_score=0, seed=1)

    assert len(archive_rows) == 2
    assert any("backpack" in row["prompt"] for row in archive_rows)


def test_build_prompt_rows_uses_bucket_quotas_with_score_fallback(tmp_path: Path) -> None:
    """Equal bucket weights with ``limit=6`` give each bucket two rows; a bucket short of posts
    above ``preferred_min_score`` falls back to posts down to ``min_score``.
    """
    metadata = tmp_path / "posts.jsonl"
    rows: list[dict] = []
    hair_tags = ["black_hair", "brown_hair", "blonde_hair", "red_hair", "blue_hair", "pink_hair"]
    for index in range(6):
        rows.append(
            {
                "id": 100 + index,
                "score": 30,
                "tag_string": f"1girl solo full_body standing boots long_hair {hair_tags[index]}",
            },
        )
    action_hair_tags = ["black_hair", "brown_hair", "blonde_hair"]
    for index in range(3):
        rows.append(
            {
                "id": 200 + index,
                "score": 6,
                "tag_string": f"1girl solo full_body jumping shoes short_hair {action_hair_tags[index]}",
            },
        )
    walking_hair_tags = ["red_hair", "blue_hair", "pink_hair"]
    for index in range(3):
        rows.append(
            {
                "id": 300 + index,
                "score": 8,
                "tag_string": f"1boy solo full_body walking shoes backpack {walking_hair_tags[index]}",
            },
        )
    _write_jsonl(metadata, rows)

    built = build_prompt_rows(
        metadata,
        min_score=5,
        preferred_min_score=20,
        limit=6,
        seed=0,
        bucket_weights={"feet_visible": 1.0, "action_pose": 1.0, "walking": 1.0},
    )

    buckets = [row["metadata"]["bucket"] for row in built]
    assert buckets.count("feet_visible") == 2
    assert buckets.count("action_pose") == 2
    assert buckets.count("walking") == 2
    assert all(row["metadata"]["source_score"] >= 5 for row in built)


def test_build_prompt_rows_allows_hand_focus_without_full_body(tmp_path: Path) -> None:
    """``hand_focus`` posts qualify without ``full_body`` and are framed as ``upper body``."""
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 400,
                "score": 15,
                "tag_string": "1girl solo upper_body hand_focus hands long_hair",
            },
        ],
    )

    rows = build_prompt_rows(metadata, min_score=5, limit=1, seed=0)

    assert rows[0]["metadata"]["bucket"] == "hand_focus"
    assert rows[0]["metadata"]["framing"] == "upper body"


@pytest.mark.real_cover(
    None,
    why=(
        "the injected fetch stands in for an HTTP GET against danbooru.donmai.us; a test "
        "that reaches the live site is neither reproducible nor free, and what is asserted "
        "here is the selection (which post ids get a request), not the transfer"
    ),
    tracked_in="docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md",
)
def test_download_danbooru_images_downloads_only_positive_selection(tmp_path: Path) -> None:
    """Only score-passing posts get a fetch, and only their files land on disk."""
    metadata = tmp_path / "posts.jsonl"
    rows = [
        {
            "id": 1,
            "score": 50,
            "tag_string": "1girl solo full_body standing",
            "file_ext": "jpg",
            "file_url": "https://example.test/1.jpg",
        },
        {
            "id": 2,
            "score": 1,
            "tag_string": "1girl solo upper_body",
            "file_ext": "jpg",
            "file_url": "https://example.test/2.jpg",
        },
    ]
    _write_jsonl(metadata, rows)
    image_root = tmp_path / "images"

    selected = positive_image_rows(
        metadata,
        image_root=image_root,
        min_score=10,
        limit=None,
        source="danbooru",
    )
    targets = {str(row["post_id"]): Path(row["image_path"]) for row in selected}

    calls: list[tuple[str, Path]] = []

    def fake_fetch(url: str, target: Path) -> None:
        calls.append((url, target))
        target.write_bytes(b"fake-image-bytes")

    downloaded, skipped, failed = download_danbooru_images(
        metadata,
        targets,
        fetch=fake_fetch,
    )

    assert downloaded == 1
    assert skipped == 0
    assert failed == 0
    assert calls == [("https://example.test/1.jpg", image_root / "1.jpg")]
    assert (image_root / "1.jpg").exists()
    assert not (image_root / "2.jpg").exists()


def test_build_positive_images_prepares_manifests_end_to_end(tmp_path: Path) -> None:
    """``build_positive_images`` keeps only posts passing the score filter, fetches just their
    images, and writes the positive and hand-crop manifests with matching report counts.
    """
    metadata = tmp_path / "posts.jsonl"
    rows = [
        {
            "id": 1,
            "score": 50,
            "tag_string": "1girl solo full_body standing",
            "file_ext": "jpg",
            "file_url": "https://example.test/1.jpg",
        },
        {
            "id": 2,
            "score": 1,
            "tag_string": "1girl solo upper_body",
            "file_ext": "jpg",
            "file_url": "https://example.test/2.jpg",
        },
    ]
    _write_jsonl(metadata, rows)
    image_root = tmp_path / "images"
    positives = tmp_path / "positive_images.jsonl"
    hand_crops = tmp_path / "hand_crops.jsonl"

    def fake_fetch(url: str, target: Path) -> None:
        target.write_bytes(b"fake-image-bytes")

    report = build_positive_images(
        metadata=metadata,
        image_root=image_root,
        output=positives,
        hand_crops_output=hand_crops,
        fetch_images=True,
        fetch=fake_fetch,
    )

    pos_rows = [json.loads(line) for line in positives.read_text().splitlines() if line.strip()]
    assert len(pos_rows) == 1
    assert pos_rows[0]["post_id"] == 1
    assert (image_root / "1.jpg").exists()
    assert not (image_root / "2.jpg").exists()

    crop_rows = [json.loads(line) for line in hand_crops.read_text().splitlines() if line.strip()]
    assert len(crop_rows) == 1
    assert crop_rows[0]["labels"] == ["hand_ok"]
    assert report["positives_written"] == 1
    assert report["hand_crops_written"] == 1


def test_positive_and_hand_crop_rows(tmp_path: Path) -> None:
    """Checks positive image rows and hand crop rows."""
    metadata = tmp_path / "posts.jsonl"
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    _write_jsonl(
        metadata,
        [
            {
                "id": 10,
                "score": 5,
                "image_path": str(image),
                "tags": ["1girl", "solo", "full_body", "standing", "hands", "boots"],
                "hand_boxes": [[1, 2, 3, 4]],
            },
        ],
    )

    positives = positive_image_rows(metadata, source="test")
    assert positives == [
        {
            "image_path": str(image),
            "source": "test",
            "post_id": 10,
            "score": 5.0,
            "tags": ["1girl", "solo", "full_body", "standing", "hands", "boots"],
            "domain": "anime",
        },
    ]

    positive_manifest = tmp_path / "positives.jsonl"
    _write_jsonl(positive_manifest, positives)
    crops = hand_crop_rows([positive_manifest], label="hand_ok", source="test")
    assert crops == []


def test_build_safety_prompts_requires_danbooru_metadata(tmp_path: Path) -> None:
    """Safety prompts cannot be built without Danbooru metadata."""
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval_baseline.jsonl"

    try:
        build_safety_prompts(
            train_output=train_output,
            eval_output=eval_output,
        )
    except ValueError as exc:
        assert "Provide metadata" in str(exc)
    else:
        raise AssertionError("expected metadata requirement error")


def test_build_danbooru_safety_prompt_rows_uses_ratings_and_nsfw_tags(
    tmp_path: Path,
) -> None:
    """Safety prompt rows come only from explicit / questionable-rated posts, carry their NSFW
    tags and a ``rating:`` token in the prompt, and drop the dataset-level provenance keys
    (safety_target, domain, template_id).
    """
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 1,
                "rating": "e",
                "score": 9,
                "tag_string": "1girl solo nude breasts long_hair",
            },
            {
                "id": 2,
                "rating": "q",
                "score": 7,
                "tag_string": "1boy solo underwear standing short_hair",
            },
            {
                "id": 3,
                "rating": "g",
                "score": 20,
                "tag_string": "1girl solo full_body standing",
            },
            {
                "id": 4,
                "rating": "e",
                "score": 20,
                "tag_string": "1girl solo loli nude",
            },
        ],
    )

    rows = build_danbooru_safety_prompt_rows(
        metadata,
        limit=4,
        seed=0,
        candidate_limit=10,
    )
    train_rows, eval_rows = split_safety_prompt_rows(rows, train_limit=1, eval_limit=1)

    assert len(rows) == 2
    assert len(train_rows) == 1
    assert len(eval_rows) == 1
    assert {row["metadata"]["rating"] for row in rows} == {"explicit", "questionable"}
    assert all(row["metadata"]["nsfw_tags"] for row in rows)
    assert all("safety_target" not in row["metadata"] for row in rows)
    assert all("domain" not in row["metadata"] for row in rows)
    assert all("template_id" not in row["metadata"] for row in rows)
    assert all("rating:" in row["prompt"] for row in rows)


def test_build_safety_prompts_from_danbooru_metadata(tmp_path: Path) -> None:
    """``build_safety_prompts`` writes train / eval splits balanced across explicit and
    questionable ratings, plus a report with per-split rating counts and the top NSFW tags.
    """
    metadata = tmp_path / "posts.jsonl"
    hair_tags = [
        "black_hair",
        "brown_hair",
        "blonde_hair",
        "red_hair",
        "blue_hair",
        "pink_hair",
        "white_hair",
        "green_hair",
    ]
    _write_jsonl(
        metadata,
        [
            {
                "id": index,
                "rating": "e" if index % 2 == 0 else "q",
                "score": 5,
                "tag_string": f"1girl solo nude long_hair standing {hair_tags[index - 1]}",
            }
            for index in range(1, 9)
        ],
    )
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval.jsonl"
    report_output = tmp_path / "report.json"

    build_safety_prompts(
        metadata=metadata,
        train_output=train_output,
        eval_output=eval_output,
        report_output=report_output,
        train_limit=4,
        eval_limit=2,
    )

    train_rows = [json.loads(line) for line in train_output.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in eval_output.read_text().splitlines()]
    report = json.loads(report_output.read_text())

    assert len(train_rows) == 4
    assert len(eval_rows) == 2
    assert report["train_ratings"] == {"explicit": 2, "questionable": 2}
    assert report["eval_ratings"] == {"explicit": 1, "questionable": 1}
    assert "dataset_metadata" not in report
    assert "rating:explicit" in report["train_nsfw_tags_top"]
