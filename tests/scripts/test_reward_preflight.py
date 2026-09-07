"""The reward preflight runs the configured reward over the configured rows.

Driven end to end on a CPU-only, model-free reward (``image_sharpness``) over a
bundled prompt manifest: the real config build, the real reward factory and
runtime, the collector's own metadata projection, one real ``score`` call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.config.loading import load_config
from vrl.scripts.rewards.preflight import _synthetic_media, main, preflight_rewards


def _config(tmp_path, *overrides: str):
    return load_config(
        "experiment/anima_preview3/online_grpo",
        overrides=[
            "+reward=image_sharpness",
            "+dataset=drawbench_train_192",
            "actor.optim.lr=1e-5",
            f"trainer.output_dir={tmp_path / 'out'}",
            *overrides,
        ],
    )


def test_preflight_scores_every_row_with_every_component(tmp_path) -> None:
    report = preflight_rewards(_config(tmp_path), prompts=2, device=torch.device("cpu"), seed=3)

    assert len(report.prompts) == 2
    assert set(report.output.components) == {"image_sharpness"}
    assert len(report.output.scores) == 2
    assert report.lines()[0].startswith("prompt")


def test_preflight_can_score_the_eval_manifest(tmp_path) -> None:
    report = preflight_rewards(
        _config(tmp_path), prompts=1, use_eval_manifest=True, device=torch.device("cpu")
    )

    assert len(report.prompts) == 1


def test_main_reports_a_missing_manifest_as_a_failure(tmp_path, capsys) -> None:
    code = main(
        [
            "--config",
            "experiment/anima_preview3/online_grpo",
            "--device",
            "cpu",
            "+reward=image_sharpness",
            "+dataset=drawbench_train_192",
            "actor.optim.lr=1e-5",
            f"trainer.output_dir={tmp_path / 'out'}",
            f"data.manifest={tmp_path / 'missing.txt'}",
        ]
    )

    assert code == 1
    assert "✓ reward preflight" not in capsys.readouterr().out


def test_main_prints_the_score_table_on_success(tmp_path, capsys) -> None:
    code = main(
        [
            "--config",
            "experiment/anima_preview3/online_grpo",
            "--prompts",
            "1",
            "--device",
            "cpu",
            "+reward=image_sharpness",
            "+dataset=drawbench_train_192",
            "actor.optim.lr=1e-5",
            f"trainer.output_dir={tmp_path / 'out'}",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "image_sharpness" in out
    assert "✓ reward preflight: 1 row(s)" in out


@pytest.mark.parametrize(
    ("task", "expected_shape"),
    [("t2i", (3, 32, 48)), ("t2v", (3, 5, 32, 48)), ("i2v", (3, 5, 32, 48))],
)
def test_synthetic_media_follows_the_collector_layout(task: str, expected_shape) -> None:
    sampling = SimpleNamespace(height=32, width=48, num_frames=5)

    media = _synthetic_media(sampling, task, seed=0)

    assert tuple(media.shape) == expected_shape
    assert float(media.min()) >= 0.0 and float(media.max()) <= 1.0
    assert torch.equal(media, _synthetic_media(sampling, task, seed=0))
