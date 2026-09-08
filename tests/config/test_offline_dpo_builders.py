"""Offline DPO trainer config bridges only the knobs the trainer consumes.

These are theorems of ``vrl.trainers.offline.OfflineDPOTrainerConfig.from_root``
over the shipped ``experiment/wan_2_1/offline_dpo_pickapic`` recipe.
"""

from __future__ import annotations

import pytest

from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.trainers.offline import OfflineDPOTrainerConfig


def _resolved_trainer_config(overrides: list[str] | None = None):
    cfg = load_config(
        "experiment/wan_2_1/offline_dpo_pickapic",
        overrides=overrides,
    )
    return OfflineDPOTrainerConfig.from_root(
        parse_config(cfg),
        DiffusionDPOConfig(beta=123.0, sft_weight=0.25),
    )


def test_offline_dpo_bridges_every_supported_adam_knob() -> None:
    resolved = _resolved_trainer_config(
        [
            "actor.optim.lr=0.01",
            "actor.optim.adam_beta1=0.7",
            "actor.optim.adam_beta2=0.8",
            "actor.optim.weight_decay=0.03",
            "actor.optim.eps=1e-6",
            "actor.scale_lr=true",
            "actor.train_batch_size=2",
            "actor.gradient_accumulation_steps=3",
        ],
    )

    assert resolved.beta == pytest.approx(123.0)
    assert resolved.sft_weight == pytest.approx(0.25)
    assert resolved.lr == pytest.approx(0.06)
    assert resolved.adam_beta1 == pytest.approx(0.7)
    assert resolved.adam_beta2 == pytest.approx(0.8)
    assert resolved.adam_weight_decay == pytest.approx(0.03)
    assert resolved.adam_epsilon == pytest.approx(1e-6)


def test_offline_dpo_uses_typed_optimizer_defaults_when_keys_are_absent() -> None:
    resolved = _resolved_trainer_config(["actor.scale_lr=false"])

    assert resolved.adam_beta1 == pytest.approx(0.9)
    assert resolved.adam_beta2 == pytest.approx(0.999)
    assert resolved.adam_weight_decay == pytest.approx(1e-4)
    assert resolved.adam_epsilon == pytest.approx(1e-8)


def test_offline_dpo_bridges_explicit_max_grad_norm() -> None:
    resolved = _resolved_trainer_config(["actor.max_norm=0.25"])

    assert resolved.max_grad_norm == pytest.approx(0.25)


def test_offline_dpo_rejects_unsupported_8bit_optimizer() -> None:
    with pytest.raises(ValueError, match=r"optim_8bit=true is not supported"):
        _resolved_trainer_config(["actor.optim.optim_8bit=true"])


def test_offline_dpo_rejects_explicit_adamw_only_knobs_for_adafactor() -> None:
    with pytest.raises(
        ValueError,
        match=r"use_adafactor=true does not consume AdamW-only key\(s\): actor\.optim\.adam_beta1",
    ):
        _resolved_trainer_config(
            ["actor.use_adafactor=true", "actor.optim.adam_beta1=0.8"],
        )


def test_offline_dpo_adafactor_keeps_shared_optimizer_knobs() -> None:
    resolved = _resolved_trainer_config(
        [
            "actor.use_adafactor=true",
            "actor.scale_lr=false",
            "actor.optim.lr=2e-7",
            "actor.optim.weight_decay=0.03",
        ],
    )

    assert resolved.use_adafactor is True
    assert resolved.lr == pytest.approx(2e-7)
    assert resolved.adam_weight_decay == pytest.approx(0.03)
