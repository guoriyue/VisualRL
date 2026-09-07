"""The new-algorithm recipes compose into runnable configs.

DanceGRPO / Flow-DPPO / GRPO-Guard each ship a base/algorithm config + an
online recipe. These pin that swapping the recipe into a real diffusion
experiment resolves to the right algorithm type and sets the knobs each
algorithm needs (and nothing trips the unknown-key check).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.grpo.continuous import FlowDPPOConfig, GRPOConfig, GRPOGuardConfig
from vrl.algorithms.grpo.token import TokenGRPOConfig
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.schema import parse_config

# A complete diffusion GRPO experiment; we swap only its /recipe/online group so
# the new recipe inherits a full model/sampling/reward/dataset environment.
_BASE = "experiment/sd3_5/online_grpo_ocr"


def _load(recipe: str):
    return load_config(_BASE, overrides=[f"/recipe/online={recipe}"])


def test_dance_grpo_recipe_resolves_with_random_timestep_selection() -> None:
    cfg = _load("flow_matching_dance_grpo")
    assert cfg.algorithm.kind == "dance_grpo"
    assert isinstance(parse_config(cfg).algorithm.hyperparameters, GRPOConfig)
    # The defining knob: random per-update timestep subset reaches TrainerConfig.
    assert build_configs(cfg).trainer.timestep_selection == "random"


def test_flow_dppo_recipe_resolves_and_enables_proposal_mean_storage() -> None:
    cfg = _load("flow_matching_dppo")
    assert cfg.algorithm.kind == "flow_dppo"
    assert isinstance(parse_config(cfg).algorithm.hyperparameters, FlowDPPOConfig)
    # Required for the trust-region loss; without it generation never stores the
    # rollout proposal mean and the loss fails fast.
    assert cfg.rollout.return_prev_sample_mean is True


def test_grpo_guard_recipe_resolves_and_enables_proposal_mean_storage() -> None:
    cfg = _load("flow_matching_grpo_guard")
    assert cfg.algorithm.kind == "grpo_guard"
    assert isinstance(parse_config(cfg).algorithm.hyperparameters, GRPOGuardConfig)
    assert cfg.rollout.return_prev_sample_mean is True


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("flow_dppo", "clip_ratio", 0.2),
        ("grpo_guard", "kl_coef", 0.1),
        ("token_grpo", "flow_kl_use_dt", True),
    ],
)
def test_algorithm_configs_reject_unconsumed_knobs(
    kind: str,
    field: str,
    value: object,
) -> None:
    cfg = OmegaConf.create({"algorithm": {"kind": kind, field: value}})

    with pytest.raises(ValueError, match=rf"algorithm\.{field}"):
        parse_config(cfg)


def test_token_grpo_keeps_its_clipping_and_reference_kl_config() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "kind": "token_grpo",
                "clip_ratio": 0.3,
                "kl_coef": 0.2,
                "kl_estimator": "k2",
            },
        },
    )

    built = parse_config(cfg).algorithm.hyperparameters

    assert isinstance(built, TokenGRPOConfig)
    assert built.clip_ratio == pytest.approx(0.3)
    assert built.kl_coef == pytest.approx(0.2)
    assert built.kl_estimator == "k2"


@pytest.mark.parametrize(
    "recipe",
    ["flow_matching_dance_grpo", "flow_matching_dppo", "flow_matching_grpo_guard", "v_grpo"],
)
def test_recipes_have_no_unknown_config_keys(recipe: str) -> None:
    """An unknown key fails parse_config, so a clean build is the assertion."""
    build_configs(_load(recipe))
