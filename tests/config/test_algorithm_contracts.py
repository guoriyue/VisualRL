"""Algorithm-owned facts control validation without becoming YAML knobs."""

from dataclasses import asdict, replace
from typing import get_args

import pytest

from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.v_grpo import VGRPOConfig
from vrl.config.algorithm import algorithm_config_class
from vrl.config.schema import AlgorithmConfig, RootConfig


@pytest.mark.parametrize("kind", get_args(AlgorithmConfig.model_fields["kind"].annotation))
def test_every_algorithm_declares_non_configurable_facts(kind: str) -> None:
    config = algorithm_config_class(kind)()
    assert isinstance(config.config_contract, AlgorithmConfigContract)
    assert "config_contract" not in asdict(config)
    with pytest.raises(ValueError, match=r"unknown algorithm.config_contract"):
        AlgorithmConfig.model_validate({"kind": kind, "config_contract": {}})


@pytest.mark.parametrize(
    ("changes", "algorithm_fields", "error"),
    [
        ({"needs_sde_rollout": True}, {}, r"rollout.sde.type"),
        (
            {"supports_step_kl_reward": False},
            {"kl_reward_coef": 0.1},
            r"does not provide one",
        ),
    ],
)
def test_rules_follow_declared_facts_without_changing_kind(
    monkeypatch, changes: dict, algorithm_fields: dict, error: str
) -> None:
    payload = {"algorithm": {"kind": "v_grpo", **algorithm_fields}}
    RootConfig.model_validate(payload)
    monkeypatch.setattr(
        VGRPOConfig, "config_contract", replace(VGRPOConfig.config_contract, **changes)
    )
    with pytest.raises(ValueError, match=error):
        RootConfig.model_validate(payload)


def test_offline_surface_is_owned_by_algorithm_config(monkeypatch) -> None:
    from vrl.algorithms.dpo import DiffusionDPOConfig

    payload = {
        "algorithm": {"kind": "diffusion_dpo"},
        "trainer": {"seed": 123},
    }
    with pytest.raises(ValueError, match=r"trainer.seed"):
        RootConfig.model_validate(payload)
    contract = DiffusionDPOConfig.config_contract
    monkeypatch.setattr(
        DiffusionDPOConfig,
        "config_contract",
        replace(
            contract,
            consumed_sections=tuple(
                (name, allowed | {"seed"} if name == "trainer" else allowed)
                for name, allowed in contract.consumed_sections
            ),
        ),
    )
    RootConfig.model_validate(payload)
