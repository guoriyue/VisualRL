"""Pairing and rollout requirements follow owner declarations, including new families."""

from dataclasses import replace

import pytest

from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.config.schema import RootConfig
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    FamilyTrainingContract,
    get_model_family_entry,
)


@pytest.mark.parametrize("family", ["nextstep_1", "nextstep"])
def test_noise_level_requirement_preserves_zero_and_algorithm_scope(family: str) -> None:
    payload = {"model": {"family": family}, "algorithm": {"kind": "token_grpo"}}
    with pytest.raises(ValueError, match=r"rollout\.noise_level"):
        RootConfig.model_validate(payload)
    RootConfig.model_validate({**payload, "rollout": {"noise_level": 0.0}})
    RootConfig.model_validate({"model": {"family": family}})
    RootConfig.model_validate({"model": {"family": family}, "algorithm": {"kind": "v_grpo"}})


@pytest.mark.parametrize("family", ["janus_pro_r1", "janus_r1"])
def test_multisegment_pairing_and_rollout_requirements(family: str) -> None:
    RootConfig.model_validate({"model": {"family": family}})
    with pytest.raises(ValueError, match=r"requires algorithm\.kind=token_grpo_multisegment"):
        RootConfig.model_validate(
            {"model": {"family": family}, "algorithm": {"kind": "token_grpo"}}
        )
    payload = {"model": {"family": family}, "algorithm": {"kind": "token_grpo_multisegment"}}
    with pytest.raises(ValueError, match=r"rollout\.final_image_policy"):
        RootConfig.model_validate(payload)
    for policy in ("always_generate", "use_selfcheck"):
        RootConfig.model_validate({**payload, "rollout": {"final_image_policy": policy}})


def test_multisegment_requires_model_section() -> None:
    with pytest.raises(ValueError, match=r"requires model\.family=janus_pro_r1"):
        RootConfig.model_validate({"algorithm": {"kind": "token_grpo_multisegment"}})


def test_new_family_constraints_work_without_new_rules(monkeypatch) -> None:
    entry = replace(
        get_model_family_entry("nextstep_1"),
        family="contract_test_family",
        training_contract=FamilyTrainingContract(
            required_algorithm="v_grpo",
            rollout_fields_by_algorithm=(("v_grpo", ("noise_level",)),),
        ),
    )
    monkeypatch.setitem(FAMILY_REGISTRY, entry.family, entry)
    with pytest.raises(ValueError, match=r"requires algorithm\.kind=v_grpo"):
        RootConfig.model_validate(
            {"model": {"family": entry.family}, "algorithm": {"kind": "token_grpo"}}
        )
    payload = {"model": {"family": entry.family}, "algorithm": {"kind": "v_grpo"}}
    with pytest.raises(ValueError, match=r"rollout\.noise_level"):
        RootConfig.model_validate(payload)
    RootConfig.model_validate({**payload, "rollout": {"noise_level": 0.0}})


def test_algorithm_pairing_and_rollout_fields_follow_its_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        MultiSegmentTokenGRPOConfig,
        "config_contract",
        replace(
            MultiSegmentTokenGRPOConfig.config_contract,
            required_model_family="janus_pro",
            required_rollout_fields=("noise_level",),
        ),
    )
    payload = {"model": {"family": "janus"}, "algorithm": {"kind": "token_grpo_multisegment"}}
    with pytest.raises(ValueError, match=r"rollout\.noise_level"):
        RootConfig.model_validate(payload)
    RootConfig.model_validate({**payload, "rollout": {"noise_level": 0.0}})


def test_declared_training_references_resolve_to_schema_fields() -> None:
    from typing import get_args

    from vrl.config.algorithm import algorithm_config_class
    from vrl.config.schema import AlgorithmConfig, RolloutConfig

    kinds = get_args(AlgorithmConfig.model_fields["kind"].annotation)
    for entry in FAMILY_REGISTRY.values():
        contract = entry.training_contract
        assert contract.required_algorithm is None or contract.required_algorithm in kinds
        for kind, fields in contract.rollout_fields_by_algorithm:
            assert kind in kinds
            assert set(fields) <= RolloutConfig.model_fields.keys()
    for kind in kinds:
        contract = algorithm_config_class(kind).config_contract
        assert (
            contract.required_model_family is None
            or contract.required_model_family in FAMILY_REGISTRY
        )
        assert set(contract.required_rollout_fields) <= RolloutConfig.model_fields.keys()


def test_rules_do_not_embed_algorithm_or_family_names() -> None:
    import ast
    import inspect
    from typing import get_args

    from vrl.config import rules
    from vrl.config.schema import AlgorithmConfig

    names = set(FAMILY_REGISTRY) | set(get_args(AlgorithmConfig.model_fields["kind"].annotation))
    literals = {
        node.value
        for node in ast.walk(ast.parse(inspect.getsource(rules)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (literals & names)
