"""The three validation tiers stay where they are declared.

Tier 1 (section shapes) is ``schema.py``; tier 2 (cross-section rules) is the
``CROSS_SECTION_RULES`` registry in ``rules.py``, run by ``RootConfig``'s own
validator; tier 3 (launch gates) is the ``TRAINING_GATES`` registry in
``validation.py``, run by ``require_training_config``. These pin the seams a
new check must go through, so a rule cannot quietly grow back into the pydantic
model or a gate into ``parse_config``.
"""

from __future__ import annotations

import inspect

import pytest
from omegaconf import OmegaConf

from vrl.config import rules, validation
from vrl.config.schema import RootConfig, parse_config


def test_every_cross_section_rule_is_a_named_root_function() -> None:
    assert rules.CROSS_SECTION_RULES
    for rule in rules.CROSS_SECTION_RULES:
        assert rule.__name__.startswith("rule_"), rule.__name__
        assert list(inspect.signature(rule).parameters) == ["root"], rule.__name__
        assert rule.__doc__, f"{rule.__name__} must say which relationship it guards"


def test_every_training_gate_takes_the_root_and_the_precision_policy() -> None:
    assert validation.TRAINING_GATES
    for gate in validation.TRAINING_GATES:
        parameters = list(inspect.signature(gate).parameters)
        assert parameters == ["root", "precision"], gate.__name__


def test_cross_section_rules_fire_on_direct_root_construction() -> None:
    """A caller that bypasses ``parse_config`` still gets tier 2."""

    with pytest.raises(ValueError, match="token_grpo_multisegment"):
        RootConfig.model_validate(
            {"model": {"family": "janus_pro_r1"}, "algorithm": {"kind": "token_grpo"}}
        )


def test_rules_module_stays_import_light() -> None:
    """Tier 2 runs on every parse, eval tools included: no torch, no runtime modules."""

    source = inspect.getsource(rules)
    for forbidden in ("import torch", "from vrl.trainers", "from vrl.models.interfaces"):
        assert forbidden not in source, forbidden


def test_launch_gates_do_not_run_inside_parse_config() -> None:
    """A gate that needs the precision policy or the filesystem must not tax
    ``parse_config`` callers: the production gate reads manifests, so a config
    that enables it parses but fails only through ``require_training_config``."""

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5"},
            "precision": {"float32_precision": "tf32", "training": {"dtype": "bf16"}},
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )

    root = parse_config(cfg)

    assert root.production is not None
    with pytest.raises(ValueError, match=r"production\.kling_video_reward requires"):
        validation.require_training_config(cfg)


def test_dataset_provenance_requires_existing_paths(tmp_path) -> None:
    from vrl.trainers.data.provenance import DatasetProvenance

    manifest = tmp_path / "train.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    root = parse_config(
        OmegaConf.create(
            {
                "data": {
                    "loader": "prompt_manifest",
                    "manifest": str(manifest),
                    "eval_manifest": str(tmp_path / "missing.jsonl"),
                    "source_report": str(tmp_path / "report.json"),
                    "task_type": "text_to_video",
                    "preprocessing": {},
                    "sampler": {"type": "random_without_replacement"},
                }
            }
        )
    )

    with pytest.raises(ValueError, match=r"data\.eval_manifest does not exist"):
        DatasetProvenance.from_config(root.data)
