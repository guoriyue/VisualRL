"""A rollout approximation must not run with every drift correction off.

Quantization is already covered without a new check: it changes the rollout
precision label, so ``PrecisionPolicy.stages_match`` goes False and the trainer
installs TIS correction plus a drift guard whose default ``mode="auto"``
resolves to ``"fail"``.

TeaCache is the uncovered case, and the reason this check exists. It reuses a
cached ``noise_pred`` on skipped denoise steps -- so the collection-time
log-prob no longer matches the trainer's exact replay forward -- while leaving
both roles' precision labels identical. Every automatic correction stays off and
the run trains on uncorrected off-policy gradients while still reporting
convergence.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.config.validation import validate_guarded_rollout_drift


def _cfg(**overrides) -> OmegaConf:
    """A minimal config carrying only what the drift check reads."""

    base = {
        "model": {"family": "sd3_5"},
        "precision": {
            "float32_precision": "tf32",
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "bf16"},
        },
    }
    base.update(overrides)
    return parse_config(OmegaConf.create(base))


def _precision(root) -> object:
    return PrecisionPolicy.from_section(root.precision)


def test_teacache_without_any_correction_is_refused() -> None:
    cfg = _cfg(sampling={"teacache": True})

    with pytest.raises(ValueError, match=r"teacache.*no drift guard"):
        validate_guarded_rollout_drift(cfg, _precision(cfg))


def test_validate_training_config_refuses_unguarded_teacache() -> None:
    """The check runs on the real entry point, not only when called directly."""

    from vrl.config.loading import load_config
    from vrl.config.validation import require_training_config

    cfg = load_config("experiment/sd3_5/online_grpo_ocr", overrides=["sampling.teacache=true"])

    with pytest.raises(ValueError, match="teacache"):
        require_training_config(cfg)


def test_teacache_mapping_form_is_refused_too() -> None:
    """``teacache: {threshold: ...}`` enables it just as ``teacache: true`` does."""

    cfg = _cfg(sampling={"teacache": {"threshold": 0.2}})

    with pytest.raises(ValueError, match="teacache"):
        validate_guarded_rollout_drift(cfg, _precision(cfg))


@pytest.mark.parametrize(
    "sampling",
    [
        {},
        {"teacache": False},
        {"teacache": {"enabled": False}},
    ],
)
def test_teacache_off_passes(sampling: dict) -> None:
    cfg = _cfg(sampling=sampling)

    validate_guarded_rollout_drift(cfg, _precision(cfg))


def test_quantized_rollout_needs_no_extra_check() -> None:
    """A precision split already arms the guard + TIS, so this must not fire.

    Firing here would reject every fp8 rollout that also uses TeaCache, even
    though such a run is exactly the covered case.
    """

    cfg = _cfg(
        sampling={"teacache": True},
        precision={
            "float32_precision": "tf32",
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "bf16", "quantization": {"format": "fp8"}},
        },
    )
    assert not _precision(cfg).stages_match

    validate_guarded_rollout_drift(cfg, _precision(cfg))


@pytest.mark.parametrize(
    "expert_block",
    ["precision_drift_guard", "precision_correction"],
)
def test_explicit_expert_block_is_honored(expert_block: str) -> None:
    """The same escape hatch the precision-split default path respects.

    ``TrainerConfig.from_root`` only fills these in when the user has not,
    so an explicit block means the correction policy was chosen deliberately.
    """

    # Each expert block's own vocabulary (the parsed root rejects a foreign key).
    explicit = {
        "precision_drift_guard": {"mode": "warn"},
        "precision_correction": {"tis_mode": "truncate"},
    }[expert_block]
    cfg = _cfg(sampling={"teacache": True}, trainer={expert_block: explicit})

    validate_guarded_rollout_drift(cfg, _precision(cfg))


# --- the compile compatibility matrix ----------------------------------------
#
# One home for every torch.compile incompatibility. Each was discovered
# separately and used to live somewhere different -- grad-checkpointing in the
# trainer, FSDP2 in the strategy builder, sequence parallelism nowhere at all --
# so adding the next one meant first finding the others.


def _compile_cfg(**overrides) -> OmegaConf:
    """A config with compile on and nothing else set, plus ``overrides``."""

    base = {"model": {"family": "sd3_5", "torch_compile": {"enable": True}}}
    base.update(overrides)
    return OmegaConf.create(base)


def test_compile_alone_has_no_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    assert compile_conflicts(parse_config(_compile_cfg())) == ()


def test_compile_off_never_conflicts() -> None:
    """Every other feature is free as long as compile is not requested."""
    from vrl.config.validation import compile_conflicts

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
            "actor": {"gradient_checkpointing": True},
            "distributed": {
                "training": {"strategy": "fsdp"},
                "resources": {"rollout": {"gpus_per_engine": 4}},
            },
        },
    )

    assert compile_conflicts(parse_config(cfg)) == ()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"actor": {"gradient_checkpointing": True}}, "gradient_checkpointing"),
        ({"distributed": {"training": {"strategy": "fsdp"}}}, "strategy=fsdp"),
        (
            {"distributed": {"resources": {"rollout": {"gpus_per_engine": 2}}}},
            "gpus_per_engine=2",
        ),
    ],
)
def test_each_incompatible_feature_is_reported(overrides: dict, expected: str) -> None:
    from vrl.config.validation import compile_conflicts

    conflicts = compile_conflicts(parse_config(_compile_cfg(**overrides)))

    assert len(conflicts) == 1
    assert expected in conflicts[0].message


def test_several_conflicts_are_reported_together() -> None:
    """One pass over the matrix, not fail-on-first: the user sees all of them."""
    from vrl.config.validation import compile_conflicts

    conflicts = compile_conflicts(
        parse_config(
            _compile_cfg(
                actor={"gradient_checkpointing": True},
                distributed={
                    "training": {"strategy": "fsdp"},
                    "resources": {"rollout": {"gpus_per_engine": 2}},
                },
            )
        ),
    )

    assert len(conflicts) == 3


def test_require_compile_compatible_raises_with_every_reason() -> None:
    from vrl.config.validation import validate_compile_compatible

    with pytest.raises(ValueError) as excinfo:
        validate_compile_compatible(
            parse_config(
                _compile_cfg(
                    actor={"gradient_checkpointing": True},
                    distributed={"training": {"strategy": "fsdp"}},
                )
            ),
        )
    message = str(excinfo.value)
    assert "gradient_checkpointing" in message
    assert "strategy=fsdp" in message


# Every conflict in the matrix binds one build role, so ``scope`` releases the
# conflicts of the role it excludes -- and ONLY those. The rollout-scope case is
# the production unlock: FSDP2 + checkpointing trainers with a compiled rollout.


def _scoped_compile_cfg(scope: str, **overrides) -> OmegaConf:
    cfg = _compile_cfg(**overrides)
    cfg.model.torch_compile.scope = scope
    return cfg


def test_rollout_scope_releases_trainer_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg(
        "rollout",
        actor={"gradient_checkpointing": True},
        distributed={"training": {"strategy": "fsdp"}},
    )

    assert compile_conflicts(parse_config(cfg)) == ()


def test_rollout_scope_keeps_rollout_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg(
        "rollout",
        distributed={"resources": {"rollout": {"gpus_per_engine": 2}}},
    )

    conflicts = compile_conflicts(parse_config(cfg))
    assert len(conflicts) == 1
    assert conflicts[0].feature == "sequence_parallel"
    assert "gpus_per_engine=2" in conflicts[0].message


def test_replay_scope_releases_rollout_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg(
        "replay",
        distributed={"resources": {"rollout": {"gpus_per_engine": 2}}},
    )

    assert compile_conflicts(parse_config(cfg)) == ()


def test_replay_scope_keeps_trainer_conflicts() -> None:
    from vrl.config.validation import compile_conflicts

    cfg = _scoped_compile_cfg("replay", actor={"gradient_checkpointing": True})

    conflicts = compile_conflicts(parse_config(cfg))
    assert len(conflicts) == 1
    assert conflicts[0].feature == "gradient_checkpointing"
    assert "gradient_checkpointing" in conflicts[0].message


def test_unknown_compile_scope_is_refused_at_config_load() -> None:
    from vrl.config.validation import compile_conflicts

    with pytest.raises(ValueError, match=r"torch_compile\.scope must be one of"):
        compile_conflicts(parse_config(_scoped_compile_cfg("trainer")))
