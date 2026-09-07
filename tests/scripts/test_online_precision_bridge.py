"""Integration: the unified precision policy drives the online trainer (P1)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tests.config.test_load_all_experiments import (
    _experiment_names,
    _load_experiment_for_static_validation,
)
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.models.dtypes import resolve_torch_dtype
from vrl.trainers.core.types import PrecisionDriftGuardConfig

# Derive every online recipe from the experiment glob (the single source of
# truth in test_load_all_experiments) rather than hand-maintaining a subset —
# the old hand list had drifted and only exercised four recipes. "online" ==
# every experiment whose final path component is not an `offline_*` recipe.
_RECIPES = [name for name in _experiment_names() if not Path(name).name.startswith("offline_")]


@pytest.mark.parametrize("experiment", _RECIPES)
def test_bridge_uses_aligned_public_precision(experiment):
    """Checks bridge derives trainer precision from public precision."""
    cfg = _load_experiment_for_static_validation(experiment)
    trainer_config = build_configs(cfg).trainer
    policy = PrecisionPolicy.from_section(parse_config(cfg).precision)
    # The bridge contract is the role-label equality below (train/rollout labels
    # equal the resolved policy labels); the label -> torch dtype resolution is
    # covered by the plain-policy cases and resolve_torch_dtype's own tests.
    assert trainer_config.train_precision == policy.training.label
    assert trainer_config.rollout_precision == policy.rollout.label
    assert policy.training == policy.rollout


def test_precision_block_drives_trainer():
    """``precision.training.dtype`` is what the trainer's ``train_precision`` resolves to; fp32
    and bf16 both round-trip through the dtype resolver.
    """
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    cfg = _with_precision("sd3_5/online_grpo_ocr", _plain_policy("fp32"))
    assert resolve_torch_dtype(build_configs(cfg).trainer.train_precision) is torch.float32

    cfg = _with_precision("sd3_5/online_grpo_ocr", _plain_policy("bf16"))
    trainer_config = build_configs(cfg).trainer
    assert trainer_config.train_precision == "bf16"
    assert resolve_torch_dtype(trainer_config.train_precision) is torch.bfloat16


def test_fp16_precision_block_drives_trainer_and_rollout():
    """Checks scalar fp16 drives both replay compute and rollout precision."""
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"precision": _plain_policy("fp16")}))

    built = build_configs(cfg)
    trainer_config = built.trainer

    assert trainer_config.train_precision == "fp16"
    assert trainer_config.rollout_precision == "fp16"
    assert built.precision.diffusion_math == "fp32"
    assert resolve_torch_dtype(trainer_config.train_precision) is torch.float16


@pytest.mark.parametrize(
    ("format_name", "expected_label"),
    [("fp8", "bf16+fp8"), ("nvfp4", "bf16+nvfp4")],
)
def test_rollout_precision_split_auto_derives_correction_policy(
    format_name: str,
    expected_label: str,
):
    """A low-precision rollout split is a user intent; correction is derived."""
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _rollout_quantization_policy(format_name),
    )

    trainer_config = build_configs(cfg).trainer

    assert trainer_config.train_precision == "bf16"
    assert trainer_config.rollout_precision == expected_label
    # The auto split-precision policy is whatever the builder helper installs;
    # assert the whole struct equals that single source, not a per-field copy of
    # its constants (which would falsely fail on any retune of the policy).
    assert trainer_config.precision_correction == PrecisionCorrectionConfig(
        tis_mode="truncate",
        rs_mode="seq_mean_k1",
    )
    assert trainer_config.precision_drift_guard == PrecisionDriftGuardConfig(
        mode="fail",
        max_abs_log_ratio=math.log(10.0),
        max_ratio_abs_dev=9.0,
        fail_on_nonfinite=True,
    )


def test_no_split_means_no_auto_correction_policy() -> None:
    """rollout == train: the builder early-returns and installs no auto policy."""
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _plain_policy("bf16"),
    )

    trainer_config = build_configs(cfg).trainer

    # No split -> the correction/guard fields keep their dataclass defaults; the
    # split-only policy (TIS truncate / drift_guard mode="fail") is NOT installed.
    assert trainer_config.precision_correction == PrecisionCorrectionConfig()
    assert trainer_config.precision_drift_guard == PrecisionDriftGuardConfig()


def test_outer_autocast_split_is_preserved_in_trainer_role_labels() -> None:
    block = _plain_policy("bf16")
    block["rollout"]["outer_autocast"] = False
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)

    trainer_config = build_configs(cfg).trainer

    assert trainer_config.train_precision == "bf16"
    assert trainer_config.rollout_precision == "bf16+no-autocast"
    assert trainer_config.precision_correction.tis_mode == "truncate"
    assert trainer_config.precision_drift_guard.mode == "fail"


def test_explicit_precision_correction_is_respected_on_rollout_split():
    """Expert correction blocks override the auto split-precision defaults."""
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _rollout_fp8_policy(),
    )
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "trainer": {
                    "precision_correction": {"tis_mode": "off", "rs_mode": "off"},
                    "precision_drift_guard": {"mode": "warn", "max_abs_log_ratio": 0.25},
                },
            },
        ),
    )

    trainer_config = build_configs(cfg).trainer

    assert trainer_config.precision_correction.tis_mode == "off"
    assert trainer_config.precision_correction.rs_mode == "off"
    assert trainer_config.precision_drift_guard.mode == "warn"
    assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(0.25)


def _with_precision(experiment, block):
    cfg = load_config(f"experiment/{experiment}")
    return OmegaConf.merge(cfg, OmegaConf.create({"precision": block}))


def _plain_policy(dtype: str) -> dict:
    return {
        "float32_precision": "tf32",
        "training": {"dtype": dtype},
        "rollout": {"dtype": dtype},
    }


def _rollout_fp8_policy() -> dict:
    return _rollout_quantization_policy("fp8")


def _rollout_quantization_policy(format_name: str) -> dict:
    block = _plain_policy("bf16")
    block["rollout"]["quantization"] = {"format": format_name}
    return block


@pytest.mark.parametrize("math,expected", [("fp32", torch.float32), ("bf16", torch.bfloat16)])
def test_math_axis_resolves_to_dtype(math, expected):
    # P2: the `math` axis resolves to the evaluator's log-prob math dtype.
    from vrl.config.precision import PrecisionPolicy
    from vrl.models.dtypes import resolve_torch_dtype

    block = _plain_policy("fp32")
    block["diffusion_math"] = {"dtype": math}
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)
    assert (
        resolve_torch_dtype(
            PrecisionPolicy.from_section(parse_config(cfg).precision).diffusion_math
        )
        is expected
    )
    assert build_configs(cfg).precision.diffusion_math == math


def test_precision_drift_guard_config_is_bridged_from_yaml():
    """Checks precision drift guard YAML config reaches TrainerConfig."""
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "trainer": {
                    "precision_drift_guard": {
                        "mode": "fail",
                        "max_abs_log_ratio": 0.02,
                    },
                },
            },
        ),
    )

    trainer_config = build_configs(cfg).trainer

    assert trainer_config.precision_drift_guard.mode == "fail"
    assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(0.02)


def test_replay_parity_config_is_bridged_from_yaml() -> None:
    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        _plain_policy("bf16"),
    )
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {"trainer": {"replay_parity": {"max_abs_logprob_diff": 2.0e-4}}},
        ),
    )

    trainer_config = build_configs(cfg).trainer

    assert trainer_config.replay_parity.max_abs_logprob_diff == pytest.approx(2.0e-4)


def test_online_metrics_csv_includes_logprob_mismatch_metrics(tmp_path):
    """Mismatch + continuous-async diagnostics are written as regular CSV columns."""
    from types import SimpleNamespace

    from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
    from vrl.algorithms.types import InitialReplayStats, PolicyUpdateStats, TrainStepMetrics
    from vrl.scripts.common.online import OnlineRecipeRun

    csv_path = tmp_path / "metrics.csv"
    # The metrics-CSV side effects live on the run controller. Unused training
    # owners can be inert objects for this row-formatting test.
    run = OnlineRecipeRun(
        bundle=SimpleNamespace(),
        trainer=SimpleNamespace(),
        strategy=SimpleNamespace(),
        family="sd3_5",
        component_names=(),
        adapter_exports=None,
        csv_path=csv_path,
        rng=None,
        resume_epoch=None,
        model_identity={"schema": "test"},
    )
    run.prepare_metrics_csv()
    run.write_metric_row(
        0,
        TrainStepMetrics(
            loss=1.0,
            policy_loss=2.0,
            weighted_kl_loss=0.025,
            update=PolicyUpdateStats(
                active_clip_fraction=0.15,
                tis_clip_fraction=0.09,
                rs_seq_masked_fraction=0.11,
            ),
            initial_replay=InitialReplayStats(
                clip_fraction=0.16,
                active_clip_fraction=0.07,
                logprob_abs_diff_max=0.008,
            ),
            logprob_mismatch=LogprobMismatchStats(
                logprob_abs_diff_mean=0.1,
                logprob_abs_diff_max=0.2,
                ratio_abs_dev_mean=0.3,
                ratio_abs_dev_max=0.4,
                mismatch_kl=-0.5,
                mismatch_k3_kl=0.6,
            ),
            phase_times={
                "continuous.stale_policy_versions": 1.0,
                "continuous.weight_sync_pause_s": 0.25,
                "continuous.lookahead_requested": 1.0,
                "continuous.producer_completed": 2.0,
            },
        ),
    )

    header, row = csv_path.read_text().splitlines()
    assert "logprob_abs_diff_mean" in header
    assert "weighted_kl_loss" in header
    assert "pre_update_logprob_abs_diff_max" in header
    assert "tis_clip_fraction" in header
    assert "ratio_abs_dev_max" in header
    assert "mismatch_k3_kl" in header
    assert "continuous_stale_versions" in header
    assert "continuous_lookahead_requested" in header
    assert "continuous_producer_completed" in header
    values = dict(zip(header.split(","), row.split(","), strict=True))
    # Assert the numeric value landed in the right column, not the CSV float
    # format width (.6f / .4f zero-padding is a display detail, not a contract).
    assert float(values["logprob_abs_diff_mean"]) == pytest.approx(0.1)
    assert float(values["weighted_kl_loss"]) == pytest.approx(0.025)
    assert float(values["active_clip_fraction"]) == pytest.approx(0.15)
    assert float(values["pre_update_clip_fraction"]) == pytest.approx(0.16)
    assert float(values["pre_update_active_clip_fraction"]) == pytest.approx(0.07)
    assert float(values["pre_update_logprob_abs_diff_max"]) == pytest.approx(0.008)
    assert float(values["tis_clip_fraction"]) == pytest.approx(0.09)
    assert float(values["rs_seq_masked_fraction"]) == pytest.approx(0.11)
    assert float(values["ratio_abs_dev_max"]) == pytest.approx(0.4)
    assert float(values["mismatch_kl"]) == pytest.approx(-0.5)
    # Continuous-async diagnostics sourced from TrainStepMetrics.phase_times.
    assert float(values["continuous_stale_versions"]) == pytest.approx(1.0)
    assert float(values["continuous_weight_sync_pause_s"]) == pytest.approx(0.25)
    assert float(values["continuous_lookahead_requested"]) == pytest.approx(1.0)
    assert float(values["continuous_producer_completed"]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "prompt_encoder,expected",
    [("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_prompt_encoder_axis_in_model_build(prompt_encoder, expected):
    """Prompt-encoder precision reaches the runtime as a real torch dtype."""
    # sd3_5 is a registry-descriptor family: its build comes from the generic
    # resolver (family resolved from cfg.model.family).
    from vrl.models.families.registry import get_model_family_entry

    block = _plain_policy("fp32")
    block["rollout"]["prompt_encoders"] = {"dtype": prompt_encoder}
    cfg = _with_precision("sd3_5/online_grpo_ocr", block)
    built = build_configs(cfg)
    build = get_model_family_entry("sd3_5").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )
    assert build.family == "sd3_5"
    assert build.rollout is not None
    assert build.rollout.prompt_encoder_dtype is expected


def test_family_parameter_dtype_is_derived_from_runtime_role() -> None:
    """Replay follows train while rollout follows the plain rollout precision."""
    from vrl.models.families.registry import get_model_family_entry

    cfg = _with_precision(
        "sd3_5/online_grpo_ocr",
        {
            "training": {"dtype": "bf16"},
            "rollout": {"dtype": "fp16"},
        },
    )

    built = build_configs(cfg)
    entry = get_model_family_entry("sd3_5")
    rollout = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )
    replay = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        for_rollout=False,
    )

    assert rollout.parameter_dtype is torch.float16
    assert replay.parameter_dtype is torch.bfloat16


def test_direct_tool_parameter_dtype_override_is_explicit() -> None:
    """Non-production tools may override storage dtype only through a named argument."""
    from vrl.models.families.registry import get_model_family_entry

    cfg = _with_precision("sd3_5/online_grpo_ocr", _plain_policy("bf16"))
    built = build_configs(cfg)
    build = get_model_family_entry("sd3_5").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        parameter_dtype_override="fp32",
    )

    assert build.parameter_dtype is torch.float32
