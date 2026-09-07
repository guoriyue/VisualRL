from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vrl.algorithms.grpo.continuous import (
    GRPO,
    FlowDPPO,
    GRPOGuard,
)
from vrl.config.builders import RewardRuntimeConfig, build_configs
from vrl.config.loading import load_config
from vrl.config.precision import RolePrecision
from vrl.config.schema import parse_config
from vrl.models.families.registry import get_model_family_entry
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import ResolvedReward, resolve_reward_inputs
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator,
    build_reward_function,
)


def _built_reward(
    weights: dict[str, float],
    kwargs: dict[str, dict],
    inference: dict[str, dict] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=RewardRuntimeConfig.from_cfg(
            OmegaConf.create(
                {
                    "reward": {
                        "components": weights,
                        "kwargs": kwargs,
                        "inference": inference or {},
                    },
                },
            ),
        ),
    )


def test_diffusion_grpo_evaluator_uses_resolved_rollout_sde_config() -> None:
    """The evaluator built for a diffusion GRPO run reads noise level and SDE type from the
    resolved rollout config, the collector's denoise options default to native mode, and the
    advantage estimator carries the reward weights.
    """
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_ocr",
        overrides=[
            "rollout.noise_level=0.37",
            "rollout.sde.type=cps",
        ],
    )
    collector_config = RolloutCollectorConfig.from_root(parse_config(cfg))
    built = build_configs(cfg)

    pair = build_algorithm_and_evaluator(
        family_entry=get_model_family_entry("wan_2_1"),
        built=built,
        collector_config=collector_config,
        scheduler=object(),
    )

    assert pair.evaluator.noise_level == 0.37
    assert pair.evaluator.sde_type == "cps"
    assert collector_config.denoise is not None
    assert collector_config.denoise.denoise_mode == "native"
    assert pair.algorithm.advantage_estimator.component_weights == built.reward.weights


@pytest.mark.parametrize(
    ("recipe", "expected_algorithm"),
    [
        ("flow_matching_grpo", GRPO),
        ("flow_matching_dppo", FlowDPPO),
        ("flow_matching_grpo_guard", GRPOGuard),
    ],
)
def test_diffusion_factory_accepts_each_kind_exact_config_type(
    recipe: str,
    expected_algorithm: type,
) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    pair = build_algorithm_and_evaluator(
        family_entry=get_model_family_entry("sd3_5"),
        built=build_configs(cfg),
        collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        scheduler=object(),
    )

    assert type(pair.algorithm) is expected_algorithm


def test_chunk_autoregressive_factory_builds_grouped_grpo_evaluator() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")

    pair = build_algorithm_and_evaluator(
        family_entry=get_model_family_entry("causvid"),
        built=build_configs(cfg),
        collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
    )

    assert type(pair.algorithm) is GRPO
    assert type(pair.evaluator).__name__ == "ChunkAutoregressiveDenoiseLogProbEvaluator"


def test_generation_only_chunk_family_fails_before_algorithm_construction() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")

    with pytest.raises(RuntimeError, match=r"generation-only.*no trainable actions"):
        build_algorithm_and_evaluator(
            family_entry=get_model_family_entry("magi_1"),
            built=build_configs(cfg),
            collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        )


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ("flow_matching_dance_grpo", "random denoise-timestep subset"),
        ("flow_matching_dppo", "reverse-SDE dt signals"),
        ("flow_matching_grpo_guard", "reverse-SDE dt signals"),
    ],
)
def test_chunk_autoregressive_factory_rejects_undefined_algorithm_semantics(
    recipe: str,
    message: str,
) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"/recipe/online={recipe}"],
    )

    with pytest.raises(ValueError, match=message):
        build_algorithm_and_evaluator(
            family_entry=get_model_family_entry("causvid"),
            built=build_configs(cfg),
            collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        )


def test_chunk_autoregressive_factory_rejects_non_fp32_transition_math() -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["precision.diffusion_math.dtype=bf16"],
    )

    with pytest.raises(ValueError, match="exact fp32 Gaussian re-noise"):
        build_algorithm_and_evaluator(
            family_entry=get_model_family_entry("causvid"),
            built=build_configs(cfg),
            collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        )


def test_chunk_autoregressive_factory_rejects_full_sequence_sft_regularizer() -> None:
    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    built = build_configs(cfg)
    built.algorithm.sft_weight = 0.1

    with pytest.raises(ValueError, match=r"grouped causal-chunk replay.*sft_weight"):
        build_algorithm_and_evaluator(
            family_entry=get_model_family_entry("causvid"),
            built=built,
            collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        )


def test_wan_empty_lora_preserves_base_policy_initially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty training LoRA on Wan keeps ``init_lora_weights`` at its True default, so a fresh
    adapter reproduces the base policy until it is trained.
    """
    monkeypatch.setattr(
        "diffusers.DiffusionPipeline.load_config",
        lambda *_args, **_kwargs: {
            "boundary_ratio": None,
            "expand_timesteps": False,
        },
    )
    cfg = load_config("experiment/wan_2_1/online_grpo_physics")
    built = build_configs(cfg)

    build = get_model_family_entry("wan_2_1").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )

    assert build.use_lora is True
    lora_config = build.lora
    assert lora_config is not None
    # Wan's apply_lora reads ``init_lora_weights`` with a True default, so an
    # empty training adapter still initially preserves base Wan output.
    assert lora_config.get("init_lora_weights", True) is True


def test_sana_aesthetic_keeps_cpu_observation_only_pickscore() -> None:
    """PickScore is logged on CPU but contributes zero optimization weight."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)

    reward = build_reward_function(
        resolve_reward_inputs(
            built,
            ResolvedDistributedResources.from_root(parse_config(cfg)),
            trainer_device="cuda:0",
        ),
    )

    assert [(name, weight) for name, weight, _ in reward.rewards] == [
        ("aesthetic", 1.0),
        ("pickscore", 0.0),
    ]
    pickscore = reward.rewards[1][2]
    assert pickscore.scorer._worker_config["device"] == "cpu"


def test_sana_family_defaults_to_native_fp16() -> None:
    """The public role precision matches the checkpoint's runtime invariant."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    built = build_configs(cfg)
    entry = get_model_family_entry("sana")
    build = entry.resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
    )

    assert cfg.model.get("dtype") is None
    expected = RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert built.trainer.train_precision == built.trainer.rollout_precision
    assert built.root.rollout is not None
    assert (
        built.trainer.batch_plan.samples_per_replay_batch
        == built.root.rollout.samples_per_generation_batch
    )
    assert build.parameter_dtype is torch.float16
    assert build.precision == expected
    assert (
        entry.resolve_model_build(
            built.root,
            torch.device("cpu"),
            precision=built.precision,
            for_rollout=False,
        ).precision
        == expected
    )
    assert build.rollout is not None
    assert build.rollout.prompt_encoder_dtype is torch.bfloat16


@pytest.mark.parametrize(("role", "dtype"), [("training", "bf16"), ("rollout", "fp32")])
def test_sana_role_precision_follows_yaml(role: str, dtype: str) -> None:
    """The selected YAML role owns SANA's transformer execution policy."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.precision[role].dtype = dtype
    built = build_configs(cfg)

    build = get_model_family_entry("sana").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        for_rollout=role == "rollout",
    )

    assert build.precision == RolePrecision(
        dtype=dtype,
        float32_precision="ieee",
        outer_autocast=False,
    )
    assert build.parameter_dtype is getattr(torch, {"bf16": "bfloat16", "fp32": "float32"}[dtype])


def test_sana_fullparam_long_is_fresh_and_pins_reward_revisions() -> None:
    """The canonical curve starts from base with immutable scorer identities."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic_fullparam_long")
    cfg.distributed.resources.visible_devices = [0]
    built = build_configs(cfg)

    assert built.resume.checkpoint_path is None
    assert built.resume.strict is True
    assert cfg.model.use_lora is False
    reward = build_reward_function(
        resolve_reward_inputs(
            built,
            ResolvedDistributedResources.from_root(parse_config(cfg)),
            trainer_device="cuda:0",
        ),
    )
    aesthetic_config = reward.rewards[0][2].scorer._worker_config
    pickscore_config = reward.rewards[1][2].scorer._worker_config
    assert aesthetic_config["model_revision"] == cfg.reward.kwargs.aesthetic.model_revision
    assert pickscore_config["device"] == "cpu"
    assert pickscore_config["processor_revision"] == cfg.reward.kwargs.pickscore.processor_revision
    assert pickscore_config["model_revision"] == cfg.reward.kwargs.pickscore.model_revision


def test_sana_rejects_redundant_or_conflicting_model_dtype() -> None:
    """A family invariant must not also survive as a user-controlled knob."""
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    cfg.model.dtype = "fp16"

    with pytest.raises(ValueError, match=r"unknown model\.dtype"):
        parse_config(cfg)


def test_sana_direct_tool_override_changes_storage_only() -> None:
    cfg = load_config("experiment/sana/online_grpo_aesthetic")
    built = build_configs(cfg)

    build = get_model_family_entry("sana").resolve_model_build(
        built.root,
        torch.device("cpu"),
        precision=built.precision,
        parameter_dtype_override="fp32",
    )

    assert build.parameter_dtype is torch.float32
    assert build.precision == RolePrecision(
        dtype="fp16",
        float32_precision="ieee",
        outer_autocast=False,
    )


def test_token_objective_rejects_unused_math_precision_override() -> None:
    cfg = load_config("experiment/emu3/online_grpo_pickscore_validation")
    cfg.precision = {
        "float32_precision": "tf32",
        "training": {"dtype": "bf16", "outer_autocast": False},
        "rollout": {"dtype": "bf16", "outer_autocast": False},
        "diffusion_math": {"dtype": "bf16"},
    }
    built = build_configs(cfg)

    with pytest.raises(ValueError, match=r"precision\.diffusion_math\.dtype.*diffusion log-prob"):
        build_algorithm_and_evaluator(
            built=built,
            family_entry=get_model_family_entry("emu3"),
            collector_config=RolloutCollectorConfig.from_root(parse_config(cfg)),
        )


def test_reward_factory_rejects_an_all_zero_objective() -> None:
    """Checks observation-only components cannot replace the training objective."""
    with pytest.raises(ValueError, match="At least one reward component"):
        build_reward_function(
            ResolvedReward(
                config=_built_reward({"pickscore": 0.0}, {}).reward,
                device="cpu",
                memory_parking_required=False,
            ),
        )


def _shared_reward_cfg(component: str) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "devices": [0],
                        "gpu_pool": "trainer",
                    },
                },
            },
            "reward": {"components": {component: 1.0}, "kwargs": {component: {}}},
        },
    )


def test_shared_reward_capability_fails_before_component_construction(monkeypatch) -> None:
    """An unsupported trainer-shared reward fails before its model constructor."""
    from vrl.rewards.functions.geneval import GenEvalReward

    constructed = False

    def fail_if_constructed(self, *args, **kwargs):
        del self, args, kwargs
        nonlocal constructed
        constructed = True
        raise AssertionError("component construction must not run")

    monkeypatch.setattr(GenEvalReward, "__init__", fail_if_constructed)
    cfg = _shared_reward_cfg("geneval")

    with pytest.raises(ValueError, match="geneval"):
        build_reward_function(
            resolve_reward_inputs(
                _built_reward({"geneval": 1.0}, {"geneval": {}}),
                ResolvedDistributedResources.from_root(parse_config(cfg)),
                trainer_device="cuda:0",
            ),
        )

    assert constructed is False


def test_reward_config_rejects_yaml_lifecycle_override() -> None:
    """Resource topology is the only public reward lifecycle source."""

    cfg = OmegaConf.create(
        {
            "reward": {
                "components": {"aesthetic": 1.0},
                "kwargs": {"aesthetic": {"sleep_offload": True}},
            },
        },
    )

    with pytest.raises(ValueError, match="sleep_offload is topology-derived"):
        RewardRuntimeConfig.from_cfg(cfg)


def test_reward_inputs_derive_device_from_resource_topology() -> None:
    """The resource plan, not a caller device, is the execution-device source."""
    cfg = _shared_reward_cfg("aesthetic")
    shared = resolve_reward_inputs(
        _built_reward({"aesthetic": 1.0}, {"aesthetic": {}}),
        ResolvedDistributedResources.from_root(parse_config(cfg)),
    )
    assert shared.device == "cuda:0"
    assert shared.memory_parking_required is True

    # A torchrun rank scores on logical cuda:0 while Ray keeps the physical ID.
    rank_local = _shared_reward_cfg("aesthetic")
    rank_local.distributed.resources.visible_devices = [2]
    rank_local.distributed.resources.trainer.devices = [2]
    rank_local.distributed.resources.rollout.devices = [2]
    rank_local_reward = resolve_reward_inputs(
        _built_reward({"aesthetic": 1.0}, {"aesthetic": {}}),
        ResolvedDistributedResources.from_root(parse_config(rank_local)),
        trainer_device="cuda:0",
    )
    assert rank_local_reward.device == "cuda:0"
    assert rank_local_reward.memory_parking_required is True

    # HTTP components own their deployment externally: no local parking policy.
    http_inference = {
        "unified_reward_video": {
            "kind": "http",
            "endpoint": "http://127.0.0.1:8300",
            "expected_model": "unified-reward-robotics",
        },
    }
    http_reward = resolve_reward_inputs(
        _built_reward({"unified_reward_video": 1.0}, {}, http_inference),
        ResolvedDistributedResources.from_root(parse_config(rank_local)),
        trainer_device="cuda:0",
    )
    assert http_reward.device == "cuda:0"
    assert http_reward.memory_parking_required is False

    cpu_cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "cpu"},
                },
            },
            "reward": {"components": {"ocr": 1.0}, "kwargs": {"ocr": {}}},
        },
    )
    # A CPU-only reward reservation wins even when the trainer runs on CUDA.
    cpu_reward = resolve_reward_inputs(
        _built_reward({"ocr": 1.0}, {"ocr": {}}),
        ResolvedDistributedResources.from_root(parse_config(cpu_cfg)),
        trainer_device="cuda:0",
    )
    assert cpu_reward.device == "cpu"


def test_multi_gpu_engine_gate_requires_family_capability() -> None:
    """gpus_per_engine > 1 fails loud for a family without an installer."""
    from vrl.models.families.registry import get_model_family_entry

    wan = get_model_family_entry("wan_2_1")
    with pytest.raises(ValueError, match=r"wan_2_1.*gpus_per_engine=2"):
        wan.validate_gpus_per_engine(2)

    # Single-GPU engines never consult the capability; a capable family passes.
    wan.validate_gpus_per_engine(1)
    get_model_family_entry("sd3_5").validate_gpus_per_engine(2)
