"""Core config-loader safety tests.

This file intentionally keeps broad experiment coverage in loop-style tests
instead of parametrizing the same assertion into dozens of collected tests.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vrl.algorithms.diffusion_nft import DiffusionNFTConfig
from vrl.algorithms.dpo import DiffusionDPOConfig
from vrl.algorithms.grpo.continuous import GRPOConfig
from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig
from vrl.algorithms.grpo.token import TokenGRPOConfig
from vrl.algorithms.v_grpo import VGRPOConfig
from vrl.config.builders import (
    RewardRuntimeConfig,
    build_configs,
)
from vrl.config.loading import (
    bundled_config_resource,
    list_bundled_configs,
    load_config,
)
from vrl.config.schema import RewardConfig, parse_config
from vrl.config.validation import require_training_config
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.orchestration import validate_rollout_schedule_topology
from vrl.scripts.common.factory import validate_reward_memory_parking


def _experiment_names() -> list[str]:
    return [
        name.removeprefix("experiment/").removesuffix(".yaml")
        for name in list_bundled_configs("experiment")
    ]


def _load_bundled_raw(name: str):
    resource = bundled_config_resource(name)
    with resource.open("r", encoding="utf-8") as stream:
        return OmegaConf.load(stream)


def _load_experiment_for_static_validation(name: str):
    """Complete runtime templates with inert test choices."""

    overrides: list[str] = []
    if name in {
        "anima_preview3/online_grpo",
        "anima_preview3/online_grpo_fullparam",
    }:
        overrides = [
            "+reward=aesthetic",
            "+dataset=anime_anatomy",
            "trainer.total_epochs=1",
            "trainer.output_dir=/test-only/anima-composition",
        ]
        if name == "anima_preview3/online_grpo":
            overrides.append("actor.optim.lr=1e-5")
    return load_config(f"experiment/{name}", overrides=overrides)


def test_load_config_enforces_mandatory_marker(tmp_path: Path) -> None:
    """Keys declared '???' must be set by the experiment or a dotlist override."""
    config = tmp_path / "exp.yaml"
    config.write_text("trainer:\n  entrypoint: ???\n  seed: 0\n")

    with pytest.raises(ValueError, match=r"trainer\.entrypoint"):
        load_config(config)

    cfg = load_config(config, overrides=["trainer.entrypoint=pkg.mod:fn"])
    assert cfg.trainer.entrypoint == "pkg.mod:fn"


def test_trainer_config_from_cfg_reports_all_missing_required_keys() -> None:
    """Missing required keys are reported together, each with its full YAML path."""
    cfg = OmegaConf.create(
        {
            "actor": {},
            "trainer": {},
            "rollout": {},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {"dtype": "bf16"},
            },
        },
    )

    from vrl.trainers.online.config import TrainerConfig

    with pytest.raises(ValueError) as exc:
        TrainerConfig.from_root(parse_config(cfg))

    message = str(exc.value)
    for path in (
        "actor.optim.lr",
        "actor.drop_zero_advantage",
        "actor.timestep_fraction",
        "trainer.output_dir",
        "rollout.prompts_per_batch",
        "rollout.n_samples_per_prompt",
    ):
        assert path in message


def test_config_groups_are_not_flattened() -> None:
    """Bundled config groups keep their nested layout: no experiment / model / sampling entry sits
    flat at the group root, and the retired ``task`` and ``profiling`` groups are empty.
    """
    flattened = [
        name
        for group in ("experiment", "model", "sampling")
        for name in list_bundled_configs(group)
        if len(Path(name).parts) == 2
    ]
    task_configs = list_bundled_configs("task")

    assert flattened == []
    assert task_configs == ()
    assert list_bundled_configs("profiling") == ()


def test_geometry_sampling_presets_do_not_own_text_encoder_lengths() -> None:
    geometry_presets = [
        name
        for name in list_bundled_configs("sampling")
        if Path(name).parts[1] in {"image", "video"}
    ]

    assert geometry_presets
    for name in geometry_presets:
        sampling = _load_bundled_raw(name).get("sampling", {})
        assert "max_sequence_length" not in sampling, name


def test_raw_yaml_has_no_user_specific_absolute_paths() -> None:
    """Committed configs must not depend on one contributor's home directory."""
    user_home = re.compile(
        r"(?:/home/[^/\s\"'{}]+/|/Users/[^/\s\"'{}]+/|/root/|"
        r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'{}]+[\\/])",
    )
    offenders = []
    for logical_name in list_bundled_configs():
        resource = bundled_config_resource(logical_name)
        with resource.open("r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        for line_number, line in enumerate(lines, start=1):
            if user_home.search(line):
                offenders.append(
                    f"vrl/config/presets/{logical_name}:{line_number}: {line.strip()}",
                )

    assert offenders == []


def test_experiments_are_grouped_by_model_family() -> None:
    """Every experiment lives at ``<family>/<recipe>``.

    Generation regime and policy-step math live in registry semantics, so
    the config path must not reintroduce the retired AR/diffusion taxonomy.
    """
    names = _experiment_names()
    groups = {Path(name).parts[0] for name in names}
    assert {"ar", "diffusion"}.isdisjoint(groups)
    assert all(len(Path(name).parts) == 2 for name in names)


def _reward_group_kwargs_keys(group_name: str) -> dict[str, set[str]]:
    """Component-name -> leaf-key set provided by a `/reward/<group>` default.

    This is the source of truth the inline-override rule is derived from, so the
    test never hand-maintains a parallel allowlist that rots when a reward group
    gains a knob.
    """
    raw = _load_bundled_raw(f"reward/{group_name}")
    kwargs = raw.get("reward", {}).get("kwargs", {}) or {}
    return {comp: set((sub or {}).keys()) for comp, sub in kwargs.items()}


def test_experiments_use_dataset_groups_and_only_override_reward_weights() -> None:
    """Experiments compose dataset groups and only fine-tune reward leaves inline.

    Two structural rules, both derived from the source of truth instead of a
    hand-kept path allowlist:

    * no experiment inlines a `data:` block (datasets come from `/dataset/...`);
    * an experiment's inline `reward.kwargs` may only override leaf scalars its
      reward group default already declared — it may NOT introduce a new
      component or a key the group never provided.
    """
    inline_data = []
    inline_reward_violations = []
    for rel in list_bundled_configs("experiment"):
        raw = _load_bundled_raw(rel)
        if "data" in raw:
            inline_data.append(rel)

        reward = raw.get("reward", None)
        if reward is None or "kwargs" not in reward:
            continue

        # Keys (per component) the imported reward groups already provide.
        provided: dict[str, set[str]] = {}
        for default in raw.get("defaults", []) or []:
            if isinstance(default, str) and default.startswith("/reward/"):
                group = default.split("/reward/", 1)[1]
                for comp, keys in _reward_group_kwargs_keys(group).items():
                    provided.setdefault(comp, set()).update(keys)

        for comp, sub in (reward.get("kwargs", {}) or {}).items():
            if comp not in provided:
                inline_reward_violations.append(f"{rel}: new component {comp!r}")
                continue
            extra = set((sub or {}).keys()) - provided[comp]
            if extra:
                inline_reward_violations.append(
                    f"{rel}: {comp} declares non-default keys {sorted(extra)}",
                )

    assert inline_data == []
    assert inline_reward_violations == []


def test_reward_configs_are_single_reward_building_blocks() -> None:
    """Every bundled ``reward/`` preset defines exactly one component, so presets compose by
    addition.
    """
    offenders = []
    for name in list_bundled_configs("reward"):
        raw = _load_bundled_raw(name)
        components = raw.get("reward", {}).get("components", {})
        if len(components) != 1:
            offenders.append(name)

    assert offenders == []


def test_all_experiments_load_and_validate() -> None:
    """Every bundled experiment loads, carries the required top-level sections and keys, has
    migrated off ``adv_estimator``, and its raw ``model`` block survives typed parsing key-for-
    key.
    """
    for name in _experiment_names():
        cfg = _load_experiment_for_static_validation(name)
        assert "model" in cfg, f"{name} missing model.*"
        assert "trainer" in cfg, f"{name} missing trainer.*"
        assert "algorithm" in cfg, f"{name} missing algorithm.*"
        assert "data" in cfg, f"{name} missing data.* source"
        if str(cfg.algorithm.kind) != "diffusion_dpo":
            assert "reward" in cfg, f"{name} missing reward.* source"
        assert "path" in cfg.model, f"{name} missing model.path"
        assert "entrypoint" in cfg.trainer, f"{name} missing trainer.entrypoint"
        assert "output_dir" in cfg.trainer, f"{name} missing trainer.output_dir"
        assert "kind" in cfg.algorithm, f"{name} missing algorithm.kind"
        assert "adv_estimator" not in cfg.algorithm, f"{name} still uses adv_estimator"
        root, _ = require_training_config(cfg)
        raw_model = OmegaConf.to_container(cfg.model, resolve=True)
        typed_model = root.model
        assert isinstance(raw_model, dict)
        assert typed_model is not None
        typed_payload = typed_model.model_dump(exclude_unset=True)
        assert {key: typed_payload[key] for key in raw_model} == raw_model


def test_all_online_experiments_pass_static_launch_preflight() -> None:
    """Every active online recipe has a valid schedule and reward topology.

    Explicit ``dedicated`` pools and cross-node recipes describe disjoint GPU
    topology, so supply the synthetic budget derived from their role requests.
    Other auto/shared recipes get one synthetic device: ``make verify`` hides
    CUDA, while this static test must still exercise single-GPU parking.
    """

    def requested_gpus(node, *, default: int) -> int:
        devices = node.get("devices", "auto")
        if devices != "auto":
            return len(devices)
        num_gpus = node.get("num_gpus", "auto")
        if num_gpus not in (None, "auto"):
            return int(num_gpus)
        num_engines = node.get("num_engines", "auto")
        if num_engines != "auto":
            # One GPU per engine; num_engines: 0 declares no fleet.
            return int(num_engines)
        return default

    failures = []
    for name in _experiment_names():
        cfg = _load_experiment_for_static_validation(name)
        if str(cfg.algorithm.kind) == "diffusion_dpo":
            continue

        resources_cfg = cfg.distributed.resources
        rollout_pool = str((resources_cfg.get("rollout") or {}).get("gpu_pool", "auto"))
        reward_pool = str(resources_cfg.get("reward", {}).get("gpu_pool", "auto"))
        if resources_cfg.get("visible_devices", "auto") == "auto":
            required = 1
            requires_disjoint_devices = "dedicated" in {rollout_pool, reward_pool} or bool(
                resources_cfg.get("cross_node", False)
            )
            if requires_disjoint_devices:
                trainer_gpus = requested_gpus(resources_cfg.get("trainer") or {}, default=1)
                rollout_gpus = requested_gpus(resources_cfg.get("rollout") or {}, default=1)
                reward_cfg = resources_cfg.get("reward", {})
                # Reward is in-process: device=gpu reserves exactly one GPU.
                reward_gpus = 1 if str(reward_cfg.get("device", "trainer")) == "gpu" else 0
                required = trainer_gpus
                if rollout_pool != "trainer":
                    required += rollout_gpus
                if reward_pool == "dedicated":
                    required += reward_gpus
            resources_cfg.visible_devices = list(range(required))

        try:
            built = build_configs(cfg)
            resources = ResolvedDistributedResources.from_root(parse_config(cfg))
            validate_rollout_schedule_topology(
                built.trainer.rollout_orchestration,
                resources,
            )
            validate_reward_memory_parking(
                resources=resources,
                built=built,
            )
        except Exception as error:  # report the full active surface in one failure
            failures.append(f"{name}: {type(error).__name__}: {error}")

    assert failures == []


def test_validate_rejects_compile_with_gradient_checkpointing() -> None:
    """compile x grad-ckpt must fail at config load, not as a mid-run dynamo crash.

    The trainer refuses the combination at startup (activation_checkpointing.py);
    this checks require_training_config rejects it too, because a model-layer
    torch_compile.enable=true default can silently flip compile on underneath an
    experiment that sets checkpointing (that exact collision shipped in four
    cosmos_predict2 240p recipes before this load-time check existed).
    """
    base = "experiment/sd3_5/online_grpo_ocr"  # resolves compile=true

    for ckpt in ("true", "full", "selective"):
        cfg = load_config(base, overrides=[f"actor.gradient_checkpointing={ckpt}"])
        with pytest.raises(ValueError, match="cannot combine"):
            require_training_config(cfg)

    # Explicit off (either spelling) keeps compile allowed.
    cfg = load_config(base, overrides=["actor.gradient_checkpointing=off"])
    require_training_config(cfg)


def test_rollout_orchestration_group_override_uses_rollout_namespace() -> None:
    """``/base/rollout/orchestration=continuous`` lands under ``trainer.rollout_orchestration``
    and types into a continuous schedule with a positive staleness bound.
    """
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["/base/rollout/orchestration=continuous"],
    )

    orchestration = cfg.trainer.rollout_orchestration
    assert orchestration.schedule_mode == "continuous"
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    typed = RolloutOrchestrationConfig(
        **OmegaConf.to_container(orchestration, resolve=True),
    )
    assert typed.schedule_mode == "continuous"
    assert typed.continuous.max_stale_policy_versions >= 1


def test_sd35_continuous_4gpu_acceptance_resolves_disjoint_resident_topology() -> None:
    """The reusable hardware gate must preserve the topology it validates."""

    cfg = load_config("experiment/sd3_5/online_grpo_ocr_continuous_4gpu_acceptance")
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == (0,)
    assert resources.rollout_devices == (1, 2, 3)
    assert resources.reward_devices == ()
    assert resources.rollout_num_engines == 3
    assert resources.lifecycle.rollout_mode == "resident"
    assert not any(
        (
            resources.lifecycle.release_rollout_before_train,
            resources.lifecycle.release_rollout_before_reward,
            resources.lifecycle.release_trainer_before_reward,
            resources.lifecycle.release_reward_after_score,
        ),
    )
    assert built.trainer.rollout_orchestration.schedule_mode == "continuous"


def test_cosmos_predict2_overfit_fsdp_4x_l4_resolves_rank_local_topology(
    cuda_devices,
) -> None:
    """The four-L4 recipe keeps one colocated rollout and CPU reward per rank."""

    cuda_devices(1)
    name = "experiment/cosmos_predict2/online_grpo_droid_overfit_validation_fsdp_4x_l4"
    cfg = load_config(name)
    parent = load_config("experiment/cosmos_predict2/online_grpo_droid_overfit_validation")
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.rollout_num_engines == 1
    assert resources.reward_devices == ()
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.release_rollout_before_train is True

    # Per-rank geometry remains inherited from the parent; only the topology
    # leaves may differ. Compare against the source of truth, not literals.
    for path in (
        "actor.ppo_epochs",
        "actor.timestep_fraction",
        "rollout.n_samples_per_prompt",
        "rollout.prompts_per_batch",
        "trainer.total_epochs",
    ):
        assert OmegaConf.select(cfg, path) == OmegaConf.select(parent, path)


def test_cosmos_predict2_full_curve_fsdp_4x_l4_preserves_training_semantics(
    cuda_devices,
) -> None:
    """The durable full-DROID run changes topology without changing its curve."""

    cuda_devices(1)
    name = "experiment/cosmos_predict2/online_grpo_droid_lora_480p_curve_fsdp_4x_l4"
    cfg = load_config(name)
    parent = load_config("experiment/cosmos_predict2/online_grpo_droid_lora_480p_curve")
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.rollout_num_engines == 1
    assert resources.reward_devices == ()
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.release_rollout_before_train is True

    # The parent is the single source of truth for learning, dataset, and reward
    # semantics. Only the validated hardware-specific leaves may differ here.
    assert OmegaConf.to_container(cfg.algorithm, resolve=True) == OmegaConf.to_container(
        parent.algorithm,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.actor, resolve=True) == OmegaConf.to_container(
        parent.actor,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.model, resolve=True) == OmegaConf.to_container(
        parent.model,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.data, resolve=True) == OmegaConf.to_container(
        parent.data,
        resolve=True,
    )
    assert OmegaConf.to_container(cfg.reward, resolve=True) == OmegaConf.to_container(
        parent.reward,
        resolve=True,
    )
    sampling = OmegaConf.to_container(cfg.sampling, resolve=True)
    parent_sampling = OmegaConf.to_container(parent.sampling, resolve=True)
    assert isinstance(sampling, dict) and isinstance(parent_sampling, dict)
    sampling.pop("guidance_scale")
    parent_sampling.pop("guidance_scale")
    assert sampling == parent_sampling

    rollout = OmegaConf.to_container(cfg.rollout, resolve=True)
    parent_rollout = OmegaConf.to_container(parent.rollout, resolve=True)
    assert isinstance(rollout, dict) and isinstance(parent_rollout, dict)
    rollout.pop("samples_per_generation_batch")
    parent_rollout.pop("samples_per_generation_batch")
    assert rollout == parent_rollout

    trainer = OmegaConf.to_container(cfg.trainer, resolve=True)
    parent_trainer = OmegaConf.to_container(parent.trainer, resolve=True)
    assert isinstance(trainer, dict) and isinstance(parent_trainer, dict)
    for key in ("output_dir", "save_freq"):
        trainer.pop(key)
        parent_trainer.pop(key)
    assert trainer == parent_trainer


def test_wan_robotics_continuous_resolves_balanced_four_l4_topology() -> None:
    """The robotics run must keep trainer, rollout, and reward GPUs disjoint."""

    cfg = load_config(
        "experiment/wan_2_1/online_grpo_robotics_physics_4x_l4_continuous",
    )
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    orchestration = built.trainer.rollout_orchestration
    assert resources.trainer_devices == (0,)
    assert resources.rollout_devices == (1, 2)
    assert resources.reward_devices == ()
    assert resources.rollout_num_engines == 2
    assert resources.lifecycle.rollout_mode == "resident"
    assert orchestration.schedule_mode == "continuous"


def test_wan_droid_fullparam_fsdp_3x_l4_preserves_launch_contract(
    cuda_devices,
) -> None:
    """The long run keeps full-param training and rank-local rollout semantics."""

    cuda_devices(1)
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_3x_l4",
    )
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.rollout_num_engines == 1
    assert resources.reward_devices == ()
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.release_rollout_before_train is True
    assert built.trainer.rollout_orchestration.schedule_mode == "strict_on_policy"


def test_wan_droid_fullparam_fsdp_4x_l4_uses_symmetric_reward_handoffs(cuda_devices) -> None:
    """All four ranks time-share local rollout and the complete robotics reward."""

    cuda_devices(1)
    cfg = load_config(
        "experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4",
    )
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.reward_devices == ()
    assert resources.reward_torch_device(trainer_device="cuda:0") == "cuda:0"
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.release_reward_after_score is True
    handoff = resources.lifecycle
    assert handoff.release_rollout_before_train is True
    assert handoff.release_rollout_before_reward is True
    assert handoff.release_trainer_before_reward is True
    assert handoff.release_reward_after_score is True


def test_masked_physical_ordinal_comes_from_the_config_knob_not_the_auto_path() -> None:
    """A non-zero rank-local ordinal is unreachable through the auto path.

    ``_auto_visible_cuda_devices`` returns ``tuple(range(n))``, so ``(1,)`` can
    never come out of it at any device count. The launcher writes the selected
    physical ordinal into ``visible_devices`` before resource resolution
    (tests/scripts/test_online_entrypoint.py), and that is the path this pins —
    the real ``_parse_devices`` / dedupe branches, on this recipe.
    """

    cfg = load_config("experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4")
    OmegaConf.update(cfg, "distributed.resources.visible_devices", [1], force_add=True)
    require_training_config(cfg)

    resources = ResolvedDistributedResources.from_root(parse_config(cfg))

    assert resources.visible_devices == (1,)
    assert resources.trainer_devices == resources.rollout_devices == (1,)
    assert resources.colocated is True


def test_algorithm_config_dispatches_representative_kinds() -> None:
    """``algorithm.kind`` dispatches to the matching hyperparameter dataclass across
    representative experiments (GRPO, token GRPO, multi-segment token GRPO, DPO, NFT, V-GRPO).
    """
    examples = {
        "sd3_5/online_grpo_ocr": GRPOConfig,
        "janus_pro/online_grpo_ocr": TokenGRPOConfig,
        "janus_pro/online_r1_grpo_ocr": MultiSegmentTokenGRPOConfig,
        "wan_2_1/offline_dpo_pickapic": DiffusionDPOConfig,
        "cosmos_predict2_5/online_nft_kling_video_reward": DiffusionNFTConfig,
        "sd3_5/online_v_grpo_pickscore": VGRPOConfig,
    }
    for name, expected_type in examples.items():
        cfg = load_config(f"experiment/{name}")
        algo_cfg = parse_config(cfg).algorithm.hyperparameters
        assert isinstance(algo_cfg, expected_type)


def test_cosmos_v2w_production_validation_accepts_source_backed_data(
    tmp_path: Path,
) -> None:
    """Cosmos V2W production validation accepts a source-backed manifest pair whose metadata
    carries the full DROID provenance (repo, split, episode, video, frame, decode method,
    conditioning).
    """
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": "episode_train",
        "source_video": "videos/camera/batch-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_first_frame",
        "conditioning": "first_frame",
    }
    train = tmp_path / "robot_train.jsonl"
    eval_manifest = tmp_path / "robot_eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves toward the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_episode="episode_eval")
    eval_manifest.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves away from the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "robot_report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "video_world_bridge",
                "source": "droid",
                "repo_id": "lerobot/droid_100",
                "source_split": "main",
                "decode_method": "pyav_http_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": reference.parent.as_posix(),
                "validation_summary": {"row_count": 1},
            },
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        "experiment/cosmos_predict2/online_grpo_v2w_reference",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    require_training_config(cfg)


def test_cosmos_target_v2w_production_validation_requires_target_clip(
    tmp_path: Path,
) -> None:
    """Checks target-reward Cosmos V2W validation requires public-source target clips."""
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    target = data_root / "video_world" / "targets" / "target.mp4"
    reference.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    target.write_bytes(b"fake-video")
    metadata = {
        "source": "droid",
        "source_repo": "lerobot/droid_100",
        "source_split": "main",
        "source_episode": "episode_train",
        "source_video": "videos/camera/batch-000/file-000.mp4",
        "source_frame_index": 0,
        "decode_method": "pyav_http_target_clip",
        "conditioning": "first_frame",
    }
    train = tmp_path / "droid_targets_train.jsonl"
    eval_manifest = tmp_path / "droid_targets_eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves toward the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "target_video": "video_world/targets/target.mp4",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_episode="episode_eval")
    eval_manifest.write_text(
        json.dumps(
            {
                "prompt": "The robot arm moves away from the cup.",
                "reference_image": "video_world/references/ref.ppm",
                "target_video": "video_world/targets/target.mp4",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "droid_targets_report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "video_world_targets",
                "source": "droid",
                "repo_id": "lerobot/droid_100",
                "source_split": "main",
                "decode_method": "pyav_http_target_clip",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": reference.parent.as_posix(),
                "target_dir": target.parent.as_posix(),
                "validation_summary": {"row_count": 1},
            },
        ),
        encoding="utf-8",
    )
    # The droid-target recipe now uses a zero-training dino+motion reward (no kling); the
    # kling production-contract path is covered by
    # test_cosmos_v2w_production_validation_accepts_source_backed_data. Here we validate
    # that target-clip-backed data resolves and validates for this recipe.
    cfg = load_config(
        "experiment/cosmos_predict2/online_grpo_droid_target_480p",
        overrides=[
            f"data.manifest={train.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    require_training_config(cfg)


def test_wan_i2v_production_validation_accepts_source_backed_data(tmp_path: Path) -> None:
    """Wan I2V production validation accepts source-backed VideoPhy manifests: image, caption,
    task type, and the CSV / video-URL / decode provenance in metadata.
    """
    data_root = tmp_path / "videophy_i2v"
    train_image = data_root / "images" / "train" / "000.ppm"
    eval_image = data_root / "images" / "eval" / "000.ppm"
    train_image.parent.mkdir(parents=True)
    eval_image.parent.mkdir(parents=True)
    train_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    eval_image.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")
    metadata = {
        "source": "videophy",
        "source_repo": "videophysics/videophy_test_public",
        "source_split": "test",
        "source_csv_row": 0,
        "source_video_url": "https://videophysics.example/train.mp4",
        "source_frame_index": 0,
        "decode_method": "imageio_ffmpeg_first_frame",
        "conditioning": "first_frame",
    }
    train_manifest = data_root / "manifests" / "train.jsonl"
    eval_manifest = data_root / "manifests" / "eval.jsonl"
    train_manifest.parent.mkdir(parents=True)
    train_manifest.write_text(
        json.dumps(
            {
                "image": "images/train/000.ppm",
                "caption": "A wheel rolls.",
                "task_type": "image_to_video",
                "metadata": metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    eval_metadata = dict(metadata, source_video_url="https://videophysics.example/eval.mp4")
    eval_manifest.write_text(
        json.dumps(
            {
                "image": "images/eval/000.ppm",
                "caption": "Honey diffuses.",
                "task_type": "image_to_video",
                "metadata": eval_metadata,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    report = data_root / "report.json"
    report.write_text(
        json.dumps(
            {
                "dataset": "videophy_i2v",
                "source_repo": "videophysics/videophy_test_public",
                "source_csv": "videophy_test_public.csv",
                "source_split": "test",
                "decode_method": "imageio_ffmpeg_first_frame",
                "train_rows": 1,
                "eval_rows": 1,
                "train_manifest": train_manifest.as_posix(),
                "eval_manifest": eval_manifest.as_posix(),
                "reference_dir": (data_root / "images").as_posix(),
            },
        ),
        encoding="utf-8",
    )

    cfg = load_config(
        "experiment/wan_2_1/online_grpo_physics_i2v",
        overrides=[
            "production.kling_video_reward.enabled=true",
            f"data.manifest={train_manifest.as_posix()}",
            f"data.eval_manifest={eval_manifest.as_posix()}",
            f"data.source_report={report.as_posix()}",
            f"data.artifact_data_root={data_root.as_posix()}",
        ],
    )

    require_training_config(cfg)


def test_wan_i2v_fsdp_2x_l4_resolves_bounded_shared_topology(cuda_devices) -> None:
    """The real-weight I2V gate shards replay and sequentially offloads rollout."""

    cuda_devices(1)
    cfg = load_config("experiment/wan_2_1/online_grpo_i2v_fsdp_2x_l4")
    require_training_config(cfg)
    built = build_configs(cfg)
    resources = ResolvedDistributedResources.from_root(parse_config(cfg))
    validate_rollout_schedule_topology(
        built.trainer.rollout_orchestration,
        resources,
    )
    validate_reward_memory_parking(resources=resources, built=built)

    assert resources.trainer_devices == resources.rollout_devices == (0,)
    assert resources.rollout_num_engines == 1
    assert resources.reward_devices == ()
    assert resources.lifecycle.rollout_mode == "on_demand"
    assert resources.lifecycle.release_rollout_before_train is True


def test_wan_video_reward_production_config_requires_reward_name() -> None:
    cfg = load_config("experiment/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.reward_name = ""

    with pytest.raises(ValueError, match="reward_name"):
        require_training_config(cfg)


def test_wan_video_reward_production_rejects_extra_loader_fields() -> None:
    """Production Kling configs reject loader-level keys (``import_path``, ``model_factory``)
    inside ``worker_config``.
    """
    cfg = load_config("experiment/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.import_path = "fake:thing"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        require_training_config(cfg)

    cfg = load_config("experiment/wan_2_1/online_grpo_kling_video_reward")
    cfg.reward.kwargs.kling_video_reward.worker_config.model_factory = "fake:factory"

    with pytest.raises(ValueError, match="remove extra loader fields"):
        require_training_config(cfg)


def test_unified_train_entrypoint_reads_yaml_entrypoint() -> None:
    """``resolve_train_target`` returns the YAML ``trainer.entrypoint`` and that dotted path
    imports to a callable.
    """
    from vrl.scripts.train import resolve_train_target
    from vrl.utils.config import import_from_path

    cfg = load_config("experiment/sd3_5/online_grpo_ocr")
    import_path = resolve_train_target(parse_config(cfg))

    assert import_path == cfg.trainer.entrypoint
    assert callable(import_from_path(import_path))


def test_cli_overrides_reach_typed_trainer_config() -> None:
    """CLI overrides for resume, torch profiler, drop-zero-advantage and batch width all reach the
    typed trainer / resume / rollout configs.
    """
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "trainer.resume_from=/tmp/checkpoint-10",
            "trainer.torch_profiler.enabled=true",
            "trainer.torch_profiler.activities=[cpu]",
            "actor.drop_zero_advantage=false",
            "rollout.samples_per_generation_batch=2",
        ],
    )
    built = build_configs(cfg)
    trainer = built.trainer

    assert built.resume.checkpoint_path == "/tmp/checkpoint-10"
    assert built.resume.strict is True
    assert trainer.torch_profiler.enabled is True
    assert trainer.torch_profiler.activities == ("cpu",)
    assert trainer.drop_zero_advantage is False
    assert built.root.rollout is not None
    assert built.root.rollout.samples_per_generation_batch == 2


def test_generation_chunk_auto_does_not_change_fixed_replay_default() -> None:
    """Generation auto remains generation-owned; replay defaults safely to one."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=["rollout.samples_per_generation_batch=auto"],
    )
    built = build_configs(cfg)

    assert built.root.rollout is not None
    assert built.root.rollout.samples_per_generation_batch == "auto"
    assert built.trainer.batch_plan.samples_per_replay_batch == 1


def test_luna_reward_overlay_changes_only_the_judge_command() -> None:
    policy = load_config("reward/codex_image_qa_anime_general_quality")
    experiment_overrides = [
        "+reward=codex_image_qa_anime_general_quality",
        "+dataset=anima_quality_ddrl",
        "actor.optim.lr=2e-5",
        "trainer.total_epochs=1",
        "trainer.output_dir=/test-only/anima-quality",
    ]
    base_experiment = load_config(
        "experiment/anima_preview3/online_grpo",
        overrides=experiment_overrides,
    )
    luna_experiment = load_config(
        "experiment/anima_preview3/online_grpo",
        overrides=[*experiment_overrides, "+reward=codex_image_qa_luna_scored"],
    )
    base_reward = base_experiment.reward.kwargs.codex_image_qa
    luna_reward = luna_experiment.reward.kwargs.codex_image_qa

    assert base_reward.prompt_template == policy.reward.kwargs.codex_image_qa.prompt_template
    assert luna_reward.prompt_template == policy.reward.kwargs.codex_image_qa.prompt_template
    assert "--model" not in policy.reward.kwargs.codex_image_qa.command
    assert "gpt-5.6-luna" not in policy.reward.kwargs.codex_image_qa.command
    assert "gpt-5.6-luna" in luna_reward.command


def test_anima_runtime_requires_explicit_experiment_choices() -> None:
    assert list_bundled_configs("experiment/anima_preview3") == (
        "experiment/anima_preview3/online_grpo.yaml",
        "experiment/anima_preview3/online_grpo_fullparam.yaml",
    )
    for name in ("online_grpo", "online_grpo_fullparam"):
        with pytest.raises(ValueError) as exc:
            load_config(f"experiment/anima_preview3/{name}")
        for key in ("reward", "data", "trainer.output_dir"):
            assert key in str(exc.value)
        if name == "online_grpo":
            assert "actor.optim.lr" in str(exc.value)

    fullparam = load_config(
        "experiment/anima_preview3/online_grpo_fullparam",
        overrides=[
            "+reward=aesthetic",
            "+dataset=anime_anatomy",
            "trainer.output_dir=/test-only/anima-fullparam",
        ],
    )
    assert fullparam.model.use_lora is False
    assert fullparam.model.lora is None


@pytest.mark.parametrize("value", ["0", "largest"])
def test_generation_chunk_rejects_non_positive_or_non_integer_values(value: str) -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[f"rollout.samples_per_generation_batch={value}"],
    )

    with pytest.raises(ValueError, match=r"rollout\.samples_per_generation_batch"):
        build_configs(cfg)


def test_negative_reward_component_weights_are_rejected() -> None:
    cfg = load_config(
        "experiment/anima_preview3/online_grpo",
        overrides=[
            "+reward=aesthetic",
            "+reward=nsfw_safety",
            "+dataset=anime_safety_stress",
            "actor.optim.lr=1e-5",
            "trainer.output_dir=/test-only/anima-safety",
        ],
    )
    cfg.reward.components.nsfw_safety = -0.5

    with pytest.raises(ValueError, match=r"reward\.components\.nsfw_safety must be >= 0"):
        RewardConfig.from_cfg(cfg)


def test_public_reward_builder_validates_its_input() -> None:
    cfg = OmegaConf.create(
        {"reward": {"components": {"ocr": 1.0}, "kwargs": {"ocr": ["invalid"]}}},
    )

    with pytest.raises(ValueError, match=r"reward\.kwargs\.ocr must be a mapping"):
        RewardRuntimeConfig.from_cfg(cfg)


def test_required_training_fields_fail_fast() -> None:
    cfg = load_config("experiment/wan_2_1/online_grpo_ocr")
    cfg.trainer.output_dir = "???"
    with pytest.raises(ValueError, match=r"trainer\.output_dir"):
        require_training_config(cfg)


def test_dpo_allows_explicit_null_max_train_samples() -> None:
    """An explicit ``data.max_train_samples=null`` is a valid DPO setting, not a missing field."""
    cfg = load_config("experiment/wan_2_1/offline_dpo_pickapic")
    cfg.data.max_train_samples = None

    assert parse_config(cfg).data.max_train_samples is None
    require_training_config(cfg)


def test_reward_collection_mode_accepts_the_three_acceptance_arms() -> None:
    """Checks the measurement override survives YAML -> typed config."""
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    for arm in ("batched_serial", "per_group_serial", "per_group_streaming"):
        typed = RolloutOrchestrationConfig(reward_collection_mode=arm)
        assert typed.reward_collection_mode == arm

    assert RolloutOrchestrationConfig().reward_collection_mode is None


def test_reward_collection_mode_rejects_unknown_arm() -> None:
    """Checks a typo fails fast instead of silently running the default arm."""
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    with pytest.raises(ValueError, match=r"reward_collection_mode must be one of"):
        RolloutOrchestrationConfig(reward_collection_mode="streaming")


def test_reward_collection_mode_rejected_under_continuous_scheduling() -> None:
    """Checks the knob is refused where it could have no effect.

    Continuous collects one group per call, so no arm can overlap inside a
    collection; accepting the key would be a no-op knob the user sets expecting
    a measurable difference.
    """
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    with pytest.raises(ValueError, match=r"strict_on_policy collection only"):
        RolloutOrchestrationConfig(
            schedule_mode="continuous",
            reward_collection_mode="per_group_streaming",
        )


def test_config_parsing_stays_torch_free() -> None:
    """Resolving any recipe must not load torch.

    Three package facades defer their torch-backed submodules to keep this true
    (vrl.trajectory, vrl.trainers.data, vrl.algorithms.grpo). Without this test
    the next eager re-export in any of them silently puts torch back on the
    config path, where nothing else would notice. Subprocess because the test
    session has already imported torch.
    """

    probe = (
        "import sys; "
        "from vrl.config.loading import load_config; "
        "from vrl.config.schema import parse_config; "
        "parse_config(load_config('experiment/sana/online_grpo_aesthetic')); "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
