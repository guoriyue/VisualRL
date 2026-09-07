"""Tests for role-level distributed resource resolution."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.ray.placement import BundleLayout
from vrl.ray.resources import (
    ResolvedDistributedResources,
    format_distributed_resource_plan,
)


def _cfg(
    resources: dict,
    *,
    rollout_runtime: dict | None = None,
    kling_video_reward: bool = False,
    reward_components: dict[str, float] | None = None,
    reward_kwargs: dict[str, dict] | None = None,
    reward_inference: dict[str, dict] | None = None,
) -> object:
    if rollout_runtime is None:
        rollout_runtime = {}
    data = {
        "distributed": {
            "resources": resources,
            "rollout": rollout_runtime,
        },
    }
    if kling_video_reward:
        reward_components = {"kling_video_reward": 1.0}
        reward_kwargs = {"kling_video_reward": {}}
    if reward_components is not None:
        data["reward"] = {
            "components": reward_components,
            "kwargs": reward_kwargs or {},
            "inference": reward_inference or {},
        }
    return OmegaConf.create(
        data,
    )


def test_auto_split_uses_remaining_visible_gpus_for_rollout() -> None:
    """``num_gpus: auto`` gives the rollout every visible GPU the trainer did not take, one engine
    per GPU, and a trainer reservation on ``cuda:0``.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2, 3],
                    "trainer": {"num_gpus": 1},
                    "rollout": {
                        "num_gpus": "auto",
                        "num_engines": "auto",
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.reward_devices == ()
    assert resolved.rollout_num_engines == 3
    assert resolved.requires_trainer_reservation is True
    assert resolved.trainer_torch_device == "cuda:0"


def test_resolved_resource_summaries_are_derived_from_topology() -> None:
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
            )
        ),
    )

    assert resolved.rollout_num_gpus == 1
    assert resolved.colocated is False
    assert resolved.requires_trainer_reservation is True
    assert BundleLayout.from_resources(resolved).bundle_gpu_ids == (0, 1)


def test_explicit_split_devices_do_not_overlap() -> None:
    """Explicit disjoint device lists resolve verbatim and are not colocated."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2, 3],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1, 2, 3]},
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.colocated is False


def test_pinned_device_intersection_declares_colocation() -> None:
    """Hand-pinning intersecting devices sets IS the sharing declaration."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                },
            )
        ),
    )

    assert resolved.colocated is True
    assert resolved.requires_trainer_reservation is False


def test_colocate_via_gpu_pool_trainer() -> None:
    """gpu_pool=trainer declares sharing by pool word instead of pinned ids."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "devices": [0],
                        "gpu_pool": "trainer",
                    },
                },
            )
        ),
    )

    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"


def test_colocate_auto_pins_rollout_to_trainer_gpu() -> None:
    """Auto rollout placement is forced onto the trainer GPU, not a spare one."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "num_gpus": "auto",
                        "gpu_pool": "trainer",
                    },
                },
            )
        ),
    )

    assert resolved.rollout_devices == (0,)
    assert resolved.colocated is True


def test_colocate_rejects_explicit_disjoint_rollout_devices() -> None:
    """Checks colocate cannot be silently reinterpreted as split-GPU resident."""
    with pytest.raises(ValueError, match="disjoint from trainer"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0]},
                        "rollout": {
                            "devices": [1],
                            "gpu_pool": "trainer",
                        },
                    },
                )
            ),
        )


def test_devices_must_be_subset_of_visible_devices() -> None:
    """Every role spells its own full config path in the failure, not a bare role name."""
    with pytest.raises(
        ValueError,
        match=(
            r"distributed\.resources\.trainer\.devices contains devices outside "
            r"distributed\.resources\.visible_devices"
        ),
    ):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [2]},
                        "rollout": {"num_gpus": 1},
                    },
                )
            ),
        )


def test_num_engines_must_match_the_resolved_gpu_count() -> None:
    """A contradicting explicit num_engines fails against the derivation."""
    with pytest.raises(ValueError, match="num_engines must equal rollout GPUs"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2],
                        "trainer": {"num_gpus": 1},
                        "rollout": {
                            "num_gpus": 2,
                            "num_engines": 1,
                        },
                    },
                )
            ),
        )


def test_gpus_per_engine_derives_engine_count() -> None:
    """4 rollout GPUs at 2 ranks per engine = 2 sequence-parallel engines."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2, 3, 4],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1, 2, 3, 4], "gpus_per_engine": 2},
                },
            )
        ),
    )

    assert resolved.rollout_devices == (1, 2, 3, 4)
    assert resolved.rollout_num_engines == 2
    assert resolved.rollout_gpus_per_engine == 2
    assert "gpus_per_engine=2" in format_distributed_resource_plan(resolved)


def test_gpus_per_engine_requires_divisible_gpu_count() -> None:
    with pytest.raises(ValueError, match="not divisible into engines of 2"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1, 2, 3], "gpus_per_engine": 2},
                    },
                )
            ),
        )


def test_explicit_num_engines_must_match_the_engine_derivation() -> None:
    with pytest.raises(ValueError, match="num_engines must equal rollout GPUs / "):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3, 4],
                        "trainer": {"devices": [0]},
                        "rollout": {
                            "devices": [1, 2, 3, 4],
                            "gpus_per_engine": 2,
                            "num_engines": 4,
                        },
                    },
                )
            ),
        )


def test_gpus_per_engine_rejects_cross_node_and_cpu_fleets() -> None:
    with pytest.raises(ValueError, match="cross_node"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": 2, "gpus_per_engine": 2},
                    },
                )
            ),
        )
    with pytest.raises(ValueError, match="CPU engine"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0],
                        "trainer": {"num_gpus": 0},
                        "rollout": {"num_gpus": 0, "gpus_per_engine": 2},
                    },
                )
            ),
        )


def test_single_gpu_auto_split_shares_the_trainer_gpu() -> None:
    """The auto pool is spare-first-else-share: no spare -> colocate on trainer."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 1},
                    "rollout": {
                        "num_gpus": 1,
                        "num_engines": 1,
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (0,)
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"


def test_single_gpu_dedicated_rollout_pool_requires_a_spare() -> None:
    """gpu_pool=dedicated never falls back to sharing the trainer GPU."""
    with pytest.raises(ValueError, match="gpu_pool=dedicated requires spare"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0],
                        "trainer": {"num_gpus": 1},
                        "rollout": {
                            "num_gpus": 1,
                            "num_engines": 1,
                            "gpu_pool": "dedicated",
                        },
                    },
                )
            ),
        )


def test_cpu_only_rollout_uses_no_gpu_bundles() -> None:
    """A GPU-less plan resolves to empty device tuples, keeps the requested engine count, and puts
    the trainer on CPU.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "num_gpus": 0,
                        "num_engines": 2,
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == ()
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_engines == 2
    assert resolved.trainer_torch_device == "cpu"


def test_pinned_devices_supersede_a_zero_gpu_count() -> None:
    """Pinned devices are the authoritative count, even over num_gpus: 0."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 0},
                    "rollout": {
                        "devices": [0],
                        "num_gpus": 0,
                        "num_engines": 1,
                    },
                },
            )
        ),
    )

    assert resolved.rollout_devices == (0,)
    assert resolved.rollout_num_engines == 1


def test_cpu_only_reward_rejects_a_gpu_device_assignment() -> None:
    """A CPU reward slot cannot carry an ignored GPU reservation."""
    with pytest.raises(ValueError, match=r"requires distributed\.resources\.reward\.device=gpu"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1]},
                        "reward": {"device": "cpu", "devices": [2]},
                    },
                )
            ),
        )


def test_trainer_only_plan_allows_zero_rollout_workers() -> None:
    """A trainer-only plan (zero rollout GPUs and engines) is valid and leaves the rollout device
    set empty.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"num_gpus": 1},
                    "rollout": {
                        "num_gpus": 0,
                        "num_engines": 0,
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == ()
    assert resolved.rollout_num_engines == 0
    assert resolved.trainer_torch_device == "cuda:0"


def test_reward_torch_device_uses_the_reserved_local_gpu() -> None:
    """A local reward reservation, not the trainer default, owns model placement."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu", "devices": [2], "gpu_pool": "dedicated"},
                },
            )
        ),
    )

    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:2"


def test_reward_torch_device_translates_narrowed_rank_plan_ordinals(monkeypatch) -> None:
    """A rank narrowed to one physical GPU addresses its reward as torch cuda:0.

    Rank-local torchrun launches keep the plan in physical ordinal space
    (visible_devices=[2] on local rank 2) while CUDA_VISIBLE_DEVICES masks the
    process to that single card; the raw physical id is then an invalid torch
    ordinal (measured: non-zero ranks of the hpsv3 fsdp 4-rank smoke).
    """
    import torch

    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [2],
                    "trainer": {"devices": [2]},
                    "rollout": {"gpu_pool": "trainer"},
                    "reward": {"device": "gpu"},
                },
            )
        ),
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:0"
    # The inverse translation returns the plan-space id Ray placement reports.
    assert resolved.plan_device_ordinal(0) == 2


def test_reward_torch_device_without_a_reservation_follows_the_rank_local_trainer() -> None:
    """An unreserved in-process reward shares the caller's actual trainer device."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                },
            )
        ),
    )

    assert resolved.reward_torch_device(trainer_device="cuda:7") == "cuda:7"


def test_reward_torch_device_honors_an_explicit_cpu_slot() -> None:
    """A CPU reward request must not silently inherit the trainer CUDA device."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "cpu"},
                },
            )
        ),
    )

    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cpu"


def test_multi_gpu_local_reward_is_rejected_at_resolution() -> None:
    """One driver-local reward runtime cannot consume an actor-pool-shaped plan."""
    with pytest.raises(ValueError, match="exactly one GPU"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1, 2, 3],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1]},
                        "reward": {"device": "gpu", "devices": [2, 3]},
                    },
                )
            ),
        )


def test_cross_node_reward_gpu_is_rejected_at_resolution() -> None:
    """A remote Ray ordinal cannot be used as a CUDA device in the driver process."""
    with pytest.raises(ValueError, match="cannot reserve a local reward GPU"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {
                            "devices": [1],
                            "num_engines": 1,
                        },
                        "reward": {"device": "gpu", "devices": [1]},
                    },
                )
            ),
        )


def test_resource_plan_formatter_includes_key_fields() -> None:
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": "auto"},
                },
            )
        ),
    )

    text = format_distributed_resource_plan(resolved)

    # The formatter renders resolved fields as `key=value`; assert the resolved
    # values reach the log line, not a frozen layout. A reword of the plan line
    # (key spelling / separators) must not break behavioral coverage.
    assert f"trainer={list(resolved.trainer_devices)}" in text
    assert f"rollout={list(resolved.rollout_devices)}" in text
    assert f"reward={list(resolved.reward_devices)}" in text
    assert f"trainer_reservation={resolved.requires_trainer_reservation}" in text
    assert f"lifecycle=rollout:{resolved.lifecycle.rollout_mode}" in text


def test_cross_node_rollout_satisfies_budget_from_explicit_counts() -> None:
    """``cross_node`` with explicit counts assigns trainer and rollout distinct ordinals, not
    colocated, with no trainer reservation.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": "auto",
                    "cross_node": True,
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 1, "num_engines": 1},
                },
            )
        ),
    )

    assert resolved.cross_node is True
    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1,)
    assert resolved.rollout_num_engines == 1
    assert resolved.colocated is False
    assert resolved.requires_trainer_reservation is False
    assert resolved.trainer_torch_device == "cuda:0"


def test_cross_node_scales_to_multiple_rollout_workers() -> None:
    """``cross_node`` scales rollout devices and engines to the explicit count."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": "auto",
                    "cross_node": True,
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 3, "num_engines": 3},
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1, 2, 3)
    assert resolved.rollout_num_engines == 3
    assert resolved.requires_trainer_reservation is False


def test_cross_node_requires_explicit_rollout_count() -> None:
    with pytest.raises(ValueError, match="cross_node"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": "auto",
                        "cross_node": True,
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": "auto"},
                    },
                )
            ),
        )


def test_cross_node_preset_resolves() -> None:
    """The bundled cross-node preset resolves to one trainer GPU, one rollout GPU and one engine
    with no reservation.
    """
    from omegaconf import OmegaConf

    from vrl.config.loading import bundled_config_resource

    preset = bundled_config_resource("base/distributed/ray_rollout_cross_node")
    with preset.open("r", encoding="utf-8") as stream:
        resolved = ResolvedDistributedResources.from_root(parse_config(OmegaConf.load(stream)))

    assert resolved.cross_node is True
    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (1,)
    assert resolved.rollout_num_engines == 1
    assert resolved.requires_trainer_reservation is False


def test_cross_node_kling_recipe_keeps_the_local_reward_on_the_driver() -> None:
    """The shipped two-host recipe has no remote reward token after pool removal."""
    from vrl.config.loading import load_config

    cfg = load_config(
        "experiment/cosmos_predict2_5/online_nft_kling_video_reward_cross_node",
    )
    resolved = ResolvedDistributedResources.from_root(parse_config(cfg))

    assert resolved.cross_node is True
    assert resolved.rollout_devices == (1,)
    assert resolved.reward_devices == ()
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cuda:0"


def test_reward_role_resolves_after_trainer_and_rollout_devices() -> None:
    """The reward role takes a GPU left over after trainer and rollout, disjoint from the rollout
    pool, so no rollout release is needed before scoring.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu"},
                },
            )
        ),
    )

    assert resolved.reward_devices == (2,)
    assert len(resolved.reward_devices) == 1
    assert resolved.reward_runs_on_cpu is False
    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))
    assert resolved.requires_trainer_reservation is True
    assert resolved.lifecycle.release_rollout_before_reward is False


def test_lifecycle_plan_resident_when_roles_disjoint() -> None:
    """Fully disjoint trainer/rollout/reward GPUs -> every role resident, no handoff."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu", "devices": [2]},
                },
            )
        ),
    )

    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))
    plan = resolved.lifecycle
    assert plan.rollout_mode == "resident"
    assert plan.release_rollout_before_train is False
    assert plan.release_rollout_before_reward is False
    assert plan.release_reward_after_score is False


def test_lifecycle_plan_on_demand_for_shared_reward() -> None:
    """Shared reward GPU -> rollout/reward on_demand, but no trainer handoff."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu", "gpu_pool": "rollout"},
                },
            )
        ),
    )

    assert set(resolved.reward_devices) & set(resolved.rollout_devices)
    plan = resolved.lifecycle
    assert plan.rollout_mode == "on_demand"
    assert plan.release_rollout_before_train is False
    assert plan.release_rollout_before_reward is True
    assert plan.release_reward_after_score is True


def test_lifecycle_plan_colocated_rollout_is_on_demand_before_train() -> None:
    """Trainer/rollout share a GPU -> rollout on_demand, releases before train only."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                },
            )
        ),
    )

    plan = resolved.lifecycle
    assert resolved.colocated is True
    assert plan.rollout_mode == "on_demand"
    assert plan.release_rollout_before_train is True
    # No reward role shares the rollout GPU, so no reward release is needed.
    assert plan.release_reward_after_score is False
    assert plan.release_rollout_before_reward is False


def test_in_process_reward_without_reservation_follows_trainer_topology() -> None:
    """An active configured reward cannot disappear behind reward_devices=[]."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "devices": [0],
                        "gpu_pool": "trainer",
                    },
                    "reward": {"device": "trainer"},
                },
                reward_components={"aesthetic": 1.0},
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.lifecycle.release_trainer_before_reward is True
    assert resolved.lifecycle.release_rollout_before_reward is True
    assert resolved.lifecycle.release_reward_after_score is True


def test_explicit_cpu_reward_does_not_create_gpu_handoffs() -> None:
    """CPU execution is a resource fact, independent of parking capability."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "cpu"},
                },
                reward_components={"ocr": 1.0},
            )
        ),
    )

    assert resolved.lifecycle.release_trainer_before_reward is False
    assert resolved.lifecycle.release_rollout_before_reward is False
    assert resolved.lifecycle.release_reward_after_score is False
    assert resolved.reward_torch_device(trainer_device="cuda:0") == "cpu"


def test_http_only_reward_owns_no_local_resource_or_handoff() -> None:
    """Inherited local reward reservations disappear for external-only scoring."""

    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    # A reward preset may carry this local default. HTTP deployment
                    # owns its accelerator externally and must not reserve GPU2.
                    "reward": {"device": "gpu", "devices": [2]},
                },
                reward_components={"videoscore2": 1.0},
                reward_inference={
                    "videoscore2": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.lifecycle.release_trainer_before_reward is False
    assert resolved.lifecycle.release_rollout_before_reward is False
    assert resolved.lifecycle.release_reward_after_score is False
    assert BundleLayout.from_resources(resolved).reward_bundle_indices == ()


def test_mixed_http_and_local_reward_resources_cover_only_local_execution() -> None:
    """A remote sibling does not erase a real local component's CPU execution."""

    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "cpu"},
                },
                reward_components={"ocr": 0.5, "videoscore2": 0.5},
                reward_inference={
                    "videoscore2": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            )
        ),
    )

    assert resolved.reward_devices == ()
    assert resolved.reward_runs_on_cpu is True
    assert resolved.reward_torch_device() == "cpu"
    # CPU rewards get no bundle: they execute in the driver process.
    assert BundleLayout.from_resources(resolved).reward_bundle_indices == ()


def test_reward_auto_placement_prefers_dedicated_spare_gpu() -> None:
    """Checks unset gpu_pool takes the spare GPU on multi-GPU boxes."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu"},
                },
            )
        ),
    )

    assert resolved.reward_devices == (2,)
    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))
    assert resolved.lifecycle.release_reward_after_score is False


def test_reward_auto_placement_falls_back_to_shared_pool_on_single_gpu() -> None:
    """Checks unset gpu_pool shares the rollout GPU when none is spare."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0]},
                    "reward": {"device": "gpu"},
                },
            )
        ),
    )

    assert resolved.reward_devices == (0,)
    assert set(resolved.reward_devices) & set(resolved.rollout_devices)
    assert resolved.lifecycle.release_rollout_before_reward is True
    assert resolved.lifecycle.release_reward_after_score is True


def test_reward_can_share_rollout_pool_when_phases_release() -> None:
    """``gpu_pool: rollout`` puts the reward on the rollout GPU, which requires the rollout to
    release before scoring and the reward to release after.
    """
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [1]},
                    "reward": {"device": "gpu", "gpu_pool": "rollout"},
                },
            )
        ),
    )

    assert resolved.reward_devices == (1,)
    assert set(resolved.reward_devices) & set(resolved.rollout_devices)
    assert resolved.lifecycle.release_rollout_before_reward is True
    assert resolved.lifecycle.release_reward_after_score is True


def test_reward_shared_pool_requires_a_rollout_gpu() -> None:
    """gpu_pool=rollout with a CPU-only rollout has no GPU to share."""
    with pytest.raises(ValueError, match="Not enough rollout GPUs"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0],
                        "trainer": {"devices": [0]},
                        "rollout": {"num_gpus": 0, "num_engines": 1},
                        "reward": {"device": "gpu", "gpu_pool": "rollout"},
                    },
                )
            ),
        )


def test_pinned_reward_on_the_trainer_gpu_declares_sharing() -> None:
    """A reward reservation pinned onto the trainer GPU is a sharing declaration."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {
                        "num_gpus": 0,
                    },
                    "reward": {"device": "gpu", "devices": [0]},
                },
                kling_video_reward=True,
            )
        ),
    )

    assert resolved.reward_devices == (0,)
    assert resolved.lifecycle.trainer_and_reward_share_gpu is True


def test_colocated_reward_on_dedicated_gpu_owns_its_own_bundle() -> None:
    """Colocated trainer+rollout on GPU 0; a dedicated reward GPU 1 owns its own
    bundle (the run-level layout targets it directly, no reservation offset)."""

    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "num_engines": 1},
                    "reward": {"device": "gpu", "devices": [1]},
                },
                kling_video_reward=True,
            )
        ),
    )
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"

    layout = BundleLayout.from_resources(resolved)
    # Colocated trainer+rollout -> no reserved trainer bundle; reward owns a
    # bundle distinct from rollout (its own GPU 1).
    assert layout.bundle_gpu_ids == (0, 1)
    assert layout.reward_bundle_indices != ()
    assert set(layout.rollout_bundle_indices).isdisjoint(layout.reward_bundle_indices)


def test_shared_single_gpu_reward_reuses_rollout_bundle() -> None:
    """Shared single-GPU reward sits on the rollout device: same bundle index."""

    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0],
                    "trainer": {"devices": [0]},
                    "rollout": {"devices": [0], "num_engines": 1},
                    "reward": {"device": "gpu", "devices": [0], "gpu_pool": "rollout"},
                },
                kling_video_reward=True,
            )
        ),
    )

    layout = BundleLayout.from_resources(resolved)
    assert layout.rollout_bundle_indices == layout.reward_bundle_indices
    assert layout.total_bundles == 1


# ---------------------------------------------------------------- fsdp (P2)


def _cfg_training(resources: dict, training: dict) -> object:
    return OmegaConf.create(
        {
            "distributed": {
                "resources": resources,
                "rollout": {},
                "training": training,
            },
        },
    )


def test_fsdp_trainer_allows_multi_gpu_disjoint_from_rollout() -> None:
    """fsdp lifts the single-device cap: trainer can own N GPUs."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg_training(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"devices": [0, 1]},
                    "rollout": {"devices": [2]},
                },
                {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
            )
        ),
    )
    assert resolved.trainer_devices == (0, 1)
    assert resolved.rollout_devices == (2,)


def test_fsdp_trainer_count_must_equal_world_size() -> None:
    """fsdp trainer device count must match num_nodes*gpus_per_node."""
    with pytest.raises(ValueError, match=r"must own num_nodes\*gpus_per_node=2"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0]},
                        "rollout": {"devices": [1]},
                    },
                    {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
                )
            ),
        )


def test_fsdp_trainer_must_be_disjoint_from_rollout_even_with_overlap() -> None:
    """fsdp rejects trainer/rollout GPU overlap even when pinned deliberately."""
    with pytest.raises(ValueError, match="fsdp requires trainer GPUs disjoint"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0, 1]},
                        "rollout": {"devices": [1]},
                    },
                    {"strategy": "fsdp", "num_nodes": 1, "gpus_per_node": 2},
                )
            ),
        )


def test_single_process_still_rejects_multi_gpu_trainer() -> None:
    """The single-device cap is preserved for single_process (default)."""
    with pytest.raises(ValueError, match="0 or 1 GPU"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg_training(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"devices": [0, 1]},
                        "rollout": {"num_gpus": 0, "num_engines": 1},
                    },
                    {"strategy": "single_process"},
                )
            ),
        )


# --- P0 surface: rollout.gpu_pool / reward.gpu_pool (single authoritative grammar) ---


def test_reward_gpu_pool_rollout_shares_rollout_gpu() -> None:
    """reward.gpu_pool=rollout forces the reward pool onto the rollout GPU."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1],
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 1},
                    "reward": {"device": "gpu", "gpu_pool": "rollout"},
                },
                reward_components={"r": 1.0},
                reward_kwargs={"r": {"execution": "pool"}},
            )
        ),
    )
    assert set(resolved.reward_devices) & set(resolved.rollout_devices)
    assert resolved.reward_devices == resolved.rollout_devices


def test_reward_gpu_pool_auto_prefers_spare_gpu() -> None:
    """reward.gpu_pool=auto takes a dedicated spare GPU when one exists."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            _cfg(
                {
                    "visible_devices": [0, 1, 2],
                    "trainer": {"num_gpus": 1},
                    "rollout": {"num_gpus": 1},
                    "reward": {"device": "gpu", "gpu_pool": "auto"},
                },
                reward_components={"r": 1.0},
                reward_kwargs={"r": {"execution": "pool"}},
            )
        ),
    )
    assert resolved.reward_devices == (2,)
    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))


def test_reward_gpu_pool_rejects_unknown_value() -> None:
    """reward.gpu_pool only accepts auto/rollout/dedicated."""
    with pytest.raises(ValueError, match=r"unknown distributed\.resources\.reward\.gpu_pool"):
        ResolvedDistributedResources.from_root(
            parse_config(
                _cfg(
                    {
                        "visible_devices": [0, 1],
                        "trainer": {"num_gpus": 1},
                        "rollout": {"num_gpus": 1},
                        "reward": {"device": "gpu", "gpu_pool": "nonsense"},
                    },
                    reward_components={"r": 1.0},
                    reward_kwargs={"r": {"execution": "pool"}},
                )
            ),
        )


# ── Symmetric colocated DDP (SPRINT_symmetric_colocated_ddp) ──


def test_ddp_colocate_resolves_per_rank_local_single_gpu() -> None:
    """Symmetric colocated DDP: each rank resolves only its LOCAL single GPU
    (trainer + colocated rollout on it); world_size drives only the grad
    all-reduce, NOT the per-rank GPU plan (ddp follows the single-GPU rule, not
    fsdp's world-covering one)."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "training": {"strategy": "ddp", "num_nodes": 2, "gpus_per_node": 1},
                        "resources": {
                            "visible_devices": [0],
                            "trainer": {"num_gpus": 1},
                            "rollout": {"gpu_pool": "trainer"},
                        },
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (0,)  # colocated on the local GPU
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"
    assert resolved.cross_node is False  # per-rank-local: no shared Ray cluster


def test_fsdp_colocate_resolves_per_rank_local_single_gpu() -> None:
    """Symmetric colocated FSDP (SPRINT_multi_gpu_training Phase 4): like ddp, each
    rank resolves only its LOCAL single GPU (trainer + colocated rollout on it),
    NOT fsdp's world-covering asymmetric plan. Signaled by rollout.gpu_pool=trainer,
    so the world-size trainer rule and the disjoint rule do not apply."""
    resolved = ResolvedDistributedResources.from_root(
        parse_config(
            OmegaConf.create(
                {
                    "distributed": {
                        "training": {"strategy": "fsdp", "num_nodes": 2, "gpus_per_node": 1},
                        "resources": {
                            "visible_devices": [0],
                            "trainer": {"num_gpus": 1},
                            "rollout": {"gpu_pool": "trainer"},
                        },
                    },
                },
            )
        ),
    )

    assert resolved.trainer_devices == (0,)
    assert resolved.rollout_devices == (0,)  # colocated on the local GPU
    assert resolved.colocated is True
    assert resolved.lifecycle.rollout_mode == "on_demand"
    assert resolved.cross_node is False  # per-rank-local: no shared Ray cluster


# ── cosmos async-reward recipe (3-GPU disjoint reward overlap) ───────────────


def test_cosmos_async_reward_recipe_resolves_resident_reward_overlap() -> None:
    """The cosmos online_grpo_async_reward recipe composes a 3-GPU disjoint layout
    whose RESOLVED plan enables reward(N)/rollout(N+1) overlap.

    Loads the recipe through the real config compose path (same as
    test_all_experiments_load_and_validate), then asserts the resolver's BEHAVIOR,
    not literal YAML values: reward devices disjoint from rollout, the handoff does
    NOT release rollout before reward, the reward lease is resident, and the
    continuous orchestration block is present/parsed. visible_devices is the only
    site-supplied knob (the recipe is GPU-count agnostic), so the test pins the
    [0,1,2] disjoint topology the recipe header documents.
    """
    from vrl.config.loading import load_config

    cfg = load_config("experiment/cosmos_predict2/online_grpo_async_reward")

    # The recipe deliberately omits visible_devices (it is a site/run knob). Supply
    # the 3-GPU box the disjoint layout targets; the resolver derives the rest.
    cfg.distributed.resources.visible_devices = [0, 1, 2]

    resolved = ResolvedDistributedResources.from_root(parse_config(cfg))

    # Disjoint reward placement is the load-bearing topology: reward must not share
    # the rollout GPU, otherwise reward(N) serializes after rollout(N).
    assert not (set(resolved.reward_devices) & set(resolved.rollout_devices))
    # Disjoint reward -> no pre-reward rollout release -> reward overlaps rollout(N+1).
    assert resolved.lifecycle.release_rollout_before_reward is False
    # A resident reward lease is what keeps reward on its own card across iterations.
    assert resolved.lifecycle.release_reward_after_score is False

    # Continuous orchestration must be composed in (the producer keeps rollout(N+1)
    # in flight while reward(N) scores). Assert the parsed schedule mode + inflight
    # bound behavior, not the literal yaml number.
    from vrl.trainers.core.types import RolloutOrchestrationConfig

    orchestration = cfg.trainer.rollout_orchestration
    assert orchestration.schedule_mode == "continuous"
    typed = RolloutOrchestrationConfig(
        **OmegaConf.to_container(orchestration, resolve=True),
    )
    # >=2 inflight groups is the invariant that lets rollout(N+1) produce while
    # reward(N) scores; assert the relation, not the literal value.
    assert typed.continuous.max_inflight_groups >= 2
