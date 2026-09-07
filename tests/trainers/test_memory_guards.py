"""Tests for host-memory guard helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.generation.ray.config import RayGenerationConfig
from vrl.ray.resources import ResolvedDistributedResources
from vrl.utils.cuda_memory import is_cuda_out_of_memory
from vrl.utils.memory import HostMemorySnapshot, format_host_memory


def _ray_config(*, colocated: bool) -> RayGenerationConfig:
    rollout = (
        {
            "gpu_pool": "trainer",
            "num_gpus": 1,
            "num_engines": 1,
        }
        if colocated
        else {
            "devices": [1],
            "num_gpus": 1,
            "num_engines": 1,
        }
    )
    cfg = OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": [0] if colocated else [0, 1],
                    "trainer": {"devices": [0]},
                    "rollout": rollout,
                },
                "rollout": {},
            },
        },
    )
    return RayGenerationConfig.from_root(
        parse_config(cfg),
        resources=ResolvedDistributedResources.from_root(parse_config(cfg)),
    )


def test_format_host_memory_omits_unknown_fields() -> None:
    """``format_host_memory`` prints only the fields the snapshot knows: ``rss`` alone when
    available/total are unknown.
    """
    snapshot = HostMemorySnapshot(rss_mb=10.0, available_mb=None, total_mb=None)

    assert format_host_memory(snapshot) == "rss=10.0MiB"


def test_cuda_oom_detection_prefers_the_typed_exception() -> None:
    """A typed CUDA OOM remains detectable even if its message format changes."""
    import torch

    assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("allocation failed"))


def test_colocated_full_generation_bundle_can_fail_strict_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under ``VRL_STRICT_REPLAY_MEMORY_GUARD`` a colocated driver bundle that loads full
    generation modules is rejected.
    """
    monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")
    bundle = SimpleNamespace(loads_full_generation_modules=True)
    config = _ray_config(colocated=True)

    with pytest.raises(ValueError, match="loads_full_generation_modules=true"):
        config.validate_driver_state(driver_bundle=bundle)


def test_non_colocated_full_generation_bundle_passes_memory_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same bundle passes when trainer and rollout do not share a GPU;
    ``validate_driver_state`` returns the config for chaining.
    """
    monkeypatch.setenv("VRL_STRICT_REPLAY_MEMORY_GUARD", "1")
    bundle = SimpleNamespace(loads_full_generation_modules=True)
    config = _ray_config(colocated=False)

    assert config.validate_driver_state(driver_bundle=bundle) is config
