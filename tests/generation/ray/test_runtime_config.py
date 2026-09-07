"""Tests for rollout runtime factory fail-fast behavior."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.builders import BuiltConfigs
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation.execution.types import BatchSizeProbeResult
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.ray.config import RayGenerationConfig
from vrl.generation.ray.engine import RayGenerationEngine
from vrl.generation.ray.executor import RayGenerationExecutor
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.launcher import (
    RayGenerationLauncher,
    _all_ranks_support_versioned_slots,
)
from vrl.generation.ray.runtime import RayGenerationRuntime
from vrl.generation.ray.session import RayGenerationSession
from vrl.generation.types import GenerationRequest
from vrl.models.families.registry import ModelFamilyEntry, get_model_family_entry
from vrl.ray.actor_group import RayActorHandle
from vrl.ray.actor_pool import RayActorDispatcher
from vrl.ray.operation_deadline import RayOperationTimeout
from vrl.ray.placement import GlobalRayPlacementOwner, RolePlacement
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import (
    OnlineRunConfig,
    ResolvedModel,
    ResolvedOnlineRun,
)
from vrl.trainers.checkpointing import TrainingResumeConfig
from vrl.utils.lifecycle import RuntimePhase


class _CudaPolicy:
    device = "cuda:0"


class _CpuPolicy:
    device = "cpu"


@dataclass
class _Bundle:
    """Driver-bundle stand-in for the CUDA-ownership checks these tests cover.

    ``loads_full_generation_modules`` defaults to the replay answer so the
    colocated-RAM guard stays out of the way; the guard has its own tests in
    ``tests/trainers/test_memory_guards.py``.
    """

    model: Any
    trainable_modules: dict[str, Any]
    loads_full_generation_modules: bool = False


_TEST_MODEL_IDENTITY = {"schema": "test"}
_TEST_RPC_TIMEOUT_S = 30.0


def _runtime(
    executor: Any,
    *,
    weight_sync: Any | None = None,
    workers: list[RayActorHandle] | None = None,
) -> RayGenerationRuntime:
    return RayGenerationRuntime(
        session=RayGenerationSession(
            executor,
            weight_sync,
            list(workers or []),
        ),
    )


def test_launch_contract_accepts_primitive_config_leaves() -> None:
    contract = GenerationRuntimeLaunchContract(
        family="unit",
        model_build={
            "model_name_or_path": "unit-test",
            "model_config": {
                "text": "value",
                "integer": 1,
                "number": 1.5,
                "enabled": True,
                "optional": None,
            },
        },
        expected_model_identity=_TEST_MODEL_IDENTITY,
    )

    model_config = contract.model_build["model_config"]
    assert model_config["enabled"] is True
    assert model_config["optional"] is None
    assert contract.expected_model_identity == _TEST_MODEL_IDENTITY


def test_launch_contract_rejects_empty_registry_identity() -> None:
    with pytest.raises(ValueError, match=r"family must be non-empty"):
        GenerationRuntimeLaunchContract(
            family="",
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        )


def test_launch_contract_rejects_empty_model_identity() -> None:
    with pytest.raises(ValueError, match=r"expected_model_identity must be non-empty"):
        GenerationRuntimeLaunchContract(
            family="unit",
            model_build={},
            expected_model_identity={},
        )


def _cfg(
    *,
    num_engines: int = 1,
    overlap: bool = False,
):
    rollout_devices = [0] if overlap else [1]
    visible_devices = [0] if overlap else [0, 1]
    distributed = {
        "resources": {
            "visible_devices": visible_devices,
            "trainer": {"devices": [0]},
            "rollout": {
                "devices": rollout_devices,
                "num_engines": num_engines,
            },
        },
        # Release scheduling is derived from topology; nothing to spell here.
        "rollout": {},
    }
    return OmegaConf.create(
        {
            "distributed": distributed,
        },
    )


def _resource_cfg(
    *,
    trainer_devices: list[int],
    rollout_devices: list[int],
):
    rollout_runtime: dict[str, Any] = {"cpus_per_worker": 1}
    rollout_resource: dict[str, Any] = {
        "devices": rollout_devices,
        "num_engines": len(rollout_devices),
    }
    return OmegaConf.create(
        {
            "distributed": {
                "resources": {
                    "visible_devices": sorted(set(trainer_devices) | set(rollout_devices)),
                    "trainer": {"devices": trainer_devices},
                    "rollout": rollout_resource,
                },
                "rollout": rollout_runtime,
            },
        },
    )


def _ray_config(cfg: Any) -> RayGenerationConfig:
    return RayGenerationConfig.from_root(
        parse_config(cfg),
        resources=ResolvedDistributedResources.from_root(parse_config(cfg)),
    )


def _launch_cfg(
    *,
    model_torch_compile: dict[str, Any] | None = None,
) -> Any:
    model_config = {
        "family": "sd3_5",
        "path": "unit-test",
        "revision": "driver-config",
        "use_lora": False,
        "torch_compile": model_torch_compile
        or {
            "enable": False,
            "mode": "default",
        },
    }
    cfg: dict[str, Any] = {
        "distributed": {
            "resources": {
                "visible_devices": [],
                "trainer": {
                    "num_gpus": 0,
                    "devices": [],
                },
                "rollout": {
                    "num_gpus": 0,
                    "devices": [],
                    "num_engines": 1,
                },
            },
        },
        "model": model_config,
        "precision": {
            "float32_precision": "tf32",
            "training": {"dtype": "bf16", "outer_autocast": True},
            # Deliberately differs from training and prompt encoding so this
            # fixture proves role-specific values survive the Ray projection.
            "rollout": {
                "dtype": "fp32",
                "outer_autocast": True,
                "prompt_encoders": {"dtype": "fp16"},
            },
        },
        "rollout": {},
    }
    return OmegaConf.create(cfg)


def _capture_launch_inputs(
    cfg: Any,
    entry: ModelFamilyEntry,
    *,
    rollout_model_identity: dict[str, Any] | None = None,
) -> RayGenerationLaunchInputs:
    """Resolve the public Ray worker boundary without starting actors."""

    config = _ray_config(cfg)
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    schedule_mode = str(
        OmegaConf.select(
            cfg,
            "trainer.rollout_orchestration.schedule_mode",
            default="strict_on_policy",
        ),
    )
    run = ResolvedOnlineRun(
        built=BuiltConfigs(
            root=root,
            algorithm=None,
            precision=precision,
            trainer=SimpleNamespace(
                rollout_orchestration=SimpleNamespace(schedule_mode=schedule_mode),
            ),
            reward=None,
            resume=TrainingResumeConfig(),
        ),
        family=entry,
        resources=config.resources,
        device=torch.device("cpu"),
        run=OnlineRunConfig(total_epochs=0),
        generation=config,
        collector=RolloutCollectorConfig(),
    )
    replay_model = ResolvedModel(
        entry=entry,
        build=entry.resolve_model_build(
            root,
            torch.device("cpu"),
            precision=precision,
            for_rollout=False,
        ),
        identity=_TEST_MODEL_IDENTITY,
    )

    with (
        patch(
            "vrl.models.checkpoint_identity.resolve_checkpoint_model_identity",
            return_value=rollout_model_identity or _TEST_MODEL_IDENTITY,
        ) as resolve_identity,
    ):
        result = run.ray_launch_inputs(replay_model)

    resolve_identity.assert_called_once()
    return result


class _SlotWorker:
    """Real Ray actor exposing the versioned-slot capability query;
    ``supports=None`` raises like a dead/broken worker."""

    def __init__(self, supports: bool | None) -> None:
        self._supports = supports

    def supports_versioned_trainable_state(self) -> bool:
        if self._supports is None:
            raise RuntimeError("actor dead")
        return self._supports


@contextlib.contextmanager
def _slot_handles(ray: Any, *supports: bool | None) -> Iterator[list[RayActorHandle]]:
    """Real ``_SlotWorker`` actors, killed on exit.

    The cluster is shared across this package, so a fleet left running would keep
    holding actors for every later test. See ``real_local_ray``'s docstring.
    """

    actor_cls = ray.remote(num_cpus=0)(_SlotWorker)
    handles = [
        RayActorHandle(
            worker_id=f"w{index}",
            actor=actor_cls.remote(value),
        )
        for index, value in enumerate(supports)
    ]
    try:
        yield handles
    finally:
        for handle in handles:
            ray.kill(handle.actor, no_restart=True)


@pytest.mark.slow_test
def test_runtime_capability_is_and_over_all_workers(local_ray) -> None:
    """supports_non_draining_weight_sync derives as the AND of every worker's
    supports_versioned_trainable_state(): all True -> True; any False -> False."""
    weight_sync = object()

    with _slot_handles(local_ray, True, True) as handles:
        assert (
            _all_ranks_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=weight_sync,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is True
        )
    with _slot_handles(local_ray, True, False) as handles:
        assert (
            _all_ranks_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=weight_sync,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is False
        )


@pytest.mark.slow_test
def test_runtime_capability_false_without_weight_sync_or_workers(local_ray) -> None:
    """No weight sync (sync_trainable_state off) or no workers -> safe draining
    barrier (False), never a silent True."""
    with _slot_handles(local_ray, True, True) as handles:
        assert (
            _all_ranks_support_versioned_slots(
                local_ray,
                handles,
                weight_sync=None,
                worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
            )
            is False
        )
    assert (
        _all_ranks_support_versioned_slots(
            local_ray,
            [],
            weight_sync=object(),
            worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
        )
        is False
    )


@pytest.mark.slow_test
def test_runtime_capability_worker_query_failure_propagates(local_ray) -> None:
    """A failed capability RPC means the candidate worker is broken."""
    with (
        _slot_handles(local_ray, True, None) as handles,
        pytest.raises(local_ray.exceptions.RayTaskError, match="actor dead"),
    ):
        _all_ranks_support_versioned_slots(
            local_ray,
            handles,
            weight_sync=object(),
            worker_rpc_timeout_s=_TEST_RPC_TIMEOUT_S,
        )


def test_launcher_capability_failure_kills_candidate_actor_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vrl.generation.ray.launcher as launcher_module

    query_error = RuntimeError("versioned-slot capability query failed")
    capability_ref = object()

    class _CapabilityMethod:
        @staticmethod
        def remote() -> object:
            return capability_ref

    class _GetTimeoutError(TimeoutError):
        pass

    class _RayApi:
        exceptions = SimpleNamespace(GetTimeoutError=_GetTimeoutError)

        @staticmethod
        def get(refs: list[object], *, timeout: float) -> None:
            assert refs == [capability_ref]
            assert timeout > 0
            raise query_error

    class _ActorGroup:
        def __init__(self) -> None:
            self.handles = [
                SimpleNamespace(
                    worker_id="rollout-0",
                    node_ip="node",
                    gpu_ids=(),
                    actor=SimpleNamespace(
                        supports_versioned_trainable_state=_CapabilityMethod(),
                    ),
                ),
            ]
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    actor_group = _ActorGroup()
    monkeypatch.setattr(launcher_module, "require_ray", lambda: _RayApi)
    monkeypatch.setattr(
        launcher_module.RayActorGroup,
        "launch",
        staticmethod(lambda **_kwargs: actor_group),
    )

    cfg = _launch_cfg()
    config = _ray_config(cfg)
    entry = get_model_family_entry("sd3_5")
    inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        ),
        gatherer=entry.new_gatherer(),
    )

    with pytest.raises(RuntimeError, match="capability query failed") as caught:
        RayGenerationLauncher(init_ray=False)._launch_session(
            config,
            inputs,
            placement=RolePlacement(
                placement_group=object(),
                bundle_indices=(0,),
                expected_gpu_ids=(),
            ),
        )

    assert caught.value is query_error
    assert actor_group.shutdown_calls == 1

    # Invalid values are now rejected at the typed schema boundary
    # (RolloutRuntimeSection Literal) at parse time, not in RayGenerationConfig —
    # see tests/config/test_schema.py::test_unknown_batch_placement_strategy_raises.


def test_worker_defaults_and_explicit_override_project_from_public_schema() -> None:
    default = _ray_config(_cfg()).worker
    assert default.cpus_per_worker == 1.0
    assert default.worker_rpc_timeout_s == 600.0
    assert default.generation_stall_timeout_s == 3600.0
    assert default.pipelined is False
    assert default.sync_trainable_state is True

    cfg = _cfg()
    cfg.distributed.rollout.cpus_per_worker = 2.5
    cfg.distributed.rollout.worker_rpc_timeout_s = 3600.0
    cfg.distributed.rollout.generation_stall_timeout_s = 1200.0
    cfg.distributed.rollout.sync_trainable_state = False
    cfg.distributed.rollout.pipelined = True
    cfg.distributed.rollout.batch_placement_strategy = "dynamic"
    override = _ray_config(cfg).worker

    assert override.cpus_per_worker == 2.5
    assert override.worker_rpc_timeout_s == 3600.0
    assert override.generation_stall_timeout_s == 1200.0
    assert override.sync_trainable_state is False
    assert override.pipelined is True
    assert override.batch_placement_strategy == "dynamic"


def test_placement_and_launcher_consume_the_same_worker_snapshot(monkeypatch) -> None:
    import vrl.generation.ray.launcher as launcher_module

    cfg = _launch_cfg()
    cfg.distributed.rollout = {
        "cpus_per_worker": 2.5,
        "health_check_interval_s": 0.0,
        "sync_trainable_state": True,
    }
    config = _ray_config(cfg)
    owner = GlobalRayPlacementOwner(config.resources, config.worker)
    assert owner.rollout_worker is config.worker
    assert owner._bundle_requirements() == [{"CPU": 2.5}]

    launch_kwargs: dict[str, Any] = {}

    class _ActorGroup:
        def __init__(self) -> None:
            self.handles = [
                SimpleNamespace(
                    worker_id="rollout-0",
                    node_ip="node",
                    gpu_ids=(),
                    actor=object(),
                ),
            ]

        @staticmethod
        def shutdown() -> None:
            return None

    def capture_actor_launch(**kwargs: Any) -> _ActorGroup:
        launch_kwargs.update(kwargs)
        return _ActorGroup()

    monkeypatch.setattr(launcher_module, "require_ray", lambda: object())
    monkeypatch.setattr(
        launcher_module.RayActorGroup,
        "launch",
        staticmethod(capture_actor_launch),
    )
    monkeypatch.setattr(
        launcher_module,
        "_all_ranks_support_versioned_slots",
        lambda *_args, **_kwargs: False,
    )
    entry = get_model_family_entry("sd3_5")
    session = RayGenerationLauncher(init_ray=False)._launch_session(
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build={},
                expected_model_identity=_TEST_MODEL_IDENTITY,
            ),
            gatherer=entry.new_gatherer(),
        ),
        placement=RolePlacement(
            placement_group=object(),
            bundle_indices=(0,),
            expected_gpu_ids=(),
        ),
    )

    assert launch_kwargs["num_cpus"] == owner.rollout_worker.cpus_per_worker == 2.5
    assert launch_kwargs["rpc_timeout_s"] == config.worker.worker_rpc_timeout_s
    assert launch_kwargs["operation_prefix"] == "rollout"
    assert session.executor.generation_stall_timeout_s == config.worker.generation_stall_timeout_s
    assert session.weight_sync is not None
    assert session.executor.actor_dispatcher is session.weight_sync.actor_dispatcher


def test_pipelined_rejects_multiple_resolved_engines() -> None:
    cfg = _resource_cfg(trainer_devices=[0], rollout_devices=[1, 2])
    cfg.distributed.rollout.pipelined = True

    with pytest.raises(ValueError, match="requires exactly one rollout engine"):
        RayGenerationConfig.from_root(
            parse_config(cfg),
            resources=ResolvedDistributedResources.from_root(parse_config(cfg)),
        )


def test_pipelined_rejects_multiple_placement_bundles_before_ray_start(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vrl.generation.ray.launcher.require_ray",
        lambda: pytest.fail("Ray must not start before pipeline placement validation"),
    )
    cfg = _launch_cfg()
    cfg.distributed.rollout = {
        "pipelined": True,
        "sync_trainable_state": False,
    }
    config = RayGenerationConfig.from_root(
        parse_config(cfg),
        resources=ResolvedDistributedResources.from_root(parse_config(cfg)),
    )
    entry = get_model_family_entry("sd3_5")
    launch_inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=entry.family,
            model_build={},
            expected_model_identity=_TEST_MODEL_IDENTITY,
        ),
        gatherer=entry.new_gatherer(),
    )
    placement = RolePlacement(
        placement_group=object(),
        bundle_indices=(0, 1),
        expected_gpu_ids=(),
    )

    with pytest.raises(ValueError, match="exactly one rollout engine"):
        RayGenerationLauncher(init_ray=False)._launch_session(
            config,
            launch_inputs,
            placement=placement,
        )


def test_generation_launch_inputs_project_model_compile_and_precision() -> None:
    """The public launch path projects model config and dtype wire values once."""
    launch_inputs = _capture_launch_inputs(
        _launch_cfg(
            model_torch_compile={
                "enable": True,
                "mode": "default",
            },
        ),
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert launch_inputs.launch_contract.family == "sd3_5"
    assert launch_inputs.launch_contract.expected_model_identity == _TEST_MODEL_IDENTITY
    assert "family" not in model_build
    assert model_build["device"] == "cpu"
    assert model_build["parameter_dtype"] == "float32"
    assert model_build["precision"] == {
        "dtype": "fp32",
        "float32_precision": "tf32",
        "quantization": None,
        "outer_autocast": True,
    }
    assert model_build["rollout"]["prompt_encoder_dtype"] == "float16"
    assert model_build["revision"] == "driver-config"
    assert "revision" not in model_build["model_config"]
    assert model_build["model_config"]["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_generation_launch_inputs_project_resolved_generation_memory() -> None:
    """The launch contract carries typed memory values, not raw model config."""
    cfg = _launch_cfg()
    cfg.model.memory = {
        "vae_decode": {
            "tiling": True,
            "slicing": False,
        },
    }

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["generation_memory"] == {
        "vae_decode": {
            "tiling": True,
            "slicing": False,
        },
    }
    assert "memory" not in model_build["model_config"]


def test_generation_launch_inputs_project_wan_offload_to_rollout_contract(monkeypatch) -> None:
    from diffusers import DiffusionPipeline

    monkeypatch.setattr(
        DiffusionPipeline,
        "load_config",
        staticmethod(lambda *a, **k: {"boundary_ratio": None}),
    )
    cfg = _launch_cfg()
    cfg.model.family = "wan_2_1_i2v"
    cfg.model.revision = "a" * 40
    cfg.model.offload_mode = "sequential"

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("wan_2_1_i2v"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["rollout"]["pipeline_offload_mode"] == "sequential"
    assert "offload_mode" not in model_build["model_config"]


def test_generation_launch_inputs_reject_rollout_identity_mismatch() -> None:
    cfg = _launch_cfg()

    with pytest.raises(
        ValueError,
        match="rollout model identity does not match the driver replay model identity",
    ):
        _capture_launch_inputs(
            cfg,
            get_model_family_entry("sd3_5"),
            rollout_model_identity={"schema": "different"},
        )


def test_generation_launch_inputs_preserve_disabled_model_compile_config() -> None:
    """Checks disabled model.torch_compile is preserved as ordinary model config."""
    launch_inputs = _capture_launch_inputs(
        _launch_cfg(),
        get_model_family_entry("sd3_5"),
    )

    model_build = launch_inputs.launch_contract.model_build
    assert model_build["revision"] == "driver-config"
    model_config = model_build["model_config"]
    assert "revision" not in model_config
    assert model_config["torch_compile"] == {
        "enable": False,
        "mode": "default",
    }


def test_generation_launch_inputs_derive_versioned_sync_from_schedule() -> None:
    strict = _capture_launch_inputs(
        _launch_cfg(),
        get_model_family_entry("sd3_5"),
    )
    assert strict.launch_contract.versioned_weight_sync is False

    continuous_fullparam_cfg = _launch_cfg()
    continuous_fullparam_cfg.trainer = {
        "rollout_orchestration": {"schedule_mode": "continuous"},
    }
    continuous_fullparam = _capture_launch_inputs(
        continuous_fullparam_cfg,
        get_model_family_entry("sd3_5"),
    )
    assert continuous_fullparam.launch_contract.versioned_weight_sync is False

    continuous_lora_cfg = _launch_cfg()
    continuous_lora_cfg.model.use_lora = True
    continuous_lora_cfg.trainer = {
        "rollout_orchestration": {"schedule_mode": "continuous"},
    }
    continuous_lora = _capture_launch_inputs(
        continuous_lora_cfg,
        get_model_family_entry("sd3_5"),
    )
    assert continuous_lora.launch_contract.versioned_weight_sync is True


def test_generation_launch_inputs_thread_resolved_base_weight_sync() -> None:
    """The rollout lifecycle, not model YAML, owns master-weight retention."""
    cfg = _launch_cfg()
    cfg.distributed.rollout = {"sync_trainable_state": False}

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_generation_launch_inputs_mark_lora_as_adapter_only_sync() -> None:
    """LoRA sync never needs retained base-precision masters on the rollout."""
    cfg = _launch_cfg()
    cfg.model.use_lora = True

    launch_inputs = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    rollout = launch_inputs.launch_contract.model_build["rollout"]
    assert rollout["base_weight_sync"] is False


def test_generation_launch_inputs_reject_model_compile_for_ar_family() -> None:
    """Checks model.torch_compile fails fast on rollout families that cannot compile."""
    cfg = _launch_cfg(
        model_torch_compile={
            "enable": True,
            "mode": "default",
        },
    )
    cfg.model.family = "janus_pro"

    with pytest.raises(ValueError, match="does not support torch compile"):
        _capture_launch_inputs(cfg, get_model_family_entry("janus_pro"))


def _runtime_factory_inputs(
    *,
    rollout_mode: str = "resident",
) -> tuple[RayGenerationConfig, RayGenerationLaunchInputs, RolePlacement]:
    config = _ray_config(_launch_cfg())
    if rollout_mode == "on_demand":
        config = replace(
            config,
            resources=replace(
                config.resources,
                lifecycle=replace(
                    config.resources.lifecycle,
                    trainer_and_rollout_share_gpu=True,
                ),
            ),
        )
    elif rollout_mode != "resident":
        raise ValueError(f"unknown rollout mode: {rollout_mode}")
    entry = get_model_family_entry("sd3_5")
    return (
        config,
        RayGenerationLaunchInputs(
            launch_contract=GenerationRuntimeLaunchContract(
                family=entry.family,
                model_build={},
                expected_model_identity=_TEST_MODEL_IDENTITY,
            ),
            gatherer=entry.new_gatherer(),
        ),
        RolePlacement(
            placement_group=object(),
            bundle_indices=(),
            expected_gpu_ids=(),
        ),
    )


class _FactorySession:
    def __init__(self) -> None:
        self.workers: list[Any] = []
        self.weight_sync = object()
        self.supports_non_draining_weight_sync = False
        self.close = AsyncMock()
        self.force_close_calls = 0
        self.kill_engines_calls = 0

    def force_close(self) -> None:
        self.force_close_calls += 1

    def kill_engines(self) -> None:
        self.kill_engines_calls += 1


def test_create_runtime_launches_resident_topology() -> None:
    config, launch_inputs, placement = _runtime_factory_inputs()
    launcher = RayGenerationLauncher(init_ray=False)
    expected_session = _FactorySession()

    with patch.object(
        RayGenerationLauncher,
        "_launch_session",
        autospec=True,
        return_value=expected_session,
    ) as launch:
        runtime = launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )

    assert runtime._session is expected_session
    assert runtime._session_factory is None
    launch.assert_called_once_with(
        launcher,
        config,
        launch_inputs,
        placement=placement,
    )
    asyncio.run(runtime.shutdown())


def test_create_runtime_kills_resident_session_when_monitor_start_fails() -> None:
    config, launch_inputs, placement = _runtime_factory_inputs()
    launcher = RayGenerationLauncher(init_ray=False)
    expected_session = _FactorySession()

    with (
        patch.object(
            RayGenerationLauncher,
            "_launch_session",
            autospec=True,
            return_value=expected_session,
        ),
        patch(
            "vrl.generation.ray.health_monitor.RolloutWorkerHealthMonitor.start",
            side_effect=RuntimeError("thread start failed"),
        ),
        pytest.raises(RuntimeError, match="thread start failed"),
    ):
        launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )

    assert expected_session.force_close_calls == 1
    assert expected_session.kill_engines_calls == 1


def test_create_runtime_defers_on_demand_topology_launch() -> None:
    config, launch_inputs, placement = _runtime_factory_inputs(
        rollout_mode="on_demand",
    )
    launcher = RayGenerationLauncher(init_ray=False)

    with patch.object(
        RayGenerationLauncher,
        "_launch_session",
        autospec=True,
    ) as launch:
        runtime = launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )

    assert runtime._session is None
    assert runtime._session_factory is not None
    launch.assert_not_called()
    asyncio.run(runtime.shutdown())


@pytest.mark.asyncio
async def test_deferred_activation_reuses_factory_launcher() -> None:
    config, launch_inputs, placement = _runtime_factory_inputs(
        rollout_mode="on_demand",
    )
    # A GPU-owning fleet (non-empty rollout devices) selects the deferred
    # activation path under test.
    config = replace(
        config,
        resources=replace(
            config.resources,
            rollout_devices=(0,),
        ),
    )
    expected_launch_inputs = replace(
        launch_inputs,
        launch_contract=replace(
            launch_inputs.launch_contract,
            sleep_offload=True,
        ),
    )
    launcher = RayGenerationLauncher(
        init_ray=False,
        ray_init_kwargs={"address": "auto"},
    )
    candidate = _FactorySession()

    with patch.object(
        RayGenerationLauncher,
        "_launch_session_async",
        autospec=True,
        return_value=candidate,
    ) as launch_async:
        runtime = launcher.create_runtime(
            config,
            launch_inputs,
            placement=placement,
        )
        await runtime.activate()

    launch_async.assert_awaited_once_with(
        launcher,
        config,
        expected_launch_inputs,
        placement=placement,
    )
    await runtime.shutdown()
    candidate.close.assert_awaited_once_with(force=False)


def test_ray_backend_rejects_unapproved_driver_cuda_overlap() -> None:
    """The runtime backstop reports the concrete conflicting devices and policy."""
    config = _ray_config(
        _resource_cfg(
            trainer_devices=[1],
            rollout_devices=[0],
        ),
    )
    # The resolved trainer owns GPU 1, but the actual driver model reports GPU
    # 0. The launch boundary must reject that real topology mismatch.

    with pytest.raises(
        ValueError,
        match=(
            r"Trainer device cuda:0 overlaps rollout devices \[0\], "
            r"but the resolved plan expected disjoint"
        ),
    ):
        config.validate_driver_state(
            driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
        )


@pytest.mark.gpu
def test_ray_backend_detects_cuda_trainable_module_when_policy_has_no_device() -> None:
    """When the policy exposes no device, the driver's CUDA device is read off its trainable
    modules, so an overlap with the rollout GPUs is still caught.
    """
    bundle = _Bundle(
        model=object(),
        trainable_modules={"transformer": torch.nn.Linear(1, 1).to("cuda:0")},
    )

    config = _ray_config(
        _resource_cfg(
            trainer_devices=[1],
            rollout_devices=[0],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Trainer device cuda:0 overlaps rollout devices \[0\]",
    ):
        config.validate_driver_state(driver_bundle=bundle)


def test_ray_backend_allows_driver_cuda_policy_with_explicit_overlap() -> None:
    """A colocated single-GPU topology derives on-demand rollout activation, so a driver CUDA
    policy overlapping the rollout GPU is allowed.
    """
    config = _ray_config(
        _resource_cfg(
            trainer_devices=[0],
            rollout_devices=[0],
        ),
    ).validate_driver_state(
        driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
    )

    assert config.resources.colocated is True
    assert config.resources.lifecycle.rollout_mode == "on_demand"


def test_ray_backend_allows_split_driver_cuda_when_devices_do_not_overlap() -> None:
    """Disjoint trainer and rollout devices are accepted as-is and reported as not colocated."""
    config = _ray_config(
        _resource_cfg(trainer_devices=[0], rollout_devices=[1]),
    ).validate_driver_state(
        driver_bundle=_Bundle(model=_CudaPolicy(), trainable_modules={}),
    )

    assert config.resources.trainer_devices == (0,)
    assert config.resources.rollout_devices == (1,)
    assert config.resources.colocated is False


# ------------------------------------- real batch-size probe fan-out (real Ray)


def _auto_chunk_request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req-probe",
        family="sd3_5",
        task="t2i",
        inputs=["p"],
        samples_per_prompt=10,
        sampling={"num_steps": 20},
        samples_per_generation_batch="auto",
        policy_version=1,
    )


@pytest.mark.asyncio
async def test_remote_batch_size_probe_timeout_is_terminal_and_cancels_refs(
    monkeypatch,
) -> None:
    import vrl.ray.operation_deadline as deadline_module

    class _NeverRef:
        def __await__(self):
            async def wait_forever() -> None:
                await asyncio.Event().wait()

            return wait_forever().__await__()

    ref = _NeverRef()

    class _RemoteProbe:
        @staticmethod
        def remote(_request: Any, *, max_samples: int) -> _NeverRef:
            assert max_samples == 10
            return ref

    engine = RayGenerationEngine(
        "w0",
        [
            RayActorHandle(
                worker_id="w0",
                actor=SimpleNamespace(probe_batch_size=_RemoteProbe()),
            ),
        ],
    )
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [engine],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=0.01,
    )

    async def execute(_request: Any) -> None:
        raise AssertionError("timed-out probe must not enter generation")

    executor.execute = execute

    class _Ray:
        cancelled: ClassVar[list[tuple[Any, bool]]] = []

        @classmethod
        def cancel(cls, value: Any, *, force: bool) -> None:
            cls.cancelled.append((value, force))

    runtime = _runtime(executor)
    monkeypatch.setattr(deadline_module, "require_ray", lambda: _Ray)

    with pytest.raises(
        RayOperationTimeout,
        match=r"rollout\.generation\.batch_size_probe",
    ):
        await runtime.generate(_auto_chunk_request())

    assert _Ray.cancelled == [(ref, False)]
    assert runtime.lifecycle.phase is RuntimePhase.TERMINATED


@pytest.mark.asyncio
async def test_concurrent_auto_chunk_requests_share_one_probe_before_submission() -> None:
    gate = asyncio.Event()
    probe_requests: list[str] = []
    executed_requests: list[GenerationRequest] = []
    probe_result = BatchSizeProbeResult(
        samples_per_generation_batch=3,
        budget_bytes=1,
        trials=(),
    )

    class _ProbeRef:
        def __await__(self):
            async def wait() -> BatchSizeProbeResult:
                await gate.wait()
                return probe_result

            return wait().__await__()

    class _RemoteProbe:
        @staticmethod
        def remote(request: GenerationRequest, *, max_samples: int) -> _ProbeRef:
            assert max_samples == 10
            probe_requests.append(request.request_id)
            return _ProbeRef()

    engine = RayGenerationEngine(
        "w0",
        [
            RayActorHandle(
                worker_id="w0",
                actor=SimpleNamespace(probe_batch_size=_RemoteProbe()),
            ),
        ],
    )
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        [engine],
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(("w0",)),
        generation_stall_timeout_s=30.0,
    )

    async def execute(request: GenerationRequest) -> GenerationRequest:
        executed_requests.append(request)
        return request

    executor.execute = execute
    runtime = _runtime(executor)
    first_request = _auto_chunk_request()
    second_request = replace(first_request, request_id="req-probe-second")

    first = asyncio.create_task(runtime.generate(first_request))
    await asyncio.sleep(0)
    second = asyncio.create_task(runtime.generate(second_request))
    await asyncio.sleep(0)

    assert probe_requests == ["req-probe"]

    gate.set()
    assert await first == executed_requests[0]
    assert await second == executed_requests[1]
    assert probe_requests == ["req-probe"]
    assert [request.samples_per_generation_batch for request in executed_requests] == [3, 3]


class _Arrivals:
    """Counts probe arrivals across actor processes so concurrency is observable."""

    def __init__(self) -> None:
        self._count = 0

    def arrived(self) -> int:
        self._count += 1
        return self._count

    def count(self) -> int:
        return self._count


class _ProbeWorker:
    """Real Ray actor exposing ``probe_batch_size`` as a remote method.

    It blocks until the whole fleet has arrived, so "probed concurrently" becomes
    a fact the test can fail on rather than a word in a name.
    """

    def __init__(self, answer: int, arrivals: Any, fleet_size: int) -> None:
        self._answer = int(answer)
        self._arrivals = arrivals
        self._fleet_size = int(fleet_size)
        self._calls = 0

    def probe_batch_size(self, request: Any, *, max_samples: int) -> BatchSizeProbeResult:
        import time

        import ray

        # Asserted inside the actor process: the request really survived Ray
        # serialization with its type and fields intact. Nothing else checks this.
        assert isinstance(request, GenerationRequest), type(request).__name__
        assert request.samples_per_generation_batch == "auto"
        assert request.inputs[0].prompt == "p"
        assert max_samples == 10
        self._calls += 1

        ray.get(self._arrivals.arrived.remote())
        deadline = time.monotonic() + 20.0
        while ray.get(self._arrivals.count.remote()) < self._fleet_size:
            if time.monotonic() > deadline:
                raise TimeoutError("probes were dispatched sequentially, not concurrently")
            time.sleep(0.01)
        return BatchSizeProbeResult(
            samples_per_generation_batch=self._answer,
            budget_bytes=32 * 1024**3,
            trials=(),
        )

    def calls(self) -> int:
        return self._calls


@pytest.mark.slow_test
def test_real_ray_probe_fan_out_resolves_auto_once_across_the_fleet(local_ray) -> None:
    """The executor-owned batch-size probe path on a live cluster.

    The executor sends N remote probes through its shared actor dispatcher, so
    generation cannot enter the same synchronous actor mailbox concurrently.
    The barrier inside ``_ProbeWorker`` makes fleet fan-out checkable: an
    implementation that waited on each worker inside the submission loop would
    deadlock instead of passing.
    """

    arrivals = local_ray.remote(num_cpus=0)(_Arrivals).remote()
    actor_cls = local_ray.remote(num_cpus=0)(_ProbeWorker)
    actors = [actor_cls.remote(answer, arrivals, 2) for answer in (6, 4)]
    executed: list[Any] = []

    engines = [
        RayGenerationEngine(
            f"w{index}",
            [RayActorHandle(worker_id=f"w{index}", actor=actor)],
        )
        for index, actor in enumerate(actors)
    ]
    executor = RayGenerationExecutor(
        SimpleNamespace(),
        engines,
        SimpleNamespace(),
        actor_dispatcher=RayActorDispatcher(tuple(engine.engine_id for engine in engines)),
        generation_stall_timeout_s=30.0,
    )

    async def execute(request: Any) -> Any:
        executed.append(request)
        return SimpleNamespace(request_id=request.request_id)

    executor.execute = execute
    runtime = _runtime(executor)

    async def go() -> None:
        await runtime.generate(_auto_chunk_request())
        await runtime.generate(_auto_chunk_request())

    try:
        asyncio.run(go())

        # Fleet answer is the min, and it is probed once: the second request is
        # rewritten from the cached verdict, so no actor sees a second probe.
        assert [request.samples_per_generation_batch for request in executed] == [4, 4]
        assert local_ray.get([actor.calls.remote() for actor in actors]) == [1, 1]
    finally:
        for actor in (*actors, arrivals):
            local_ray.kill(actor, no_restart=True)
