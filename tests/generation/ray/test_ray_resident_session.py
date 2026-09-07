"""Ray generation worker resident-session tests."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import BatchPayload
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.generation.ray.worker import RayGenerationWorker
from vrl.generation.types import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import (
    ModelBuild,
    ReplayResult,
    RolloutBuildOptions,
    RuntimeBundle,
)
from vrl.trajectory import TrajectoryBatch


class _TinyRuntimeModel:
    device = "cpu"

    def replay_forward(self, batch: Any, timestep_idx: int = 0, **kwargs: Any) -> ReplayResult:
        raise NotImplementedError("Ray worker idempotency test never replays")

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> None:
        self.loaded_state = dict(state_dict)


class _TinyGatherer:
    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        return GenerationOutput(
            output=list(batches),
            trajectory=TrajectoryBatch(
                request_id=request.request_id,
                family=request.family,
                task=request.task,
                sample_rows=list(sample_rows),
                axes={},
                segments={},
            ),
        )


class _TinyChunkExecutor:
    build_count = 0
    family = "janus_pro"
    task = "ar_t2i"

    def __init__(
        self,
        model: _TinyRuntimeModel,
        *,
        gatherer: Any | None = None,
    ) -> None:
        type(self).build_count += 1
        self.model = model
        self.gatherer = gatherer

    def forward_batch(self, *args: Any, **kwargs: Any) -> BatchPayload:
        raise NotImplementedError("Ray worker idempotency test never executes batches")

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        assert self.gatherer is not None
        return self.gatherer.gather_batches(request, sample_rows, batches)


def build_tiny_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    assert str(build.device) == "cpu"
    assert build.parameter_dtype is torch.float16
    assert isinstance(build.rollout, RolloutBuildOptions)
    assert build.precision == RolePrecision("fp16", "tf32", outer_autocast=False)
    assert build.rollout.prompt_encoder_dtype is torch.float32
    return RuntimeBundle(
        model=_TinyRuntimeModel(),
        trainable_modules={},
        scheduler=None,
        raw_handle=None,
        precision=build.precision,
        loads_full_generation_modules=True,
    )


def _build_tiny_rollout(_entry: Any, build: ModelBuild) -> RuntimeBundle:
    return build_tiny_runtime_bundle(build)


def _install_tiny_family(monkeypatch: pytest.MonkeyPatch) -> None:
    import vrl.models.checkpoint_identity as checkpoint_identity
    import vrl.models.families.registry as registry

    entry = registry.FAMILY_REGISTRY["janus_pro"]
    monkeypatch.setitem(
        registry.FAMILY_REGISTRY,
        "janus_pro",
        replace(
            entry,
            executor_cls=("tests.generation.ray.test_ray_resident_session:_TinyChunkExecutor"),
        ),
    )
    monkeypatch.setattr(
        registry.ModelFamilyEntry,
        "build_rollout",
        _build_tiny_rollout,
    )
    monkeypatch.setattr(
        checkpoint_identity,
        "resolve_checkpoint_model_identity",
        lambda _build: {"schema": "test"},
    )


def _launch_contract() -> GenerationRuntimeLaunchContract:
    return GenerationRuntimeLaunchContract(
        family="janus_pro",
        model_build={
            "model_name_or_path": "unit-test",
            "revision": None,
            "device": "cpu",
            "parameter_dtype": "float16",
            "precision": {
                "dtype": "fp16",
                "float32_precision": "tf32",
                "quantization": None,
                "outer_autocast": False,
            },
            "rollout": {
                "prompt_encoder_dtype": "float32",
                "base_weight_sync": False,
            },
        },
        expected_model_identity={"schema": "test"},
        policy_version=1,
    )


def _launch_inputs() -> RayGenerationLaunchInputs:
    return RayGenerationLaunchInputs(
        launch_contract=_launch_contract(),
        gatherer=_TinyGatherer(),
    )


def test_ray_generation_worker_load_policy_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second ``load_policy`` is a no-op: the executor is built once and keeps the launch
    inputs' gatherer.
    """
    _install_tiny_family(monkeypatch)
    _TinyChunkExecutor.build_count = 0
    launch_inputs = _launch_inputs()
    worker = RayGenerationWorker("rollout-0", launch_inputs)

    worker.load_policy()
    first_executor = worker.core.executor
    worker.load_policy()

    assert _TinyChunkExecutor.build_count == 1
    assert worker.core.executor is first_executor
    assert first_executor.gatherer is launch_inputs.gatherer


def test_ray_generation_worker_rebuilds_executor_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The resident-session contract: a released worker tears down its executor and
    # rebuilds a fresh one on the next load_policy (not the cached idempotent path).
    _install_tiny_family(monkeypatch)
    _TinyChunkExecutor.build_count = 0
    worker = RayGenerationWorker("rollout-0", _launch_inputs())

    worker.load_policy()
    first_executor = worker.core.executor

    worker.release_policy()
    assert worker.core.executor is None

    worker.load_policy()

    assert _TinyChunkExecutor.build_count == 2
    assert worker.core.executor is not None
    assert worker.core.executor is not first_executor
