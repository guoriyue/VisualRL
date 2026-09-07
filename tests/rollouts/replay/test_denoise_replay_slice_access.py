"""Regression tests for diffusion replay step slicing before device moves."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.models.steps.denoise import DiffusionModelBase
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.batch.ops import move_training_batch_to_device
from vrl.rollouts.evaluators.denoise.sde_logprob import DiffusionSDELogProbEvaluator
from vrl.trajectory import build_diffusion_trajectory

_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


def test_diffusion_replay_slices_timestep_before_device_move(monkeypatch) -> None:
    """The SDE evaluator slices the requested timestep out of deferred replay tensors before any
    device move, so no full-tensor ``.to`` ever happens on the batch.
    """

    batch, sentinels = _batch_with_sentinel_timestep_tensors()
    evaluator = DiffusionSDELogProbEvaluator(_Scheduler())
    model = _ReplayModel()

    assert evaluator.supports_deferred_replay_tensor_move is True
    moved = move_training_batch_to_device(
        batch,
        torch.device("cpu"),
        defer_replay_tensors=True,
    )
    signals = evaluator.evaluate(
        model,
        moved,
        timestep_idx=1,
    )

    assert signals.primary.log_prob.shape == (2,)
    assert all(sentinel.full_to_calls == 0 for sentinel in sentinels)
    assert all(sentinel.slice_count > 0 for sentinel in sentinels)


def _batch_with_sentinel_timestep_tensors() -> tuple[RolloutBatch, list[_NoFullMoveTensor]]:
    observations = _NoFullMoveTensor(torch.ones(2, 2, 3))
    actions = _NoFullMoveTensor(torch.ones(2, 2, 3) * 0.5)
    timesteps = _NoFullMoveTensor(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    old_log_prob = torch.zeros(2, 2)
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0", "p1"],
        samples_per_prompt=1,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=index,
            sample_index=0,
            prompt=f"p{index}",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.ones(2, 2, 3),
        actions=torch.ones(2, 2, 3) * 0.5,
        old_log_prob=old_log_prob,
        timesteps=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        kl=torch.zeros(2, 2),
        replay_tensors={},
        context={"guidance_scale": 1.0, "cfg": False, "model_family": "sd3_5"},
    )
    segment = trajectory.segments["denoise"]
    segment.tensors["observations"].value = observations
    segment.tensors["actions"].value = actions
    segment.tensors["timesteps"].value = timesteps
    return (
        RolloutBatch(
            rewards=torch.ones(2),
            group_ids=torch.arange(2),
            trajectory=trajectory,
        ),
        [observations, actions, timesteps],
    )


class _NoFullMoveTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.full_to_calls = 0
        self.slice_count = 0

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, key: Any) -> _StepTensor:
        self.slice_count += 1
        return _StepTensor(self.tensor[key])

    def to(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        self.full_to_calls += 1
        raise AssertionError("full timestep tensor moved before slicing")


class _StepTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.tensor.to(*args, **kwargs)


class _Scheduler:
    sigmas = torch.tensor([1.0, 0.5, 0.1])

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        return int(timestep.item())


class _ReplayModel(DiffusionModelBase):
    precision = _PRECISION
    family = "test"
    device = torch.device("cpu")

    def encode_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def prepare_sampling(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_step(self, state: Any, step_idx: int) -> dict[str, torch.Tensor]:
        del step_idx
        return {"noise_pred": torch.zeros_like(state.latents)}

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def export_batch_context(self, state: Any) -> dict[str, Any]:
        return {}

    def export_replay_tensors(self, state: Any) -> dict[str, Any]:
        return {}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del batch_context, step_idx
        return type("State", (), {"latents": latents, "timesteps": replay_tensors["timesteps"]})()

    def disable_adapter(self) -> Any:
        return nullcontext()

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: Any | None = None,
    ) -> ReplayResult:
        del request
        replay_tensors, batch_context, latents = self._replay_inputs_for_step(
            batch,
            timestep_idx,
        )
        state = self.restore_eval_state(replay_tensors, batch_context, latents, timestep_idx)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=self.forward_step(state, 0),
                ),
            },
        )
