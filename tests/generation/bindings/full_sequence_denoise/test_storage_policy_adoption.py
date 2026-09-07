"""Tests for diffusion trajectory storage policy adoption."""

from __future__ import annotations

import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchGatherer,
    DiffusionBatchResult,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.trajectory import TrajectoryResolver, TrajectoryStoragePolicy


def test_diffusion_rollout_batch_builder_applies_storage_policy() -> None:
    """The trajectory storage policy (device, dtype) is applied to the denoise observations and
    actions when the batch is built, while the trajectory object stays the gatherer's, not a
    copy.
    """
    request = GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0"],
        samples_per_prompt=1,
        sampling={"num_steps": 2},
    )
    output = DiffusionBatchGatherer().gather_batches(
        request,
        request.sample_rows(),
        [_chunk()],
    )

    batch = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(
            metadata={},
            trajectory_storage_policy=TrajectoryStoragePolicy(device="cpu", dtype="float16"),
        ),
    ).build(torch.tensor([1.0]))

    resolver = TrajectoryResolver.from_batch(batch)
    observations = resolver.role_value("denoise", "observation")
    actions = resolver.role_value("denoise", "action")
    assert observations.device.type == "cpu"
    assert observations.dtype == torch.float16
    assert actions.dtype == torch.float16
    assert batch.trajectory is output.trajectory


def _chunk() -> DiffusionBatchResult:
    return DiffusionBatchResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1),
        observations=torch.ones(1, 2, 3, dtype=torch.float32),
        actions=torch.ones(1, 2, 3, dtype=torch.float32) * 2,
        log_probs=torch.ones(1, 2, dtype=torch.float32) * 3,
        timesteps=torch.arange(2, dtype=torch.float32).view(1, 2),
        kl=torch.ones(1, 2, dtype=torch.float32) * 4,
        video=torch.ones(1, 3, 4, 4, dtype=torch.float32),
        replay_tensors={},
        context={"guidance_scale": 4.5, "cfg": False, "model_family": "sd3_5"},
    )
