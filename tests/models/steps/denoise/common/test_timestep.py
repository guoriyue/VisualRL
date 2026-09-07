from __future__ import annotations

import torch

from vrl.models.steps.denoise.common import (
    broadcast_spatial_timestep,
    expand_batch_timestep,
    pack_eval_timestep,
)


def test_expand_batch_timestep_handles_rollout_scalar_and_eval_vector() -> None:
    """A rollout scalar timestep expands to a [B] vector on the scalar's dtype and device; an
    eval-time vector passes through as the same object.
    """
    scalar = torch.tensor(7.0, dtype=torch.float32)
    expanded = expand_batch_timestep(scalar, 3)
    assert expanded.shape == (3,)
    assert expanded.dtype == scalar.dtype
    assert expanded.device == scalar.device
    assert torch.equal(expanded, torch.tensor([7.0, 7.0, 7.0]))

    vector = torch.tensor([1.0, 2.0, 3.0])
    assert expand_batch_timestep(vector, 3) is vector


def test_pack_eval_timestep_matches_forward_step_index_zero_convention() -> None:
    """``pack_eval_timestep`` selects one column of a [B,T] timestep table as a [1,B] row, the
    forward-step index-0 convention the eval path relies on.
    """
    timesteps = torch.tensor([[10, 11], [20, 21], [30, 31]])

    packed = pack_eval_timestep(timesteps, 1)

    assert packed.shape == (1, 3)
    assert torch.equal(packed[0], torch.tensor([11, 21, 31]))


def test_broadcast_spatial_timestep_returns_cosmos_shape() -> None:
    """Cosmos takes a per-frame timestep of shape [B,1,T,1,1] broadcast from one scalar."""
    timestep = torch.tensor(0.25)

    spatial = broadcast_spatial_timestep(timestep, batch_size=2, frames=4)

    assert spatial.shape == (2, 1, 4, 1, 1)
    assert torch.equal(spatial[:, :, :, 0, 0], torch.full((2, 1, 4), 0.25))
