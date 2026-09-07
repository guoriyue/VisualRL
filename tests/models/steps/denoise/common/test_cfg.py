from __future__ import annotations

import pytest
import torch

from vrl.models.steps.denoise.common.cfg import (
    DiffusionBranch,
    combine_cfg,
    pack_batched_cfg,
    split_batched_cfg_output,
)


def test_pack_and_split_batched_cfg_preserves_uncond_cond_order() -> None:
    """Packing puts the uncond rows first and the cond rows second across hidden states,
    timesteps, encoder states and extra kwargs; splitting the output honors the same order.
    """
    cond = DiffusionBranch(
        hidden_states=torch.full((2, 1), 2.0),
        timestep=torch.tensor([10.0, 11.0]),
        encoder_hidden_states=torch.full((2, 3), 20.0),
        extra_kwargs={"pooled_projections": torch.full((2, 4), 200.0)},
    )
    uncond = DiffusionBranch(
        hidden_states=torch.full((2, 1), -2.0),
        timestep=torch.tensor([1.0, 2.0]),
        encoder_hidden_states=torch.full((2, 3), -20.0),
        extra_kwargs={"pooled_projections": torch.full((2, 4), -200.0)},
    )

    packed = pack_batched_cfg(cond=cond, uncond=uncond)

    assert torch.equal(
        packed.hidden_states,
        torch.tensor([[-2.0], [-2.0], [2.0], [2.0]]),
    )
    assert torch.equal(packed.timestep, torch.tensor([1.0, 2.0, 10.0, 11.0]))
    assert packed.encoder_hidden_states.shape == (4, 3)
    assert packed.extra_kwargs["pooled_projections"].shape == (4, 4)

    uncond_out, cond_out = split_batched_cfg_output(torch.arange(8).view(4, 2))
    assert torch.equal(uncond_out, torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(cond_out, torch.tensor([[4, 5], [6, 7]]))


def test_combine_cfg_supports_sd_and_cosmos_bases() -> None:
    """``combine_cfg`` supports both bases, the SD form ``uncond + g*(cond - uncond)`` and the
    cosmos form ``cond + g*(cond - uncond)``, and returns cond untouched when CFG is off.
    """
    cond = torch.tensor([3.0])
    uncond = torch.tensor([1.0])

    assert torch.equal(
        combine_cfg(cond, uncond, guidance_scale=4.0, do_cfg=True),
        torch.tensor([9.0]),
    )
    assert torch.equal(
        combine_cfg(cond, uncond, guidance_scale=4.0, do_cfg=True, base="cond"),
        torch.tensor([11.0]),
    )
    assert torch.equal(
        combine_cfg(cond, None, guidance_scale=1.0, do_cfg=False),
        cond,
    )


def test_split_batched_cfg_requires_even_batch() -> None:
    with pytest.raises(ValueError, match="must be even"):
        split_batched_cfg_output(torch.zeros(3, 1))
