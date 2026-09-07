from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn as nn

from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBranch,
)


class _RecordingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        hidden = kwargs["hidden_states"]
        timestep = kwargs["timestep"]
        while timestep.ndim < hidden.ndim:
            timestep = timestep.unsqueeze(-1)
        return (hidden + timestep, None)


class _Adapter:
    cfg_normalization = False

    def __init__(self, *, cfg_mode: str, cfg_base: str = "uncond") -> None:
        self.cfg_mode = cfg_mode
        self.cfg_base = cfg_base

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds
        assert embeds is not None
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=dict(request.extra),
        )

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        del request, branch
        return raw_output

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        del request, cond, uncond
        return combined


def test_backbone_batched_cfg_calls_transformer_once_and_returns_contract() -> None:
    """Batched CFG is a single transformer call on the doubled batch (hidden states and every
    extra kwarg such as pooled projections become 2B rows) that unpacks back into B-row
    noise_pred / cond / uncond.
    """
    transformer = _RecordingTransformer()
    module = DiffusionBackboneCaller(transformer, _Adapter(cfg_mode="batched_cfg"))

    output = module(
        DiffusionBackboneInput(
            hidden_states=torch.ones(2, 1),
            timestep=torch.tensor([2.0, 3.0]),
            prompt_embeds=torch.ones(2, 4),
            negative_prompt_embeds=torch.zeros(2, 4),
            guidance_scale=2.0,
            do_cfg=True,
            extra={"pooled_projections": torch.ones(2, 3)},
        )
    )

    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape == (4, 1)
    assert transformer.calls[0]["pooled_projections"].shape == (4, 3)
    assert output.noise_pred.shape == (2, 1)
    assert output.noise_pred_cond.shape == (2, 1)
    assert output.noise_pred_uncond.shape == (2, 1)


def test_backbone_separate_cfg_calls_transformer_twice() -> None:
    """Separate CFG issues two transformer calls (cond, uncond) and still returns B-row outputs."""
    transformer = _RecordingTransformer()
    module = DiffusionBackboneCaller(
        transformer,
        _Adapter(cfg_mode="separate_cfg", cfg_base="cond"),
    )

    output = module(
        DiffusionBackboneInput(
            hidden_states=torch.ones(2, 1),
            timestep=torch.tensor([2.0, 3.0]),
            prompt_embeds=torch.ones(2, 4),
            negative_prompt_embeds=torch.zeros(2, 4),
            guidance_scale=2.0,
            do_cfg=True,
        )
    )

    assert len(transformer.calls) == 2
    assert output.noise_pred.shape == (2, 1)


def test_backbone_single_branch_calls_transformer_once_without_cfg() -> None:
    transformer = _RecordingTransformer()
    module = DiffusionBackboneCaller(transformer, _Adapter(cfg_mode="single_branch"))

    output = module(
        DiffusionBackboneInput(
            hidden_states=torch.ones(2, 1),
            timestep=torch.tensor([2.0, 3.0]),
            prompt_embeds=torch.ones(2, 4),
            guidance_scale=2.0,
            do_cfg=False,
        )
    )

    assert len(transformer.calls) == 1
    assert torch.equal(output.noise_pred, output.noise_pred_cond)
    assert torch.count_nonzero(output.noise_pred_uncond) == 0


def test_backbone_single_branch_rejects_cfg() -> None:
    transformer = _RecordingTransformer()
    module = DiffusionBackboneCaller(transformer, _Adapter(cfg_mode="single_branch"))

    with pytest.raises(ValueError, match="single_branch runner cannot run CFG"):
        module(
            DiffusionBackboneInput(
                hidden_states=torch.ones(2, 1),
                timestep=torch.tensor([2.0, 3.0]),
                prompt_embeds=torch.ones(2, 4),
                negative_prompt_embeds=torch.zeros(2, 4),
                guidance_scale=2.0,
                do_cfg=True,
            )
        )
