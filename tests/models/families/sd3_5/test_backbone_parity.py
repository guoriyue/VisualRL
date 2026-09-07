"""Real-inference tests for the SD3.5 CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``SD3Transformer2DModel`` runs
through ``forward_step``; the wrapper's classifier-free-guidance behavior is
verified against the model's OWN real outputs, never a re-derived formula.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_SD3_JOINT_DIM,
    TINY_SD3_LATENT_SHAPE,
    TINY_SD3_POOLED_DIM,
    build_tiny_sd3_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.sd3_5.model import SD3_5Model, SD3SamplingState

_TEXT_LEN = 3


def _model(transformer: torch.nn.Module) -> SD3_5Model:
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def test_sd3_forward_step_runs_real_batched_cfg() -> None:
    """Batched CFG is one forward with cond and uncond concatenated (2B rows), combined as
    ``uncond + g*(cond - uncond)`` on the real branch outputs.
    """
    transformer = build_tiny_sd3_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = SD3SamplingState(
        latents=torch.randn(TINY_SD3_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_SD3_JOINT_DIM),
        pooled_prompt_embeds=torch.randn(2, TINY_SD3_POOLED_DIM),
        negative_prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_SD3_JOINT_DIM),
        negative_pooled_prompt_embeds=torch.randn(2, TINY_SD3_POOLED_DIM),
        guidance_scale=3.0,
        do_cfg=True,
    )

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == 2 * TINY_SD3_LATENT_SHAPE[0]
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 3.0 * (cond - uncond))
    assert out["noise_pred"].shape == TINY_SD3_LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_sd3_forward_step_single_branch_skips_cfg() -> None:
    transformer = build_tiny_sd3_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = SD3SamplingState(
        latents=torch.randn(TINY_SD3_LATENT_SHAPE),
        timesteps=torch.tensor([[5.0, 6.0]]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_SD3_JOINT_DIM),
        pooled_prompt_embeds=torch.randn(2, TINY_SD3_POOLED_DIM),
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
    )

    out = model.forward_step(state, 0)

    # No CFG: a single conditional forward (batch not doubled), noise == cond,
    # and the uncond branch is a zero placeholder.
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == TINY_SD3_LATENT_SHAPE[0]
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    torch.testing.assert_close(
        out["noise_pred_uncond"], torch.zeros_like(out["noise_pred_uncond"])
    )


def test_sd3_forward_step_casts_replay_tensors_to_transformer_dtype() -> None:
    """Checks bf16 rollout tensors replay cleanly through an fp32 transformer."""
    transformer = build_tiny_sd3_transformer()
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = SD3SamplingState(
        latents=torch.randn(TINY_SD3_LATENT_SHAPE, dtype=torch.bfloat16),
        timesteps=torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_SD3_JOINT_DIM, dtype=torch.bfloat16),
        pooled_prompt_embeds=torch.randn(2, TINY_SD3_POOLED_DIM, dtype=torch.bfloat16),
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
    )

    out = model.forward_step(state, 0)

    assert out["noise_pred"].dtype == torch.float32
    assert calls[0]["hidden_states"].dtype == torch.float32
    assert calls[0]["timestep"].dtype == torch.float32
    assert calls[0]["encoder_hidden_states"].dtype == torch.float32
    assert calls[0]["pooled_projections"].dtype == torch.float32
