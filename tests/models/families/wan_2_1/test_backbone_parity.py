"""Real-inference tests for the Wan T2V/I2V CFG backbone wrapper.

No fake transformer: these run a real (tiny, cache-free) ``WanTransformer3DModel``
through ``forward_step`` and verify the wrapper's classifier-free-guidance
behavior against the model's OWN real outputs — never a hand-re-derived formula.

What is pinned, all from real inference:
- batched CFG issues exactly ONE transformer forward (un/cond concatenated);
- ``noise_pred == uncond + scale * (cond - uncond)`` holds on the real branches,
  so a sign/formula error in the wrapper is caught;
- ``cond != uncond`` — the real model actually responds to the conditioning, which
  a closed-form fake cannot demonstrate;
- I2V threads the condition latent into the channel axis and the image embeds into
  the image cross-attention branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    build_tiny_wan_i2v_transformer,
    build_tiny_wan_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.wan_2_1.model import (
    WanI2VDiffusersModel,
    WanI2VSamplingState,
    WanT2VDiffusersModel,
    WanT2VReplayModel,
    WanT2VSamplingState,
)

_TEXT_LEN = 3
_TEXT_DIM = 16
_LATENT_SHAPE = (2, 4, 1, 4, 4)


def _model(
    cls: type,
    transformer: torch.nn.Module,
    *,
    transformer_2: torch.nn.Module | None = None,
    boundary_ratio: float | None = None,
) -> Any:
    model = cls(
        pipeline=SimpleNamespace(
            transformer=transformer,
            transformer_2=transformer_2,
            config=SimpleNamespace(boundary_ratio=boundary_ratio),
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def test_wan_t2v_forward_step_runs_real_batched_cfg() -> None:
    transformer = build_tiny_wan_transformer()
    calls = record_forward_calls(transformer)
    model = _model(WanT2VDiffusersModel, transformer)
    state = WanT2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        guidance_scale=3.0,
        do_cfg=True,
    )

    out = model.forward_step(state, 0)

    # One batched forward: un/cond concatenated along batch -> 2*B rows.
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == 2 * _LATENT_SHAPE[0]
    # CFG combination verified on the real model's own branch outputs.
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 3.0 * (cond - uncond))
    assert out["noise_pred"].shape == _LATENT_SHAPE
    # The real transformer genuinely responds to the prompt (not a no-op fake).
    assert not torch.allclose(cond, uncond)


def test_wan_t2v_forward_step_casts_replay_tensors_to_transformer_dtype() -> None:
    """Checks bf16 rollout tensors replay cleanly through an fp32 transformer."""
    transformer = build_tiny_wan_transformer()
    calls = record_forward_calls(transformer)
    model = _model(WanT2VDiffusersModel, transformer)
    state = WanT2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE, dtype=torch.bfloat16),
        timesteps=torch.tensor([5.0], dtype=torch.bfloat16),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM, dtype=torch.bfloat16),
        negative_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
    )

    out = model.forward_step(state, 0)

    assert out["noise_pred"].dtype == torch.float32
    assert calls[0]["hidden_states"].dtype == torch.float32
    assert calls[0]["timestep"].dtype == torch.float32
    assert calls[0]["encoder_hidden_states"].dtype == torch.float32


def test_wan_i2v_forward_step_threads_condition_and_image_embeds() -> None:
    """I2V concatenates the condition latent into the channel axis (4 latent + 4 cond) and passes
    the image embeds, all inside one batched CFG forward.
    """
    transformer = build_tiny_wan_i2v_transformer()
    calls = record_forward_calls(transformer)
    model = _model(WanI2VDiffusersModel, transformer)
    state = WanI2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        image_embeds=torch.randn(2, 2, _TEXT_DIM),
        condition=torch.randn(_LATENT_SHAPE),
        guidance_scale=2.0,
        do_cfg=True,
    )

    out = model.forward_step(state, 0)

    assert len(calls) == 1
    call = calls[0]
    # Condition latent is concatenated into the channel axis (4 latent + 4 cond),
    # and batched CFG doubles the batch.
    assert call["hidden_states"].shape == (2 * _LATENT_SHAPE[0], 8, 1, 4, 4)
    assert call["encoder_hidden_states_image"] is not None
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 2.0 * (cond - uncond))
    assert out["noise_pred"].shape == _LATENT_SHAPE
    assert not torch.allclose(cond, uncond)


def test_wan_i2v_dual_stage_routes_by_diffusers_boundary() -> None:
    """Checks Wan 2.2 A14B routes high/low timesteps to the matching transformer."""
    high = build_tiny_wan_i2v_transformer(seed=1)
    low = build_tiny_wan_i2v_transformer(seed=2)
    high_calls = record_forward_calls(high)
    low_calls = record_forward_calls(low)
    model = _model(
        WanI2VDiffusersModel,
        high,
        transformer_2=low,
        boundary_ratio=0.5,
    )
    state = WanI2VSamplingState(
        latents=torch.randn(_LATENT_SHAPE),
        timesteps=torch.tensor([750.0, 250.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=None,
        image_embeds=torch.randn(2, 2, _TEXT_DIM),
        condition=torch.randn(_LATENT_SHAPE),
        guidance_scale=1.0,
        do_cfg=False,
        guidance_scale_2=1.0,
        boundary_ratio=0.5,
        num_train_timesteps=1000,
    )

    high_out = model.forward_step(state, 0)
    low_out = model.forward_step(state, 1)

    assert len(high_calls) == 1
    assert len(low_calls) == 1
    torch.testing.assert_close(high_calls[0]["timestep"], torch.full((2,), 750.0))
    torch.testing.assert_close(low_calls[0]["timestep"], torch.full((2,), 250.0))
    assert set(model.trainable_modules) == {"transformer_2"}
    assert not torch.allclose(high_out["noise_pred"], low_out["noise_pred"])


def test_wan_dual_expert_slot_install_rejects_partial_state_before_mutation() -> None:
    high = torch.nn.Linear(2, 2)
    low = torch.nn.Linear(2, 2)
    model = WanT2VReplayModel(
        transformer=high,
        transformer_2=low,
        scheduler=None,
        boundary_ratio=0.5,
        trainable_transformers="both",
    )
    high.requires_grad_(True)
    low.requires_grad_(True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    partial = {
        f"transformer.{name}": torch.full_like(value, 3.0)
        for name, value in high.state_dict().items()
    }

    with pytest.raises(ValueError, match="empty state dict"):
        model.install_trainable_state(1, partial)

    assert not model.has_trainable_state(1)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_wan_i2v_replay_state_roundtrip_keeps_conditioning_tensors() -> None:
    # Pure state plumbing (no forward): export -> restore must preserve the
    # conditioning tensors and slice the per-step timestep.
    model = _model(WanI2VDiffusersModel, build_tiny_wan_i2v_transformer())
    state = WanI2VSamplingState(
        latents=torch.zeros(_LATENT_SHAPE),
        timesteps=torch.tensor([9.0, 7.0]),
        scheduler=None,
        prompt_embeds=torch.ones(2, _TEXT_LEN, _TEXT_DIM),
        negative_prompt_embeds=torch.zeros(2, _TEXT_LEN, _TEXT_DIM),
        image_embeds=torch.full((2, 2, _TEXT_DIM), 0.5),
        condition=torch.full(_LATENT_SHAPE, 10.0),
        guidance_scale=2.0,
        do_cfg=True,
        guidance_scale_2=3.0,
        boundary_ratio=0.5,
        num_train_timesteps=1000,
    )
    replay = model.export_replay_tensors(state)
    replay["timesteps"] = torch.tensor([[9.0, 7.0], [9.0, 7.0]])

    restored = model.restore_eval_state(
        replay,
        model.export_batch_context(state),
        state.latents,
        1,
    )

    torch.testing.assert_close(restored.condition, state.condition)
    torch.testing.assert_close(restored.image_embeds, state.image_embeds)
    torch.testing.assert_close(restored.timesteps, torch.tensor([[7.0, 7.0]]))
    assert restored.guidance_scale_2 == 3.0
    assert restored.boundary_ratio == 0.5
    assert restored.num_train_timesteps == 1000
