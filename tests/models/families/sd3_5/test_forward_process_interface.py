"""SD3.5's forward-process objective surface (DiffusionNFT / V-GRPO).

Three things the objectives dispatch to on a model must work for SD3.5 without
the real checkpoint: ``diffusion_nft_prepare_transformer_input`` must yield
kwargs the real (tiny) ``SD3Transformer2DModel`` consumes and that match the
conditional branch ``forward_step`` runs, ``latents_clean`` must be exported,
and the frozen ``previous`` adapter must attach and sync through the shared
``LoraModelMixin`` on a real PEFT-wrapped transformer.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_SD3_JOINT_DIM,
    TINY_SD3_LATENT_SHAPE,
    TINY_SD3_POOLED_DIM,
    build_tiny_sd3_transformer,
    stamp_model_precision,
)
from vrl.models.families.sd3_5.model import SD3_5Model, SD3_5ReplayModel, SD3SamplingState

_TEXT_LEN = 3
_LORA_TARGETS = ["to_q", "to_v"]


def _model(transformer: torch.nn.Module) -> SD3_5Model:
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _conditioning() -> tuple[torch.Tensor, torch.Tensor]:
    batch = TINY_SD3_LATENT_SHAPE[0]
    return (
        torch.randn(batch, _TEXT_LEN, TINY_SD3_JOINT_DIM),
        torch.randn(batch, TINY_SD3_POOLED_DIM),
    )


def test_forward_process_input_matches_the_conditional_forward_step_branch() -> None:
    """The objectives' raw kwargs drive the real transformer to the same output
    ``forward_step`` produces for the conditional branch at the same timestep."""
    torch.manual_seed(0)
    model = _model(build_tiny_sd3_transformer())
    latents = torch.randn(TINY_SD3_LATENT_SHAPE)
    prompt_embeds, pooled = _conditioning()
    timestep = torch.full((TINY_SD3_LATENT_SHAPE[0],), 500.0)

    inputs = model.diffusion_nft_prepare_transformer_input(
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=None,
        pooled_prompt_embeds=pooled,
        timestep=timestep,
        num_frames=1,
        height=64,
        width=64,
        guidance_scale=1.0,
    )
    with torch.no_grad():
        direct = model.transformer(**inputs)[0]
        state = SD3SamplingState(
            latents=latents,
            timesteps=torch.tensor([[500.0] * TINY_SD3_LATENT_SHAPE[0]]),
            scheduler=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            guidance_scale=1.0,
            do_cfg=False,
        )
        via_step = model.forward_step(state, 0)["noise_pred"]

    assert set(inputs) == {
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "pooled_projections",
        "return_dict",
    }
    assert inputs["return_dict"] is False
    assert inputs["timestep"].shape == (TINY_SD3_LATENT_SHAPE[0],)
    torch.testing.assert_close(direct, via_step)


def test_forward_process_input_requires_the_pooled_projection() -> None:
    model = _model(build_tiny_sd3_transformer())
    prompt_embeds, _ = _conditioning()
    try:
        model.diffusion_nft_prepare_transformer_input(
            latents=torch.randn(TINY_SD3_LATENT_SHAPE),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=None,
            pooled_prompt_embeds=None,
            timestep=torch.full((TINY_SD3_LATENT_SHAPE[0],), 500.0),
            num_frames=1,
            height=64,
            width=64,
        )
    except ValueError as exc:
        assert "pooled_prompt_embeds" in str(exc)
    else:
        raise AssertionError("missing pooled projection must be refused")


def test_replay_tensors_carry_the_final_latent_for_the_forward_process_objectives() -> None:
    model = _model(build_tiny_sd3_transformer())
    prompt_embeds, pooled = _conditioning()
    latents = torch.randn(TINY_SD3_LATENT_SHAPE)
    state = SD3SamplingState(
        latents=latents,
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        guidance_scale=1.0,
        do_cfg=False,
    )

    exported = model.export_replay_tensors(state)

    assert torch.equal(exported["latents_clean"], latents)
    assert exported["latents_clean"].requires_grad is False


def _peft_default_only_replay_model() -> SD3_5ReplayModel:
    from peft import LoraConfig, get_peft_model

    base = build_tiny_sd3_transformer()
    base.requires_grad_(False)
    peft_t = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, init_lora_weights="gaussian", target_modules=_LORA_TARGETS),
    )
    return SD3_5ReplayModel(transformer=peft_t, scheduler=None, device="cpu")


def test_previous_policy_adapter_attaches_frozen_and_syncs_through_the_shared_mixin() -> None:
    """The replay model (no pipeline) reaches ``LoraModelMixin``'s attach/sync:
    a frozen ``previous`` mirror seeded from ``default`` and refreshed on sync."""
    model = _peft_default_only_replay_model()
    build = SimpleNamespace(lora={"rank": 4, "alpha": 8, "target_modules": _LORA_TARGETS})

    model.attach_previous_policy_adapter(build)

    named = dict(model.transformer.named_parameters())
    previous = {n: p for n, p in named.items() if ".previous." in n}
    assert previous and all(not p.requires_grad for p in previous.values())
    assert any(".default." in n and p.requires_grad for n, p in named.items())
    a_name = next(n for n in previous if "lora_A" in n)
    d_name = a_name.replace(".previous.", ".default.")
    assert torch.allclose(named[a_name], named[d_name])
    with torch.no_grad():
        named[d_name].add_(1.0)
    assert not torch.allclose(named[a_name], named[d_name])

    model.sync_previous_policy_adapter(decay=0.0)

    assert torch.allclose(named[a_name], named[d_name])
    assert model.transformer.active_adapter == "default"
