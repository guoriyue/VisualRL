"""Real-object tests for the Cosmos3 Omni family wrapper.

Everything is a genuine diffusers 0.39 object built from config on CPU
(``build_tiny_cosmos3_pipeline``): the pipeline's own ``prepare_latents`` /
segment builders assemble ``packed_static``, the real ``Cosmos3OmniTransformer``
runs the per-step call with the eleven kwargs ``forward_step`` passes, and the
real ``AutoencoderKLWan`` decodes. What this cannot say is anything about the
numbers a trained checkpoint would produce; that needs an e2e case, see the
audit in ``docs/sprints/done/SPRINT_tiny-real-diffusers-fixtures_audit.md`` §9.
"""

from __future__ import annotations

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS3_LATENT_CHANNELS,
    TINY_COSMOS3_LATENT_PATCH_SIZE,
    build_tiny_cosmos3_pipeline,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import DenoiseRequest
from vrl.models.families.cosmos.cosmos3.model import (
    Cosmos3Model,
    Cosmos3ReplayModel,
    Cosmos3SamplingState,
)

_GUIDANCE = 2.0
_NUM_STEPS = 3
_FRAMES, _HEIGHT, _WIDTH = 5, 32, 32

# The kwargs forward_step passes the transformer, minus the two spliced per step.
_PACKED_STATIC_TRANSFORMER_KWARGS = frozenset(
    {
        "input_ids",
        "text_indexes",
        "position_ids",
        "und_len",
        "sequence_length",
        "vision_token_shapes",
        "vision_sequence_indexes",
        "vision_mse_loss_indexes",
        "vision_noisy_frame_indexes",
    }
)


def _model(**vae_stats: float) -> Cosmos3Model:
    model = Cosmos3Model(
        pipeline=build_tiny_cosmos3_pipeline(**vae_stats), device=torch.device("cpu")
    )
    stamp_model_precision(model)
    return model


def _sampling_state(
    model: Cosmos3Model, *, guidance_scale: float = _GUIDANCE
) -> Cosmos3SamplingState:
    encoded = model.encode_prompt(
        "a cat video",
        "the",
        num_frames=_FRAMES,
        height=_HEIGHT,
        width=_WIDTH,
        fps=24,
        guidance_scale=guidance_scale,
    )
    request = DenoiseRequest(
        width=_WIDTH,
        height=_HEIGHT,
        frame_count=_FRAMES,
        num_steps=_NUM_STEPS,
        guidance_scale=guidance_scale,
        seed=0,
        fps=24,
    )
    return model.prepare_sampling(request, encoded)


def test_prepare_sampling_packs_one_sample_from_the_real_pipeline_builders() -> None:
    """One sample, ``num_steps`` timesteps, and a packed_static the transformer can consume.

    T2V conditions no frame, so every latent frame is noisy: the noisy-token
    count is the whole vision grid, derived from the VAE latent and the
    transformer's patch size rather than declared.
    """
    model = _model()

    state = _sampling_state(model)

    assert state.latents.shape[0] == 1
    assert state.latents.shape[1] == TINY_COSMOS3_LATENT_CHANNELS
    assert state.timesteps.numel() == _NUM_STEPS
    assert state.do_cfg is True
    _, _, latent_t, latent_h, latent_w = state.latents.shape
    patch = TINY_COSMOS3_LATENT_PATCH_SIZE
    assert state.num_noisy_vision_tokens == latent_t * (latent_h // patch) * (latent_w // patch)
    for packed in (state.cond_packed_static, state.uncond_packed_static):
        assert set(packed) >= _PACKED_STATIC_TRANSFORMER_KWARGS
        assert packed["sequence_length"] == packed["und_len"] + packed["num_vision_tokens"]
    # Cond and uncond differ only in their text: the vision segment is shared.
    assert (
        state.cond_packed_static["num_vision_tokens"]
        == state.uncond_packed_static["num_vision_tokens"]
    )
    assert state.cond_input_ids != state.uncond_input_ids


def test_forward_step_returns_raw_velocity_and_the_cosmos3_cfg_combine() -> None:
    """Two real forwards, combined as ``uncond + g * (cond - uncond)``, kept in the flow domain.

    The raw velocity (no ``/ sigma``) is the contract ``sde_step_with_logprob``
    relies on for UniPC's ``[0, 1]`` sigmas; predict2's EDM combine would be
    wrong here.
    """
    model = _model()
    calls = record_forward_calls(model.transformer)
    state = _sampling_state(model)

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    assert set(calls[0]) == _PACKED_STATIC_TRANSFORMER_KWARGS | {
        "vision_tokens",
        "vision_timesteps",
    }
    cond, uncond = out["noise_pred_cond"], out["noise_pred_uncond"]
    assert cond.shape == uncond.shape == out["noise_pred"].shape == state.latents.shape
    assert out["noise_pred"].dtype is torch.float32
    torch.testing.assert_close(out["noise_pred"], (uncond + _GUIDANCE * (cond - uncond)).float())
    assert not torch.allclose(cond, uncond)
    torch.testing.assert_close(
        calls[0]["vision_timesteps"],
        torch.full((state.num_noisy_vision_tokens,), float(state.timesteps[0])),
    )


def test_forward_step_without_cfg_runs_one_forward_and_reports_a_zero_uncond() -> None:
    model = _model()
    calls = record_forward_calls(model.transformer)
    state = _sampling_state(model, guidance_scale=1.0)

    out = model.forward_step(state, 0)

    assert state.do_cfg is False
    assert len(calls) == 1
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"].float())
    assert torch.count_nonzero(out["noise_pred_uncond"]) == 0


def test_replay_model_sets_its_own_scheduler_through_the_shared_set_num_steps() -> None:
    """The replay model holds the scheduler itself; the inherited ``set_num_steps`` reaches
    it through the ``scheduler`` property (UniPC is static, so the set is eager)."""
    from diffusers import UniPCMultistepScheduler

    pipe = build_tiny_cosmos3_pipeline()
    replay = Cosmos3ReplayModel(
        pipeline_shell=pipe,
        scheduler=UniPCMultistepScheduler(),
        device=torch.device("cpu"),
    )

    replay.set_num_steps(3)

    assert replay.scheduler is not pipe.scheduler
    assert replay.scheduler.timesteps.numel() == 3


def test_decode_latents_denormalizes_with_the_real_vae_stats() -> None:
    """``decode_latents`` undoes the pipeline's latent normalization before the VAE.

    Non-identity stats make the order observable: ``z / inv_std + mean`` decoded
    by the real VAE, then the pipeline's own ``video_processor`` postprocess.
    """
    model = _model(latents_mean=0.5, latents_std=2.0)
    pipe = model.pipeline
    latents = _sampling_state(model).latents

    video = model.decode_latents(latents)

    with torch.no_grad():
        raw = latents / pipe._vae_latents_inv_std.view(
            1, -1, 1, 1, 1
        ) + pipe._vae_latents_mean.view(1, -1, 1, 1, 1)
        expected = pipe.video_processor.postprocess_video(
            pipe.vae.decode(raw).sample, output_type="pt"
        )[0]
    assert video.shape[-2:] == (_HEIGHT, _WIDTH)
    torch.testing.assert_close(video, expected)
    # The stats really enter: identity stats decode to something else.
    plain = pipe.video_processor.postprocess_video(
        pipe.vae.decode(latents).sample, output_type="pt"
    )[0]
    assert not torch.allclose(video, plain)
