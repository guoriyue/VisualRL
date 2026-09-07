"""Real-object tests for the MiniMax-H3 family wrapper.

Everything is a genuine diffusers 0.40 / transformers object built from config
on CPU (``build_tiny_minimax_h3_components``): the diffusers layout and
row-timestep statics build the packed sequence, the real
``MiniMaxH3Transformer3DModel`` runs the joint forward, the real
``MiniMaxH3Scheduler`` pair supplies both schedules, the real Qwen3-VL stack
conditions, and both real VAEs decode. What this cannot say is anything about
the numbers a trained checkpoint produces: the released weights need a
multi-GPU deployment, see the family preset.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    TINY_MINIMAX_H3_LATENT_CHANNELS,
    TINY_MINIMAX_H3_PATCH_SIZE,
    TINY_MINIMAX_H3_TEXT_DIM,
    build_tiny_minimax_h3_components,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import DenoiseRequest
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.models.families.minimax_h3.model import (
    MiniMaxH3Model,
    MiniMaxH3ReplayModel,
    MiniMaxH3SamplingState,
    audio_euler_step,
    build_flow_scheduler_class,
    patchify_video_latents,
    unpatchify_video_rows,
)

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

_NUM_STEPS = 3
_FRAMES, _HEIGHT, _WIDTH = 8, 16, 16
_MAX_TEXT_TOKENS = 8
_LAYOUT_KWARGS = frozenset(
    {"token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"}
)


def _model(**vae_stats: float) -> MiniMaxH3Model:
    model = MiniMaxH3Model(
        pipeline=build_tiny_minimax_h3_components(**vae_stats), device=torch.device("cpu")
    )
    stamp_model_precision(model)
    return model


def _request(**overrides: Any) -> DenoiseRequest:
    fields: dict[str, Any] = {
        "width": _WIDTH,
        "height": _HEIGHT,
        "frame_count": _FRAMES,
        "num_steps": _NUM_STEPS,
        "guidance_scale": 1.0,
        "seed": 0,
        "fps": 24,
    }
    fields.update(overrides)
    return DenoiseRequest(**fields)


def _sampling_state(model: MiniMaxH3Model, prompt: str = "a cat video") -> MiniMaxH3SamplingState:
    encoded = model.encode_prompt(
        prompt, None, guidance_scale=1.0, max_sequence_length=_MAX_TEXT_TOKENS
    )
    return model.prepare_sampling(_request(), encoded)


def _rollout(
    model: MiniMaxH3Model,
) -> tuple[MiniMaxH3SamplingState, list[torch.Tensor], list[torch.Tensor]]:
    """Drive the shared SDE step over the state the way the denoise loop does."""

    state = _sampling_state(model)
    observations, noise_preds = [], []
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for step_idx in range(_NUM_STEPS):
            observations.append(state.latents.clone())
            out = model.forward_step(state, step_idx)
            noise_preds.append(out["noise_pred"].clone())
            result = sde_step_with_logprob(
                state.scheduler,
                out["noise_pred"],
                state.timesteps[step_idx].unsqueeze(0),
                state.latents,
                generator=generator,
                step_index=step_idx,
            )
            state.latents = result.prev_sample
    return state, observations, noise_preds


def test_patchify_matches_the_diffusers_row_order_and_round_trips() -> None:
    """Same rows as the reference helper for batch one; the inverse restores the latent."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        patchify_video_latents as reference_patchify,
    )

    latents = torch.randn(2, TINY_MINIMAX_H3_LATENT_CHANNELS, 5, 4, 4)

    rows = patchify_video_latents(latents, TINY_MINIMAX_H3_PATCH_SIZE)

    torch.testing.assert_close(
        rows[:1].reshape(-1, rows.shape[-1]),
        reference_patchify(latents[:1], TINY_MINIMAX_H3_PATCH_SIZE),
    )
    restored = unpatchify_video_rows(
        rows,
        channels=TINY_MINIMAX_H3_LATENT_CHANNELS,
        frames=5,
        height=4,
        width=4,
        patch_size=TINY_MINIMAX_H3_PATCH_SIZE,
    )
    torch.testing.assert_close(restored, latents)


def test_encode_prompt_reads_the_configured_qwen3vl_hidden_state_without_special_tokens() -> None:
    """``[1, tokens, 5120-analogue]`` from ``hidden_states[layer]``; the prompt is bare
    (no chat template, no BOS/EOS) and longer prompts are cut to ``max_sequence_length``."""
    model = _model()
    components = model.pipeline

    encoded = model.encode_prompt("a cat video", None, max_sequence_length=_MAX_TEXT_TOKENS)

    ids = components.tokenizer("a cat video", add_special_tokens=False)["input_ids"]
    assert encoded["prompt_embeds"].shape == (1, len(ids), TINY_MINIMAX_H3_TEXT_DIM)
    assert encoded["max_text_tokens"] == _MAX_TEXT_TOKENS
    with torch.no_grad():
        reference = components.text_encoder.model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            mm_token_type_ids=torch.zeros(1, len(ids), dtype=torch.long),
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states[components.text_encoder_layer]
    torch.testing.assert_close(encoded["prompt_embeds"], reference)
    truncated = model.encode_prompt(
        "a cat video the dog runs on grass", None, max_sequence_length=2
    )
    assert truncated["prompt_embeds"].shape[1] == 2


def test_prepare_sampling_builds_the_reference_layout_and_one_extra_sigma() -> None:
    """``num_steps`` model evaluations need ``num_steps + 1`` sigmas (terminal 0
    included) on both schedules; rows are ``[text | audio | video]`` in the
    reference order with audio channel-major; the row plan's first step holds
    one distinct timestep because video and audio both start at ``t = 0``."""
    model = _model()

    state = _sampling_state(model)

    assert state.latents.shape == (1, TINY_MINIMAX_H3_LATENT_CHANNELS, 5, 4, 4)
    assert state.timesteps.numel() == _NUM_STEPS
    assert state.scheduler.sigmas.numel() == _NUM_STEPS + 1
    assert float(state.scheduler.sigmas[-1]) == 0.0
    assert state.audio_scheduler.timesteps.numel() == _NUM_STEPS
    assert float(state.scheduler.sigmas.max()) <= 1.0  # rectified-flow domain for the SDE
    layout = state.layout
    num_audio_latents = round(_FRAMES / 24 * 40)
    assert layout.num_audio_latents == num_audio_latents
    assert state.audio_rows.shape == (
        1,
        2 * num_audio_latents,
        TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    )
    rows_per_frame = (4 // 2) * (4 // 2)
    assert layout.video_indices.numel() == 5 * rows_per_frame
    assert layout.text_indices.tolist() == list(range(layout.num_text_tokens))
    assert int(layout.audio_indices[0]) == layout.num_text_tokens
    assert int(layout.video_indices[0]) == layout.num_text_tokens + 2 * num_audio_latents
    assert layout.position_ids.shape == (
        layout.num_text_tokens + 2 * num_audio_latents + 5 * rows_per_frame,
        3,
    )
    unique_timesteps, timestep_indices = layout.row_timestep_plan[0]
    assert unique_timesteps.tolist() == [0.0]
    assert timestep_indices.shape == (layout.position_ids.shape[0],)
    last_timesteps, _ = layout.row_timestep_plan[-1]
    assert last_timesteps.numel() == 2  # video (shift 12) and audio (shift 3) diverge


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"guidance_scale": 2.0}, "guidance-distilled"),
        ({"fps": 16}, "24 fps"),
        ({"frame_count": 9}, "next valid count is 13"),
        ({"height": 12}, "multiples of 8"),
    ],
)
def test_prepare_sampling_refuses_geometry_the_checkpoint_cannot_serve(overrides, message) -> None:
    """Off-grid frame counts, off-multiple canvases, a second fps and any guidance
    are refused up front instead of being rounded or silently ignored."""
    model = _model()
    encoded = model.encode_prompt("a cat video", None, max_sequence_length=_MAX_TEXT_TOKENS)

    with pytest.raises(ValueError, match=message):
        model.prepare_sampling(_request(**overrides), encoded)


def test_forward_step_runs_one_joint_forward_and_reports_the_negated_video_velocity() -> None:
    """One transformer call per step with the layout kwargs; ``noise_pred`` is the
    unpatchified video head output negated (data-ward -> ``noise - x0``)."""
    model = _model()
    calls = record_forward_calls(model.transformer)
    state = _sampling_state(model)

    with torch.no_grad():
        out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert set(calls[0]) >= _LAYOUT_KWARGS | {
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "timestep",
        "timestep_indices",
    }
    assert out["noise_pred"].shape == state.latents.shape
    assert out["noise_pred"].dtype is torch.float32
    with torch.no_grad():
        video_rows, _audio = model.transformer(
            hidden_states=patchify_video_latents(state.latents, TINY_MINIMAX_H3_PATCH_SIZE),
            audio_hidden_states=state.audio_rows,
            encoder_hidden_states=state.prompt_embeds,
            timestep=state.layout.row_timestep_plan[0][0],
            timestep_indices=state.layout.row_timestep_plan[0][1],
            return_dict=False,
            **state.layout.transformer_kwargs(),
        )
    expected = -unpatchify_video_rows(
        video_rows.float(),
        channels=TINY_MINIMAX_H3_LATENT_CHANNELS,
        frames=5,
        height=4,
        width=4,
        patch_size=TINY_MINIMAX_H3_PATCH_SIZE,
    )
    torch.testing.assert_close(out["noise_pred"], expected)


def test_audio_side_stream_advances_one_step_behind_the_loop_and_is_reentrant() -> None:
    """Step ``i`` consumes the audio rows prepared by step ``i - 1``; a repeated
    call at the same step (the frozen reference forward) sees the same rows;
    a skipped step is refused."""
    model = _model()
    state = _sampling_state(model)
    initial_audio = state.audio_rows.clone()

    with torch.no_grad():
        first = model.forward_step(state, 0)
        again = model.forward_step(state, 0)
        assert torch.equal(first["noise_pred"], again["noise_pred"])
        assert torch.equal(state.audio_rows, initial_audio)
        assert state.audio_step == 0
        model.forward_step(state, 1)
    assert state.audio_step == 1
    assert not torch.equal(state.audio_rows, initial_audio)
    expected_next = audio_euler_step(
        initial_audio,
        first["audio_velocity"],
        timestep=state.audio_scheduler.timesteps[0],
        sigma=state.audio_scheduler.sigmas[0],
        sigma_next=state.audio_scheduler.sigmas[1],
    )
    torch.testing.assert_close(state.audio_rows, expected_next)
    assert len(state.audio_rows_by_step) == 2
    with pytest.raises(RuntimeError, match="out of step"):
        model.forward_step(state, 3)


def test_audio_euler_step_matches_the_reference_scheduler_update() -> None:
    """The stateless audio update is byte-equal to ``MiniMaxH3Scheduler.step``."""
    from diffusers import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift=3.0)
    scheduler.set_timesteps(_NUM_STEPS + 1)
    rows = torch.randn(1, 26, TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS)
    velocity = torch.randn_like(rows)

    ours = audio_euler_step(
        rows,
        velocity,
        timestep=scheduler.timesteps[1],
        sigma=scheduler.sigmas[1],
        sigma_next=scheduler.sigmas[2],
    )

    reference = scheduler.step(velocity, scheduler.timesteps[1], rows, return_dict=False)[0]
    assert torch.equal(ours, reference)


def test_flow_scheduler_step_reproduces_the_reference_sampler_from_the_negated_velocity() -> None:
    """``denoise_mode: native`` hands ``-v`` to ``scheduler.step``; the subclass
    undoes the sign so the update equals the checkpoint's own Euler step."""
    from diffusers import MiniMaxH3Scheduler

    reference = MiniMaxH3Scheduler(shift=12.0)
    flow = build_flow_scheduler_class()(shift=12.0)
    for scheduler in (reference, flow):
        scheduler.set_timesteps(_NUM_STEPS + 1)
    sample = torch.randn(1, TINY_MINIMAX_H3_LATENT_CHANNELS, 5, 4, 4)
    velocity = torch.randn_like(sample)

    ours = flow.step(-velocity, flow.timesteps[1], sample, return_dict=False)[0]

    expected = reference.step(velocity, reference.timesteps[1], sample, return_dict=False)[0]
    assert torch.equal(ours, expected)
    assert flow.config.shift == 12.0


def test_replay_restores_every_step_bit_exactly_from_the_exported_tensors() -> None:
    """The transformer-only replay model rebuilds the layout from the batch
    context and the audio rows from ``audio_rows_by_step``, so the replayed
    ``noise_pred`` equals the rollout's at every step, from any restore order."""
    from diffusers import MiniMaxH3Scheduler

    model = _model()
    state, observations, noise_preds = _rollout(model)
    context = model.export_batch_context(state)
    replay_tensors = model.export_replay_tensors(state)
    replay = MiniMaxH3ReplayModel(
        transformer=model.transformer,
        scheduler=build_flow_scheduler_class()(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        device=torch.device("cpu"),
    )
    stamp_model_precision(replay)

    with torch.no_grad():
        replayed = {
            step_idx: replay.forward_step(
                replay.restore_eval_state(
                    replay_tensors, context, observations[step_idx], step_idx
                ),
                step_idx,
            )["noise_pred"]
            for step_idx in (2, 0, 1)
        }

    for step_idx, expected in enumerate(noise_preds):
        assert torch.equal(replayed[step_idx], expected), step_idx
    assert replay_tensors["prompt_embeds"].shape == (1, _MAX_TEXT_TOKENS, TINY_MINIMAX_H3_TEXT_DIM)
    assert replay_tensors["num_text_tokens"].tolist() == [state.layout.num_text_tokens]
    assert replay_tensors["audio_rows_by_step"].shape[:2] == (1, _NUM_STEPS)
    assert context["vae_geometry"] == [5, 3, 4]


def test_replay_refuses_a_micro_batch_that_mixes_prompt_lengths() -> None:
    """One packed layout per forward: rows of different text length cannot share it."""
    from diffusers import MiniMaxH3Scheduler

    model = _model()
    state, observations, _ = _rollout(model)
    context = model.export_batch_context(state)
    replay_tensors = model.export_replay_tensors(state)
    replay = MiniMaxH3ReplayModel(
        transformer=model.transformer,
        scheduler=build_flow_scheduler_class()(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        device=torch.device("cpu"),
    )
    mixed = {key: torch.cat([value, value], dim=0) for key, value in replay_tensors.items()}
    mixed["num_text_tokens"] = torch.tensor([3, 2])

    with pytest.raises(ValueError, match="share one prompt length"):
        replay.restore_eval_state(mixed, context, torch.cat([observations[0]] * 2), 0)


def test_decode_latents_undoes_both_normalizations_through_the_real_vae() -> None:
    """Latent stats then ImageNet pixel stats, clamped, batch-first ``[B, C, T, H, W]``."""
    model = _model(latents_mean=0.5, latents_std=2.0)
    vae = model.pipeline.vae
    latents = _sampling_state(model).latents

    video = model.decode_latents(latents)

    assert video.shape == (1, 3, _FRAMES, _HEIGHT, _WIDTH)
    assert float(video.min()) >= 0.0 and float(video.max()) <= 1.0
    with torch.no_grad():
        raw = vae.decode(latents * 2.0 + 0.5, return_dict=False)[0]
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(video, (raw.float() * std + mean).clamp(0, 1))


def test_decode_audio_unpacks_channel_major_rows_into_a_stereo_waveform() -> None:
    """The final audio rows (after the last loop step) decode to ``[2, samples]``
    at the audio VAE's rate; before the last step there is nothing final."""
    model = _model()
    state, _, _ = _rollout(model)

    waveform, sample_rate = model.decode_audio(model.final_audio_rows(state))

    hop = model.pipeline.audio_vae.hop_length
    assert waveform.shape == (2, state.layout.num_audio_latents * hop)
    assert sample_rate == 100
    fresh = _sampling_state(model)
    with pytest.raises(RuntimeError, match="last denoise step"):
        model.final_audio_rows(fresh)
