"""Real ``from_pretrained`` pipeline-assembly tests on tiny HF repos (~1 MB).

The config-init fixtures (``build_tiny_*``) build only the transformer in memory,
so they never exercise the production loading path: ``from_pretrained`` reading
``model_index.json`` and assembling transformer + VAE + text_encoder + scheduler.
These tests do exactly that, using ``hf-internal-testing/tiny-*`` pipelines that
are small enough (~1 MB, random weights) to download per-PR.

Scope: this pins pipeline *assembly / wiring* and that the production model wrapper
accepts a really-loaded pipeline. Weights are random, so it does NOT check numeric
correctness — that stays in the gated real-checkpoint e2e suite. Skips cleanly when
the Hub is unreachable (offline CI).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.steps.denoise.fixtures import stamp_model_precision
from tests.models.steps.denoise.registry import (
    TinyPipelineUnavailable,
    architectures_with_tiny_pipeline,
    load_tiny_pipeline,
)
from vrl.models.families.sd3_5.model import SD3_5Model, SD3SamplingState
from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel, WanT2VSamplingState


@pytest.mark.parametrize("arch", architectures_with_tiny_pipeline())
def test_tiny_pipeline_assembles_real_components(arch: str) -> None:
    """``from_pretrained`` on the tiny repo assembles a real transformer and VAE with parameters,
    for every listed architecture.
    """
    try:
        pipe = load_tiny_pipeline(arch)
    except TinyPipelineUnavailable as exc:
        pytest.skip(str(exc))

    # The real from_pretrained path assembled every component from the repo layout.
    assert pipe.transformer is not None
    assert pipe.vae is not None
    assert sum(p.numel() for p in pipe.transformer.parameters()) > 0


def test_wan_wrapper_runs_on_real_loaded_pipeline() -> None:
    """The Wan wrapper steps a genuinely ``from_pretrained``-loaded tiny pipeline: CFG combine on
    real branches and the latent shape preserved.
    """
    try:
        pipe = load_tiny_pipeline("wan-t2v")
    except TinyPipelineUnavailable as exc:
        pytest.skip(str(exc))

    # The production wrapper must accept a genuinely-loaded pipeline and step it.
    tf = pipe.transformer
    model = WanT2VDiffusersModel(
        pipeline=SimpleNamespace(transformer=tf, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    c, dim = tf.config.in_channels, tf.config.text_dim
    state = WanT2VSamplingState(
        latents=torch.randn(2, c, 1, 4, 4),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, 3, dim),
        negative_prompt_embeds=torch.randn(2, 3, dim),
        guidance_scale=3.0,
        do_cfg=True,
    )

    out = model.forward_step(state, 0)

    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 3.0 * (cond - uncond))
    assert out["noise_pred"].shape == (2, c, 1, 4, 4)


def test_sd3_wrapper_runs_on_real_loaded_pipeline() -> None:
    """The SD3 wrapper steps a genuinely ``from_pretrained``-loaded tiny pipeline with the SD-form
    CFG combine.
    """
    try:
        pipe = load_tiny_pipeline("sd3")
    except TinyPipelineUnavailable as exc:
        pytest.skip(str(exc))

    tf = pipe.transformer
    model = SD3_5Model(
        pipeline=SimpleNamespace(transformer=tf, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    size = tf.config.sample_size
    joint, pooled = tf.config.joint_attention_dim, tf.config.pooled_projection_dim
    state = SD3SamplingState(
        latents=torch.randn(2, tf.config.in_channels, size, size),
        timesteps=torch.tensor([5.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, 3, joint),
        pooled_prompt_embeds=torch.randn(2, pooled),
        negative_prompt_embeds=torch.randn(2, 3, joint),
        negative_pooled_prompt_embeds=torch.randn(2, pooled),
        guidance_scale=3.0,
        do_cfg=True,
    )

    out = model.forward_step(state, 0)

    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], uncond + 3.0 * (cond - uncond))
