"""Real-inference test for the Cosmos Predict2.5 CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``CosmosTransformer3DModel`` runs
through ``forward_step``. Predict2.5 issues TWO separate forwards and blends the
ground-truth velocity over conditioned regions via ``cond_mask``; the wrapper's
CFG combination is verified against the model's OWN real branch outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS_LATENT_SHAPE,
    TINY_COSMOS_TEXT_DIM,
    build_tiny_cosmos_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.cosmos.predict2_5.model import (
    CosmosPredict25Model,
    CosmosPredict25SamplingState,
)

_GUIDANCE = 2.0


def _state() -> CosmosPredict25SamplingState:
    b = TINY_COSMOS_LATENT_SHAPE[0]
    # Partial conditioning mask: top rows are clamped to the ground-truth
    # velocity, the rest carry the (prompt-dependent) transformer output, so the
    # cond/uncond branches genuinely differ.
    cond_mask = torch.zeros(b, 1, 1, 4, 4)
    cond_mask[..., :2, :] = 1.0
    return CosmosPredict25SamplingState(
        latents=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        timesteps=torch.tensor([0.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([0.75])),
        prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        negative_prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        guidance_scale=_GUIDANCE,
        do_cfg=True,
        cond_latent=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        cond_mask=cond_mask,
        cond_indicator=torch.zeros(b, 1, 1, 1, 1),
        padding_mask=torch.zeros(1, 1, 4, 4),
        height=4,
        width=4,
        num_frames=1,
        fps=16,
    )


def test_cosmos_predict25_forward_step_runs_real_unbatched_cfg() -> None:
    """Two separate real forwards; predict2.5 keeps predict2's ``cond + g*(cond - uncond)``
    combine but emits it as-is (no EDM sigma conversion), and the prompt still shows through
    the cond_mask blend.
    """
    transformer = build_tiny_cosmos_transformer()
    calls = record_forward_calls(transformer)
    model = CosmosPredict25Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    state = _state()

    out = model.forward_step(state, 0)

    assert len(calls) == 2
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    torch.testing.assert_close(out["noise_pred"], cond + _GUIDANCE * (cond - uncond))
    assert out["noise_pred"].shape == TINY_COSMOS_LATENT_SHAPE
    # Prompt response survives the cond_mask blend in the unconditioned region.
    assert not torch.allclose(cond, uncond)
