"""Real-inference test for the Cosmos Predict2 CFG backbone wrapper.

No fake transformer: a real (tiny, cache-free) ``CosmosTransformer3DModel`` runs
through ``forward_step``. Predict2 issues TWO separate forwards (cond + uncond,
not batched) and applies sigma preconditioning; the wrapper's CFG combination is
verified against the model's OWN real branch outputs.
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
from vrl.models.families.cosmos.predict2.model import (
    CosmosPredict2Model,
    CosmosPredict2SamplingState,
)

_SIGMA = 3.0
_GUIDANCE = 2.0


def _state() -> CosmosPredict2SamplingState:
    b = TINY_COSMOS_LATENT_SHAPE[0]
    mask = torch.ones(1, 1, 1, 4, 4)
    no_cond = torch.zeros(1, 1, 1, 1, 1)
    return CosmosPredict2SamplingState(
        latents=torch.randn(TINY_COSMOS_LATENT_SHAPE),
        timesteps=torch.tensor([0.0]),
        scheduler=SimpleNamespace(sigmas=torch.tensor([_SIGMA])),
        prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        negative_prompt_embeds=torch.randn(b, 3, TINY_COSMOS_TEXT_DIM),
        guidance_scale=_GUIDANCE,
        do_cfg=True,
        init_latents=torch.zeros(1, TINY_COSMOS_LATENT_SHAPE[1], 1, 4, 4),
        cond_mask=mask,
        uncond_mask=mask,
        padding_mask=torch.zeros(1, 1, 4, 4),
        cond_indicator=no_cond,
        uncond_indicator=no_cond,
        fps=16,
    )


def test_cosmos_predict2_forward_step_runs_real_unbatched_cfg() -> None:
    """Two separate real forwards (not a batched CFG), combined as ``cond + g*(cond - uncond)``
    and then turned into an EDM noise estimate ``(latents - combined) / sigma``, unlike the
    flow-velocity families.
    """
    transformer = build_tiny_cosmos_transformer()
    calls = record_forward_calls(transformer)
    model = CosmosPredict2Model(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    state = _state()

    out = model.forward_step(state, 0)

    # Predict2 runs cond and uncond as two separate forwards (not batched).
    assert len(calls) == 2
    uncond, cond = out["noise_pred_uncond"], out["noise_pred_cond"]
    # CFG combination + sigma->noise conversion verified on the real branches.
    combined = cond + _GUIDANCE * (cond - uncond)
    torch.testing.assert_close(out["noise_pred"], (state.latents - combined) / _SIGMA)
    assert out["noise_pred"].shape == TINY_COSMOS_LATENT_SHAPE
    # The real transformer responds to the prompt (cond vs uncond differ).
    assert not torch.allclose(cond, uncond)
