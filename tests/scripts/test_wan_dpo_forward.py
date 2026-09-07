"""The Wan DPO forward adapter must call the backbone, not the policy wrapper.

``wan_forward`` is the one concrete ``ForwardFn`` this recipe hands to the
family-neutral ``OfflineDPOTrainer``, so it lives beside ``_build_encoders`` in
the recipe module. The policy object it receives is the Wan model wrapper, whose
own ``forward`` is the per-step denoise API — calling it instead of the
registered ``transformer`` would silently train against the wrong signature.
"""

from __future__ import annotations

import torch

from tests.models.steps.denoise.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
    build_tiny_wan_transformer,
    record_forward_calls,
)
from vrl.scripts.families.wan_2_1.train_dpo import wan_forward


class _PolicyWrapperGuard(torch.nn.Module):
    """A policy wrapper owning ``.transformer``; its own forward is a red line.

    ``wan_forward`` must unwrap to the backbone, never call the wrapper itself —
    this is a test instrument (a guard), NOT a stand-in for the model: the real
    backbone is the shared tiny ``WanTransformer3DModel`` it wraps.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(self, *args, **kwargs):  # pragma: no cover - red-line guard
        raise AssertionError("wan_forward must call the registered transformer")


def _wan_backbone_inputs(batch: int = 2):
    """Realistic Wan backbone inputs (5D latents + text embeds) for ``wan_forward``."""

    channels, frames, height, width = TINY_WAN_LATENT_SHAPE[1:]
    noisy = torch.randn(batch, channels, frames, height, width)
    timesteps = torch.full((batch,), 5.0)
    encoder_hidden_states = torch.randn(batch, TINY_WAN_TEXT_LEN, TINY_WAN_TEXT_DIM)
    return noisy, timesteps, encoder_hidden_states


def _real_backbone_output(transformer, noisy, timesteps, encoder_hidden_states):
    """The genuine transformer's own output for the exact call ``wan_forward`` makes."""

    with torch.no_grad():
        return transformer(
            hidden_states=noisy,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def test_wan_forward_unwraps_model_transformer() -> None:
    transformer = build_tiny_wan_transformer().eval()
    noisy, timesteps, encoder = _wan_backbone_inputs()
    # Pin against the real backbone's own output, computed BEFORE the recorder is
    # attached so the recorded count reflects only wan_forward's own invocation.
    expected = _real_backbone_output(transformer, noisy, timesteps, encoder)
    calls = record_forward_calls(transformer)

    out = wan_forward(_PolicyWrapperGuard(transformer), noisy, timesteps, encoder)

    # Unwrapped to the real backbone and returned ITS output (tuple[0]).
    torch.testing.assert_close(out, expected)
    # Exactly one backbone forward, carrying the kwargs the REAL Wan signature
    # consumes. A rename/addition in that signature breaks this here — which the
    # old hand-written stub would have silently absorbed.
    assert len(calls) == 1
    assert {"hidden_states", "timestep", "encoder_hidden_states"} <= calls[0].keys()
    assert calls[0]["return_dict"] is False


def test_wan_forward_still_accepts_raw_transformer() -> None:
    """``wan_forward`` also accepts a bare transformer (no policy wrapper) and calls it exactly
    once.
    """
    transformer = build_tiny_wan_transformer().eval()
    noisy, timesteps, encoder = _wan_backbone_inputs(batch=1)
    expected = _real_backbone_output(transformer, noisy, timesteps, encoder)
    calls = record_forward_calls(transformer)

    out = wan_forward(transformer, noisy, timesteps, encoder)

    torch.testing.assert_close(out, expected)
    assert len(calls) == 1
