"""Real-object tests for the VDN-H3 family wrapper.

The hybrid attention is the vendored upstream module (``third_party/vdn-minimax-h3``,
a pinned submodule), grafted onto a tiny real ``MiniMaxH3Transformer3DModel`` with
the released 8-NFE artifact's own transform config. So these exercise the shipped
delta rule, bridge, anchor mode, short convolutions and text state, on CPU,
through our own ``forward_step``.

What they cannot say is anything about the numbers a trained VDN checkpoint
produces: the branch weights are torch-default init here, and the real
checkpoint needs a multi-GPU deployment (see the model preset).
"""

from __future__ import annotations

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_MINIMAX_H3_LATENT_CHANNELS,
    TINY_VDN_H3_TRANSFORM_CONFIG,
    build_tiny_minimax_h3_components,
    build_tiny_vdn_h3_model,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.generation.types import DenoiseRequest
from vrl.models.families.minimax_h3.model import MiniMaxH3Model
from vrl.models.families.vdn_h3.model import VDNH3Model, VDNH3ReplayModel
from vrl.models.families.vdn_h3.vendor import load_vdn

pytest.importorskip("src.models.hybrid_attention")

_FRAMES, _HEIGHT, _WIDTH = 8, 16, 16
_NUM_STEPS = 3
_MAX_TEXT_TOKENS = 8


def _request() -> DenoiseRequest:
    return DenoiseRequest(
        width=_WIDTH,
        height=_HEIGHT,
        frame_count=_FRAMES,
        num_steps=_NUM_STEPS,
        guidance_scale=1.0,
        seed=0,
        fps=24,
    )


def _state(model: VDNH3Model):
    encoded = model.encode_prompt("a cat video", None, max_sequence_length=_MAX_TEXT_TOKENS)
    return model.prepare_sampling(_request(), encoded)


def test_the_transform_replaces_every_block_attention_and_keeps_the_original() -> None:
    """Upstream's contract: ``block.attn`` becomes a HybridAttention wrapping the
    original module, which stays reachable at ``attn.orig`` -- so the dense
    teacher is always recoverable and the backbone weights are untouched."""
    vendor = load_vdn()
    components = build_tiny_minimax_h3_components()
    original = [block.attn for block in components.transformer.transformer_blocks]

    model = build_tiny_vdn_h3_model()

    assert model.hybrid_blocks == len(model.transformer.transformer_blocks)
    for block in model.transformer.transformer_blocks:
        assert isinstance(block.attn, vendor.hybrid_attention_cls)
    # A freshly built tiny model has the same block count as the untransformed one.
    assert len(original) == model.hybrid_blocks


def test_forward_step_runs_the_hybrid_and_keeps_the_minimax_h3_contract() -> None:
    """One joint forward through the grafted transformer: the video velocity comes
    back in the latent's own shape, finite, fp32, negated into the flow convention."""
    model = build_tiny_vdn_h3_model()
    calls = record_forward_calls(model.transformer)
    state = _state(model)

    with torch.no_grad():
        out = model.forward_step(state, 0)

    assert len(calls) == 1
    assert out["noise_pred"].shape == state.latents.shape
    assert out["noise_pred"].shape[1] == TINY_MINIMAX_H3_LATENT_CHANNELS
    assert out["noise_pred"].dtype is torch.float32
    assert bool(torch.isfinite(out["noise_pred"]).all())
    assert "audio_velocity" in out


def test_the_packed_geometry_reaches_every_hybrid_before_the_forward() -> None:
    """``HybridAttention`` reads its ``SequenceLayout`` off itself, so the layout
    must be installed on every block, and describe the same packed sequence
    MiniMax-H3 built: contiguous video rows, F frames of S tokens, the text span."""
    vendor = load_vdn()
    model = build_tiny_vdn_h3_model()
    state = _state(model)
    for attn in vendor.iter_hybrids(model.transformer):
        assert attn.layout is None

    model.set_hybrid_layout(state)

    layouts = [attn.layout for attn in vendor.iter_hybrids(model.transformer)]
    assert layouts and all(item is layouts[0] for item in layouts)
    layout, packed = layouts[0], state.layout
    _, patch_h, patch_w = model.pipeline.patch_size
    grid = (packed.latent_height // patch_h, packed.latent_width // patch_w)
    assert layout.seq_len == int(packed.position_ids.shape[0])
    assert layout.num_frames == packed.num_latent_frames
    assert layout.tokens_per_frame == grid[0] * grid[1]
    assert layout.frame_size == grid
    assert layout.video_start == int(packed.video_indices[0])
    assert layout.video_end == int(packed.video_indices[-1]) + 1
    assert layout.text_range == (0, packed.num_text_tokens)


def test_the_hybrid_changes_the_prediction_it_wraps() -> None:
    """The graft is not a pass-through: the same latents through the dense base and
    through the hybrid differ, and upstream's ``teacher_mode`` recovers the base."""
    vendor = load_vdn()
    components = build_tiny_minimax_h3_components()
    dense = MiniMaxH3Model(pipeline=components, device=torch.device("cpu"))
    stamp_model_precision(dense)
    state = _state(dense)
    with torch.no_grad():
        base = dense.forward_step(state, 0)["noise_pred"].clone()

    # Graft in place, so both paths share one set of backbone weights.
    vendor.apply_hybrid_attention_transform(components.transformer, TINY_VDN_H3_TRANSFORM_CONFIG)
    vendor.set_softmax_backend(components.transformer, "ref")
    hybrid = VDNH3Model(pipeline=components, device=torch.device("cpu"))
    stamp_model_precision(hybrid)
    hybrid_state = _state(hybrid)
    with torch.no_grad():
        grafted = hybrid.forward_step(hybrid_state, 0)["noise_pred"]
        for attn in vendor.iter_hybrids(components.transformer):
            attn.teacher_mode = True
        teacher = hybrid.forward_step(hybrid_state, 0)["noise_pred"]

    assert not torch.allclose(base, grafted)
    torch.testing.assert_close(teacher, base)


def test_gradients_reach_the_branch_the_policy_would_train() -> None:
    """RL trains through this module, so the graft must stay differentiable: the
    linear branch's own output projection receives gradient from the video velocity."""
    model = build_tiny_vdn_h3_model()
    state = _state(model)
    projection = model.transformer.transformer_blocks[0].attn.to_out_linear
    projection.requires_grad_(True)

    model.forward_step(state, 0)["noise_pred"].square().mean().backward()

    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert float(projection.weight.grad.abs().sum()) > 0.0


def test_the_forward_only_inference_kernels_stay_off() -> None:
    """Upstream's ``set_inference_mode`` installs no-grad fused block bodies; a
    trained policy must never carry them, so the graft leaves both flags off."""
    model = build_tiny_vdn_h3_model()

    for attn in load_vdn().iter_hybrids(model.transformer):
        assert attn.inference_mode is False
        assert attn.hybrid_inference_mode is False


def test_a_missing_vdn_checkpoint_is_refused_with_the_dense_alternative() -> None:
    """Without an artifact there is no transform config and no branch: that
    configuration is the dense base, which is a different family."""
    from types import SimpleNamespace

    model = build_tiny_vdn_h3_model()
    build = SimpleNamespace(model_config={}, parameter_dtype=torch.float32)

    with pytest.raises(ValueError, match="minimax_h3"):
        model.install_hybrid_attention(build)


def test_replay_takes_its_forward_from_vdn_and_its_plumbing_from_minimax_h3() -> None:
    """Base order is load-bearing: the replay plumbing must win while the forward
    and the layout hand-off still resolve to the VDN class."""

    def owner(cls: type, name: str) -> type | None:
        return next((klass for klass in cls.__mro__ if name in klass.__dict__), None)

    from vrl.models.families.minimax_h3.model import MiniMaxH3ReplayModel

    for name in ("forward_step", "set_hybrid_layout", "install_hybrid_attention"):
        assert owner(VDNH3ReplayModel, name) is VDNH3Model, name
    assert owner(VDNH3ReplayModel, "__init__") is MiniMaxH3ReplayModel
