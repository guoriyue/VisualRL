"""VDN-H3: MiniMax-H3 with the hybrid window-softmax / linear-attention branch.

VDN-H3 (VideoDeltaNet, arXiv/blog: openvdn.github.io) is a derivative of
MiniMax-H3: it leaves the backbone weights alone and replaces every DiT block's
attention with a two-branch module -- a local sliding-window softmax over nearby
frames plus a bidirectional delta-rule linear branch for long range -- then adds
a linear-branch output projection, gates, and two small LoRA adapters folded
into the backbone at load.

Three facts decide this file:

1. **The architecture is a module swap, not a fork.** Upstream's
   ``apply_hybrid_attention_transform`` sets ``block.attn = HybridAttention(block.attn, ...)``
   and keeps the original attention reachable at ``attn.orig``; the transformer's
   own ``forward`` signature never changes. So everything the ``minimax_h3``
   family already does -- the packed layout, the row-timestep plan, the two
   schedulers, the audio side stream, the replay contract -- is inherited
   unchanged, and this class only adds loading plus one per-forward hand-off.

2. **The hybrid needs the packed geometry.** ``HybridAttention`` reads a
   ``SequenceLayout`` off itself before each forward to know where the video
   rows are and how they tile into frames. ``MiniMaxH3Layout`` already carries
   every number that needs, so ``forward_step`` projects one into upstream's
   type and installs it.

3. **The inference kernels are forward-only.** Upstream's ``set_inference_mode``
   installs fused, no-grad block bodies ("the CALLER states no graph will be
   built"), so an RL policy must never enable it -- the defaults are the eager
   bodies its own trainers use. Only the window-softmax backend is selectable
   here (``model.softmax_backend``), because that one keeps gradients on every
   setting.

The vendored sources are a pinned git submodule (``third_party/vdn-minimax-h3``)
reached through :mod:`vrl.models.families.vdn_h3.vendor`; the attention math is
upstream's, never transcribed here.
"""

from __future__ import annotations

from typing import Any

import torch

from vrl.models.families.minimax_h3.model import (
    MiniMaxH3Model,
    MiniMaxH3ReplayModel,
    MiniMaxH3SamplingState,
)
from vrl.models.families.vdn_h3.vendor import VDNEntryPoints, load_vdn
from vrl.models.interfaces.runtime import ModelBuild
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

# Upstream's own default (``HybridAttention.softmax_impl``) is resolved per GPU
# architecture by ``set_softmax_backend("auto")``.
_DEFAULT_SOFTMAX_BACKEND = "auto"


class VDNH3Model(MiniMaxH3Model):
    """MiniMax-H3 policy carrying VDN-H3's hybrid attention."""

    @classmethod
    def from_build(cls, build: ModelBuild) -> VDNH3Model:
        """Load the MiniMax-H3 base, then graft the VDN artifact onto it.

        The order is upstream's (``src/inference/assemble.py``): base, transform,
        branch weights, folded adapters. What this deliberately does not do is
        upstream's inference overlay -- no ``set_inference_mode``, no fp8 -- because
        the policy is trained through this module.
        """

        model = super().from_build(build)
        model.install_hybrid_attention(build)
        return model

    # -- loading -----------------------------------------------------------

    def install_hybrid_attention(self, build: ModelBuild) -> None:
        """Apply the transform in the artifact's own spec and load its weights."""

        vendor = load_vdn()
        model_config = build.model_config or {}
        checkpoint = str(model_config.get("vdn_checkpoint") or "").strip()
        if not checkpoint:
            raise ValueError(
                "model.family=vdn_h3 requires model.vdn_checkpoint (the VDN artifact "
                "supplying the transform config, the linear branch and the adapters); "
                "without one this is the dense MiniMax-H3 base, which is model.family=minimax_h3",
            )
        artifact = vendor.load_checkpoint(vendor.resolve_weights(checkpoint))
        if artifact.metadata.get("truncated_blocks"):
            raise ValueError(f"{checkpoint} is a truncated smoke-test artifact")
        transforms = (artifact.model_spec or {}).get("transforms") or ()
        if len(transforms) != 1 or transforms[0].get("type") != vendor.transform_type:
            raise ValueError(
                f"{checkpoint} does not declare exactly one {vendor.transform_type!r} "
                f"transform; got {[t.get('type') for t in transforms]}",
            )
        transformer = self.transformer
        vendor.apply_hybrid_attention_transform(transformer, transforms[0]["config"])
        weights = artifact.weights or {}
        branch = {name: value for name, value in weights.items() if "lora_" not in name}
        adapters = {name: value for name, value in weights.items() if "lora_" in name}
        loaded = vendor.load_model_weights(transformer, branch)
        merged = vendor.merge_lora_state(transformer, adapters) if adapters else 0
        # The transform's new modules are created at torch's default dtype while
        # the backbone is the build's; upstream casts them the same way before
        # running. The backbone's own mixed-precision islands
        # (``_keep_in_fp32_modules``) are untouched -- only what the transform added.
        for attn in vendor.iter_hybrids(transformer):
            for parameter in attn.parameters():
                if parameter.dtype is torch.float32:
                    parameter.data = parameter.data.to(build.parameter_dtype)
        backend = str(model_config.get("softmax_backend") or _DEFAULT_SOFTMAX_BACKEND)
        resolved = vendor.set_softmax_backend(transformer, backend)
        logger.info(
            "grafted VDN-H3 hybrid attention %s",
            kv(
                checkpoint=checkpoint,
                branch_tensors=loaded,
                lora_pairs=merged,
                softmax_backend=resolved,
                dtype=build.parameter_dtype,
            ),
        )

    # -- per-forward geometry ---------------------------------------------

    def forward_step(self, state: MiniMaxH3SamplingState, step_idx: int) -> dict[str, Any]:
        """Install the packed geometry, then run MiniMax-H3's own joint forward."""

        self.set_hybrid_layout(state)
        return super().forward_step(state, step_idx)

    def set_hybrid_layout(self, state: MiniMaxH3SamplingState) -> None:
        """Project ``MiniMaxH3Layout`` into upstream's ``SequenceLayout``.

        Upstream derives it from the same ``build_packed_sequence`` outputs our
        layout is built from, and verifies rather than assumes that the video
        rows are contiguous and t-major. That holds for ``t2va`` (no keyframe
        conditioning rows precede the generated ones); a layout with keyframe
        anchors would fail there, loudly, which is the correct answer until this
        family carries an ``fl2va`` task.
        """

        vendor = load_vdn()
        layout = state.layout
        _, patch_h, patch_w = self.pipeline.patch_size
        grid_h = layout.latent_height // patch_h
        grid_w = layout.latent_width // patch_w
        vendor.set_layout(
            self.transformer,
            vendor.layout_from_indices(
                layout.video_indices,
                layout.num_latent_frames,
                grid_h * grid_w,
                int(layout.position_ids.shape[0]),
                frame_size=(grid_h, grid_w),
                text_indices=layout.text_indices,
            ),
        )

    # -- diagnostics -------------------------------------------------------

    @property
    def hybrid_blocks(self) -> int:
        """How many DiT blocks carry the hybrid attention (0 = dense base)."""

        return sum(1 for _ in load_vdn().iter_hybrids(self.transformer))


class VDNH3ReplayModel(MiniMaxH3ReplayModel, VDNH3Model):
    """Trainer-side replay model: the hybrid transformer and the two schedulers.

    Base order is load-bearing, as for Wan I2V: ``MiniMaxH3ReplayModel`` first so
    the replay plumbing (constructor, pipeline-shell accessors) wins, while
    ``forward_step`` and the layout hand-off still resolve to ``VDNH3Model``.
    The replay policy must carry the same graft as the rollout policy or the
    recomputed log-probs describe a different model, so the replay builder calls
    ``install_hybrid_attention`` too.
    """


__all__ = ["VDNEntryPoints", "VDNH3Model", "VDNH3ReplayModel"]
