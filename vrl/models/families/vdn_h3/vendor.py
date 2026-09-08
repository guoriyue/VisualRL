"""The one place VRL touches the vendored VDN-H3 package.

VDN-H3 names its own top-level package ``src`` (its ``pyproject.toml`` declares
``packages.find include = ["src*"]``), and it is vendored verbatim as a pinned
git submodule under ``third_party/vdn-minimax-h3`` so the novel attention math
is upstream's, not a transcription. Renaming the tree would break its own
absolute ``from src.models...`` imports, so the generic name stays and every
use funnels through this module: one import site to audit, one error message
when the submodule is not installed.

Nothing here wraps or reinterprets the upstream API. ``load_vdn()`` returns the
entry points ``VDNH3Model`` calls, in the order
``src/inference/assemble.py::build_inference_model`` calls them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SETUP_HINT = (
    "the VDN-H3 sources are vendored as a git submodule; run `make setup` "
    "(git submodule update --init --recursive && pip install -e third_party) "
    "from the repository root"
)


@dataclass(frozen=True, slots=True)
class VDNEntryPoints:
    """Upstream callables and constants, resolved once."""

    load_checkpoint: Any
    resolve_weights: Any
    apply_hybrid_attention_transform: Any
    load_model_weights: Any
    merge_lora_state: Any
    set_softmax_backend: Any
    set_layout: Any
    iter_hybrids: Any
    layout_from_indices: Any
    hybrid_attention_cls: type
    sequence_layout_cls: type
    # The transform's own name/version, stamped into every checkpoint's ModelSpec.
    transform_type: str
    transform_version: int


def load_vdn() -> VDNEntryPoints:
    """Resolve the vendored VDN-H3 entry points, or say how to install them."""

    try:
        from src.checkpoints import load_checkpoint
        from src.inference.lora import merge_lora_state
        from src.models.factory import load_model_weights
        from src.models.hybrid_attention import HybridAttention
        from src.models.hybrid_transform import (
            TRANSFORM_TYPE,
            TRANSFORM_VERSION,
            apply_hybrid_attention_transform,
            iter_hybrids,
            set_layout,
            set_softmax_backend,
        )
        from src.models.sequence_layout import SequenceLayout, layout_from_indices
        from src.paths import resolve_weights
    except ImportError as exc:  # pragma: no cover - exercised by the install-hint test
        raise ImportError(
            f"model.family=vdn_h3 requires the VDN-H3 package: {_SETUP_HINT}"
        ) from exc

    return VDNEntryPoints(
        load_checkpoint=load_checkpoint,
        resolve_weights=resolve_weights,
        apply_hybrid_attention_transform=apply_hybrid_attention_transform,
        load_model_weights=load_model_weights,
        merge_lora_state=merge_lora_state,
        set_softmax_backend=set_softmax_backend,
        set_layout=set_layout,
        iter_hybrids=iter_hybrids,
        layout_from_indices=layout_from_indices,
        hybrid_attention_cls=HybridAttention,
        sequence_layout_cls=SequenceLayout,
        transform_type=TRANSFORM_TYPE,
        transform_version=TRANSFORM_VERSION,
    )


__all__ = ["VDNEntryPoints", "load_vdn"]
