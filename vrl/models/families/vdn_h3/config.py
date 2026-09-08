"""Lightweight public config schema for the VDN-H3 model family."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class VDNH3ModelSection(ModelSection):
    """VDN-H3 public model keys.

    ``path`` is the MiniMax-H3 base the hybrid transform is grafted onto;
    ``vdn_checkpoint`` is the VDN artifact that supplies the transform config,
    the linear-branch weights and the adapters folded into the backbone. Two
    sources, so two checkpoint-identity entries: a run is only reproducible
    when both are pinned.
    """

    vdn_checkpoint: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source",
            source="vdn",
            revision_field="vdn_revision",
        ),
    )
    vdn_revision: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata(
            "source_revision",
            source="vdn",
        ),
    )
    # Window-softmax implementation, upstream's one runtime knob
    # (hybrid_transform.set_softmax_backend): auto | flex | decomposed | ref.
    # "auto" resolves per GPU architecture; "ref" is the eager reference that
    # runs anywhere, including CPU.
    softmax_backend: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default="auto"),
    )


__all__ = ["VDNH3ModelSection"]
