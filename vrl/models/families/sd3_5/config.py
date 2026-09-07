"""Lightweight public config schema for the SD 3.5 model family."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vrl.config.model_schema import ModelSection
from vrl.models.checkpoint_identity import checkpoint_identity_metadata


class SD3_5ModelSection(ModelSection):
    """SD 3.5 public model keys."""

    # The frozen ``previous`` LoRA mirror DiffusionNFT and V-GRPO evaluate the
    # behaviour policy through (``LoraModelMixin.attach_previous_policy_adapter``).
    nft_previous_adapter: Any = Field(
        default=None,
        json_schema_extra=checkpoint_identity_metadata("value", default=False),
    )


__all__ = ["SD3_5ModelSection"]
