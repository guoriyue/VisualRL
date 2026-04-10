"""Composable pipeline framework for staged video generation.

This package provides the runtime skeleton for assembling and executing
generation pipelines as sequences of ``PipelineStage`` objects.  It sits
between the northbound ``VideoGenerationModel`` contract (in ``models/``)
and the low-level engine scheduler (in ``engine/``).

Concrete model implementations in ``models/`` can delegate to a
``ComposedPipeline`` to reuse shared stages (postprocess, VAE decode
passthrough, etc.) while keeping model-specific logic in custom stages.
"""

from wm_infra.pipeline.base import ComposedPipeline
from wm_infra.pipeline.config import PipelineConfig
from wm_infra.pipeline.stages import (
    PassthroughDecodeStage,
    PipelineStage,
    Uint8PostprocessStage,
)

__all__ = [
    "ComposedPipeline",
    "PassthroughDecodeStage",
    "PipelineConfig",
    "PipelineStage",
    "Uint8PostprocessStage",
]
