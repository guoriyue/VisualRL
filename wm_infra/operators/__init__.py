"""Model invocation operators for dynamics and generation backends.

Operators isolate model- and runner-specific invocation details from backend
contracts and execution scheduling. Backends adapt northbound requests into
operator calls; execution and queueing layers should not need to know how a
particular model is invoked.
"""

from .base import ModelOperator, OperatorFamily
from .dynamics import RolloutEngineDynamicsOperator
from .generation import CosmosGenerationOperator, WanInProcessGenerationOperator
from wm_infra.runtime import (
    CallableGenerationStage,
    ComposedGenerationPipeline,
    GenerationPipelineRun,
    GenerationPipelineStageSpec,
    GenerationRuntimeBackend,
    GenerationRuntimeConfig,
    GenerationStageUpdate,
)

__all__ = [
    "CallableGenerationStage",
    "ComposedGenerationPipeline",
    "CosmosGenerationOperator",
    "GenerationPipelineRun",
    "GenerationPipelineStageSpec",
    "GenerationRuntimeBackend",
    "GenerationRuntimeConfig",
    "GenerationStageUpdate",
    "ModelOperator",
    "OperatorFamily",
    "RolloutEngineDynamicsOperator",
    "WanInProcessGenerationOperator",
]
