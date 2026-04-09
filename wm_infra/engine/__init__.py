"""Unified runtime engine for world-model inference.

Public API re-exports for the engine module.
"""

from wm_infra.engine.types import (
    EngineRunConfig,
    EntityRequest,
    Phase,
    SchedulerOutput,
    StepResult,
    SwapHandle,
)
from wm_infra.engine.managers.engine_loop import EngineLoop
from wm_infra.engine.model_executor.worker import (
    DynamicsStage,
    EncodeStage,
    StageRunner,
    StageSpec,
)
from wm_infra.engine.model_executor.task_graph import TaskEdge, TaskGraph, TaskNode
from wm_infra.engine.managers.scheduler import ContinuousBatchingScheduler, EntityState
from wm_infra.engine.mem_cache.paged_pool import PagedLatentPool, PageTable
from wm_infra.engine.mem_cache.radix_cache import RadixNode, RadixStateCache
from wm_infra.engine.model_executor.worker import AsyncQueue, RequestQueue, ResultQueue
from wm_infra.engine.model_executor.worker import Worker

__all__ = [
    # Types
    "EngineRunConfig",
    "EntityRequest",
    "Phase",
    "SchedulerOutput",
    "StepResult",
    "SwapHandle",
    # Loop
    "EngineLoop",
    # Scheduler
    "ContinuousBatchingScheduler",
    "EntityState",
    # State
    "PagedLatentPool",
    "PageTable",
    "RadixNode",
    "RadixStateCache",
    # Pipeline
    "DynamicsStage",
    "EncodeStage",
    "StageRunner",
    "StageSpec",
    "TaskEdge",
    "TaskGraph",
    "TaskNode",
    # Workers
    "AsyncQueue",
    "RequestQueue",
    "ResultQueue",
    "Worker",
]
