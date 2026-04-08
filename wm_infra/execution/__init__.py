"""Execution substrate for temporal runtime scheduling."""

from wm_infra.execution.profiling import ExecutionRuntimeTrace, ExecutionStageRecord
from wm_infra.execution.scheduler import (
    GroupedChunkDecision,
    HomogeneousChunkScheduler,
    SchedulerDecision,
    build_execution_chunks,
    schedule_grouped_chunks,
)
from wm_infra.execution.types import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionEntity,
    ExecutionStats,
    ExecutionWorkItem,
    chunk_fill_ratio,
    summarize_execution_chunks,
)

__all__ = [
    "BatchSignature",
    "ExecutionBatchPolicy",
    "ExecutionChunk",
    "ExecutionEntity",
    "ExecutionRuntimeTrace",
    "ExecutionStageRecord",
    "ExecutionStats",
    "GroupedChunkDecision",
    "ExecutionWorkItem",
    "HomogeneousChunkScheduler",
    "SchedulerDecision",
    "build_execution_chunks",
    "schedule_grouped_chunks",
    "chunk_fill_ratio",
    "summarize_execution_chunks",
]
