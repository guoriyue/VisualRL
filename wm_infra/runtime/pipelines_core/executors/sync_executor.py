"""Sequential executor for generation pipelines."""

from __future__ import annotations

from typing import Any

from .pipeline_executor import PipelineExecutor


class SyncPipelineExecutor(PipelineExecutor):
    """Simple in-process executor for ordered stage execution."""

    async def execute(self, stages: list[Any], context: Any, state: dict[str, Any]) -> list[tuple[Any, Any]]:
        results: list[tuple[Any, Any]] = []
        for stage in stages:
            results.append((stage, await stage(context, state)))
        return results
