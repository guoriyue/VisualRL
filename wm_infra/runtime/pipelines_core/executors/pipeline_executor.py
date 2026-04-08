"""Executor contract for composed generation pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PipelineExecutor(ABC):
    """Execute an ordered list of pipeline stages."""

    @abstractmethod
    async def execute(self, stages: list[Any], context: Any, state: dict[str, Any]) -> list[tuple[Any, Any]]:
        """Run stages and return ``(stage, update)`` pairs."""
