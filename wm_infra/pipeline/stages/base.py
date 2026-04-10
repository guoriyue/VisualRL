"""Base class for composable pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wm_infra.models.video_generation import StageResult, VideoGenerationRequest


class PipelineStage(ABC):
    """Single composable unit in a generation pipeline.

    Each stage takes a request and mutable state dict, performs one
    logical step of generation, and returns a ``StageResult`` describing
    what changed.  Stages are assembled by ``ComposedPipeline`` and
    executed in order.
    """

    name: str = "unnamed"

    @abstractmethod
    async def forward(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Execute the stage transformation."""

    async def verify_input(  # noqa: B027
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> None:
        """Optional pre-execution validation (override to add checks)."""

    async def verify_output(  # noqa: B027
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
        result: StageResult,
    ) -> None:
        """Optional post-execution validation (override to add checks)."""

    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata about this stage."""
        return {"name": self.name, "stage_class": self.__class__.__name__}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
