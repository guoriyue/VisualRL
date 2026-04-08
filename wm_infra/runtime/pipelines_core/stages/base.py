"""Stage contract for SGLang-style generation pipelines."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StageParallelismType(str, Enum):
    """Execution semantics borrowed from SGLang Diffusion."""

    REPLICATED = "replicated"
    MAIN_RANK_ONLY = "main_rank_only"
    CFG_PARALLEL = "cfg_parallel"


@dataclass(frozen=True, slots=True)
class StageProfile:
    """Lightweight per-stage timing profile."""

    elapsed_ms: float


class PipelineStage(ABC):
    """Base class for one pipeline stage.

    The wrapper centralizes validation and timing so concrete stages can focus
    on stage logic rather than lifecycle boilerplate.
    """

    parallelism: StageParallelismType = StageParallelismType.REPLICATED

    def verify_input(self, _context: Any, _state: dict[str, Any]) -> None:
        """Stage-specific input validation hook."""

    def verify_output(self, _output: Any) -> None:
        """Stage-specific output validation hook."""

    @abstractmethod
    async def forward(self, context: Any, state: dict[str, Any]) -> Any:
        """Run the stage and return a stage update."""

    async def __call__(self, context: Any, state: dict[str, Any]) -> Any:
        self.verify_input(context, state)
        started = time.perf_counter()
        output = await self.forward(context, state)
        self.verify_output(output)
        profile = StageProfile(elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2))
        outputs = getattr(output, "outputs", None)
        if isinstance(outputs, dict):
            outputs.setdefault("stage_profile", {"elapsed_ms": profile.elapsed_ms})
        else:
            try:
                setattr(output, "_stage_profile", profile)
            except AttributeError:
                pass
        return output
