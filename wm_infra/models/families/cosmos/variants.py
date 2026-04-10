"""Cosmos variant definitions and executor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class CosmosVariant(str, Enum):
    """Cosmos world-generation variant."""

    PREDICT1_TEXT2WORLD = "predict1_text2world"
    PREDICT1_VIDEO2WORLD = "predict1_video2world"
    PREDICT2_VIDEO2WORLD = "predict2_video2world"
    PREDICT2_TEXT2IMAGE = "predict2_text2image"


class CosmosLocalExecutor(ABC):
    """Executor interface for local/in-process Cosmos generation."""

    execution_mode: str = "in_process"

    async def load(self) -> None:  # noqa: B027
        """Resolve local resources needed for inference (optional override)."""

    @abstractmethod
    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Produce prompt-side conditioning."""

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Produce reference-side conditioning."""
        return StageResult(outputs={"reference_count": len(request.references)})

    @abstractmethod
    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Run the local generation loop and return latent/video state."""

    @abstractmethod
    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Decode latent state into video frames."""

    @abstractmethod
    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        """Standardise decoded frames for backend persistence."""

    def describe(self) -> dict[str, Any]:
        return {"execution_mode": self.execution_mode, "executor": self.__class__.__name__}
