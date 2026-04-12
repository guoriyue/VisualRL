"""Cosmos variant definitions and executor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from vrl.models.base import ModelResult, VideoGenerationRequest


class CosmosVariant(str, Enum):
    """Cosmos world-generation variant."""

    PREDICT1_TEXT2WORLD = "predict1_text2world"
    PREDICT1_VIDEO2WORLD = "predict1_video2world"
    PREDICT2_VIDEO2WORLD = "predict2_video2world"
    PREDICT2_TEXT2IMAGE = "predict2_text2image"
    PREDICT25_VIDEO2WORLD = "predict25_video2world"


class CosmosLocalExecutor(ABC):
    """Executor interface for local/in-process Cosmos generation."""

    execution_mode: str = "in_process"

    async def load(self) -> None:  # noqa: B027
        """Optional: resolve local resources."""

    @abstractmethod
    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Produce prompt-side conditioning."""

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Produce reference-side conditioning."""
        return ModelResult(outputs={"reference_count": len(request.references)})

    @abstractmethod
    async def generate(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Run generation loop."""

    @abstractmethod
    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Decode latent state into video frames."""

    @abstractmethod
    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> ModelResult:
        """Normalize decoded frames."""

    def describe(self) -> dict[str, Any]:
        return {"execution_mode": self.execution_mode, "executor": self.__class__.__name__}
