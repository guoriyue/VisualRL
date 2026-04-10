"""Shared postprocess stage: clip and convert to uint8."""

from __future__ import annotations

from typing import Any

from wm_infra.engine.model_executor.stages.base import PipelineStage
from wm_infra.models.video_generation import StageResult, VideoGenerationRequest


class Uint8PostprocessStage(PipelineStage):
    """Clip float frames to [0, 1] and convert to uint8.

    This logic is identical across diffusers-based models (Wan I2V,
    Cosmos Predict1, etc.) and is extracted here to avoid duplication.
    """

    name = "postprocess_uint8"

    async def forward(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        import numpy as np

        frames = np.asarray(state["video_frames"])
        if frames.dtype != np.uint8:
            frames = np.clip(frames * 255.0, 0.0, 255.0).astype(np.uint8)

        fps = request.fps or 16
        return StageResult(
            state_updates={
                "video_frames": frames,
                "output_fps": fps,
                "_pipeline_output": frames,
            },
            runtime_state_updates={
                "frame_count": int(frames.shape[0]),
                "output_fps": fps,
            },
            outputs={"frame_count": int(frames.shape[0])},
            notes=["Postprocess normalized frames to uint8."],
        )
