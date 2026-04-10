"""Shared VAE decode stages."""

from __future__ import annotations

from typing import Any

from wm_infra.engine.model_executor.stages.base import PipelineStage
from wm_infra.models.video_generation import StageResult, VideoGenerationRequest


class PassthroughDecodeStage(PipelineStage):
    """Passthrough decode for pipelines where VAE decode is handled internally.

    Diffusers-based models that use ``output_type="np"`` already have
    decoded frames in ``state["video_frames"]``.  This stage simply
    forwards them and records spatial metadata.
    """

    name = "decode_vae_passthrough"

    async def forward(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        import numpy as np

        frames = np.asarray(state["video_frames"])
        return StageResult(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "decoded_frame_count": int(frames.shape[0]),
                "decoded_spatial_size": [int(frames.shape[2]), int(frames.shape[1])],
            },
            outputs={"fps": request.fps or 16},
            notes=[
                "VAE decode is a passthrough — frames already decoded by diffusers pipeline."
            ],
        )
