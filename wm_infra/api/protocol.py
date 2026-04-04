"""Request/Response protocol for world model rollout API."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field


class RolloutRequest(BaseModel):
    """Client request for a world model rollout."""

    model: str = Field(default="latent_dynamics", description="Model name from registry")
    initial_observation_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded initial observation (image/video frame)",
    )
    initial_latent: Optional[list[list[float]]] = Field(
        default=None,
        description="Pre-encoded latent state [N, D]",
    )
    actions: Optional[list[list[float]]] = Field(
        default=None,
        description="Action sequence [T, A] for each prediction step",
    )
    num_steps: int = Field(default=1, ge=1, le=128, description="Number of prediction steps")
    return_frames: bool = Field(default=True, description="Decode latents back to pixel frames")
    return_latents: bool = Field(default=False, description="Return raw latent states")
    stream: bool = Field(default=False, description="Stream results step by step via SSE")


class StepResult(BaseModel):
    """One prediction step result (used in streaming)."""

    step: int
    latent: Optional[list[list[float]]] = None
    frame_b64: Optional[str] = None

    def to_sse(self) -> str:
        """Format as a Server-Sent Event data line."""
        return f"data: {self.model_dump_json()}\n\n"


SSE_DONE = "data: [DONE]\n\n"


class RolloutResponse(BaseModel):
    """Complete rollout response."""

    job_id: str
    model: str
    steps_completed: int
    elapsed_ms: float
    latents: Optional[list[list[list[float]]]] = Field(
        default=None,
        description="Predicted latent states [T, N, D]",
    )
    frames_b64: Optional[list[str]] = Field(
        default=None,
        description="Base64-encoded predicted frames",
    )


class HealthResponse(BaseModel):
    """Server health check."""

    status: str = "ok"
    model_loaded: bool = False
    active_rollouts: int = 0
    memory_used_gb: float = 0.0


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    num_parameters: int
    device: str
    dtype: str
