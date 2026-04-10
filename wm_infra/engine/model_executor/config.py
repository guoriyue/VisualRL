"""Pipeline configuration for composable stage execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    """Shared configuration accessible by all stages in a pipeline.

    Model-specific pipelines subclass this to add callbacks or parameters
    that customise shared stage behaviour without requiring stage subclasses.
    """

    device_id: int = 0
    dtype: str = "bfloat16"
    enable_cpu_offload: bool = True
    enable_vae_tiling: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def device_str(self) -> str:
        return f"cuda:{self.device_id}"
