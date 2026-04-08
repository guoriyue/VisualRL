"""SGLang-style server/runtime args for generation workloads.

Adapted for wm-infra so generation backends can share one serving-runtime
descriptor without importing SGLang's full scheduler/worker stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class GenerationRuntimeBackend(str, Enum):
    """Execution backend family for one runtime."""

    NATIVE = "native"
    EXTERNAL = "external"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class GenerationServerArgs:
    """Runtime shape for a generation server/process topology."""

    model_path: str | None = None
    host: str | None = None
    port: int | None = None
    scheduler_port: int | None = None
    num_gpus: int = 1
    tp_size: int = 1
    sp_degree: int = 1
    enable_cfg_parallel: bool = False
    warmup: bool = False
    local_scheduler: bool = True
    attention_backend: str | None = None
    output_path: str | None = None
    backend: GenerationRuntimeBackend = GenerationRuntimeBackend.NATIVE

    @property
    def local_mode(self) -> bool:
        return self.host is None or self.port is None

    def describe(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "scheduler_port": self.scheduler_port,
            "num_gpus": self.num_gpus,
            "tp_size": self.tp_size,
            "sp_degree": self.sp_degree,
            "enable_cfg_parallel": self.enable_cfg_parallel,
            "warmup": self.warmup,
            "local_scheduler": self.local_scheduler,
            "local_mode": self.local_mode,
            "attention_backend": self.attention_backend,
            "output_path": self.output_path,
            "backend": self.backend.value,
        }
