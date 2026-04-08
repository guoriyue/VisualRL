"""Minimal generation request/output state containers.

SGLang Diffusion uses a richer request object that every stage mutates in
place. wm-infra keeps the same idea but trims it down to the fields needed by
its current video/world-generation backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationRequestState:
    """Mutable per-request state passed across generation stages."""

    request_id: str
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    terminal_output: Any | None = None


@dataclass(slots=True)
class GenerationOutputBatch:
    """Final outputs emitted by a generation pipeline execution."""

    outputs: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
