"""Collector protocol — collects training experience from model rollouts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vrl.experience.types import ExperienceBatch


@runtime_checkable
class Collector(Protocol):
    """Collects training experience from model rollouts."""

    async def collect(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> ExperienceBatch:
        ...
