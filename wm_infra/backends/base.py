"""Backend interface for producing control-plane samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from wm_infra.controlplane.schemas import ProduceSampleRequest, SampleRecord, WorldModelKind


class ProduceSampleBackend(ABC):
    """Interface for a runtime/backend that can materialize a sample request."""

    backend_name: str
    world_model_kind: WorldModelKind | None = None
    capability_flags: frozenset[str] = frozenset()

    def validate_world_model_kind(self, request: ProduceSampleRequest) -> None:
        """Reject requests that explicitly declare a mismatched world-model family."""

        if (
            request.world_model_kind is not None
            and self.world_model_kind is not None
            and request.world_model_kind != self.world_model_kind
        ):
            raise ValueError(
                f"Backend {self.backend_name} expects world_model_kind={self.world_model_kind.value}, "
                f"got {request.world_model_kind.value}"
            )

    def backend_descriptor(self) -> dict[str, Any]:
        """Return backend-family metadata shared by northbound listing surfaces."""

        return {
            "world_model_kind": None if self.world_model_kind is None else self.world_model_kind.value,
            "capabilities": sorted(self.capability_flags),
        }

    @abstractmethod
    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        """Execute the request and return a populated sample record."""
