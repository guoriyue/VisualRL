"""Shared profiling primitives for backend execution traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(slots=True)
class ExecutionStageRecord:
    """One stage execution sample in a backend runtime trace."""

    stage: str
    entity_id: str
    queue_lane: str
    elapsed_ms: float
    started_at: float | None = None
    chunk_id: str | None = None
    chunk_size: int | None = None
    expected_occupancy: float | None = None
    estimated_transfer_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionRuntimeTrace:
    """Shared per-request runtime trace for staged backend execution."""

    include_estimated_transfer_bytes: ClassVar[bool] = False

    records: list[ExecutionStageRecord] = field(default_factory=list)
    queue_lanes_seen: list[str] = field(default_factory=list)

    def record(self, record: ExecutionStageRecord) -> None:
        self.records.append(record)
        if record.queue_lane not in self.queue_lanes_seen:
            self.queue_lanes_seen.append(record.queue_lane)

    def stage_timings_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.records:
            totals[record.stage] = round(totals.get(record.stage, 0.0) + record.elapsed_ms, 3)
        return totals

    def chunk_summary(self) -> dict[str, Any]:
        chunk_count = 0
        chunk_sizes: list[int] = []
        occupancies: list[float] = []
        transfer_bytes = 0
        for record in self.records:
            if record.chunk_id is None:
                continue
            chunk_count += 1
            if record.chunk_size is not None:
                chunk_sizes.append(record.chunk_size)
            if record.expected_occupancy is not None:
                occupancies.append(record.expected_occupancy)
            if record.estimated_transfer_bytes is not None:
                transfer_bytes += record.estimated_transfer_bytes

        summary = {
            "chunk_count": chunk_count,
            "avg_chunk_size": (sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0.0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_expected_occupancy": (sum(occupancies) / len(occupancies)) if occupancies else 0.0,
        }
        if self.include_estimated_transfer_bytes:
            summary["estimated_transfer_bytes"] = transfer_bytes
        return summary
