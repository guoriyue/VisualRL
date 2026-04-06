"""Cross-request transition batching for Genie stage runtime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from wm_infra.backends.genie_runner import GeniePreparedRun, GenieRunner, GenieWindowResult
from wm_infra.backends.genie_runtime import GenieBatchSignature, GenieExecutionChunk


@dataclass(slots=True)
class TransitionBatchOutcome:
    batch_id: str
    batch_size: int
    sample_ids: list[str]
    elapsed_ms: float
    window_result: GenieWindowResult


@dataclass(slots=True)
class PendingTransitionChunk:
    sample_id: str
    prepared: GeniePreparedRun
    signature: GenieBatchSignature
    frame_start: int
    frame_end: int
    future: asyncio.Future[TransitionBatchOutcome]
    enqueued_at: float = field(default_factory=time.perf_counter)


class GenieTransitionBatcher:
    """Batch compatible transition windows across concurrent Genie rollouts."""

    def __init__(self, runner: GenieRunner, *, max_batch_size: int = 8, batch_wait_ms: float = 2.0) -> None:
        self._runner = runner
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self._lock = asyncio.Lock()
        self._pending: dict[tuple[GenieBatchSignature, int, int], list[PendingTransitionChunk]] = {}
        self._flush_tasks: dict[tuple[GenieBatchSignature, int, int], asyncio.Task[None]] = {}
        self._stats = {
            "total_requests": 0,
            "total_flushed_batches": 0,
            "batched_requests": 0,
            "largest_batch_size": 0,
        }

    async def run(
        self,
        *,
        sample_id: str,
        prepared: GeniePreparedRun,
        signature: GenieBatchSignature,
        frame_start: int,
        frame_end: int,
    ) -> TransitionBatchOutcome:
        if self.max_batch_size <= 1:
            t0 = time.perf_counter()
            window_result = self._runner.run_chunk([prepared], frame_start=frame_start, frame_end=frame_end)[0]
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            self._stats["total_requests"] += 1
            self._stats["total_flushed_batches"] += 1
            self._stats["largest_batch_size"] = max(self._stats["largest_batch_size"], 1)
            return TransitionBatchOutcome(
                batch_id=f"{signature.stage}:{frame_start}:{frame_end}:1",
                batch_size=1,
                sample_ids=[sample_id],
                elapsed_ms=elapsed_ms,
                window_result=window_result,
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[TransitionBatchOutcome] = loop.create_future()
        pending = PendingTransitionChunk(
            sample_id=sample_id,
            prepared=prepared,
            signature=signature,
            frame_start=frame_start,
            frame_end=frame_end,
            future=future,
        )
        key = (signature, frame_start, frame_end)

        async with self._lock:
            group = self._pending.setdefault(key, [])
            group.append(pending)
            self._stats["total_requests"] += 1
            if len(group) >= self.max_batch_size:
                task = self._flush_tasks.pop(key, None)
                if task is not None:
                    task.cancel()
                asyncio.create_task(self._flush(key))
            elif key not in self._flush_tasks:
                self._flush_tasks[key] = asyncio.create_task(self._delayed_flush(key))

        return await future

    async def run_transition(
        self,
        *,
        sample_id: str,
        prepared: GeniePreparedRun,
        chunk: GenieExecutionChunk,
    ) -> TransitionBatchOutcome:
        return await self.run(
            sample_id=sample_id,
            prepared=prepared,
            signature=chunk.signature,
            frame_start=chunk.frame_range[0],
            frame_end=chunk.frame_range[1],
        )

    async def _delayed_flush(self, key: tuple[GenieBatchSignature, int, int]) -> None:
        try:
            await asyncio.sleep(self.batch_wait_ms / 1000.0)
            await self._flush(key)
        except asyncio.CancelledError:
            return

    async def _flush(self, key: tuple[GenieBatchSignature, int, int]) -> None:
        async with self._lock:
            items = self._pending.pop(key, [])
            task = self._flush_tasks.pop(key, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()

        if not items:
            return

        batch_size = len(items)
        self._stats["total_flushed_batches"] += 1
        self._stats["largest_batch_size"] = max(self._stats["largest_batch_size"], batch_size)
        if batch_size > 1:
            self._stats["batched_requests"] += batch_size

        try:
            batch_id = f"{items[0].signature.stage}:{items[0].frame_start}:{items[0].frame_end}:{batch_size}"
            sample_ids = [item.sample_id for item in items]
            t0 = time.perf_counter()
            results = self._runner.run_chunk(
                [item.prepared for item in items],
                frame_start=items[0].frame_start,
                frame_end=items[0].frame_end,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            for item, result in zip(items, results, strict=True):
                if not item.future.done():
                    item.future.set_result(
                        TransitionBatchOutcome(
                            batch_id=batch_id,
                            batch_size=batch_size,
                            sample_ids=sample_ids,
                            elapsed_ms=elapsed_ms,
                            window_result=result,
                        )
                    )
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)

    def snapshot(self) -> dict[str, Any]:
        total_requests = int(self._stats["total_requests"])
        total_flushed_batches = int(self._stats["total_flushed_batches"])
        batched_requests = int(self._stats["batched_requests"])
        return {
            "total_requests": total_requests,
            "total_flushed_batches": total_flushed_batches,
            "batched_requests": batched_requests,
            "largest_batch_size": int(self._stats["largest_batch_size"]),
            "max_observed_batch_size": int(self._stats["largest_batch_size"]),
            "mean_batch_size": (total_requests / total_flushed_batches) if total_flushed_batches else 0.0,
            "batch_wait_ms": self.batch_wait_ms,
            "max_batch_size": self.max_batch_size,
        }
