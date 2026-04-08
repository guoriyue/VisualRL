"""In-process async job queue for sample production backends."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from wm_infra.controlplane.schemas import SampleRecord, SampleStatus

if TYPE_CHECKING:
    from wm_infra.controlplane.schemas import ProduceSampleRequest
    from wm_infra.controlplane.storage import SampleManifestStore

logger = logging.getLogger("wm_infra.job_queue")


@dataclass
class JobEntry:
    """Tracks a single queued/running/completed job."""

    sample_id: str
    request: ProduceSampleRequest
    status: str = "queued"  # queued | running | succeeded | failed
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


class SampleJobQueue:
    """Bounded async job queue with background workers."""

    def __init__(
        self,
        execute_fn: Callable[[ProduceSampleRequest, str], Coroutine[Any, Any, SampleRecord]],
        store: SampleManifestStore,
        *,
        queue_name: str = "sample",
        max_queue_size: int = 64,
        max_concurrent: int = 2,
        execute_many_fn: Callable[[list[tuple[ProduceSampleRequest, str]]], Coroutine[Any, Any, list[SampleRecord]]] | None = None,
        batch_key_fn: Callable[[ProduceSampleRequest], str | tuple[Any, ...] | None] | None = None,
        batch_select_fn: Callable[[ProduceSampleRequest, ProduceSampleRequest], float | None] | None = None,
        max_batch_size: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> None:
        self._execute_fn = execute_fn
        self._execute_many_fn = execute_many_fn
        self._batch_key_fn = batch_key_fn
        self._batch_select_fn = batch_select_fn
        self._store = store
        self._queue_name = queue_name
        self._max_queue_size = max_queue_size
        self._max_concurrent = max_concurrent
        self._max_batch_size = max(1, max_batch_size)
        self._batch_wait_s = max(batch_wait_ms, 0.0) / 1000.0

        self._queue: asyncio.Queue[JobEntry] = asyncio.Queue(maxsize=max_queue_size)
        self._jobs: dict[str, JobEntry] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    @property
    def total_count(self) -> int:
        return len(self._jobs)

    def get_job(self, sample_id: str) -> JobEntry | None:
        return self._jobs.get(sample_id)

    def position(self, sample_id: str) -> int | None:
        if sample_id not in self._jobs:
            return None
        ahead = 0
        for entry in self._jobs.values():
            if entry.sample_id == sample_id:
                break
            if entry.status == "queued":
                ahead += 1
        return ahead

    def snapshot(self) -> dict[str, Any]:
        queued = [job for job in self._jobs.values() if job.status == "queued"]
        running = [job for job in self._jobs.values() if job.status == "running"]
        return {
            "pending": self.pending_count,
            "running": self.running_count,
            "total_tracked": self.total_count,
            "max_queue_size": self._max_queue_size,
            "max_concurrent": self._max_concurrent,
            "max_batch_size": self._max_batch_size,
            "batch_select_enabled": self._batch_select_fn is not None,
            "queued_sample_ids": [job.sample_id for job in queued],
            "running_sample_ids": [job.sample_id for job in running],
        }

    def submit(self, sample_id: str, request: ProduceSampleRequest) -> JobEntry:
        if self._queue.full():
            raise RuntimeError(f"{self._queue_name} job queue is full ({self._max_queue_size} pending)")
        entry = JobEntry(sample_id=sample_id, request=request)
        self._jobs[sample_id] = entry
        self._queue.put_nowait(entry)
        return entry

    def _retire_job(self, sample_id: str) -> None:
        """Drop terminal bookkeeping so the in-memory job table stays bounded."""
        self._jobs.pop(sample_id, None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for i in range(self._max_concurrent):
            task = asyncio.get_event_loop().create_task(self._worker_loop(i))
            self._workers.append(task)
        logger.info("%s SampleJobQueue started with %d workers", self._queue_name, self._max_concurrent)

    async def stop(self) -> None:
        self._running = False
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("%s SampleJobQueue stopped", self._queue_name)

    async def _worker_loop(self, worker_id: int) -> None:
        logger.info("%s worker %d started", self._queue_name, worker_id)
        while self._running:
            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            batch = [entry]
            batch_key = self._batch_key(entry.request) if self._execute_many_fn is not None else None
            if self._execute_many_fn is not None and self._max_batch_size > 1:
                batch.extend(await self._collect_batch(entry.request, batch_key))
            for batch_entry in batch:
                batch_entry.status = "running"
                batch_entry.started_at = time.time()
            logger.info(
                "%s worker %d executing batch size=%d [%s]",
                self._queue_name,
                worker_id,
                len(batch),
                ", ".join(item.sample_id for item in batch),
            )

            try:
                for batch_entry in batch:
                    queued_record = self._store.get(batch_entry.sample_id)
                    if queued_record is not None:
                        queued_record.status = SampleStatus.RUNNING
                        history = list(queued_record.runtime.get("status_history", []))
                        history.append({"status": SampleStatus.RUNNING.value, "timestamp": batch_entry.started_at})
                        queued_record.runtime["status_history"] = history
                        queued_record.runtime["started_at"] = batch_entry.started_at
                        self._store.put(queued_record)

                if self._execute_many_fn is not None and len(batch) > 1:
                    records = await self._execute_many_fn([(batch_entry.request, batch_entry.sample_id) for batch_entry in batch])
                else:
                    records = [await self._execute_fn(batch_entry.request, batch_entry.sample_id) for batch_entry in batch]

                records_by_id = {record.sample_id: record for record in records}
                for batch_entry in batch:
                    record = records_by_id[batch_entry.sample_id]
                    batch_entry.status = record.status.value
                    batch_entry.completed_at = time.time()
                    self._store.put(record)
                    self._retire_job(batch_entry.sample_id)
                    logger.info(
                        "%s worker %d completed job %s → %s",
                        self._queue_name,
                        worker_id,
                        batch_entry.sample_id,
                        batch_entry.status,
                    )
            except Exception as exc:
                for batch_entry in batch:
                    batch_entry.status = "failed"
                    batch_entry.completed_at = time.time()
                    batch_entry.error = str(exc)
                    logger.exception("%s worker %d failed job %s", self._queue_name, worker_id, batch_entry.sample_id)
                    failed_record = self._store.get(batch_entry.sample_id)
                    if failed_record is not None:
                        failed_record.status = SampleStatus.FAILED
                        history = list(failed_record.runtime.get("status_history", []))
                        history.append({"status": SampleStatus.FAILED.value, "timestamp": batch_entry.completed_at})
                        failed_record.runtime["status_history"] = history
                        failed_record.runtime["completed_at"] = batch_entry.completed_at
                        failed_record.runtime["queue_error"] = str(exc)
                        failed_record.metadata["runner_error"] = str(exc)
                        self._store.put(failed_record)
                    self._retire_job(batch_entry.sample_id)
            finally:
                for _batch_entry in batch:
                    self._queue.task_done()

    def _batch_key(self, request: ProduceSampleRequest) -> str | tuple[Any, ...] | None:
        if self._batch_key_fn is None:
            return None
        return self._batch_key_fn(request)

    async def _collect_batch(
        self,
        reference_request: ProduceSampleRequest,
        batch_key: str | tuple[Any, ...] | None,
    ) -> list[JobEntry]:
        collected: list[JobEntry] = []
        if self._max_batch_size <= 1:
            return collected

        if self._batch_wait_s > 0:
            await asyncio.sleep(self._batch_wait_s)

        pending: list[JobEntry] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        ranked: list[tuple[float, int, JobEntry]] = []
        for index, candidate in enumerate(pending):
            candidate_key = self._batch_key(candidate.request)
            if batch_key is not None and candidate_key == batch_key:
                ranked.append((10_000.0, index, candidate))
                continue
            if self._batch_select_fn is None:
                continue
            score = self._batch_select_fn(reference_request, candidate.request)
            if score is None or score <= 0:
                continue
            ranked.append((float(score), index, candidate))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected_ids = {entry.sample_id for _, _, entry in ranked[: self._max_batch_size - 1]}
        for candidate in pending:
            if candidate.sample_id in selected_ids:
                collected.append(candidate)
            else:
                self._queue.put_nowait(candidate)
        return collected


WanJobQueue = SampleJobQueue
CosmosJobQueue = SampleJobQueue
