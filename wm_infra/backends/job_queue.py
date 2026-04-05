"""In-process async job queue for sample production backends.

Originally introduced for Wan video generation, but now reusable for any backend
that wants queued sample execution (e.g. Genie temporal rollouts).
"""

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
    ) -> None:
        self._execute_fn = execute_fn
        self._store = store
        self._queue_name = queue_name
        self._max_queue_size = max_queue_size
        self._max_concurrent = max_concurrent

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

            entry.status = "running"
            entry.started_at = time.time()
            logger.info("%s worker %d executing job %s", self._queue_name, worker_id, entry.sample_id)

            try:
                queued_record = self._store.get(entry.sample_id)
                if queued_record is not None:
                    queued_record.status = SampleStatus.RUNNING
                    history = list(queued_record.runtime.get("status_history", []))
                    history.append({"status": SampleStatus.RUNNING.value, "timestamp": entry.started_at})
                    queued_record.runtime["status_history"] = history
                    queued_record.runtime["started_at"] = entry.started_at
                    self._store.put(queued_record)

                record = await self._execute_fn(entry.request, entry.sample_id)
                entry.status = record.status.value
                entry.completed_at = time.time()
                self._store.put(record)
                self._retire_job(entry.sample_id)
                logger.info("%s worker %d completed job %s → %s", self._queue_name, worker_id, entry.sample_id, entry.status)
            except Exception as exc:
                entry.status = "failed"
                entry.completed_at = time.time()
                entry.error = str(exc)
                logger.exception("%s worker %d failed job %s", self._queue_name, worker_id, entry.sample_id)
                failed_record = self._store.get(entry.sample_id)
                if failed_record is not None:
                    failed_record.status = SampleStatus.FAILED
                    history = list(failed_record.runtime.get("status_history", []))
                    history.append({"status": SampleStatus.FAILED.value, "timestamp": entry.completed_at})
                    failed_record.runtime["status_history"] = history
                    failed_record.runtime["completed_at"] = entry.completed_at
                    failed_record.runtime["queue_error"] = str(exc)
                    failed_record.metadata["runner_error"] = str(exc)
                    self._store.put(failed_record)
                self._retire_job(entry.sample_id)
            finally:
                self._queue.task_done()


WanJobQueue = SampleJobQueue
GenieJobQueue = SampleJobQueue
