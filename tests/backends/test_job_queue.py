"""Focused tests for queue retention and terminal-job cleanup."""

from __future__ import annotations

import asyncio

import pytest

from wm_infra.backends.job_queue import SampleJobQueue
from wm_infra.controlplane import ProduceSampleRequest, SampleRecord, SampleSpec, SampleStatus, TaskType


class _MemoryStore:
    def __init__(self, record: SampleRecord) -> None:
        self._records = {record.sample_id: record}

    def get(self, sample_id: str) -> SampleRecord | None:
        return self._records.get(sample_id)

    def put(self, record: SampleRecord) -> SampleRecord:
        self._records[record.sample_id] = record
        return record


@pytest.mark.asyncio
async def test_job_queue_retire_terminal_jobs():
    sample_id = "sample-queue"
    record = SampleRecord(
        sample_id=sample_id,
        task_type=TaskType.TEMPORAL_ROLLOUT,
        backend="rollout-engine",
        model="latent_dynamics",
        status=SampleStatus.QUEUED,
        sample_spec=SampleSpec(prompt="queue cleanup"),
    )
    store = _MemoryStore(record)

    async def execute_fn(request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        current = store.get(sample_id)
        assert current is not None
        return current.model_copy(update={"status": SampleStatus.SUCCEEDED})

    queue = SampleJobQueue(
        execute_fn=execute_fn,
        store=store,  # type: ignore[arg-type]
        queue_name="test",
        max_queue_size=4,
        max_concurrent=1,
    )

    queue.start()
    queue.submit(
        sample_id,
        ProduceSampleRequest(
            task_type=TaskType.TEMPORAL_ROLLOUT,
            backend="rollout-engine",
            model="latent_dynamics",
            sample_spec=SampleSpec(prompt="queue cleanup"),
            task_config={"num_steps": 1},
        ),
    )

    deadline = asyncio.get_event_loop().time() + 2.0
    while queue.get_job(sample_id) is not None and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)

    await queue.stop()

    assert queue.get_job(sample_id) is None
    assert queue.total_count == 0


@pytest.mark.asyncio
async def test_job_queue_batches_compatible_jobs_by_score():
    def _record(sample_id: str) -> SampleRecord:
        return SampleRecord(
            sample_id=sample_id,
            task_type=TaskType.TEXT_TO_VIDEO,
            backend="wan-video",
            model="wan2.2-t2v-A14B",
            status=SampleStatus.QUEUED,
            sample_spec=SampleSpec(prompt=sample_id),
        )

    store = _MemoryStore(_record("sample-a"))
    store.put(_record("sample-b"))
    store.put(_record("sample-c"))
    executed_batches: list[list[str]] = []

    async def execute_fn(request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        current = store.get(sample_id)
        assert current is not None
        return current.model_copy(update={"status": SampleStatus.SUCCEEDED})

    async def execute_many_fn(items: list[tuple[ProduceSampleRequest, str]]) -> list[SampleRecord]:
        executed_batches.append([sample_id for _request, sample_id in items])
        records: list[SampleRecord] = []
        for _request, sample_id in items:
            current = store.get(sample_id)
            assert current is not None
            records.append(current.model_copy(update={"status": SampleStatus.SUCCEEDED}))
        return records

    def batch_select_fn(reference_request: ProduceSampleRequest, candidate_request: ProduceSampleRequest) -> float | None:
        reference_bucket = reference_request.sample_spec.metadata.get("bucket")
        candidate_bucket = candidate_request.sample_spec.metadata.get("bucket")
        if reference_bucket != candidate_bucket:
            return None
        return 50.0

    queue = SampleJobQueue(
        execute_fn=execute_fn,
        execute_many_fn=execute_many_fn,
        batch_select_fn=batch_select_fn,
        store=store,  # type: ignore[arg-type]
        queue_name="test",
        max_queue_size=8,
        max_concurrent=1,
        max_batch_size=2,
        batch_wait_ms=20.0,
    )

    queue.start()
    queue.submit(
        "sample-a",
        ProduceSampleRequest(
            task_type=TaskType.TEXT_TO_VIDEO,
            backend="wan-video",
            model="wan2.2-t2v-A14B",
            sample_spec=SampleSpec(prompt="a", metadata={"bucket": "match"}),
        ),
    )
    queue.submit(
        "sample-b",
        ProduceSampleRequest(
            task_type=TaskType.TEXT_TO_VIDEO,
            backend="wan-video",
            model="wan2.2-t2v-A14B",
            sample_spec=SampleSpec(prompt="b", metadata={"bucket": "other"}),
        ),
    )
    queue.submit(
        "sample-c",
        ProduceSampleRequest(
            task_type=TaskType.TEXT_TO_VIDEO,
            backend="wan-video",
            model="wan2.2-t2v-A14B",
            sample_spec=SampleSpec(prompt="c", metadata={"bucket": "match"}),
        ),
    )

    deadline = asyncio.get_event_loop().time() + 2.0
    while queue.total_count > 0 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)

    await queue.stop()

    assert executed_batches
    assert executed_batches[0] == ["sample-a", "sample-c"]
