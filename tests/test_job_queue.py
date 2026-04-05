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
        task_type=TaskType.WORLD_MODEL_ROLLOUT,
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
            task_type=TaskType.WORLD_MODEL_ROLLOUT,
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
