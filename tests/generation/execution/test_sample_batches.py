"""Tests for generation executor sample batch helpers."""

from __future__ import annotations


def test_sample_batch_plan_prompt_major() -> None:
    """Checks build prompt batches prompt major."""
    from vrl.generation.execution.sample_batches import GenerationSampleBatch

    batches = GenerationSampleBatch.plan(
        2,
        samples_per_prompt=5,
        max_samples_per_batch=2,
    )

    got = [(batch.prompt_index, batch.sample_start, batch.sample_count) for batch in batches]
    assert got == [
        (0, 0, 2),
        (0, 2, 2),
        (0, 4, 1),
        (1, 0, 2),
        (1, 2, 2),
        (1, 4, 1),
    ]


def test_run_sample_batches_with_oom_retry_splits_until_success() -> None:
    """An OOM on a 5-sample batch halves recursively (2, then 3 -> 1 + 2) and the results come
    back in sample order.
    """
    from vrl.generation.execution.sample_batches import (
        GenerationSampleBatch,
        run_sample_batches_with_oom_retry,
    )

    seen: list[tuple[int, int]] = []

    def run_one(batch: GenerationSampleBatch) -> int:
        seen.append((batch.sample_start, batch.sample_count))
        if batch.sample_count > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return batch.sample_count

    results = run_sample_batches_with_oom_retry(
        [
            GenerationSampleBatch(
                prompt_index=0,
                sample_start=0,
                sample_count=5,
            )
        ],
        run_one,
    )

    assert results == [2, 1, 2]
    assert seen == [(0, 5), (0, 2), (2, 3), (2, 1), (3, 2)]
