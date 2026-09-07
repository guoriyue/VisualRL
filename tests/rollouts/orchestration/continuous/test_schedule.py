"""Integration tests for the continuous rollout schedule."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.rollouts.orchestration.continuous._helpers import owner_snapshot
from vrl.generation.execution.types import StaleSlotDiscard
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.orchestration import (
    ContinuousRolloutSchedule,
    build_rollout_schedule,
)
from vrl.trainers.data.prompts import PromptExample
from vrl.trainers.strategy import SingleProcessStrategy, TrainingMemoryState


def _batch(prompts: list[str], group_size: int) -> RolloutBatch:
    batch_size = len(prompts) * group_size
    group_ids = torch.tensor(
        [idx for idx in range(len(prompts)) for _ in range(group_size)],
        dtype=torch.long,
    )
    return RolloutBatch(
        rewards=torch.arange(batch_size, dtype=torch.float32),
        group_ids=group_ids,
        # Test-fake provenance used only to verify finite lookahead ordering.
        context={"fixture_prompts": tuple(prompts)},
    )


class _Runtime:
    def __init__(self) -> None:
        self.current_policy_version = 0
        self.requires_driver_model_offload = False
        # Default False keeps every existing test on the draining barrier; the
        # non-draining tests flip it True to exercise the slot-backed path.
        self.supports_non_draining_weight_sync = False


class _Syncer:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.calls: list[dict[str, Any]] = []

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.calls.append(dict(state_dict))
        self.runtime.current_policy_version += 1

    async def pull(self) -> dict[str, Any]:
        return dict(self.calls[-1])

    @property
    def current_policy_version(self) -> int | None:
        # Mirrors RayRuntimeWeightSyncer's concrete version property.
        return self.runtime.current_policy_version


class _FailingPostTrainSyncer(_Syncer):
    def __init__(self, runtime: _Runtime) -> None:
        super().__init__(runtime)
        self.push_attempts = 0

    async def push(self, state_dict: dict[str, Any]) -> None:
        self.push_attempts += 1
        if self.push_attempts == 2:
            raise RuntimeError("worker install ACK mismatch")
        await super().push(state_dict)


class _Collector:
    def __init__(self, runtime: _Runtime) -> None:
        self.generation_runtime = runtime
        self.requires_generation_offload_before_reward = False
        self.requires_driver_model_offload_for_reward = False
        self.supports_reward_generation_overlap = False
        self.supports_continuous_reward_execution = True
        self.calls: list[dict[str, Any]] = []
        self.activation_calls = 0
        self.offload_calls = 0
        self.shutdown_calls = 0
        self.shutdown_failures = 0

    async def collect_unscored(self, inputs: Any, **kwargs: Any) -> RolloutBatch:
        prompts = [getattr(item, "prompt", item) for item in inputs]
        self.calls.append({"prompts": prompts, **dict(kwargs)})
        return _batch(prompts, int(kwargs["group_size"]))

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        return list(pendings)

    async def activate_generation_runtime(self) -> None:
        self.activation_calls += 1

    async def offload_generation_runtime_memory(self) -> None:
        self.offload_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("collector cleanup failed")


def _continuous_config(**continuous: Any) -> SimpleNamespace:
    defaults = {
        "max_inflight_groups": 1,
        "max_ready_bytes_mb": 8192,
        "max_stale_policy_versions": 1,
        "wait_timeout_s": 5.0,
        "queue_poll_interval_s": 0.001,
        "fail_fast_errors": 3,
    }
    defaults.update(continuous)
    return SimpleNamespace(
        schedule_mode="continuous",
        continuous=SimpleNamespace(**defaults),
    )


def _iteration_stat(iteration: Any, name: str) -> float:
    return iteration.stats.as_phase_dict()[name]


def _build(
    config: SimpleNamespace,
    collector: _Collector,
    syncer: _Syncer | None,
    *,
    algorithm_tolerates_off_policy_staleness: bool = True,
    initially_initialized: bool = False,
):
    initialized = {"value": initially_initialized}

    def _set(value: bool) -> None:
        initialized["value"] = bool(value)

    model = nn.Linear(1, 1)
    return build_rollout_schedule(
        config,
        collector=collector,
        strategy=SingleProcessStrategy(),
        training_state_getter=lambda: TrainingMemoryState(
            model=model,
            ref_model=None,
            optimizer=None,
            ema=None,
            grad_scaler=None,
            device=torch.device("cpu"),
        ),
        weight_syncer=syncer,
        sync_state_getter=(lambda: {"w": 1}) if syncer is not None else None,
        weights_initialized=lambda: initialized["value"],
        set_weights_initialized=_set,
        algorithm_tolerates_off_policy_staleness=(algorithm_tolerates_off_policy_staleness),
    )


async def _snapshot_when(
    schedule: ContinuousRolloutSchedule,
    condition: Callable[[Any], bool],
    *,
    timeout_s: float = 5.0,
) -> Any:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        snapshot = await owner_snapshot(schedule._owner)
        if condition(snapshot):
            return snapshot
        if loop.time() >= deadline:
            raise AssertionError("owner snapshot condition not reached before timeout")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_shutdown_joins_owner_and_is_idempotent() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    schedule = _build(
        _continuous_config(),
        collector,
        _Syncer(runtime),
    )

    await schedule.next_iteration(["p0"], group_size=1)
    running = await owner_snapshot(schedule._owner)
    assert running.producer_state is not None
    assert running.producer_state.running is True

    await asyncio.gather(schedule.shutdown(), schedule.shutdown())
    await schedule.shutdown()

    assert collector.shutdown_calls == 1
    assert schedule._owner._stopped.is_set()
    thread = schedule._owner._thread
    assert thread is not None and not thread.is_alive()


@pytest.mark.asyncio
async def test_shutdown_failure_retries_cleanup_before_closing_owner() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.shutdown_failures = 1
    schedule = _build(_continuous_config(), collector, _Syncer(runtime))

    await schedule.next_iteration(["p0"], group_size=1)

    with pytest.raises(RuntimeError, match="collector cleanup failed"):
        await schedule.shutdown()

    thread = schedule._owner._thread
    assert collector.shutdown_calls == 1
    assert schedule._owner._closed is False
    assert thread is not None and thread.is_alive()

    await schedule.shutdown()
    await schedule.shutdown()

    assert collector.shutdown_calls == 2
    assert schedule._owner._closed is True
    assert thread is not None and not thread.is_alive()


class _SlowCollector(_Collector):
    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        await asyncio.sleep(0.02)
        return await super().collect_unscored(prompts, **kwargs)


@pytest.mark.asyncio
async def test_owner_production_advances_while_trainer_loop_is_blocked() -> None:
    runtime = _Runtime()
    collector = _SlowCollector(runtime)
    schedule = _build(_continuous_config(), collector, _Syncer(runtime))

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=["p1"],
        )
        before = await owner_snapshot(schedule._owner)
        assert before.producer_state is not None
        assert before.producer_state.submitted_count == 2

        # This blocks the trainer asyncio loop exactly like synchronous backward.
        time.sleep(0.12)

        after = await owner_snapshot(schedule._owner)
        assert after.producer_state is not None
        assert after.producer_state.tick_count > before.producer_state.tick_count
        assert after.producer_state.submitted_count == before.producer_state.submitted_count
        assert after.producer_state.completed_count > before.producer_state.completed_count
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_initialized_runtime_does_not_receive_redundant_initial_push() -> None:
    runtime = _Runtime()
    runtime.current_policy_version = 7
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(),
        collector,
        syncer,
        initially_initialized=True,
    )

    try:
        iteration = await schedule.next_iteration(["p0"], group_size=1)

        assert _iteration_stat(iteration, "continuous.rollout_policy_version") == 7.0
        assert runtime.current_policy_version == 7
        assert syncer.calls == []
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_reset_reuses_committed_runtime_weights_without_version_bump() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        first = await schedule.next_iteration(["p0"], group_size=1)
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0
        assert len(syncer.calls) == 1

        schedule.reset()
        second = await schedule.next_iteration(["p0"], group_size=1)

        assert _iteration_stat(second, "continuous.rollout_policy_version") == 1.0
        assert len(syncer.calls) == 1
        assert collector.shutdown_calls == 0
    finally:
        await schedule.shutdown()


def test_continuous_rejects_stale_window_for_intolerant_algorithm() -> None:
    """A likelihood-free algorithm + max_stale>0 must fail fast as unsound."""
    runtime = _Runtime()
    with pytest.raises(ValueError, match="likelihood-free"):
        _build(
            _continuous_config(max_stale_policy_versions=1),
            _Collector(runtime),
            _Syncer(runtime),
            algorithm_tolerates_off_policy_staleness=False,
        )


def test_continuous_rejects_zero_window_before_algorithm_gate() -> None:
    """Zero-window execution belongs to strict_on_policy, not continuous."""
    runtime = _Runtime()
    with pytest.raises(ValueError, match=r"max_stale_policy_versions >= 1"):
        _build(
            _continuous_config(max_stale_policy_versions=0),
            _Collector(runtime),
            _Syncer(runtime),
            algorithm_tolerates_off_policy_staleness=False,
        )


@pytest.mark.asyncio
async def test_continuous_drains_full_homogeneous_iteration() -> None:
    """A homogeneous continuous iteration drains the full set: one policy version for rollout and
    consume, zero staleness, distinct group ids, and the item-age / ready-groups / queue-wait
    phases all reported.
    """
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        iteration = await schedule.next_iteration(["p0", "p1"], group_size=2)

        # Full set, one fresh policy version, distinct group ids 0..1.
        phases = iteration.stats.as_phase_dict()
        assert phases["continuous.rollout_policy_version"] == 1.0
        assert phases["continuous.consume_policy_version"] == 1.0
        assert phases["continuous.stale_policy_versions"] == 0.0
        assert phases["continuous.item_age_s"] >= 0.0
        assert phases["continuous.ready_groups_at_demand"] >= 0.0
        assert len(iteration.batches) == 2
        assert sum(batch.rewards.numel() for batch in iteration.batches) == 4
        group_ids = sorted(int(b.group_ids[0]) for b in iteration.batches)
        assert group_ids == [0, 1]
        assert "continuous.queue_wait_s" in phases
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_weight_sync_barrier_advances_version_and_resumes() -> None:
    """``after_train_step`` performs exactly one weight sync, bumps the runtime policy version and
    unpauses admission; the next iteration is produced and consumed at the new version with
    zero staleness.
    """
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        first = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0

        sync_calls_before = len(syncer.calls)
        await schedule.after_train_step()
        # Barrier performed exactly one post-train sync and resumed admission.
        assert len(syncer.calls) == sync_calls_before + 1
        snapshot = await owner_snapshot(schedule._owner)
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
        assert runtime.current_policy_version == 2

        second = await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert _iteration_stat(second, "continuous.rollout_policy_version") == 2.0
        assert _iteration_stat(second, "continuous.consume_policy_version") == 2.0
        assert _iteration_stat(second, "continuous.stale_policy_versions") == 0.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_partial_commit_failure_closes_admission_and_runtime() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _FailingPostTrainSyncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        await schedule.next_iteration(["p0"], group_size=1)

        with pytest.raises(RuntimeError, match="worker install ACK mismatch"):
            await schedule.after_train_step()

        calls_after_failure = len(collector.calls)
        failed = await owner_snapshot(schedule._owner)
        assert failed.producer_state is None
        assert failed.queue_stats == {}
        assert "worker install ACK mismatch" in str(failed.terminal_error)
        assert collector.shutdown_calls == 1

        await asyncio.sleep(0.02)
        assert len(collector.calls) == calls_after_failure
        with pytest.raises(RuntimeError, match="owner has failed"):
            await schedule.next_iteration(["p0"], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_draining_sync_finishes_the_active_prompt_batch_before_commit() -> None:
    """A draining backend completes every finite lookahead slot at one version."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(max_stale_policy_versions=1),
        collector,
        syncer,
    )

    try:
        first = await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert _iteration_stat(first, "continuous.rollout_policy_version") == 1.0
        await schedule.after_train_step()
        assert runtime.current_policy_version == 2
        after = await owner_snapshot(schedule._owner)
        assert after.queue_stats["ready_items"] == 2

        second = await schedule.next_iteration(["p2", "p3"], group_size=2)
        assert _iteration_stat(second, "continuous.rollout_policy_version") == 1.0
        assert _iteration_stat(second, "continuous.stale_policy_versions") == 1.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_queue_capacity_fits_the_finite_prompt_batch() -> None:
    """The internal queue must hold every group required by one iteration."""
    runtime = _Runtime()
    collector = _Collector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        iteration = await schedule.next_iteration(["a", "b", "c"], group_size=2)
        assert len(iteration.batches) == 3
        assert sum(batch.rewards.numel() for batch in iteration.batches) == 6
        assert sorted(int(b.group_ids[0]) for b in iteration.batches) == [0, 1, 2]
    finally:
        await schedule.shutdown()


def test_rejects_runtime_that_requires_driver_model_offload() -> None:
    runtime = _Runtime()
    runtime.requires_driver_model_offload = True
    with pytest.raises(RuntimeError, match="requires driver model offload"):
        _build(_continuous_config(), _Collector(runtime), None)


def test_rejects_mid_iteration_reward_offload() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.requires_generation_offload_before_reward = True
    with pytest.raises(RuntimeError, match=r"does not offload.*mid-iteration"):
        _build(_continuous_config(), collector, _Syncer(runtime))


def test_rejects_reward_scoring_on_trainer_gpu() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.requires_driver_model_offload_for_reward = True

    with pytest.raises(RuntimeError, match="trainer GPU while backward overlaps"):
        _build(_continuous_config(), collector, _Syncer(runtime))


def test_rejects_unverified_external_reward_accelerator() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    collector.supports_continuous_reward_execution = False

    with pytest.raises(RuntimeError, match="verified reward accelerator isolation"):
        _build(_continuous_config(), collector, _Syncer(runtime))


class _FailingCollector(_Collector):
    def __init__(self, runtime: _Runtime, message: str = "boom") -> None:
        super().__init__(runtime)
        self.message = message

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        raise RuntimeError(self.message)


@pytest.mark.asyncio
async def test_persistent_producer_failure_fails_fast_with_root_cause() -> None:
    # Every generation fails. The consumer must surface the producer's root
    # cause well before the (long) wait timeout, not an opaque timeout.
    runtime = _Runtime()
    collector = _FailingCollector(runtime, message="reward model OOM")
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    try:
        with pytest.raises(RuntimeError, match="reward model OOM") as excinfo:
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert "failing every generation" in str(excinfo.value)
    finally:
        await schedule.shutdown()


class _RewardFailingCollector(_Collector):
    """Generation succeeds; reward scoring always fails."""

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        raise RuntimeError("reward model exploded")


@pytest.mark.asyncio
async def test_reward_failure_fails_fast_and_never_reaches_queue() -> None:
    # Reward scoring (not generation) fails persistently: the consumer must
    # surface that root cause and the ready queue must stay empty.
    runtime = _Runtime()
    collector = _RewardFailingCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    try:
        with pytest.raises(RuntimeError, match="reward model exploded"):
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert (await owner_snapshot(schedule._owner)).queue_stats == {}
    finally:
        await schedule.shutdown()


class _GatedScoreCollector(_Collector):
    """Reward scoring blocks until the test opens the gate."""

    def __init__(self, runtime: _Runtime) -> None:
        super().__init__(runtime)
        self.allow_score = threading.Event()
        self.allow_score.set()
        self.score_blocked = threading.Event()
        self.block_after_scores = 0
        self.score_calls = 0

    async def score_rollouts(self, pendings: Any) -> list[RolloutBatch]:
        self.score_calls += 1
        if self.score_calls > self.block_after_scores and not self.allow_score.is_set():
            self.score_blocked.set()
            await asyncio.to_thread(self.allow_score.wait)
        return await super().score_rollouts(pendings)


@pytest.mark.asyncio
async def test_weight_sync_waits_for_inflight_reward() -> None:
    # after_train_step must drain in-flight generation+reward before pushing
    # weights; syncing earlier would mix two policies inside one request.
    """Checks weight sync waits for in-flight reward scoring."""
    runtime = _Runtime()
    collector = _GatedScoreCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        collector.block_after_scores = 2
        collector.score_blocked.clear()
        collector.allow_score.clear()
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert await asyncio.to_thread(collector.score_blocked.wait, 5.0)

        sync_calls_before = len(syncer.calls)
        barrier = asyncio.create_task(schedule.after_train_step())
        await asyncio.sleep(0.05)
        # Reward still in flight: admission paused, sync not yet performed.
        blocked = await owner_snapshot(schedule._owner)
        assert blocked.producer_state is not None
        assert blocked.producer_state.paused_for_weight_sync is True
        assert len(syncer.calls) == sync_calls_before
        assert not barrier.done()

        collector.allow_score.set()
        await asyncio.wait_for(barrier, 5.0)
        assert len(syncer.calls) == sync_calls_before + 1
        resumed = await owner_snapshot(schedule._owner)
        assert resumed.producer_state is not None
        assert resumed.producer_state.paused_for_weight_sync is False
    finally:
        collector.allow_score.set()
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_draining_barrier_reports_mode_zero() -> None:
    """Default (no versioned slots): the barrier mode metric is 0 (draining)."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(["p0", "p1"], group_size=2)
        phases = await schedule.after_train_step()
        assert phases.as_phase_dict()["continuous.weight_sync_barrier_mode"] == 0.0
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_non_draining_sync_skips_inflight_wait() -> None:
    # The whole point of versioned slots: when the runtime advertises non-draining
    # support, after_train_step must NOT wait for in-flight generation/reward — it
    # syncs and returns while the gated reward is still blocked (the in-flight
    # request keeps its own version's slot and finishes concurrently).
    """Checks non-draining sync does not drain in-flight work."""
    runtime = _Runtime()
    runtime.supports_non_draining_weight_sync = True
    collector = _GatedScoreCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(_continuous_config(), collector, syncer)

    try:
        collector.block_after_scores = 2
        collector.score_blocked.clear()
        collector.allow_score.clear()
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p2", "p3"],
        )
        assert await asyncio.to_thread(collector.score_blocked.wait, 5.0)

        sync_calls_before = len(syncer.calls)
        # Must complete WITHOUT opening the reward gate (contrast with the draining
        # canary test, where this would block until allow_score.set()).
        phases = await asyncio.wait_for(schedule.after_train_step(), 5.0)

        assert phases.as_phase_dict()["continuous.weight_sync_barrier_mode"] == 1.0
        assert len(syncer.calls) == sync_calls_before + 1
        snapshot = await owner_snapshot(schedule._owner)
        assert snapshot.producer_state is not None
        assert snapshot.producer_state.paused_for_weight_sync is False
    finally:
        collector.allow_score.set()
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_three_gas2_updates_consume_exact_finite_lookahead_sequence() -> None:
    """Three GAS2 updates consume each announced prompt batch once within stale bound."""
    runtime = _Runtime()
    runtime.supports_non_draining_weight_sync = True
    collector = _Collector(runtime)
    schedule = _build(
        _continuous_config(
            max_inflight_groups=6,
            max_stale_policy_versions=1,
        ),
        collector,
        _Syncer(runtime),
    )
    prompts = [[f"update-{update}-micro-{micro}"] for update in range(3) for micro in range(2)]
    iterations = []
    sync_stats = []

    try:
        for index, current_prompts in enumerate(prompts):
            next_prompts = prompts[index + 1] if index + 1 < len(prompts) else None
            iterations.append(
                await schedule.next_iteration(
                    current_prompts,
                    group_size=4,
                    next_prompts=next_prompts,
                ),
            )
            if index % 2 == 1:
                sync_stats.append(await schedule.after_train_step())

        assert [
            [batch.context["fixture_prompts"][0] for batch in iteration.batches]
            for iteration in iterations
        ] == prompts
        assert [
            _iteration_stat(iteration, "continuous.rollout_policy_version")
            for iteration in iterations
        ] == [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        assert [
            _iteration_stat(iteration, "continuous.consume_policy_version")
            for iteration in iterations
        ] == [
            1,
            1,
            2,
            2,
            3,
            3,
        ]
        assert [
            _iteration_stat(iteration, "continuous.stale_policy_versions")
            for iteration in iterations
        ] == [
            0,
            0,
            1,
            0,
            1,
            0,
        ]
        assert [
            iteration.stats.as_phase_dict()["continuous.lookahead_requested"]
            for iteration in iterations
        ] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        assert [
            (call["prompts"], call["group_size"], call["policy_version"])
            for call in collector.calls
        ] == [
            (current_prompts, 4, rollout_version)
            for current_prompts, rollout_version in zip(
                prompts,
                [1, 1, 1, 2, 2, 3],
                strict=True,
            )
        ]
        assert [
            stats.as_phase_dict()["continuous.weight_sync_barrier_mode"] for stats in sync_stats
        ] == [1.0, 1.0, 1.0]
        assert runtime.current_policy_version == 4

        snapshot = await owner_snapshot(schedule._owner)
        assert snapshot.producer_state is not None
        assert snapshot.queue_stats["ready_items"] == 0
    finally:
        await schedule.shutdown()


class _StaleSlotCollector(_Collector):
    """Generation always hits an evicted trainable-state slot (non-draining sync).

    Mirrors what the Ray executor raises when a request outlives its worker's slot
    window: a typed StaleSlotDiscard, NOT a generation failure.
    """

    async def collect_unscored(self, prompts: Any, **kwargs: Any) -> RolloutBatch:
        raise StaleSlotDiscard("trainable-state slot evicted for policy_version=1")


@pytest.mark.asyncio
async def test_stale_slot_discard_fails_the_fixed_version_prompt_batch() -> None:
    """A prompt batch cannot replace one slot without violating version identity."""
    runtime = _Runtime()
    runtime.supports_non_draining_weight_sync = True
    collector = _StaleSlotCollector(runtime)
    syncer = _Syncer(runtime)
    schedule = _build(
        _continuous_config(wait_timeout_s=30.0, fail_fast_errors=2),
        collector,
        syncer,
    )

    try:
        with pytest.raises(RuntimeError, match="producer control loop failed") as exc_info:
            await schedule.next_iteration(["p0", "p1"], group_size=2)
        assert exc_info.value.__cause__ is not None
        assert "lost its fixed policy-version slot" in str(exc_info.value.__cause__)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_installs_the_next_prompt_batch() -> None:
    """The producer advances to exactly the prompt batch announced as lookahead."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=2,
            next_prompts=["p0", "p2"],
        )
        await schedule.next_iteration(["p0", "p2"], group_size=2)
        assert (await owner_snapshot(schedule._owner)).prompts == ("p0", "p2")
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_freezes_runtime_debug_at_generation_time() -> None:
    runtime = _Runtime()
    collector = _Collector(runtime)
    schedule = _build(_continuous_config(), collector, _Syncer(runtime))

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            runtime_debug=True,
            next_prompts=["p1"],
        )
        await schedule.next_iteration(
            ["p1"],
            group_size=1,
            runtime_debug=False,
        )
        assert [call["runtime_debug"] for call in collector.calls] == [True, True]
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_mismatch_fails_instead_of_training_the_wrong_prompt_batch() -> None:
    """The trainer must consume the exact prompt batch it announced as lookahead."""
    runtime = _Runtime()
    schedule = _build(
        _continuous_config(
            max_stale_policy_versions=1,
            max_inflight_groups=2,
            wait_timeout_s=1.0,
        ),
        _Collector(runtime),
        _Syncer(runtime),
    )

    try:
        await schedule.next_iteration(
            ["p0", "p1"],
            group_size=1,
            next_prompts=["p2", "p3"],
        )
        with pytest.raises(RuntimeError, match="lookahead prompt batch does not match"):
            await schedule.next_iteration(["p2", "different"], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_group_size_mismatch_fails_before_consumption() -> None:
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=["p1"],
        )
        with pytest.raises(RuntimeError, match="lookahead group size does not match"):
            await schedule.next_iteration(["p1"], group_size=2)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_accepts_identical_prompt_with_non_scalar_metadata() -> None:
    """The sampler may present the same object whose structural equality is invalid."""
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))
    prompt = PromptExample(
        prompt="p1",
        metadata={"embedding": torch.tensor([1.0, 2.0])},
    )

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=[prompt],
        )
        await schedule.next_iteration([prompt], group_size=1)
    finally:
        await schedule.shutdown()


@pytest.mark.asyncio
async def test_lookahead_fails_closed_when_prompt_equality_is_non_scalar() -> None:
    runtime = _Runtime()
    schedule = _build(_continuous_config(), _Collector(runtime), _Syncer(runtime))
    installed = PromptExample(
        prompt="p1",
        metadata={"embedding": torch.tensor([1.0, 2.0])},
    )
    presented = PromptExample(
        prompt="p1",
        metadata={"embedding": torch.tensor([1.0, 2.0])},
    )

    try:
        await schedule.next_iteration(
            ["p0"],
            group_size=1,
            next_prompts=[installed],
        )
        with pytest.raises(RuntimeError, match="lookahead prompt batch does not match"):
            await schedule.next_iteration([presented], group_size=1)
    finally:
        await schedule.shutdown()
