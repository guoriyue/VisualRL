"""Failure-path tests for the strict on-policy rollout schedule.

The risk these guard: if collection raises mid-rollout, the driver model may be
restored only after rollout runtime memory was released successfully. Otherwise
both models can become resident after an incomplete GPU handoff. We also pin
that weight sync happens in after_train_step, never before collect.

These drive the REAL RolloutRuntimeCoordinator (the phase-transition ordering
under test lives in its ``rollout_phase`` manager) over recording strategy and
collector fakes; only the transition primitives are faked, never the ordering.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vrl.rollouts.orchestration.rollout_runtime import (
    RolloutPhaseCleanupError,
    RolloutRuntimeCoordinator,
)
from vrl.rollouts.orchestration.strict_on_policy import StrictOnPolicyRolloutSchedule


class _Strategy:
    """Trainer-side parking fake recording call order."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.fail_restore = False

    def validate_training_state_parking(self) -> None:
        self.calls.append("validate_training_state_parking")

    def park_training_state(self, _state: Any) -> None:
        self.calls.append("park_trainer")

    def restore_training_state(self, _state: Any) -> None:
        self.calls.append("restore_trainer")
        if self.fail_restore:
            raise RuntimeError("trainer restore blew up")


class _Collector:
    """RolloutCollectorControl fake whose generation raises mid-rollout."""

    requires_generation_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    supports_reward_generation_overlap = False
    supports_continuous_reward_execution = True

    def __init__(
        self,
        calls: list[str],
        *,
        collect_raises: bool = True,
        trainer_shares_gpu: bool = True,
    ) -> None:
        self.calls = calls
        self.collect_raises = collect_raises
        self.fail_rollout_offload = False
        # requires_driver_model_offload drives whether the coordinator parks
        # the trainer at phase entry (the shared-GPU topology fact).
        self.generation_runtime = SimpleNamespace(
            requires_driver_model_offload=trainer_shares_gpu,
            current_policy_version=0,
        )

    async def activate_generation_runtime(self) -> None:
        self.calls.append("activate_rollout")

    async def offload_generation_runtime_memory(self) -> None:
        self.calls.append("offload_rollout")
        if self.fail_rollout_offload:
            raise RuntimeError("rollout offload blew up")

    async def shutdown(self) -> None:
        self.calls.append("collector_shutdown")

    async def collect_unscored(self, *_args: object, **_kwargs: object) -> object:
        if self.collect_raises:
            raise RuntimeError("collect blew up")
        raise AssertionError("collect_unscored should not run for empty prompts")


class _Syncer:
    current_policy_version = 1

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def push(self, _prepared: Any) -> None:
        self.calls.append("sync_weights_after_train")


def _schedule(
    collector: _Collector,
    strategy: _Strategy,
    *,
    calls: list[str],
    with_syncer: bool = False,
) -> StrictOnPolicyRolloutSchedule:
    syncer = _Syncer(calls) if with_syncer else None
    lifecycle = RolloutRuntimeCoordinator(
        collector=collector,
        strategy=strategy,
        training_state_getter=lambda: object(),
        weight_syncer=syncer,
        sync_state_getter=(lambda: {}) if with_syncer else None,
        weights_initialized=lambda: True,
        set_weights_initialized=lambda _value: None,
    )
    return StrictOnPolicyRolloutSchedule(lifecycle=lifecycle)


@pytest.mark.asyncio
async def test_cleanup_runs_when_collect_raises() -> None:
    calls: list[str] = []
    schedule = _schedule(_Collector(calls), _Strategy(calls), calls=calls)

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    # Both cleanup hooks ran despite the collect failure, in handoff order.
    assert "activate_rollout" in calls
    assert "offload_rollout" in calls
    assert "restore_trainer" in calls
    assert calls.index("park_trainer") < calls.index("activate_rollout")
    assert calls.index("offload_rollout") < calls.index("restore_trainer")
    # Weights were never synced as part of a failed collection.
    assert "sync_weights_after_train" not in calls


@pytest.mark.asyncio
async def test_driver_not_restored_when_not_parked() -> None:
    """A topology that never parked the trainer must not restore it either."""
    calls: list[str] = []
    collector = _Collector(calls, trainer_shares_gpu=False)
    schedule = _schedule(collector, _Strategy(calls), calls=calls)

    with pytest.raises(RuntimeError, match="collect blew up"):
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert "park_trainer" not in calls
    assert "offload_rollout" in calls
    assert "restore_trainer" not in calls


@pytest.mark.asyncio
async def test_failed_rollout_offload_keeps_trainer_parked() -> None:
    calls: list[str] = []
    collector = _Collector(calls)
    collector.fail_rollout_offload = True
    strategy = _Strategy(calls)
    # A restore failure would be observable if the phase manager incorrectly
    # attempted to wake the trainer after an incomplete rollout handoff.
    strategy.fail_restore = True
    schedule = _schedule(collector, strategy, calls=calls)

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "rollout offload blew up"
    assert raised.value.__cause__ is raised.value.root_cause
    assert "restore_trainer" not in calls


@pytest.mark.asyncio
async def test_offload_failure_after_collection_keeps_trainer_parked() -> None:
    calls: list[str] = []
    collector = _Collector(calls, collect_raises=False)
    collector.fail_rollout_offload = True
    schedule = _schedule(collector, _Strategy(calls), calls=calls)

    with pytest.raises(RuntimeError, match="rollout offload blew up"):
        await schedule.next_iteration([], group_size=2)

    assert "offload_rollout" in calls
    assert "restore_trainer" not in calls


@pytest.mark.asyncio
async def test_restore_failure_is_reported_after_successful_rollout_offload() -> None:
    calls: list[str] = []
    strategy = _Strategy(calls)
    strategy.fail_restore = True
    schedule = _schedule(_Collector(calls), strategy, calls=calls)

    with pytest.raises(RolloutPhaseCleanupError) as raised:
        await schedule.next_iteration(["a prompt"], group_size=2)

    assert str(raised.value.root_cause) == "collect blew up"
    assert str(raised.value.cleanup_error) == "trainer restore blew up"
    assert calls.index("offload_rollout") < calls.index("restore_trainer")


@pytest.mark.asyncio
async def test_weight_sync_only_in_after_train_step() -> None:
    """Weight sync belongs to after_train_step, not the collect path."""
    calls: list[str] = []
    collector = _Collector(calls, collect_raises=False)
    schedule = _schedule(collector, _Strategy(calls), calls=calls, with_syncer=True)

    await schedule.next_iteration([], group_size=2)
    assert "sync_weights_after_train" not in calls

    await schedule.after_train_step()
    assert calls.count("sync_weights_after_train") == 1
