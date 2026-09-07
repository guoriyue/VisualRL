"""The skip-backward decision must be unanimous across training ranks.

A backward pass fires cross-rank collectives — FSDP2 per-layer all-gather +
reduce-scatter, or DDP's gradient all-reduce. If one rank skips an all-filtered
(zero-advantage) microbatch while another rank runs it, those collectives
mismatch and the job DEADLOCKS (an unrecoverable NCCL hang). ``_all_ranks_have_work``
all-reduces the local ``has_work`` flag with MIN so every rank takes the SAME
branch: the microbatch runs only when ALL ranks have work. This spawns a real
gloo 2-rank group and asserts the agreed result is the logical AND of the ranks'
local flags.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.trainers._strategy_policies import free_port
from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _diffusion_rollout_batch,
    _EvaluatorAlgorithmFake,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.algorithms.logprob_mismatch import LogprobMismatchStats
from vrl.algorithms.types import InitialReplayStats, PolicyUpdateStats, TrainStepMetrics
from vrl.rollouts.batch import RolloutBatch
from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
from vrl.trainers.online import trainer as trainer_module
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import (
    OnlineTrainer,
    PhaseTimer,
    TrainingBatch,
    _all_ranks_have_work,
    _balanced_training_sample_batches,
    _distributed_initial_replay_stats,
    _distributed_parity_verdict,
    _ReplayMetrics,
)

# (rank0_has_work, rank1_has_work) -> the agreed result both ranks must return.
_CASES = {
    "both_have_work": ([True, True], True),
    "one_rank_empty": ([True, False], False),
    "both_empty": ([False, False], False),
}


def _rollout_batch(sample_count: int) -> RolloutBatch:
    return _diffusion_rollout_batch(
        rewards=torch.arange(sample_count, dtype=torch.float32),
        group_ids=torch.zeros(sample_count, dtype=torch.long),
        num_steps=1,
    )


def _run_rank(rank: int, world_size: int, port: int, local_flags: list[bool], q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        agreed = _all_ranks_have_work(local_flags[rank], torch.device("cpu"))
        q.put((rank, agreed))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("local_flags", "expected"),
    list(_CASES.values()),
    ids=list(_CASES),
)
def test_skip_backward_decision_is_unanimous(local_flags: list[bool], expected: bool) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, local_flags, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, agreed = q.get(timeout=50)
        results[rank] = agreed
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0
    # Both ranks must agree, and on the AND of the local flags.
    assert results[0] is expected
    assert results[1] is expected


def test_falls_back_to_local_without_process_group() -> None:
    assert _all_ranks_have_work(True, torch.device("cpu")) is True
    assert _all_ranks_have_work(False, torch.device("cpu")) is False


def test_zero_weight_initial_replay_is_fully_neutral() -> None:
    resolved, has_measurements = _distributed_initial_replay_stats(
        InitialReplayStats(
            clip_fraction=float("nan"),
            active_clip_fraction=float("inf"),
            logprob_abs_diff_max=float("inf"),
            finite=False,
        ),
        local_weight=0.0,
        device=torch.device("cpu"),
    )

    assert has_measurements is False
    assert resolved == InitialReplayStats()


def _run_parity_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        finite_result = _distributed_parity_verdict(
            local_finite=True,
            local_max_abs_diff=(0.1, 0.9)[rank],
            limit=0.5,
            device=torch.device("cpu"),
        )
        nonfinite_result = _distributed_parity_verdict(
            local_finite=(rank == 0),
            local_max_abs_diff=(0.1, 0.9)[rank],
            limit=1.0,
            device=torch.device("cpu"),
        )
        initial_replay, initial_has_measurements = _distributed_initial_replay_stats(
            InitialReplayStats(
                clip_fraction=(0.2, 0.8)[rank],
                active_clip_fraction=(0.1, 0.4)[rank],
                logprob_abs_diff_max=(0.1, 0.9)[rank],
            ),
            local_weight=(1.0, 3.0)[rank],
            device=torch.device("cpu"),
        )
        mixed_aggregate = _ReplayMetrics()
        if rank == 0:
            mixed_aggregate.add(
                TrainStepMetrics(
                    update=PolicyUpdateStats(
                        clip_fraction=0.2,
                        active_clip_fraction=0.1,
                    ),
                    logprob_mismatch=LogprobMismatchStats(
                        logprob_abs_diff_max=0.1,
                    ),
                ),
                weight=1.0,
                capture_initial_replay=True,
            )
        mixed_local, mixed_weight = mixed_aggregate.initial_replay_snapshot()
        mixed_rank_replay, mixed_has_measurements = _distributed_initial_replay_stats(
            mixed_local,
            local_weight=mixed_weight,
            device=torch.device("cpu"),
        )

        empty_local, empty_weight = _ReplayMetrics().initial_replay_snapshot()
        empty_rank_replay, empty_has_measurements = _distributed_initial_replay_stats(
            empty_local,
            local_weight=empty_weight,
            device=torch.device("cpu"),
        )
        q.put(
            (
                rank,
                finite_result,
                nonfinite_result,
                initial_replay,
                initial_has_measurements,
                mixed_rank_replay,
                mixed_has_measurements,
                empty_rank_replay,
                empty_has_measurements,
            )
        )
    finally:
        dist.destroy_process_group()


def test_parity_verdict_is_rank_consistent() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_parity_rank, args=(r, 2, port, q)) for r in range(2)]
    for process in procs:
        process.start()
    results = [q.get(timeout=50) for _ in range(2)]
    for process in procs:
        process.join(timeout=10)
        assert process.exitcode == 0

    for (
        _rank,
        finite_result,
        nonfinite_result,
        initial_replay,
        initial_has_measurements,
        mixed_rank_replay,
        mixed_has_measurements,
        empty_rank_replay,
        empty_has_measurements,
    ) in results:
        assert finite_result == pytest.approx((True, 0.9, False))
        assert nonfinite_result[0] is False
        assert nonfinite_result[1] == float("inf")
        assert nonfinite_result[2] is False
        assert initial_replay.clip_fraction == pytest.approx(0.65)
        assert initial_replay.active_clip_fraction == pytest.approx(0.325)
        assert initial_replay.logprob_abs_diff_max == pytest.approx(0.9)
        assert initial_replay.finite is True
        assert initial_has_measurements is True
        assert mixed_has_measurements is True
        assert mixed_rank_replay.finite is True
        assert mixed_rank_replay.clip_fraction == pytest.approx(0.2)
        assert mixed_rank_replay.active_clip_fraction == pytest.approx(0.1)
        assert mixed_rank_replay.logprob_abs_diff_max == pytest.approx(0.1)
        assert empty_has_measurements is False
        assert empty_rank_replay == InitialReplayStats()


def test_replay_planner_pads_to_global_slot_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trainer_module,
        "_distributed_max_int",
        lambda value, device: 8,
    )

    rank0_chunks = _balanced_training_sample_batches(
        [_rollout_batch(8)],
        [torch.ones(8)],
        samples_per_replay_batch=1,
        device=torch.device("cpu"),
    )
    rank1_chunks = _balanced_training_sample_batches(
        [_rollout_batch(3)],
        [torch.ones(3)],
        samples_per_replay_batch=1,
        device=torch.device("cpu"),
    )

    assert len(rank0_chunks) == 8
    assert len(rank1_chunks) == 8
    assert sum(batch.is_dummy for batch in rank0_chunks) == 0
    assert sum(batch.is_dummy for batch in rank1_chunks) == 5
    assert sum(batch.loss_weight for batch in rank0_chunks) == pytest.approx(1.0)
    assert sum(batch.loss_weight for batch in rank1_chunks) == pytest.approx(1.0)
    assert all(batch.loss_weight == 0.0 for batch in rank1_chunks[3:])
    assert all(torch.count_nonzero(batch.advantages) == 0 for batch in rank1_chunks[3:])


def _run_replay_planner_rank(
    rank: int,
    world_size: int,
    port: int,
    local_counts: list[int],
    q: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        sample_count = local_counts[rank]
        batches = _balanced_training_sample_batches(
            [_rollout_batch(sample_count)],
            [torch.ones(sample_count)],
            samples_per_replay_batch=1,
            device=torch.device("cpu"),
        )
        q.put(
            (
                rank,
                len(batches),
                sum(batch.is_dummy for batch in batches),
                sum(batch.loss_weight for batch in batches),
            )
        )
    finally:
        dist.destroy_process_group()


def test_replay_planner_slot_count_is_unanimous_under_gloo() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target=_run_replay_planner_rank, args=(r, 2, port, [8, 3], q))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, slot_count, dummy_count, weight_sum = q.get(timeout=50)
        results[rank] = (slot_count, dummy_count, weight_sum)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    assert results[0] == pytest.approx((8, 0, 1.0))
    assert results[1] == pytest.approx((8, 5, 1.0))


def _run_replay_loop_rank(
    rank: int,
    world_size: int,
    port: int,
    local_counts: list[int],
    q: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        evaluate_calls: list[int] = []
        backward_calls: list[float] = []

        class _Algorithm(_EvaluatorAlgorithmFake):
            required_signal_keys = ("log_prob",)
            required_data_keys: tuple[str, ...] = ()

            class _Config:
                global_std = False
                eps = 1e-8
                adv_clip_max = 5.0
                kl_coef = 0.0

            config = _Config()

            def compute_loss(self, inputs):
                signals = inputs.signals.primary
                loss = signals.log_prob.mean()
                return loss, TrainStepMetrics(
                    loss=float(loss.detach().item()),
                    policy_loss=float(loss.detach().item()),
                )

        class _Evaluator:
            def evaluate(self, model, batch, timestep_idx, **kw):
                del kw
                evaluate_calls.append(int(batch.rewards.shape[0]))
                log_prob = model.weight.reshape(()) * batch.rewards
                return _trajectory_signals(batch, log_prob, timestep_idx)

        model = nn.Linear(1, 1, bias=False)
        _stamp_model_precision(model)
        with torch.no_grad():
            model.weight.fill_(1.0)
        trainer = OnlineTrainer(
            algorithm=_Algorithm(),
            collector=CollectorControlFake(),
            evaluator=_Evaluator(),
            model=model,
            config=TrainerConfig(
                batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=8),
                timestep_fraction=1.0,
                drop_zero_advantage=False,
                output_dir="outputs/",
                optim=OptimConfig(lr=0.0),
                ema=EMAConfig(),
                debug=DebugConfig(),
            ),
            device="cpu",
        )

        def _record_backward(loss: torch.Tensor) -> None:
            backward_calls.append(float(loss.detach().item()))

        trainer._backward = _record_backward  # type: ignore[method-assign]
        trainer.begin_optimizer_update()
        sample_count = local_counts[rank]
        rollout_batch = _rollout_batch(sample_count)
        rollout_batch.rewards = rollout_batch.rewards + 1.0
        batch = TrainingBatch(
            iteration=object(),
            timer=PhaseTimer(enabled=False),
            batches=[rollout_batch],
            advantages=[torch.ones(sample_count)],
            group_size=float(sample_count),
            trained_prompt_num=1,
            adv_zero_rate=0.0,
            adv_saturation=0.0,
            pre_filter_reward_mean=0.0,
            pre_filter_reward_std=0.0,
            pre_filter_adv_mean=1.0,
            reward_components={},
        )
        trainer.backward_on_training_batch(batch, total_groups=1)
        q.put(
            (
                rank,
                len(evaluate_calls),
                len(backward_calls),
                len(trainer._update_agg_metrics.losses),
                sum(1 for value in backward_calls if value == 0.0),
            )
        )
    finally:
        dist.destroy_process_group()


def test_replay_loop_balances_evaluate_and_backward_counts_under_gloo() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target=_run_replay_loop_rank, args=(r, 2, port, [8, 3], q)) for r in range(2)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, evaluate_count, backward_count, metric_count, zero_backward_count = q.get(
            timeout=50,
        )
        results[rank] = (
            evaluate_count,
            backward_count,
            metric_count,
            zero_backward_count,
        )
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    assert results[0] == (8, 8, 8, 0)
    assert results[1] == (8, 8, 3, 5)
