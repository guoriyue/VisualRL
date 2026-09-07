"""OnlineTrainer resume: optimizer/EMA/rollout-init state restore and pre-collect driver-weight push."""

from __future__ import annotations

import pytest

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _algorithm_inputs,
    _diffusion_rollout_batch,
    _EvaluatorAlgorithmFake,
    _stamp_model_precision,
    _trajectory_signals,
)
from vrl.config.precision import RolePrecision


class TestOnlineTrainerResumeState:
    """Groups tests for online trainer resume state."""

    def test_load_state_dict_accepts_legacy_total_keys(self) -> None:
        """Checkpoints written before the totals were dropped still resume."""

        source = _make_resume_trainer()
        source.state.step = 3
        source.state.global_step = 5

        state = source.state_dict()
        state.update({"total_reward": 99.0, "total_loss": 101.0})
        restored = _make_resume_trainer()
        restored.load_state_dict(state, strict=True)

        assert restored.state.step == 3
        assert restored.state.global_step == 5

    def test_load_state_dict_initializes_and_restores_optimizer_state(self) -> None:
        """``load_state_dict`` on a fresh trainer creates the optimizer on demand and restores its
        Adam moments together with the step counters.
        """
        import torch

        source = _make_resume_trainer()
        optimizer = source._ensure_optimizer()
        loss = source.model(torch.ones(1, 1)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        source.state.step = 3
        source.state.global_step = 5
        state = source.state_dict()

        restored = _make_resume_trainer()
        restored.load_state_dict(state, strict=True)

        assert restored.state.step == 3
        assert restored.state.global_step == 5
        assert restored._optimizer is not None
        assert _adam_exp_avg_values(restored._optimizer) == pytest.approx(
            _adam_exp_avg_values(optimizer),
        )

    def test_strict_resume_rejects_master_state_for_plain_optimizer(self) -> None:
        source = _make_resume_trainer()
        source._ensure_optimizer()
        state = source.state_dict()
        state["optimizer"]["fp32_master_weights"] = {
            "version": 1,
            "parameters": [],
        }

        with pytest.raises(ValueError, match="master-weight state does not match"):
            _make_resume_trainer().load_state_dict(state, strict=True)

    def test_load_state_dict_initializes_and_restores_ema_state(self) -> None:
        """``load_state_dict`` creates the EMA on demand when EMA is enabled and restores its
        shadow parameters.
        """
        import torch

        source = _make_resume_trainer(ema=True)
        ema = source._ensure_ema()
        assert ema is not None
        ema.ema_parameters[0].fill_(7.0)
        state = source.state_dict()

        restored = _make_resume_trainer(ema=True)
        restored.load_state_dict(state, strict=True)

        assert restored._ema is not None
        assert torch.equal(
            restored._ema.ema_parameters[0],
            torch.full_like(restored._ema.ema_parameters[0], 7.0),
        )

    def test_state_dict_includes_initial_ema_state_before_first_step(self) -> None:
        """Checks zero-step EMA checkpoints can resume strictly."""
        source = _make_resume_trainer(ema=True)

        state = source.state_dict()
        restored = _make_resume_trainer(ema=True)
        restored.load_state_dict(state, strict=True)

        assert "ema" in state
        assert state["ema"]["num_updates"] == 0
        assert restored._ema is not None

    def test_load_state_dict_rejects_ema_state_when_ema_is_disabled(self) -> None:
        source = _make_resume_trainer(ema=True)
        source._ensure_ema()
        state = source.state_dict()

        restored = _make_resume_trainer(ema=False)
        with pytest.raises(ValueError, match="EMA state"):
            restored.load_state_dict(state, strict=True)

    def test_load_state_dict_resets_rollout_weight_initialization(self) -> None:
        """Loading a checkpoint resets the rollout-weights-initialized and replay-parity flags, so
        the next step re-pushes weights and re-checks parity against the restored state.
        """
        trainer = _make_resume_trainer()
        trainer._rollout_weights_initialized = True
        trainer._replay_parity_passed = True

        trainer.load_state_dict({"step": 9, "global_step": 9}, strict=True)

        assert trainer._rollout_weights_initialized is False
        assert trainer._replay_parity_passed is False

    @pytest.mark.parametrize("strict", [True, False])
    def test_resume_rechecks_parity_at_nonzero_training_step(self, strict: bool) -> None:
        from vrl.algorithms.types import InitialReplayStats

        trainer = _make_resume_trainer()
        trainer._replay_parity_passed = True
        assert "_replay_parity_passed" not in trainer.state_dict()

        trainer.load_state_dict({"step": 4, "global_step": 4}, strict=strict)

        with pytest.raises(RuntimeError, match="replay parity failed"):
            trainer._validate_first_update_parity(
                InitialReplayStats(logprob_abs_diff_max=0.02, finite=True),
                local_weight=1.0,
            )
        assert trainer.state.step == 4
        assert trainer.state.global_step == 4
        assert trainer._replay_parity_passed is False

    def test_strict_resume_requires_master_optimizer_state_after_first_step(self) -> None:
        trainer = _make_resume_trainer()
        trainer.model.half()

        with pytest.raises(ValueError, match=r"missing optimizer state.*master residuals"):
            trainer.load_state_dict({"step": 1, "global_step": 1}, strict=True)

        trainer.load_state_dict({"step": 1, "global_step": 1}, strict=False)

        zero_step = _make_resume_trainer()
        zero_step.model.half()
        zero_step.load_state_dict({"step": 0, "global_step": 0}, strict=True)

    def test_low_precision_master_gate_runs_before_distributed_prepare(self) -> None:
        from types import SimpleNamespace

        import torch

        from vrl.trainers.strategy import SingleProcessStrategy

        class _SpyStrategy:
            def __init__(self, name: str) -> None:
                self.context = SimpleNamespace(strategy=name)
                self.prepared = False
                self._delegate = SingleProcessStrategy()

            def prepare_model(self, model):
                self.prepared = True
                return model

            def __getattr__(self, name):
                return getattr(self._delegate, name)

        distributed = _SpyStrategy("ddp")
        with pytest.raises(NotImplementedError, match="use BF16 or single_process"):
            _make_resume_trainer(model_dtype="float16", strategy=distributed)
        assert distributed.prepared is False

        compatible = _SpyStrategy("single_process")
        trainer = _make_resume_trainer(model_dtype="float16", strategy=compatible)
        assert compatible.prepared is True
        assert trainer.model.weight.dtype is torch.float16

    @pytest.mark.gpu
    def test_fp16_cuda_state_dict_round_trips_grad_scaler(self) -> None:
        """CUDA fp16 training must save and restore GradScaler state."""
        import torch

        source = _make_resume_trainer(device="cuda", train_precision="fp16")
        assert source._grad_scaler is not None

        optimizer = source._ensure_optimizer()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = source.model(torch.ones(1, 1, device="cuda")).sum()
        source._backward(loss)
        source._clip_and_step(optimizer)
        state = source.state_dict()

        assert "grad_scaler" in state

        restored = _make_resume_trainer(device="cuda", train_precision="fp16")
        restored.load_state_dict(state, strict=True)

        assert restored._grad_scaler is not None
        assert restored._grad_scaler.state_dict()["scale"] == state["grad_scaler"]["scale"]

    @pytest.mark.gpu
    def test_strict_resume_rejects_nonzero_fp16_checkpoint_without_scaler(self) -> None:
        trainer = _make_resume_trainer(device="cuda", train_precision="fp16")

        with pytest.raises(ValueError, match="missing GradScaler state"):
            trainer.load_state_dict({"step": 1, "global_step": 1}, strict=True)

        trainer.load_state_dict({"step": 1, "global_step": 1}, strict=False)

    def test_resume_pushes_restored_driver_weights_before_next_collect(self) -> None:
        """After a resume the restored driver weights are pushed to the rollout before the first
        collect: the collector observes exactly one sync.
        """
        import asyncio

        syncer = _Syncer()
        collect_seen_sync_counts: list[int] = []
        trainer = _make_resume_trainer(
            weight_syncer=syncer,
            collector=_SyncCountingCollector(syncer, collect_seen_sync_counts),
        )
        trainer._rollout_weights_initialized = True
        trainer.load_state_dict({"step": 4, "global_step": 4}, strict=True)

        asyncio.run(trainer.step(["prompt-a"]))

        assert collect_seen_sync_counts == [1]


class _ResumeAlgorithm(_EvaluatorAlgorithmFake):
    required_signal_keys = ("log_prob",)
    required_data_keys: tuple[str, ...] = ()

    class _Config:
        global_std = False
        eps = 1e-8
        adv_clip_max = 5.0
        kl_coef = 0.0

    config = _Config()

    def compute_advantages_from_tensors(self, rewards, group_ids):
        del group_ids
        return rewards - rewards.mean()

    def compute_loss(self, inputs):
        signals, advantages, _old_log_probs = _algorithm_inputs(inputs)
        from vrl.algorithms.types import TrainStepMetrics

        loss = signals.log_prob.mean() + advantages.mean() * 0.0
        return loss, TrainStepMetrics(
            loss=loss.item(),
            policy_loss=loss.item(),
        )


class _ResumeCollector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)

    async def collect_unscored(self, prompts, **kwargs):
        import torch

        group_size = int(kwargs["group_size"])
        return _diffusion_rollout_batch(
            rewards=torch.arange(group_size, dtype=torch.float32),
            group_ids=torch.zeros(group_size, dtype=torch.long),
            num_steps=2,
        )


class _SyncCountingCollector(_ResumeCollector):
    def __init__(self, syncer: _Syncer, seen_counts: list[int]) -> None:
        self.syncer = syncer
        self.seen_counts = seen_counts

    async def collect_unscored(self, prompts, **kwargs):
        self.seen_counts.append(len(self.syncer.calls))
        return await super().collect_unscored(prompts, **kwargs)


class _ResumeEvaluator:
    def evaluate(self, model, batch, timestep_idx, **kw):
        del kw

        return _trajectory_signals(
            batch, model.weight.view(1).expand(batch.rewards.shape[0]), timestep_idx
        )


class _Syncer:
    current_policy_version = None  # This fake does not track a policy version.

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def push(self, state_dict):
        self.calls.append(dict(state_dict))

    async def pull(self):
        return dict(self.calls[-1])


def _make_resume_trainer(
    *,
    ema: bool = False,
    weight_syncer=None,
    collector=None,
    device: str = "cpu",
    train_precision: str = "",
    model_dtype: str = "float32",
    strategy=None,
):
    import torch
    import torch.nn as nn

    from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.to(device=device, dtype=getattr(torch, model_dtype))
    role_dtype = train_precision if train_precision in {"fp16", "bf16"} else "fp32"
    _stamp_model_precision(
        model,
        precision=RolePrecision(
            dtype=role_dtype,
            float32_precision="ieee",
            outer_autocast=train_precision in {"fp16", "bf16"},
        ),
    )
    return OnlineTrainer(
        algorithm=_ResumeAlgorithm(),
        collector=collector or _ResumeCollector(),
        evaluator=_ResumeEvaluator(),
        model=model,
        weight_syncer=weight_syncer,
        sync_state_getter=(
            lambda: (
                {"linear.weight": model.weight.detach().clone()}
                if weight_syncer is not None
                else None
            )
        ),
        strategy=strategy,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            output_dir="outputs/",
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(enable=ema),
            debug=DebugConfig(),
            train_precision=train_precision,
        ),
        device=device,
    )


def _adam_exp_avg_values(optimizer) -> list[float]:
    values: list[float] = []
    for slot in optimizer.state.values():
        exp_avg = slot.get("exp_avg")
        if exp_avg is not None:
            values.extend(float(v) for v in exp_avg.reshape(-1).detach().cpu().tolist())
    return values
