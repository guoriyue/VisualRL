"""FSDP acceptance for BF16 model shards with positional FP32 masters."""

from __future__ import annotations

import os
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.trainers._strategy_policies import free_port
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.optimizer import FP32MasterWeightOptimizer
from vrl.trainers.strategy import FSDPStrategy


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _Transformer(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer()

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer}

    def set_module_root(self, name: str, module: nn.Module) -> None:
        if name != "transformer":
            raise ValueError(f"unknown trainable root: {name!r}")
        self.transformer = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.transformer(value)


def _strategy() -> FSDPStrategy:
    return FSDPStrategy(
        DistributedTrainingContext(
            strategy="fsdp",
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
        mesh_dims=["dp_shard"],
        precision_policy="none",
        reshard_after_forward=True,
        cpu_offload=False,
    )


def _build() -> tuple[_Policy, FSDPStrategy, FP32MasterWeightOptimizer]:
    policy = _Policy()
    strategy = _strategy()
    strategy.prepare_model(policy)
    optimizer = FP32MasterWeightOptimizer(
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        lambda parameters: torch.optim.AdamW(parameters, lr=1.0e-3),
    )
    return policy, strategy, optimizer


def _step(
    policy: _Policy,
    strategy: FSDPStrategy,
    optimizer: FP32MasterWeightOptimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8
    strategy.backward(policy(inputs).float().square().mean())
    optimizer.prepare_gradients()
    norm = strategy.clip_grad_norm(optimizer.parameters(), 1.0)
    assert torch.isfinite(torch.tensor(norm))
    assert norm > 0
    optimizer.step()


def test_fsdp_bf16_fp32_master_step_and_checkpoint_round_trip(
    cpu_process_group,
) -> None:
    from torch.distributed.tensor import DTensor

    torch.manual_seed(7)
    policy, strategy, optimizer = _build()
    masters = tuple(optimizer.parameters())
    assert masters
    assert all(isinstance(master, DTensor) for master in masters)
    assert all(master.dtype == torch.float32 for master in masters)

    _step(policy, strategy, optimizer)
    model_checkpoint = strategy.export_checkpoint_state(policy)
    checkpoint = strategy.export_optimizer_state(policy, optimizer)
    assert "fp32_master_weights" in checkpoint
    assert all(
        not isinstance(value, DTensor)
        for value in checkpoint["fp32_master_weights"]["parameters"]
        if value is not None
    )

    torch.manual_seed(7)
    restored_policy, restored_strategy, restored = _build()
    restored_strategy.load_checkpoint_state(restored_policy, model_checkpoint)
    restored_strategy.load_optimizer_state(restored_policy, restored, checkpoint)
    for actual, expected in zip(
        restored.parameters(),
        optimizer.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.full_tensor(), expected.full_tensor())

    _step(policy, strategy, optimizer)
    _step(restored_policy, restored_strategy, restored)
    for actual, expected in zip(
        restored.parameters(),
        optimizer.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual.full_tensor(), expected.full_tensor())


def _run_two_rank_master_round_trip(
    rank: int,
    world_size: int,
    port: int,
    queue: mp.Queue,
) -> None:
    from vrl.trainers.fsdp import (
        apply_fsdp,
        build_fsdp_mesh,
        gather_full_optimizer_state_dict,
        gather_trainable_state_dict,
        load_checkpoint_state_dict,
        load_full_optimizer_state_dict,
        mixed_precision_policy,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    context = DistributedTrainingContext(
        strategy="fsdp",
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
    )

    def build() -> tuple[_Transformer, FP32MasterWeightOptimizer]:
        torch.manual_seed(11)
        model = _Transformer()
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(context, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
        )
        optimizer = FP32MasterWeightOptimizer(
            model.parameters(),
            lambda parameters: torch.optim.AdamW(parameters, lr=1.0e-3),
        )
        return model, optimizer

    def step(model: _Transformer, optimizer: FP32MasterWeightOptimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4) / 8
        model(inputs).float().square().mean().backward()
        optimizer.prepare_gradients()
        optimizer.step()

    try:
        model, optimizer = build()
        step(model, optimizer)
        model_state = gather_trainable_state_dict(model)
        optimizer_state = gather_full_optimizer_state_dict(model, optimizer)

        restored_model, restored_optimizer = build()
        load_checkpoint_state_dict(restored_model, model_state)
        load_full_optimizer_state_dict(
            restored_model,
            restored_optimizer,
            optimizer_state,
        )
        step(model, optimizer)
        step(restored_model, restored_optimizer)

        masters_match = all(
            torch.equal(actual.full_tensor(), expected.full_tensor())
            for actual, expected in zip(
                restored_optimizer.parameters(),
                optimizer.parameters(),
                strict=True,
            )
        )
        model_matches = all(
            torch.equal(actual, model_state_after)
            for actual, model_state_after in zip(
                gather_trainable_state_dict(restored_model).values(),
                gather_trainable_state_dict(model).values(),
                strict=True,
            )
        )
        checkpoint_model = gather_trainable_state_dict(model, rank0_only=True)
        checkpoint_optimizer = gather_full_optimizer_state_dict(
            model,
            optimizer,
            rank0_only=True,
        )
        primary_only_checkpoint = (
            bool(checkpoint_model) and bool(checkpoint_optimizer)
            if rank == 0
            else checkpoint_model == {} and checkpoint_optimizer == {}
        )
        queue.put((rank, masters_match, model_matches, primary_only_checkpoint))
    finally:
        dist.destroy_process_group()


def test_two_rank_bf16_fp32_master_round_trip() -> None:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = free_port()
    processes = [
        context.Process(
            target=_run_two_rank_master_round_trip,
            args=(rank, 2, port, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = {queue.get(timeout=60) for _ in range(2)}
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert results == {(0, True, True, True), (1, True, True, True)}
