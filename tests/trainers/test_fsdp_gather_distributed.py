"""Checkpoint-owned FSDP state must materialize selected tensors on EVERY rank.

Regression for the two selection traps: gathering full state materializes the
frozen base, while rank0-only state skips the symmetric collective contract used
by checkpoint save. The checkpoint gather must instead select owned keys while
they are still sharded, then materialize exactly those keys on every rank.

This spawns a real gloo 2-rank group, shards a PEFT-LoRA toy transformer with
FSDP2, and asserts BOTH ranks call ``full_tensor`` only for LoRA tensors, then
scatter them back on load without changing frozen base state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

peft = pytest.importorskip("peft")

from tests.trainers._state_dict_helpers import gather_full_state_dict  # noqa: E402
from tests.trainers._strategy_policies import free_port  # noqa: E402
from vrl.trainers.distributed import DistributedTrainingContext  # noqa: E402
from vrl.trainers.fsdp import (  # noqa: E402
    apply_fsdp,
    build_fsdp_mesh,
    gather_checkpoint_state_dict,
    load_checkpoint_state_dict,
    mixed_precision_policy,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(16, 16)
        self.to_k = nn.Linear(16, 16)
        self.to_v = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x) + self.to_k(x) + self.to_v(x)


class _ToyTransformer(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    from peft import LoraConfig, get_peft_model

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        base = _ToyTransformer()
        base.register_buffer("frozen_cache", torch.ones(8))
        model = get_peft_model(
            base,
            LoraConfig(r=4, target_modules=["to_q", "to_k", "to_v"], lora_alpha=8),
        )
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        frozen_before = {
            name: value
            for name, value in gather_full_state_dict(model).items()
            if name not in trainable
        }

        from torch.distributed.tensor import DTensor

        full_tensor_calls = 0
        original_full_tensor = DTensor.full_tensor

        def _record_full_tensor(self, *args, **kwargs):
            nonlocal full_tensor_calls
            full_tensor_calls += 1
            return original_full_tensor(self, *args, **kwargs)

        DTensor.full_tensor = _record_full_tensor
        try:
            gathered = gather_checkpoint_state_dict(model)
        finally:
            DTensor.full_tensor = original_full_tensor

        all_cpu = all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in gathered.values()
        )
        selective = set(gathered) == trainable and full_tensor_calls == len(trainable)

        replacement = {name: torch.full_like(value, 9.0) for name, value in gathered.items()}
        load_checkpoint_state_dict(model, replacement, strict=True)
        restored = gather_checkpoint_state_dict(model)
        trainable_restored = all(
            torch.equal(value, torch.full_like(value, 9.0)) for value in restored.values()
        )
        frozen_after = {
            name: value
            for name, value in gather_full_state_dict(model).items()
            if name not in trainable
        }
        frozen_unchanged = frozen_before.keys() == frozen_after.keys() and all(
            torch.equal(value, frozen_after[name]) for name, value in frozen_before.items()
        )
        q.put((rank, selective, all_cpu, trainable_restored, frozen_unchanged))
    finally:
        dist.destroy_process_group()


def test_checkpoint_state_is_selective_and_round_trips_on_every_rank() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=60)
        results[rank] = flags
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    for rank in (0, 1):
        selective, all_cpu, trainable_restored, frozen_unchanged = results[rank]
        assert selective, f"rank{rank} gathered a frozen tensor or skipped a LoRA tensor"
        assert all_cpu, f"rank{rank} gathered non-CPU tensors"
        assert trainable_restored, f"rank{rank} did not restore the full LoRA tensors"
        assert frozen_unchanged, f"rank{rank} changed frozen base state during LoRA load"


def _run_optim_ema_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    """Real 2-rank shard semantics for optimizer resume + EMA (ws=1 gathers are no-ops)."""

    from torch.distributed.tensor import DTensor

    from vrl.trainers.fsdp import (
        gather_full_optimizer_state_dict,
        load_full_optimizer_state_dict,
    )
    from vrl.trainers.online.ema import EMAModuleWrapper

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)  # identical init on both ranks pre-shard
        model = _ToyTransformer()
        global_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        params = list(model.parameters())

        # One real training step so Adam moments exist and are rank-sharded.
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        torch.manual_seed(7)  # identical batch on both ranks
        model(torch.randn(2, 16)).sum().backward()
        optimizer.step()

        exported = gather_full_optimizer_state_dict(model, optimizer)
        moments_full = all(
            isinstance(entry.get("exp_avg"), torch.Tensor)
            and not isinstance(entry["exp_avg"], DTensor)
            and tuple(entry["exp_avg"].shape) == global_shapes[fqn]
            for fqn, entry in exported["state"].items()
        )
        # Round trip into a fresh optimizer (resume precondition: no grads).
        for p in params:
            p.grad = None
        fresh = torch.optim.AdamW(params, lr=1e-2)
        load_full_optimizer_state_dict(model, fresh, exported)
        reexported = gather_full_optimizer_state_dict(model, fresh)
        optim_round_trip = all(
            torch.equal(entry["exp_avg"], reexported["state"][fqn]["exp_avg"])
            and torch.equal(entry["exp_avg_sq"], reexported["state"][fqn]["exp_avg_sq"])
            for fqn, entry in exported["state"].items()
        )
        checkpoint_optimizer = gather_full_optimizer_state_dict(
            model,
            fresh,
            rank0_only=True,
        )
        primary_only_optimizer = (
            bool(checkpoint_optimizer.get("state")) if rank == 0 else checkpoint_optimizer == {}
        )

        # EMA over sharded params: step + checkpoint gather + reshard on load.
        ema = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
        with torch.no_grad():
            for p in params:
                p.add_(1.0)
        ema.step(params, optimization_step=0)
        state = ema.state_dict()
        ema_full = all(
            isinstance(t, torch.Tensor)
            and not isinstance(t, DTensor)
            and tuple(t.shape) == tuple(shadow.shape)  # DTensor.shape is global
            for t, shadow in zip(state["ema_parameters"], ema.ema_parameters, strict=True)
        )
        restored = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
        restored.load_state_dict(state)
        ema_round_trip = all(
            isinstance(got, DTensor) and torch.equal(got.full_tensor(), want.full_tensor())
            for got, want in zip(restored.ema_parameters, ema.ema_parameters, strict=True)
        )
        checkpoint_ema = ema.checkpoint_state_dict(is_primary=rank == 0)
        primary_only_ema = (
            bool(checkpoint_ema.get("ema_parameters")) if rank == 0 else checkpoint_ema == {}
        )

        q.put(
            (
                rank,
                moments_full,
                optim_round_trip,
                ema_full,
                ema_round_trip,
                primary_only_optimizer,
                primary_only_ema,
            ),
        )
    finally:
        dist.destroy_process_group()


def test_optimizer_and_ema_full_state_on_every_rank() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_run_optim_ema_rank, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=120)
        results[rank] = flags
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    for rank in (0, 1):
        (
            moments_full,
            optim_round_trip,
            ema_full,
            ema_round_trip,
            primary_only_optimizer,
            primary_only_ema,
        ) = results[rank]
        assert moments_full, f"rank{rank}: optimizer moments not full plain tensors"
        assert optim_round_trip, f"rank{rank}: optimizer state round trip diverged"
        assert ema_full, f"rank{rank}: EMA checkpoint state not full plain tensors"
        assert ema_round_trip, f"rank{rank}: EMA reshard-on-load diverged"
        assert primary_only_optimizer, f"rank{rank}: checkpoint optimizer ownership diverged"
        assert primary_only_ema, f"rank{rank}: checkpoint EMA ownership diverged"


def _run_checkpoint_ema_export_rank(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
    q: mp.Queue,
    fail_swap_rank: int | None = None,
    fail_artifact_write: bool = False,
    fail_rollback_rank: int | None = None,
) -> None:
    """Exercise raw checkpoint + EMA artifact ordering on real FSDP shards."""

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        AdapterExport,
        load_training_checkpoint,
        save_training_checkpoint,
    )
    from vrl.trainers.online.ema import EMAModuleWrapper
    from vrl.trainers.strategy import FSDPStrategy

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = get_peft_model(
            _ToyTransformer(),
            LoraConfig(
                r=2,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["to_q", "to_k", "to_v"],
            ),
        )
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        model = apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        bundle = type(
            "_Bundle",
            (),
            {"model": model, "trainable_modules": {"transformer": model}},
        )()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        with torch.no_grad():
            for parameter in parameters:
                parameter.fill_(3.0)
        ema = EMAModuleWrapper(parameters, decay=0.9)
        with torch.no_grad():
            for shadow in ema.ema_parameters:
                shadow.fill_(7.0)
        ema.num_updates = 1

        class _Trainer:
            def state_dict(self):
                return {"step": 1, "global_step": 1, "ema": ema.state_dict()}

        strategy = FSDPStrategy(
            ctx,
            mesh_dims=["dp_shard"],
            precision_policy="none",
            reshard_after_forward=True,
            cpu_offload=False,
        )
        if rank == fail_swap_rank:

            def _fail_swap(parameters, *, store_temp=True):
                del parameters, store_temp
                raise RuntimeError("rank-local EMA swap failed")

            ema.copy_ema_to = _fail_swap
        if rank == fail_rollback_rank:

            def _fail_rollback(parameters):
                del parameters
                raise RuntimeError("rank-local EMA rollback failed")

            ema.copy_temp_to = _fail_rollback
        if fail_artifact_write and rank == 0:

            def _fail_artifact(*_args, **_kwargs):
                raise RuntimeError("rank0 artifact write failed")

            model.save_pretrained = _fail_artifact

        save_error: BaseException | None = None
        try:
            save_training_checkpoint(
                output_dir,
                trainer=_Trainer(),
                bundle=bundle,
                family="toy",
                model_identity={"schema": "toy/v1"},
                progress={"completed_epoch": 1, "next_epoch": 1},
                rng_state={},
                adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(model)},
                export_ema=ema,
                strategy=strategy,
            )
        except BaseException as error:
            save_error = error
        dist.barrier()

        if fail_rollback_rank is not None:
            failed_together = isinstance(save_error, RuntimeError)
            reported_rollback = save_error is not None and "restore raw training weights" in str(
                save_error
            )
            wrote_nothing = rank != 0 or not Path(output_dir).exists()
            q.put((rank, failed_together, reported_rollback, wrote_nothing))
            return

        live = strategy.export_checkpoint_state(bundle)["transformer"]
        live_restored = all(
            torch.equal(value, torch.full_like(value, 3.0)) for value in live.values()
        )
        if fail_swap_rank is not None or fail_artifact_write:
            failed_together = isinstance(save_error, RuntimeError)
            wrote_nothing = rank != 0 or not Path(output_dir).exists()
            q.put((rank, live_restored, failed_together, wrote_nothing))
            return
        if save_error is not None:
            raise save_error
        raw_checkpoint = True
        ema_artifact = True
        if rank == 0:
            checkpoint = load_training_checkpoint(output_dir)
            raw_checkpoint = all(
                torch.equal(value, torch.full_like(value, 3.0))
                for value in checkpoint.checkpoint_state["transformer"].values()
            )
            artifact = load_file(
                str(
                    Path(output_dir) / LORA_WEIGHTS_NAME / "adapter_model.safetensors",
                ),
            )
            ema_artifact = bool(artifact) and all(
                torch.equal(value, torch.full_like(value, 7.0)) for value in artifact.values()
            )
        q.put((rank, live_restored, raw_checkpoint, ema_artifact))
    finally:
        dist.destroy_process_group()


def test_checkpoint_keeps_raw_state_and_exports_ema_on_every_fsdp_rank(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    output_dir = str(tmp_path / "checkpoint-ema")
    procs = [
        ctx.Process(
            target=_run_checkpoint_ema_export_rank,
            args=(rank, 2, port, output_dir, q),
        )
        for rank in range(2)
    ]
    for process in procs:
        process.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=180)
        results[rank] = flags
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0

    for rank in (0, 1):
        live_restored, raw_checkpoint, ema_artifact = results[rank]
        assert live_restored, f"rank{rank}: live parameters were not restored"
        assert raw_checkpoint, f"rank{rank}: checkpoint.pt contains EMA weights"
        assert ema_artifact, f"rank{rank}: adapter artifact does not contain EMA weights"


def test_peer_ema_swap_failure_aborts_before_second_fsdp_gather(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    output_dir = str(tmp_path / "checkpoint-peer-swap-failure")
    procs = [
        ctx.Process(
            target=_run_checkpoint_ema_export_rank,
            args=(rank, 2, port, output_dir, q, 0),
        )
        for rank in range(2)
    ]
    for process in procs:
        process.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=180)
        results[rank] = flags
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0

    for rank in (0, 1):
        live_restored, failed_together, wrote_nothing = results[rank]
        assert live_restored, f"rank{rank}: peer swap failure left EMA weights live"
        assert failed_together, f"rank{rank}: peer swap failure did not propagate"
        assert wrote_nothing, f"rank{rank}: failed EMA export published a checkpoint"


def test_peer_ema_swap_and_rollback_failures_propagate_to_every_fsdp_rank(
    tmp_path,
) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    output_dir = str(tmp_path / "checkpoint-peer-rollback-failure")
    procs = [
        ctx.Process(
            target=_run_checkpoint_ema_export_rank,
            args=(rank, 2, port, output_dir, q, 0, False, 1),
        )
        for rank in range(2)
    ]
    for process in procs:
        process.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=180)
        results[rank] = flags
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0

    for rank in (0, 1):
        failed_together, reported_rollback, wrote_nothing = results[rank]
        assert failed_together, f"rank{rank}: EMA rollback failure did not propagate"
        assert reported_rollback, f"rank{rank}: rollback failure was not explicit"
        assert wrote_nothing, f"rank{rank}: failed EMA rollback published a checkpoint"


def test_rank0_artifact_write_failure_propagates_to_every_fsdp_rank(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = free_port()
    output_dir = str(tmp_path / "checkpoint-rank0-write-failure")
    procs = [
        ctx.Process(
            target=_run_checkpoint_ema_export_rank,
            args=(rank, 2, port, output_dir, q, None, True),
        )
        for rank in range(2)
    ]
    for process in procs:
        process.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=180)
        results[rank] = flags
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0

    for rank in (0, 1):
        live_restored, failed_together, wrote_nothing = results[rank]
        assert live_restored, f"rank{rank}: publication failure left EMA weights live"
        assert failed_together, f"rank{rank}: rank0 publication failure did not propagate"
        assert wrote_nothing, f"rank{rank}: failed publication left a checkpoint"
