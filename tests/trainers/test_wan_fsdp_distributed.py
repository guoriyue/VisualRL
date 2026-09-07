"""Wan I2V LoRA replay through a real multi-rank FSDP2 lifecycle.

The production checkpoint is too large for ordinary CI, so this uses the real
diffusers ``WanTransformer3DModel`` with a tiny config. It still exercises the
family wrapper, image-conditioning branch, PEFT adapter, FSDP2 collectives,
optimizer state, and serialized resume contract instead of substituting a fake
linear model.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from tests.models.steps.denoise.fixtures import (
    TINY_WAN_LATENT_SHAPE,
    TINY_WAN_TEXT_DIM,
    TINY_WAN_TEXT_LEN,
    build_tiny_wan_i2v_transformer,
    stamp_model_precision,
)
from tests.trainers._strategy_policies import free_port
from vrl.config.precision import RolePrecision
from vrl.config.schema import FSDPConfig
from vrl.models.families.wan_2_1.model import (
    WanI2VReplayModel,
    WanI2VSamplingState,
)
from vrl.models.interfaces.runtime import ModelBuild
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy


def _fsdp_strategy(
    context: DistributedTrainingContext,
    **overrides: object,
) -> FSDPStrategy:
    config = FSDPConfig.model_validate(overrides)
    return FSDPStrategy(
        context,
        mesh_dims=config.mesh,
        precision_policy=config.precision_policy,
        reshard_after_forward=config.reshard_after_forward,
        cpu_offload=config.cpu_offload,
    )


def _build_policy(seed: int = 0) -> WanI2VReplayModel:
    policy = WanI2VReplayModel(
        transformer=build_tiny_wan_i2v_transformer(seed=seed),
        scheduler=None,
        device=torch.device("cpu"),
    )
    build = ModelBuild(
        model_name_or_path="tiny-wan-i2v",
        revision=None,
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="wan_2_1_i2v",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        model_config={
            "use_lora": True,
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_v"],
            },
        },
        defer_trainable_device_move=True,
    )
    policy.apply_lora(build)
    stamp_model_precision(policy)
    return policy


def _build_dual_policy(seed: int = 0) -> WanI2VReplayModel:
    policy = WanI2VReplayModel(
        transformer=build_tiny_wan_i2v_transformer(seed=seed),
        transformer_2=build_tiny_wan_i2v_transformer(seed=seed + 1),
        scheduler=None,
        device=torch.device("cpu"),
        boundary_ratio=0.5,
        trainable_transformers="both",
    )
    build = ModelBuild(
        model_name_or_path="tiny-wan2.2-i2v",
        revision=None,
        device=torch.device("cpu"),
        parameter_dtype=torch.float32,
        family="wan_2_1_i2v",
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        model_config={
            "use_lora": True,
            "trainable_transformers": "both",
            "lora": {
                "rank": 2,
                "alpha": 4,
                "target_modules": ["to_q", "to_v"],
            },
        },
        defer_trainable_device_move=True,
    )
    policy.apply_lora(build)
    stamp_model_precision(policy)
    return policy


def _input_state(
    seed: int = 7,
    *,
    device: torch.device | str = "cpu",
    timestep: float = 5.0,
    boundary_ratio: float | None = None,
) -> WanI2VSamplingState:
    generator = torch.Generator().manual_seed(seed)
    state = WanI2VSamplingState(
        latents=torch.randn(TINY_WAN_LATENT_SHAPE, generator=generator),
        timesteps=torch.tensor([timestep]),
        scheduler=None,
        prompt_embeds=torch.randn(
            1,
            TINY_WAN_TEXT_LEN,
            TINY_WAN_TEXT_DIM,
            generator=generator,
        ),
        negative_prompt_embeds=None,
        image_embeds=torch.randn(1, 2, TINY_WAN_TEXT_DIM, generator=generator),
        condition=torch.randn(TINY_WAN_LATENT_SHAPE, generator=generator),
        guidance_scale=1.0,
        do_cfg=False,
        boundary_ratio=boundary_ratio,
        num_train_timesteps=1000 if boundary_ratio is not None else None,
    )
    target = torch.device(device)
    if target.type == "cpu":
        return state
    state.latents = state.latents.to(target)
    state.timesteps = state.timesteps.to(target)
    state.prompt_embeds = state.prompt_embeds.to(target)
    state.image_embeds = state.image_embeds.to(target)
    state.condition = state.condition.to(target)
    return state


def _bundle(policy: WanI2VReplayModel) -> SimpleNamespace:
    return SimpleNamespace(trainable_modules=policy.trainable_modules)


def _train_once(
    policy: WanI2VReplayModel,
    strategy: FSDPStrategy,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    timestep: float = 5.0,
    boundary_ratio: float | None = None,
) -> float:
    optimizer.zero_grad()
    noise_pred = policy.forward_step(
        _input_state(
            device=device,
            timestep=timestep,
            boundary_ratio=boundary_ratio,
        ),
        0,
    )["noise_pred"]
    strategy.backward(noise_pred.float().square().mean())
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    grad_norm = strategy.clip_grad_norm(trainable, max_norm=10.0)
    optimizer.step()
    return grad_norm


def _module_changed(before: dict, after: dict, name: str) -> bool:
    return any(not torch.equal(value, after[name][key]) for key, value in before[name].items())


def _tensor_tree_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _tensor_tree_equal(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes))
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _tensor_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_path: str,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        context = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        strategy = _fsdp_strategy(context, precision_policy="none")
        policy = strategy.prepare_model(_build_policy())
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )

        before = strategy.export_checkpoint_state(_bundle(policy))
        grad_norm = _train_once(policy, strategy, optimizer)
        after = strategy.export_checkpoint_state(_bundle(policy))
        optimizer_state = strategy.export_optimizer_state(policy, optimizer)

        adapter_state = after["transformer"]
        adapter_only = bool(adapter_state) and all("lora_" in key for key in adapter_state)
        changed = any(
            not torch.equal(value, after["transformer"][key])
            for key, value in before["transformer"].items()
        )

        if rank == 0:
            torch.save(
                {"checkpoint": after, "optimizer": optimizer_state},
                checkpoint_path,
            )
        dist.barrier()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resumed = strategy.prepare_model(_build_policy())
        resumed_optimizer = torch.optim.AdamW(
            [parameter for parameter in resumed.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        strategy.load_checkpoint_state(_bundle(resumed), checkpoint["checkpoint"], strict=True)
        strategy.load_optimizer_state(resumed, resumed_optimizer, checkpoint["optimizer"])
        restored = strategy.export_checkpoint_state(_bundle(resumed))
        resume_matches = all(
            torch.equal(value, restored["transformer"][key])
            for key, value in after["transformer"].items()
        )

        resumed_before = restored
        resumed_grad_norm = _train_once(resumed, strategy, resumed_optimizer)
        resumed_after = strategy.export_checkpoint_state(_bundle(resumed))
        continued = any(
            not torch.equal(value, resumed_after["transformer"][key])
            for key, value in resumed_before["transformer"].items()
        )
        queue.put(
            (
                rank,
                grad_norm,
                adapter_only,
                changed,
                resume_matches,
                resumed_grad_norm,
                continued,
            ),
        )
    finally:
        dist.destroy_process_group()


def test_wan_i2v_fsdp_step_checkpoint_resume_and_continue(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = free_port()
    checkpoint_path = tmp_path / "checkpoint.pt"
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, 2, port, str(checkpoint_path), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(2):
        rank, *values = queue.get(timeout=120)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert checkpoint_path.is_file()
    for rank in (0, 1):
        (
            grad_norm,
            adapter_only,
            changed,
            resume_matches,
            resumed_grad_norm,
            continued,
        ) = results[rank]
        assert grad_norm > 0, f"rank{rank} produced a zero first-step gradient"
        assert adapter_only, f"rank{rank} checkpoint materialized frozen Wan weights"
        assert changed, f"rank{rank} optimizer did not change a LoRA parameter"
        assert resume_matches, f"rank{rank} did not restore the serialized LoRA state"
        assert resumed_grad_norm > 0, f"rank{rank} produced a zero resumed gradient"
        assert continued, f"rank{rank} did not update after resume"


def _run_dual_rank(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_path: str,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        context = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        strategy = _fsdp_strategy(context, precision_policy="none")
        policy = strategy.prepare_model(_build_dual_policy())
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )

        before = strategy.export_checkpoint_state(_bundle(policy))
        high_grad = _train_once(
            policy,
            strategy,
            optimizer,
            timestep=750.0,
            boundary_ratio=0.5,
        )
        after_high = strategy.export_checkpoint_state(_bundle(policy))
        high_only = _module_changed(before, after_high, "transformer") and not _module_changed(
            before,
            after_high,
            "transformer_2",
        )

        low_grad = _train_once(
            policy,
            strategy,
            optimizer,
            timestep=250.0,
            boundary_ratio=0.5,
        )
        after_low = strategy.export_checkpoint_state(_bundle(policy))
        low_only = _module_changed(
            after_high,
            after_low,
            "transformer_2",
        ) and not _module_changed(after_high, after_low, "transformer")
        rollout_state = strategy.export_rollout_state(_bundle(policy))
        sync_has_both = any(key.startswith("transformer.") for key in rollout_state) and any(
            key.startswith("transformer_2.") for key in rollout_state
        )
        optimizer_state = strategy.export_optimizer_state(policy, optimizer)

        if rank == 0:
            torch.save(
                {"checkpoint": after_low, "optimizer": optimizer_state},
                checkpoint_path,
            )
        dist.barrier()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        resumed = strategy.prepare_model(_build_dual_policy())
        resumed_optimizer = torch.optim.AdamW(
            [parameter for parameter in resumed.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        strategy.load_checkpoint_state(_bundle(resumed), checkpoint["checkpoint"], strict=True)
        strategy.load_optimizer_state(resumed, resumed_optimizer, checkpoint["optimizer"])
        restored = strategy.export_checkpoint_state(_bundle(resumed))
        restored_optimizer = strategy.export_optimizer_state(resumed, resumed_optimizer)
        queue.put(
            (
                rank,
                high_grad,
                low_grad,
                high_only,
                low_only,
                sync_has_both,
                _tensor_tree_equal(after_low, restored),
                _tensor_tree_equal(optimizer_state, restored_optimizer),
            ),
        )
    finally:
        dist.destroy_process_group()


def test_wan_dual_expert_fsdp_stage_isolation_sync_and_resume(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = free_port()
    checkpoint_path = tmp_path / "dual-expert-checkpoint.pt"
    processes = [
        context.Process(
            target=_run_dual_rank,
            args=(rank, 2, port, str(checkpoint_path), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(2):
        rank, *values = queue.get(timeout=180)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert checkpoint_path.is_file()
    for rank in (0, 1):
        (
            high_grad,
            low_grad,
            high_only,
            low_only,
            sync_has_both,
            weights_match,
            optimizer_matches,
        ) = results[rank]
        assert high_grad > 0, f"rank{rank} high-noise expert produced zero gradient"
        assert low_grad > 0, f"rank{rank} low-noise expert produced zero gradient"
        assert high_only, f"rank{rank} high stage changed the wrong expert"
        assert low_only, f"rank{rank} low stage changed the wrong expert"
        assert sync_has_both, f"rank{rank} rollout sync omitted an expert"
        assert weights_match, f"rank{rank} did not restore both expert weights"
        assert optimizer_matches, f"rank{rank} did not restore both expert optimizer slots"


def _run_cuda_rank(rank: int, world_size: int, port: int, queue: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        context = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=device,
        )
        strategy = _fsdp_strategy(context, precision_policy="none")
        policy = strategy.prepare_model(_build_policy())

        from torch.distributed.tensor import DTensor

        cuda_sharded = all(
            isinstance(parameter, DTensor)
            and parameter.to_local().device == device
            and parameter.numel() >= parameter.to_local().numel()
            for parameter in policy.parameters()
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        before = strategy.export_checkpoint_state(_bundle(policy))
        grad_norm = _train_once(policy, strategy, optimizer, device=device)
        after = strategy.export_checkpoint_state(_bundle(policy))
        changed = any(
            not torch.equal(value, after["transformer"][key])
            for key, value in before["transformer"].items()
        )
        adapter_only = all("lora_" in key for key in after["transformer"])
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        queue.put((rank, cuda_sharded, grad_norm, changed, adapter_only, peak_bytes))
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


def _run_dual_cuda_offload_rank(
    rank: int,
    world_size: int,
    port: int,
    queue: mp.Queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        context = DistributedTrainingContext(
            strategy="fsdp",
            rank=rank,
            world_size=world_size,
            device=device,
        )
        strategy = _fsdp_strategy(
            context,
            precision_policy="none",
            cpu_offload=True,
        )
        policy = _build_dual_policy()
        policy._device = device
        policy._expert_lifecycle_profiling = True
        policy = strategy.prepare_model(policy)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=1e-2,
        )
        high_grad = _train_once(
            policy,
            strategy,
            optimizer,
            device=device,
            timestep=750.0,
            boundary_ratio=0.5,
        )
        low_grad = _train_once(
            policy,
            strategy,
            optimizer,
            device=device,
            timestep=250.0,
            boundary_ratio=0.5,
        )
        local_shards_on_cpu = all(
            getattr(parameter, "_local_tensor", parameter).device.type == "cpu"
            for parameter in policy.parameters()
        )
        queue.put(
            (
                rank,
                high_grad,
                low_grad,
                local_shards_on_cpu,
                int(torch.cuda.max_memory_allocated(device)),
            ),
        )
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [1, 2])
def test_wan_dual_expert_fsdp_cuda_cpu_offload(world_size: int) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA device(s)")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = free_port()
    processes = [
        context.Process(
            target=_run_dual_cuda_offload_rank,
            args=(rank, world_size, port, queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(world_size):
        rank, *values = queue.get(timeout=180)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    for rank in range(world_size):
        high_grad, low_grad, local_shards_on_cpu, peak_bytes = results[rank]
        assert high_grad > 0, f"rank{rank} high expert produced zero CUDA gradient"
        assert low_grad > 0, f"rank{rank} low expert produced zero CUDA gradient"
        assert local_shards_on_cpu, f"rank{rank} retained an inactive expert shard on CUDA"
        assert peak_bytes > 0, f"rank{rank} reported no CUDA allocation"


@pytest.mark.gpu
@pytest.mark.distributed
def test_wan_i2v_fsdp_four_rank_cuda_step() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    port = free_port()
    processes = [
        context.Process(target=_run_cuda_rank, args=(rank, 4, port, queue)) for rank in range(4)
    ]
    for process in processes:
        process.start()

    results = {}
    for _ in range(4):
        rank, *values = queue.get(timeout=180)
        results[rank] = values
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    for rank in range(4):
        cuda_sharded, grad_norm, changed, adapter_only, peak_bytes = results[rank]
        assert cuda_sharded, f"rank{rank} did not own CUDA DTensor shards"
        assert grad_norm > 0, f"rank{rank} produced a zero CUDA gradient"
        assert changed, f"rank{rank} did not update a CUDA LoRA shard"
        assert adapter_only, f"rank{rank} gathered frozen Wan weights"
        assert peak_bytes > 0, f"rank{rank} reported no CUDA allocation"
