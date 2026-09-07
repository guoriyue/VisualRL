"""DDP training strategy (SPRINT_symmetric_colocated_ddp.md).

We cannot run real multi-GPU here, but ``DistributedDataParallel`` runs on a
single CPU rank (``world_size=1`` + gloo): the module wraps, forward/backward
works, and because DDP keeps FULL (replicated, not sharded) params the rollout/
checkpoint export is the single-process full-state path on the unwrapped module.
That is enough to exercise the strategy for real — wrapping, clip, and the
invariant that the rollout-facing key space is identical to single-process.

The build_strategy dispatch + the model-handle guard need no process group and
run unconditionally.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.trainers._strategy_policies import (
    Bundle,
    DualStagePolicy,
    FakePolicy,
    ToyTransformer,
)
from vrl.config.schema import DDPConfig, RootConfig
from vrl.models.interfaces.runtime import register_checkpoint_owned_state
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import (
    DDPStrategy,
    SingleProcessStrategy,
    build_strategy,
)

# ── fixtures / fakes ────────────────────────────────────────────────────────


def _cpu_ddp_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="ddp",
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )


def _ddp_strategy(
    context: DistributedTrainingContext,
    **overrides: bool,
) -> DDPStrategy:
    config = DDPConfig.model_validate(overrides)
    return DDPStrategy(
        context,
        find_unused_parameters=config.find_unused_parameters,
    )


class _ARLikePolicy:
    """No trainable_modules / writer (AR family shape)."""


def _ddp_wrap(module: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(module, device_ids=None)  # CPU rank


# ── build_strategy dispatch + model-handle guard (no process group) ──────────


@pytest.mark.parametrize("find_unused_parameters", [False, True])
def test_build_strategy_reads_ddp_public_default_and_override(
    find_unused_parameters: bool,
) -> None:
    ddp = {} if not find_unused_parameters else {"find_unused_parameters": True}
    strategy = build_strategy(
        RootConfig.model_validate(
            {
                "distributed": {
                    "training": {
                        "strategy": "ddp",
                        "ddp": ddp,
                    },
                },
            },
        ),
        _cpu_ddp_context(),
    )

    assert isinstance(strategy, DDPStrategy)
    assert strategy._find_unused_parameters is find_unused_parameters


def test_ddp_rejects_shared_gpu_training_state_parking_preflight() -> None:
    with pytest.raises(NotImplementedError, match="Use disjoint rollout GPUs"):
        _ddp_strategy(_cpu_ddp_context()).validate_training_state_parking()


def test_ddp_wraps_resolved_device_after_per_rank_mask(monkeypatch) -> None:
    """Physical rank 3 passes logical cuda:0 to DDP after launcher masking."""

    import torch.nn.parallel

    context = DistributedTrainingContext(
        strategy="ddp",
        rank=3,
        world_size=4,
        device=torch.device("cuda:0"),
    )
    wrap_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "vrl.trainers.strategy.init_training_process_group",
        lambda _context, *, backend: None,
    )

    def _wrap(module, **kwargs):
        wrap_calls.append(kwargs)
        return module

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _wrap)

    DDPStrategy(context, find_unused_parameters=False).prepare_model(FakePolicy(ToyTransformer()))

    assert wrap_calls == [{"device_ids": [0], "find_unused_parameters": False}]


def test_ddp_prepare_model_wraps_multi_transformer_model(cpu_process_group) -> None:
    """Dual-stage Wan wraps both named roots and writes both aliases back."""
    from torch.nn.parallel import DistributedDataParallel

    policy = DualStagePolicy(ToyTransformer())
    out = _ddp_strategy(_cpu_ddp_context()).prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert policy.set_2_calls == 1
    assert isinstance(policy.transformer, DistributedDataParallel)
    assert isinstance(policy.transformer_2, DistributedDataParallel)


# ── wrapping + export (need a process group) ─────────────────────────────────


def test_ddp_prepare_model_wraps_transformer(cpu_process_group) -> None:
    from torch.nn.parallel import DistributedDataParallel

    policy = FakePolicy(ToyTransformer())
    out = _ddp_strategy(_cpu_ddp_context()).prepare_model(policy)

    assert out is policy
    assert policy.set_calls == 1
    assert isinstance(policy.transformer, DistributedDataParallel)


def test_ddp_rollout_export_matches_single_process_key_space(cpu_process_group) -> None:
    """The invariant: DDP-wrapped rollout state == single-process rollout state.

    Same keys (clean ``transformer.*``, no ``.module.`` leak), same values — a
    rollout worker is oblivious to whether the trainer ran DDP.
    """
    ref = ToyTransformer()
    snapshot = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    replicated = ToyTransformer()
    replicated.load_state_dict(snapshot)
    wrapped = _ddp_wrap(replicated)

    got = _ddp_strategy(_cpu_ddp_context()).export_rollout_state(Bundle(wrapped))
    expected = SingleProcessStrategy().export_rollout_state(Bundle(ref))

    assert got.keys() == expected.keys()
    assert all(key.startswith("transformer.") for key in got)
    assert all("module" not in key for key in got)
    for key in got:
        assert torch.allclose(got[key], expected[key])


def test_ddp_rollout_export_filters_frozen_params(cpu_process_group) -> None:
    """Rollout excludes frozen state while checkpoint includes registered state."""
    net = ToyTransformer()
    net.head.requires_grad_(False)
    register_checkpoint_owned_state(
        net,
        [name for name, _ in net.named_parameters() if name.startswith("head.")],
    )
    wrapped = _ddp_wrap(net)
    strategy = _ddp_strategy(_cpu_ddp_context())

    rollout = strategy.export_rollout_state(Bundle(wrapped))
    assert rollout
    assert not any("head" in key for key in rollout)

    checkpoint = strategy.export_checkpoint_state(Bundle(wrapped))["transformer"]
    assert any("head" in key for key in checkpoint)


def test_ddp_checkpoint_state_excludes_unregistered_frozen_params(cpu_process_group) -> None:
    net = ToyTransformer()
    net.head.requires_grad_(False)
    strategy = _ddp_strategy(_cpu_ddp_context())

    checkpoint = strategy.export_checkpoint_state(Bundle(_ddp_wrap(net)))["transformer"]

    assert not any("head" in key for key in checkpoint)


def test_ddp_export_then_load_checkpoint_state_round_trip(cpu_process_group) -> None:
    strategy = _ddp_strategy(_cpu_ddp_context())
    src = _ddp_wrap(ToyTransformer())
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(3.0)

    snapshot = strategy.export_checkpoint_state(Bundle(src))
    assert set(snapshot) == {"transformer"}
    first_name, first_value = next(iter(snapshot["transformer"].items()))
    live = dict(src.module.state_dict())[first_name]
    assert first_value.data_ptr() != live.data_ptr()

    dst = _ddp_wrap(ToyTransformer())
    strategy.load_checkpoint_state(Bundle(dst), snapshot)
    for value in dst.module.state_dict().values():
        assert torch.allclose(value, torch.full_like(value, 3.0))


def test_ddp_restore_protocol_loads_schema_v1_full_frozen_state(
    cpu_process_group,
    tmp_path,
) -> None:
    from vrl.trainers.checkpointing import TrainingCheckpoint, restore_model_checkpoint

    source_module = ToyTransformer()
    source_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in source_module.parameters():
            parameter.fill_(7.0)
    source = _ddp_wrap(source_module)

    restored_module = ToyTransformer()
    restored_module.head.requires_grad_(False)
    with torch.no_grad():
        for parameter in restored_module.parameters():
            parameter.fill_(3.0)
    restored = _ddp_wrap(restored_module)
    checkpoint = TrainingCheckpoint(
        checkpoint_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        payload={
            "schema_version": 1,
            "family": "toy",
            "trainer": {},
            "model": {"trainable_modules": {"transformer": dict(source.module.state_dict())}},
            "progress": {},
            "rng": {},
        },
        meta={},
    )

    restore_model_checkpoint(
        checkpoint,
        bundle=Bundle(restored),
        family="toy",
        strict=True,
        strategy=_ddp_strategy(_cpu_ddp_context()),
    )

    assert all(
        torch.allclose(value, torch.full_like(value, 7.0))
        for value in restored.module.state_dict().values()
    )


def test_ddp_adapter_export_uses_gathered_checkpoint_state(
    cpu_process_group,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    from vrl.trainers.checkpointing import (
        LORA_WEIGHTS_NAME,
        AdapterExport,
        save_training_checkpoint,
    )

    module = get_peft_model(
        ToyTransformer(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["lin"]),
    )
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.fill_(5.0)
    wrapped = _ddp_wrap(module)

    save_training_checkpoint(
        tmp_path,
        trainer=SimpleNamespace(state_dict=lambda: {"step": 0, "global_step": 0}),
        bundle=Bundle(wrapped),
        family="toy",
        model_identity={"schema": "toy/v1"},
        progress={"next_epoch": 1},
        rng_state={},
        adapter_exports={LORA_WEIGHTS_NAME: AdapterExport(module)},
        strategy=_ddp_strategy(_cpu_ddp_context()),
    )

    artifact = load_file(
        str(tmp_path / LORA_WEIGHTS_NAME / "adapter_model.safetensors"),
    )
    assert artifact
    assert all(torch.equal(value, torch.full_like(value, 5.0)) for value in artifact.values())
