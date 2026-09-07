"""Tests for the shared diffusion model base."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.models.steps.denoise.fixtures import (
    RecordingModule,
    add_lora_adapters,
    build_tiny_wan_transformer,
)
from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.types import DenoiseRequest
from vrl.models.families.cosmos import CosmosReplayForward
from vrl.models.families.cosmos.predict2.model import CosmosPredict2Model
from vrl.models.families.flux.model import FluxModel
from vrl.models.families.mochi.model import MochiModel
from vrl.models.families.sd3_5.model import SD3_5Model
from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel
from vrl.models.interfaces import ReplayResult
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.models.steps.denoise import DiffusionModelBase
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import build_diffusion_trajectory


class _AdapterTransformer(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.disabled = False
        self.active_adapter = "default"

    def set_adapter(self, name: str) -> None:
        self.active_adapter = name

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class _PluralAdapterTransformer(nn.Linear):
    """A diffusers ``PeftAdapterMixin``-style module: only the plural surface.

    Cosmos/Wan transformers expose ``disable_adapters``/``enable_adapters`` (and
    no singular ``disable_adapter`` context manager), so this pins that the
    boundary disables via the plural pair rather than silently no-op'ing.
    """

    def __init__(self) -> None:
        super().__init__(2, 2)
        self.adapters_enabled = True

    def disable_adapters(self) -> None:
        self.adapters_enabled = False

    def enable_adapters(self) -> None:
        self.adapters_enabled = True


class _CompiledWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module


class _ModelBaseStub(DiffusionModelBase):
    precision = RolePrecision(
        dtype="fp32",
        float32_precision="ieee",
        outer_autocast=False,
    )
    device = torch.device("cpu")
    family = "stub"

    def __init__(self) -> None:
        super().__init__()
        pipeline = SimpleNamespace(transformer=_AdapterTransformer())
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self.forward_models: list[Any] = []
        self.forward_step_indices: list[int] = []

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, negative_prompt, kwargs
        return {}

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        del request, encoded, kwargs
        return object()

    def forward_step(
        self,
        state: Any,
        step_idx: int,
    ) -> dict[str, Any]:
        del state
        self.forward_models.append(self.transformer)
        self.forward_step_indices.append(step_idx)
        return {"noise_pred": torch.ones(1)}

    def decode_latents(self, latents: Any) -> Any:
        return latents

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Any:
        del replay_tensors, batch_context, latents, step_idx
        return object()

    @property
    def scheduler(self) -> Any:
        return None

    @property
    def raw_handle(self) -> Any:
        return self.pipeline


class _BackendPipelineStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.device = torch.device("cpu")


class _LoadedPipeline:
    def __init__(self) -> None:
        self.transformer = RecordingModule()
        self.vae = RecordingModule()
        self.text_encoder = RecordingModule()


def _bare_build() -> ModelBuild:
    return ModelBuild(
        model_name_or_path="genmo/mochi-1-preview",
        revision=None,
        device="cuda:0",
        parameter_dtype=torch.bfloat16,
        family="mochi",
        precision=RolePrecision("bf16", "tf32"),
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.float16),
    )


def test_shared_from_build_parks_the_prompt_encoder_off_device(monkeypatch) -> None:
    """``_prompt_encoder_on_cpu`` is the only thing that decides encoder placement."""
    from diffusers import MochiPipeline

    pipeline = _LoadedPipeline()
    monkeypatch.setattr(
        MochiPipeline,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: pipeline),
    )

    model = MochiModel.from_build(_bare_build())

    assert model.pipeline is pipeline
    assert pipeline.text_encoder.requires_grad_enabled is False
    assert pipeline.text_encoder.to_calls == [("cpu", torch.float16)]
    # The VAE never follows the encoder: fp32 on the compute device, always.
    assert pipeline.vae.requires_grad_enabled is False
    assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]


def test_shared_from_build_skips_an_absent_declared_encoder(monkeypatch) -> None:
    """A pipeline variant without one of the declared encoders loads anyway.

    The loop reads encoders with ``getattr(..., None)``, so a checkpoint that
    ships fewer encoders than the family declares is a skip, not a crash.
    """
    from diffusers import StableDiffusion3Pipeline

    pipeline = _LoadedPipeline()  # only ``text_encoder``, not _2 / _3
    monkeypatch.setattr(
        StableDiffusion3Pipeline,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: pipeline),
    )

    model = SD3_5Model.from_build(_bare_build())

    assert model.pipeline is pipeline
    assert pipeline.text_encoder.to_calls == [("cuda:0", torch.float16)]


def test_pipeline_model_base_discovers_primary_encoder_device() -> None:
    pipeline = _BackendPipelineStub()
    runtime = SD3_5Model(pipeline=pipeline, device=torch.device("meta"))

    assert runtime._encoder_device() == torch.device("cpu")


def test_pipeline_model_base_falls_back_for_parameterless_encoder() -> None:
    pipeline = _BackendPipelineStub()
    pipeline.text_encoder = nn.Identity()
    runtime = SD3_5Model(pipeline=pipeline, device=torch.device("meta"))

    assert runtime._encoder_device() == torch.device("meta")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(guidance_embeds=True), True),
        (SimpleNamespace(guidance_embeds=False), False),
        (SimpleNamespace(), False),
    ],
)
def test_pipeline_model_base_reads_guidance_embeds(config: Any, expected: bool) -> None:
    pipeline = _BackendPipelineStub()
    pipeline.transformer.config = config
    runtime = SD3_5Model(pipeline=pipeline, device=torch.device("cpu"))

    assert runtime._guidance_embeds is expected


def test_flux_encoder_device_prefers_second_encoder() -> None:
    pipeline = _BackendPipelineStub()
    pipeline.text_encoder_2 = nn.Linear(2, 2, device="meta")
    runtime = FluxModel(pipeline=pipeline, device=torch.device("cpu"))

    assert runtime._encoder_device() == torch.device("meta")


def _peft_disable_flags(module: nn.Module) -> list[bool]:
    flags: list[bool] = []
    for child in module.modules():
        disabled = getattr(child, "disable_adapters", None)
        if isinstance(disabled, bool):
            flags.append(disabled)
    return flags


def test_trainable_modules_default_reports_the_registered_transformer() -> None:
    """The stub declares no ``trainable_modules``; the base default answers."""
    runtime = _ModelBaseStub()

    assert runtime.trainable_modules == {"transformer": runtime.transformer}


def test_trainable_modules_default_fails_loud_without_a_transformer() -> None:
    """The default routes through ``_require_transformer``, not a bare attribute."""
    runtime = _ModelBaseStub()
    runtime.transformer = None

    with pytest.raises(RuntimeError, match="no registered trainable transformer"):
        _ = runtime.trainable_modules


def test_diffusion_model_base_registers_only_transformer_child() -> None:
    """Only the transformer is a registered child: the pipeline (with its frozen VAE / encoders)
    stays out of ``state_dict``, so checkpoints and weight sync carry trainable state alone.
    """
    runtime = _ModelBaseStub()

    assert isinstance(runtime, nn.Module)
    assert dict(runtime.named_children()) == {"transformer": runtime.transformer}
    keys = set(runtime.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith("pipeline.") for key in keys)


@pytest.mark.parametrize(
    "runtime_cls",
    [SD3_5Model, WanT2VDiffusersModel, CosmosPredict2Model],
)
def test_concrete_diffusion_runtimes_register_only_transformer(
    runtime_cls: type[DiffusionModelBase],
) -> None:
    """Every concrete diffusion runtime registers only the transformer; the pipeline's VAE and
    text encoders never enter ``named_children`` or ``state_dict``.
    """
    pipeline = _BackendPipelineStub()
    runtime = runtime_cls(pipeline=pipeline, device=torch.device("cpu"))

    assert isinstance(runtime, nn.Module)
    assert runtime.pipeline is pipeline
    assert runtime.transformer is pipeline.transformer
    assert dict(runtime.named_children()) == {"transformer": pipeline.transformer}
    keys = set(runtime.state_dict())
    assert keys == {"transformer.weight", "transformer.bias"}
    assert not any(key.startswith(("vae.", "text_encoder.", "_pipeline.")) for key in keys)


@pytest.mark.parametrize(
    "runtime_cls",
    [SD3_5Model, WanT2VDiffusersModel, CosmosPredict2Model],
)
def test_concrete_diffusion_runtimes_keep_pipeline_transformer_in_sync(
    runtime_cls: type[DiffusionModelBase],
) -> None:
    """``_set_transformer`` swaps both the registered child and the pipeline's transformer, so the
    runtime and the pipeline can never point at different modules.
    """
    pipeline = _BackendPipelineStub()
    runtime = runtime_cls(pipeline=pipeline, device=torch.device("cpu"))
    replacement = nn.Linear(2, 2)

    runtime._set_transformer(replacement)

    assert runtime.transformer is replacement
    assert runtime.pipeline.transformer is replacement
    assert dict(runtime.named_children()) == {"transformer": replacement}


def test_forward_resolves_runtime_self_to_registered_transformer() -> None:
    """``forward`` runs on the registered transformer child, the module weight sync and
    checkpoints see, not on a stale reference.
    """
    runtime = _ModelBaseStub()

    runtime.forward(object(), 0)

    assert runtime.forward_models == [runtime.transformer]


def test_replay_forward_returns_typed_replay_result() -> None:
    """``replay_forward`` yields a ``ReplayResult`` carrying the ``denoise`` segment's noise_pred;
    the plain base replays every timestep at index 0 while ``CosmosReplayForward`` passes the
    real ``timestep_idx`` through, because Cosmos indexes sigmas per step.
    """
    runtime = _ModelBaseStub()
    observations = torch.zeros(2, 2, 1)
    actions = torch.ones(2, 2, 1)
    old_log_prob = torch.zeros(2, 2)
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="diffusion",
            family="sd3_5",
            task="t2i",
            inputs=["a", "b"],
            samples_per_prompt=1,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=index,
                sample_index=0,
                prompt=prompt,
                sample_id=f"s{index}",
            )
            for index, prompt in enumerate(("a", "b"))
        ],
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=torch.tensor([[1, 0], [1, 0]]),
        kl=torch.zeros(2, 2),
        replay_tensors={"prompt_embeds": torch.zeros(2, 3, 4)},
        context={"scheduler": "stub"},
    )
    batch = RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 1]),
        trajectory=trajectory,
        context=trajectory.context,
    )

    result = runtime.replay_forward(batch, timestep_idx=1)
    caller_latent_result = runtime.replay_forward_with_latents(
        batch,
        timestep_idx=1,
        latents=torch.zeros(2, 1),
    )

    assert isinstance(result, ReplayResult)
    assert result.segments["denoise"].values["noise_pred"].shape == (1,)
    assert caller_latent_result["noise_pred"].shape == (1,)
    assert runtime.forward_step_indices == [0, 0]

    class _CosmosReplayStub(CosmosReplayForward, _ModelBaseStub):
        pass

    cosmos_runtime = _CosmosReplayStub()
    cosmos_runtime.replay_forward(batch, timestep_idx=1)
    cosmos_runtime.replay_forward_with_latents(
        batch,
        timestep_idx=1,
        latents=torch.zeros(2, 1),
    )
    assert cosmos_runtime.forward_step_indices == [1, 1]


def test_disable_adapter_forwards_to_transformer_context() -> None:
    """``disable_adapter`` is a context manager delegating to the transformer's own toggle and re-
    enables on exit.
    """
    runtime = _ModelBaseStub()

    with runtime.disable_adapter():
        assert runtime.transformer.disabled is True

    assert runtime.transformer.disabled is False


def test_disable_adapter_without_transformer_adapter_is_noop() -> None:
    """Checks disable adapter without transformer adapter is no-op."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(nn.Linear(2, 2))

    with runtime.disable_adapter():
        assert runtime.transformer.training is True


def test_disable_adapter_uses_plural_diffusers_surface() -> None:
    """A diffusers PeftAdapterMixin module (plural disable/enable only) is disabled.

    Regression: checking only the singular ``disable_adapter`` left cosmos/wan
    adapters ON during the reference forward (silent no-op).
    """
    runtime = _ModelBaseStub()
    runtime._set_transformer(_PluralAdapterTransformer())
    transformer = runtime.transformer

    with runtime.disable_adapter():
        assert transformer.adapters_enabled is False
    assert transformer.adapters_enabled is True


def test_disable_adapter_noops_for_diffusers_transformer_without_lora() -> None:
    """A plain diffusers PeftAdapterMixin transformer has plural methods but no LoRA."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(build_tiny_wan_transformer())

    with runtime.disable_adapter():
        assert getattr(runtime.transformer, "_hf_peft_config_loaded", False) is False


def test_disable_adapter_disables_real_diffusers_lora_layers() -> None:
    """A real diffusers-native LoRA transformer is disabled through the plural API."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(add_lora_adapters(build_tiny_wan_transformer()))
    flags = _peft_disable_flags(runtime.transformer)
    assert flags and not any(flags)

    with runtime.disable_adapter():
        assert all(_peft_disable_flags(runtime.transformer))

    assert not any(_peft_disable_flags(runtime.transformer))


def test_disable_adapter_preserves_already_disabled_diffusers_lora_state() -> None:
    """Nested plural disable contexts must not re-enable an already-disabled model."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(add_lora_adapters(build_tiny_wan_transformer()))
    runtime.transformer.disable_adapters()
    assert all(_peft_disable_flags(runtime.transformer))

    with runtime.disable_adapter():
        assert all(_peft_disable_flags(runtime.transformer))

    assert all(_peft_disable_flags(runtime.transformer))
    runtime.transformer.enable_adapters()


def test_activate_adapter_sets_named_adapter_and_restores_default() -> None:
    """activate_adapter routes to transformer.set_adapter and restores 'default'."""
    runtime = _ModelBaseStub()

    with runtime.activate_adapter("previous"):
        assert runtime.transformer.active_adapter == "previous"

    assert runtime.transformer.active_adapter == "default"


def test_activate_adapter_without_set_adapter_raises() -> None:
    """Activating a named adapter on a module that cannot is a loud misconfig."""
    runtime = _ModelBaseStub()
    runtime._set_transformer(nn.Linear(2, 2))  # plain module, no set_adapter

    with pytest.raises(RuntimeError, match="set_adapter"), runtime.activate_adapter("previous"):
        pass


def test_load_trainable_state_accepts_trainable_keys() -> None:
    """A state whose keys are exactly the ``transformer.``-prefixed trainable keys loads into the
    transformer in place.
    """
    runtime = _ModelBaseStub()
    replacement = {
        "weight": torch.full_like(runtime.transformer.weight, 2.0),
        "bias": torch.full_like(runtime.transformer.bias, 3.0),
    }
    state = {f"transformer.{key}": value for key, value in replacement.items()}

    runtime.load_trainable_state(state)

    assert torch.equal(runtime.transformer.weight, replacement["weight"])
    assert torch.equal(runtime.transformer.bias, replacement["bias"])


def test_load_trainable_state_accepts_compiled_transformer_wrapper() -> None:
    """Checks weight sync loads into torch.compile wrapped modules."""
    runtime = _ModelBaseStub()
    original = runtime.transformer
    runtime._set_transformer(_CompiledWrapper(original))
    replacement = {
        "weight": torch.full_like(original.weight, 2.0),
        "bias": torch.full_like(original.bias, 3.0),
    }
    state = {f"transformer.{key}": value for key, value in replacement.items()}

    runtime.load_trainable_state(state)

    assert torch.equal(original.weight, replacement["weight"])
    assert torch.equal(original.bias, replacement["bias"])


def test_load_trainable_state_rejects_all_unmatched_keys() -> None:
    runtime = _ModelBaseStub()

    with pytest.raises(ValueError, match="trainable keys prefixed"):
        runtime.load_trainable_state({"unknown": torch.ones(1)})


# -- versioned trainable-state slots ------------------------------------------


def _slot_state(runtime: _ModelBaseStub, weight: float, bias: float) -> dict[str, Any]:
    return {
        "transformer.weight": torch.full_like(runtime.transformer.weight, weight),
        "transformer.bias": torch.full_like(runtime.transformer.bias, bias),
    }


def test_install_retains_old_version_after_newer_install() -> None:
    """The core non-draining invariant: an old version stays activatable after a
    newer one is installed, and each activates to its OWN weights."""
    runtime = _ModelBaseStub()
    runtime.install_trainable_state(1, _slot_state(runtime, 1.0, 1.0))
    runtime.install_trainable_state(2, _slot_state(runtime, 2.0, 2.0))

    assert runtime.has_trainable_state(1)
    assert runtime.has_trainable_state(2)

    runtime.activate_trainable_state(2)
    assert torch.equal(
        runtime.transformer.weight, torch.full_like(runtime.transformer.weight, 2.0)
    )

    # Old v1 still resolves to v1's weights even though v2 was installed after it.
    runtime.activate_trainable_state(1)
    assert torch.equal(
        runtime.transformer.weight, torch.full_like(runtime.transformer.weight, 1.0)
    )
    assert torch.equal(runtime.transformer.bias, torch.full_like(runtime.transformer.bias, 1.0))


def test_install_does_not_mutate_live_weights() -> None:
    """install only retains; only activate touches the live model."""
    runtime = _ModelBaseStub()
    before = runtime.transformer.weight.detach().clone()
    runtime.install_trainable_state(7, _slot_state(runtime, 9.0, 9.0))

    assert torch.equal(runtime.transformer.weight, before)  # unchanged until activate
    runtime.activate_trainable_state(7)
    assert torch.equal(runtime.transformer.weight, torch.full_like(before, 9.0))


def test_activate_is_idempotent_for_active_version() -> None:
    """Re-activating the live version is a no-op (skips the reload)."""
    runtime = _ModelBaseStub()
    runtime.install_trainable_state(1, _slot_state(runtime, 1.0, 1.0))
    runtime.activate_trainable_state(1)
    # Mutate live weights, then re-activate the SAME version: skip-if-active means
    # the live (mutated) weights are NOT reloaded over.
    with torch.no_grad():
        runtime.transformer.weight.fill_(5.0)
    runtime.activate_trainable_state(1)
    assert torch.equal(
        runtime.transformer.weight, torch.full_like(runtime.transformer.weight, 5.0)
    )


def test_forward_step_runs_under_the_stamped_contract() -> None:
    """The base hook applies the model's own contract around forward_step only."""

    states: list[bool] = []

    class _ContractStub(_ModelBaseStub):
        def forward_step(self, state, step_idx):
            states.append(torch.is_autocast_enabled("cpu"))
            return {"noise_pred": torch.zeros(1)}

    stub = _ContractStub()
    stub.precision = RolePrecision(
        dtype="bf16",
        float32_precision="ieee",
        outer_autocast=True,
    )
    stub.forward_step(object(), 0)
    stub.precision = RolePrecision(
        dtype="bf16",
        float32_precision="ieee",
        outer_autocast=False,
    )
    stub.forward_step(object(), 0)

    assert states == [True, False]
    assert torch.is_autocast_enabled("cpu") is False
