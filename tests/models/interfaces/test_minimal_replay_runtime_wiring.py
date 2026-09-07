from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.config.precision import QuantizationPolicy, RolePrecision
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions


def test_role_precision_rejects_quantized_base_dtype() -> None:
    """Quantization formats are selective GEMM policies, not autocast dtypes."""

    with pytest.raises(ValueError, match="precision role dtype must be one of"):
        RolePrecision(
            dtype="fp8",
            float32_precision="tf32",
        )


def test_role_precision_rejects_unknown_float32_backend() -> None:
    with pytest.raises(ValueError, match=r"precision\.float32_precision must be one of"):
        RolePrecision(
            dtype="fp32",
            float32_precision="fast",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dtype", ["fp8"])
def test_model_build_rejects_subbyte_parameter_storage(dtype: str) -> None:
    with pytest.raises(ValueError, match="neither is parameter storage"):
        ModelBuild(
            model_name_or_path="fake/repo",
            revision=None,
            device="cpu",
            parameter_dtype=dtype,
            family="sd3_5",
            precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        )


@pytest.mark.parametrize(
    ("quantization_format", "message"),
    [("fp4", "replaced.*nvfp4"), ("int8", "quantization.format")],
)
def test_quantization_policy_rejects_unknown_or_ambiguous_format(
    quantization_format: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        QuantizationPolicy(format=quantization_format)


def test_quantization_policy_normalizes_fp8_default_recipe() -> None:
    quantization = QuantizationPolicy(format="FP8")

    assert quantization.format == "fp8"
    assert quantization.recipe == "rowwise"


def test_quantization_policy_accepts_nvfp4_without_recipe() -> None:
    quantization = QuantizationPolicy(format="NVFP4")

    assert quantization.format == "nvfp4"
    assert quantization.recipe is None


def test_quantization_policy_rejects_nvfp4_recipe() -> None:
    with pytest.raises(ValueError, match=r"nvfp4.*does not accept.*recipe"):
        QuantizationPolicy(format="nvfp4", recipe="rowwise")


def _compile_scope_build(scope: str | None, *, rollout: bool) -> ModelBuild:
    torch_compile: dict[str, Any] = {"enable": True, "mode": "default"}
    if scope is not None:
        torch_compile["scope"] = scope
    return ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype=torch.float32,
        family="sd3_5",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        model_config={"torch_compile": torch_compile},
        rollout={"prompt_encoder_dtype": "fp32"} if rollout else None,
    )


@pytest.mark.parametrize(
    ("scope", "rollout_compiles", "replay_compiles"),
    [
        (None, True, True),  # absent scope means "all": today's behavior
        ("all", True, True),
        ("rollout", True, False),
        ("replay", False, True),
    ],
)
def test_torch_compile_property_resolves_scope_per_role(
    scope: str | None,
    rollout_compiles: bool,
    replay_compiles: bool,
) -> None:
    """One knob, two builds: each role reads only its own compile decision."""

    rollout_build = _compile_scope_build(scope, rollout=True)
    replay_build = _compile_scope_build(scope, rollout=False)

    expected = {"enable": True, "mode": "default"}
    assert rollout_build.torch_compile == (expected if rollout_compiles else None)
    assert replay_build.torch_compile == (expected if replay_compiles else None)


def test_torch_compile_property_refuses_unknown_scope() -> None:
    build = _compile_scope_build("trainer", rollout=False)

    with pytest.raises(ValueError, match=r"torch_compile\.scope must be one of"):
        _ = build.torch_compile


def test_model_build_reconstructs_nested_rollout_payload() -> None:
    """The primitive Ray mapping becomes the typed one-layer rollout contract."""

    build = ModelBuild(
        model_name_or_path="fake/repo",
        revision=None,
        device="cpu",
        parameter_dtype="fp16",
        family="sd3_5",
        precision={
            "dtype": "bf16",
            "float32_precision": "tf32",
            "quantization": {"format": "fp8", "recipe": "rowwise"},
            "outer_autocast": True,
        },
        rollout={
            "prompt_encoder_dtype": "fp32",
            "base_weight_sync": False,
        },
    )

    assert build.parameter_dtype is torch.float16
    assert build.precision == RolePrecision(
        "bf16",
        "tf32",
        QuantizationPolicy(format="fp8", recipe="rowwise"),
    )
    assert build.precision.outer_autocast is True
    assert isinstance(build.rollout, RolloutBuildOptions)
    assert build.rollout.prompt_encoder_dtype is torch.float32
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == "fp8"
    assert build.rollout.base_weight_sync is False


def test_model_build_resolver_projects_nvfp4_over_the_rollout_base_dtype() -> None:
    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    cfg = OmegaConf.create(
        {
            "model": {"family": "sd3_5", "path": "fake/repo"},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {
                    "dtype": "bf16",
                    "quantization": {"format": "nvfp4"},
                    "prompt_encoders": {"dtype": "fp16"},
                },
            },
        },
    )
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)

    build = get_model_family_entry("sd3_5").resolve_model_build(
        root,
        "cuda",
        precision=precision,
        for_rollout=True,
    )
    rollout = build.require_rollout()

    assert build.parameter_dtype is torch.bfloat16
    assert build.precision == RolePrecision(
        "bf16",
        "tf32",
        QuantizationPolicy(format="nvfp4"),
    )
    assert build.precision.outer_autocast is True
    assert rollout.prompt_encoder_dtype is torch.float16
    assert build.precision.quantization is not None
    assert build.precision.quantization.format == "nvfp4"
    assert build.precision.quantization.recipe is None


def test_full_generation_build_with_training_role_excludes_rollout_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline DPO may load generation modules without adopting rollout kernels."""

    from omegaconf import OmegaConf

    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config
    from vrl.models.families.registry import get_model_family_entry

    monkeypatch.setattr(
        "diffusers.DiffusionPipeline.load_config",
        lambda *_args, **_kwargs: {
            "boundary_ratio": None,
            "expand_timesteps": False,
        },
    )
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1",
                "path": "fake/repo",
                "revision": "a" * 40,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "bf16"},
                "rollout": {
                    "dtype": "bf16",
                    "quantization": {"format": "fp8"},
                },
            },
        },
    )
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)

    build = get_model_family_entry("wan_2_1").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=True,
        precision_role="training",
    )

    assert isinstance(build.require_rollout(), RolloutBuildOptions)
    assert build.precision == RolePrecision("bf16", "tf32")
    assert build.precision.quantization is None


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype


def _flow_match_scheduler(*_args: Any, **_kwargs: Any) -> Any:
    """Stand-in for ``load_flow_match_scheduler``: the real class from its defaults."""

    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler()


def _diffusers_scheduler(_build: Any, class_name: str, **_kwargs: Any) -> Any:
    """Stand-in for ``load_diffusers_scheduler``.

    The registry's ``scheduler_classname`` is its own source of truth, so build
    exactly that real class from its defaults instead of echoing a double.
    """

    import diffusers

    return getattr(diffusers, class_name)()


class _TinyRuntimeModel(nn.Module):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.language_model = nn.Linear(1, 1)

    def replay_forward(self, *_args: Any, **_kwargs: Any) -> Any:
        return None

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> Any:
        return self.load_state_dict(state_dict, strict=False)

    @property
    def adapter_roots(self) -> dict[str, Any]:
        # Mirrors ARModelBase: the checkpoint root is the wrapper, the adapter
        # is one hop in on language_model.
        return {"model": self.language_model}


def _build(**overrides: Any) -> ModelBuild:
    """Build a ModelBuild from friendly overrides.

    Translates the legacy ``use_lora`` / ``lora_config`` / ``scheduler_config``
    test kwargs into the carried ``model_config`` / ``sampling_config`` blocks
    so tests exercise the same read helpers the families use.
    """

    use_lora = bool(overrides.pop("use_lora", False))
    lora_config = overrides.pop("lora_config", None)
    scheduler_config = overrides.pop("scheduler_config", {"num_steps": 2})
    extra = overrides.pop("extra", None)

    model_config: dict[str, Any] = {"path": "fake/repo", "use_lora": use_lora}
    if lora_config is not None:
        model_config["lora"] = dict(lora_config)
    if extra is not None:
        # Legacy ``extra`` test fields (anima artifact paths, scheduler_shift)
        # now ride directly in the carried model block.
        model_config.update(dict(extra))

    values: dict[str, Any] = {
        "model_name_or_path": "fake/repo",
        "revision": None,
        "device": "cpu",
        "parameter_dtype": torch.float32,
        "family": "sd3_5",
        "precision": RolePrecision("fp32", "tf32", outer_autocast=False),
        "model_config": model_config,
        "sampling_config": dict(scheduler_config),
    }
    values.update(overrides)
    if str(values["family"]).startswith("wan_2_1"):
        model_config.setdefault("boundary_ratio", None)
        model_config.setdefault(
            "trainable_transformers",
            ["transformer_2"] if model_config["boundary_ratio"] is not None else ["transformer"],
        )
    return ModelBuild(**values)


@pytest.mark.parametrize(
    "family",
    ["sd3_5", "sana", "cogvideox", "wan_2_1"],
)
def test_registry_descriptor_replay_builder_returns_minimal_bundle(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    """The descriptor-driven generic replay builder (descriptor families).

    These families ship no builder functions: the registry entry's
    ``DenoiseFamilyBuild`` recipe drives the generic builder, keyed by
    ``build.family``. Behavioral contract matches the per-family builders above.
    """
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    loaded_builds: list[ModelBuild] = []

    def fake_transformer_loader(build: ModelBuild, *_args: Any, **_kwargs: Any):
        loaded_builds.append(build)
        return _TinyTransformer()

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        fake_transformer_loader,
    )
    monkeypatch.setattr(_shared_build, "load_flow_match_scheduler", _flow_match_scheduler)
    # Families with a scheduler_classname (cogvideox) load through the
    # classname path instead of the flow-match default.
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    entry = get_model_family_entry(family)
    bundle = entry.build_replay(
        _build(
            family=family,
            parameter_dtype=torch.float16 if family == "sana" else torch.float32,
        ),
    )

    assert bundle.loads_full_generation_modules is False
    assert loaded_builds
    if family == "sana":
        assert loaded_builds[-1].parameter_dtype is torch.float16
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline

    # ModelBuild rejects a missing identity before any registry dispatch.
    with pytest.raises(ValueError, match=r"ModelBuild\.family"):
        _build(family="")


def test_wan_replay_builder_uses_wan_pipeline_scheduler_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wan descriptor's scheduler_classname drives the generic replay loader."""
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    scheduler_classes: list[str] = []

    def fake_scheduler_loader(build: Any, class_name: str, **kwargs: Any) -> Any:
        scheduler_classes.append(class_name)
        return _diffusers_scheduler(build, class_name, **kwargs)

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", fake_scheduler_loader)

    bundle = get_model_family_entry("wan_2_1").build_replay(
        _build(family="wan_2_1"),
    )

    from diffusers import UniPCMultistepScheduler

    assert scheduler_classes == ["UniPCMultistepScheduler"]
    assert isinstance(bundle.scheduler, UniPCMultistepScheduler)


@pytest.mark.real_cover(
    "tests/models/steps/denoise/test_scheduler_logprob_parity.py"
    "::test_family_scheduler_sample_replay_parity",
    why=(
        "the transformer is a double and the rebuilt ladder is pinned by class and length only "
        "(literal timestep values move with the diffusers version); element-wise parity between "
        "this replay ladder and the rollout's runs on real schedulers in the counterpart"
    ),
)
@pytest.mark.parametrize(
    ("family", "scheduler_class"),
    [("mochi", "FlowMatchEulerDiscreteScheduler"), ("pixart_sigma", "DDIMScheduler")],
)
def test_replay_builders_standardize_the_loaded_scheduler_onto_the_rollout_ladder(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    scheduler_class: str,
) -> None:
    """``prepare_replay`` replaces the scheduler the loader handed over.

    Mochi ships ``invert_sigmas`` (ascending time) and PixArt ships a DPM-Solver;
    both replay models rebuild the rollout's own ladder from the shipped config
    and ``build.num_steps``. The loaded instance must therefore NOT survive.
    """
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    loaded: list[Any] = []

    def loader(*args: Any, **kwargs: Any) -> Any:
        scheduler = (
            _diffusers_scheduler(*args, **kwargs)
            if len(args) > 1
            else _flow_match_scheduler(*args, **kwargs)
        )
        loaded.append(scheduler)
        return scheduler

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_flow_match_scheduler", loader)
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", loader)

    bundle = get_model_family_entry(family).build_replay(
        _build(family=family, scheduler_config={"num_steps": 2}),
    )

    (shipped,) = loaded
    assert bundle.scheduler is not shipped
    assert type(bundle.scheduler).__name__ == scheduler_class
    assert bundle.scheduler.timesteps.numel() == 2


def test_wan_i2v_replay_builder_uses_i2v_replay_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The i2v registry entry's replay_cls selects the I2V replay model."""
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.families.wan_2_1.model import WanI2VReplayModel
    from vrl.models.steps.denoise import build as _shared_build

    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(family="wan_2_1_i2v"),
    )

    assert bundle.loads_full_generation_modules is False
    assert isinstance(bundle.model, WanI2VReplayModel)


def test_wan_dual_stage_replay_builder_loads_low_noise_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wan 2.2 dual-stage: prepare_replay late-loads transformer_2 and trains it."""
    import vrl.models.loader as _loader
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    loaded_subfolders: list[str] = []

    def fake_transformer_loader(
        _build: Any,
        _class_name: str,
        *,
        subfolder: str = "transformer",
    ) -> _TinyTransformer:
        loaded_subfolders.append(subfolder)
        return _TinyTransformer()

    # The generic builder loads the primary transformer; prepare_replay
    # late-loads transformer_2 through vrl.models.loader — patch both.
    monkeypatch.setattr(_shared_build, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(_loader, "load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)

    bundle = get_model_family_entry("wan_2_1_i2v").build_replay(
        _build(
            family="wan_2_1_i2v",
            extra={
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
            },
        ),
    )

    assert bundle.loads_full_generation_modules is False
    # Primary first (generic ctor), then the prepare_replay late-load.
    assert loaded_subfolders == ["transformer", "transformer_2"]
    assert set(bundle.trainable_modules) == {"transformer_2"}
    # boundary_ratio is behavior-consumed on the model (dual-stage transformer
    # routing), not bundle metadata — assert the consumed surface.
    assert bundle.model.boundary_ratio == 0.9


def test_cosmos_predict25_replay_builder_keeps_diffusion_nft_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predict2.5 replay bundle is minimal (no pipeline, no raw handle) yet still exposes
    ``diffusion_nft_prepare_transformer_input``, the surface DiffusionNFT needs from a replay
    model.
    """
    from vrl.models.families.cosmos.predict2_5 import model as predict25_model
    from vrl.models.families.registry import get_model_family_entry
    from vrl.models.steps.denoise import build as _shared_build

    # predict2_5 is a registry-descriptor family: the generic replay builder
    # constructs it, so the loaders are patched on the shared build module.
    monkeypatch.setattr(
        _shared_build,
        "load_diffusers_transformer",
        lambda *_args, **_kwargs: _TinyTransformer(),
    )
    monkeypatch.setattr(_shared_build, "load_diffusers_scheduler", _diffusers_scheduler)
    monkeypatch.setattr(
        predict25_model.CosmosPredict25ReplayModel,
        "apply_lora",
        lambda self, _build: self.transformer.requires_grad_(True),
    )

    bundle = get_model_family_entry("cosmos-predict2.5").build_replay(
        _build(
            family="cosmos-predict2.5",
            use_lora=True,
            lora_config={"rank": 1, "alpha": 1, "target_modules": ["to_q"]},
        ),
    )

    assert bundle.loads_full_generation_modules is False
    assert bundle.raw_handle is None
    assert callable(bundle.model.diffusion_nft_prepare_transformer_input)
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline


def test_anima_replay_builder_uses_only_transformer_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anima's replay bundle loads only the transformer checkpoint: no pipeline, no raw handle,
    the transformer as sole trainable module, and prompt encoding refused because no text
    encoder exists.
    """
    from vrl.models.families.cosmos.anima import runtime

    monkeypatch.setattr(
        runtime,
        "load_anima_transformer",
        lambda _build: _TinyTransformer(),
    )

    bundle = runtime.build_anima_replay_runtime_bundle(
        _build(
            family="cosmos-predict2-anima",
            extra={
                "transformer_path": "/tmp/anima-preview3-base.safetensors",
                "scheduler_shift": 3.0,
            },
        ),
    )

    assert bundle.loads_full_generation_modules is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"transformer"}
    with pytest.raises(RuntimeError, match="pipeline"):
        _ = bundle.model.pipeline
    with pytest.raises(RuntimeError, match="encode prompts"):
        bundle.model.encode_prompt("prompt")


def test_anima_empty_prompts_are_replaced_before_tokenization() -> None:
    """Empty or whitespace-only prompts become "." before tokenization so the tokenizer never sees
    an empty string.
    """
    from vrl.models.families.cosmos.anima.model import _non_empty_prompts

    assert _non_empty_prompts(["", "  ", "anime"]) == [".", ".", "anime"]


def test_anima_model_build_uses_explicit_local_paths() -> None:
    """Every explicit local path in the anima model section (transformer, text encoder, VAE, both
    tokenizers) reaches ``model_config`` unchanged for both the rollout and the replay build;
    only the rollout build carries ``RolloutBuildOptions``.
    """
    from vrl.config.loading import load_config
    from vrl.models.families.registry import get_model_family_entry

    cfg = load_config(
        "model/cosmos/anima_preview3",
        overrides=[
            "precision.training.dtype=bf16",
            "precision.float32_precision=tf32",
            "model.path=/models/anima",
            "model.transformer_path=/models/anima/transformer.safetensors",
            "model.text_encoder_path=/models/anima/text_encoder.safetensors",
            "model.vae_path=/models/anima/vae.safetensors",
            "model.qwen_tokenizer_path=/tokenizers/qwen",
            "model.t5_tokenizer_path=/tokenizers/t5",
            "sampling.num_steps=1",
            "model.use_lora=false",
        ],
    )
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    entry = get_model_family_entry("cosmos-predict2-anima")
    full = entry.resolve_model_build(root, "cpu", precision=precision)
    replay = entry.resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )

    assert isinstance(full.rollout, RolloutBuildOptions)
    assert replay.rollout is None
    assert full.model_config["transformer_path"] == "/models/anima/transformer.safetensors"
    assert full.model_config["text_encoder_path"] == "/models/anima/text_encoder.safetensors"
    assert full.model_config["vae_path"] == "/models/anima/vae.safetensors"
    assert full.model_config["qwen_tokenizer_path"] == "/tokenizers/qwen"
    assert full.model_config["t5_tokenizer_path"] == "/tokenizers/t5"
    assert replay.model_config["transformer_path"] == "/models/anima/transformer.safetensors"


def test_anima_artifact_resolution_fails_loud_when_hub_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub-fetch failure surfaces the config knob, not a raw download error."""
    from vrl.config.loading import load_config
    from vrl.models.families.registry import get_model_family_entry

    cfg = load_config(
        "model/cosmos/anima_preview3",
        overrides=[
            "precision.training.dtype=bf16",
            "precision.float32_precision=tf32",
            "sampling.num_steps=1",
            "model.use_lora=false",
        ],
    )
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import parse_config

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("cosmos-predict2-anima").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )

    # Resolution delegates to hf_hub_download (auto-fetch, same contract as
    # from_pretrained); when the hub fetch fails the error names the config
    # knob to set instead of leaking a bare download traceback.
    import huggingface_hub

    def _refuse(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _refuse)

    with pytest.raises(ValueError, match=r"model\.path='circlestone-labs/Anima'"):
        build.model_config["transformer_path"] = ""
        from vrl.models.families.cosmos.anima.runtime import _resolve_artifact

        _resolve_artifact(
            build.model_name_or_path,
            explicit_path="",
            relative_file=build.model_config["transformer_file"],
            field_name="transformer_path",
        )


@pytest.mark.parametrize(
    ("family", "model_module_path", "model_attr", "use_lora"),
    [
        (
            "janus_pro",
            "vrl.models.families.janus_pro.model",
            "JanusProReplayModel",
            True,
        ),
        (
            "nextstep_1",
            "vrl.models.families.nextstep_1.model",
            "NextStep1ReplayModel",
            False,
        ),
    ],
)
def test_ar_replay_builders_return_minimal_bundles(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_module_path: str,
    model_attr: str,
    use_lora: bool,
) -> None:
    """Every AR family's replay bundle is minimal: full generation modules not loaded, no raw
    handle, and ``model`` as the single trainable module, with or without LoRA.
    """
    from vrl.models.families.registry import get_model_family_entry

    model_module = __import__(model_module_path, fromlist=[model_attr])
    monkeypatch.setattr(model_module, model_attr, _TinyRuntimeModel)

    bundle = get_model_family_entry(family).build_replay(
        _build(family=family, use_lora=use_lora),
    )

    assert bundle.loads_full_generation_modules is False
    assert bundle.raw_handle is None
    assert set(bundle.trainable_modules) == {"model"}


@pytest.mark.parametrize(
    ("family", "model_module_path", "model_attr"),
    [
        ("janus_pro", "vrl.models.families.janus_pro.model", "JanusProModel"),
        ("emu3", "vrl.models.families.emu3.model", "Emu3Model"),
    ],
)
def test_ar_rollout_builders_follow_registry_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model_module_path: str,
    model_attr: str,
) -> None:
    from vrl.models.families.registry import get_model_family_entry

    model_module = __import__(model_module_path, fromlist=[model_attr])
    monkeypatch.setattr(model_module, model_attr, _TinyRuntimeModel)

    bundle = get_model_family_entry(family).build_rollout(
        _build(
            family=family,
            rollout=RolloutBuildOptions(
                prompt_encoder_dtype=torch.float16,
            ),
        ),
    )

    assert bundle.loads_full_generation_modules is True
    assert bundle.raw_handle is bundle.model
