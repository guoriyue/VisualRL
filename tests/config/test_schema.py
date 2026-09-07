"""Tests for the typed config schema boundary (vrl/config/schema.py)."""

from __future__ import annotations

import typing
from dataclasses import fields

import pytest
from omegaconf import OmegaConf

from tests.config.helpers import unknown_keys
from vrl.config.model_schema import (
    LoraSection,
    ModelExecutorSection,
    ModelMemorySection,
    ModelSection,
    TorchCompileSection,
    VaeDecodeMemorySection,
)
from vrl.config.sampling_schema import (
    ARSamplingSection,
    DenoiseImageSamplingSection,
    JanusProSamplingSection,
    TextEncodedImageSamplingSection,
    VideoSamplingSection,
)
from vrl.config.schema import (
    AlgorithmConfig,
    DataConfig,
    RewardConfig,
    RolloutRuntimeSection,
    parse_config,
)
from vrl.models.families.causvid.config import CausVidModelSection
from vrl.models.families.cosmos.anima.config import CosmosAnimaModelSection
from vrl.models.families.cosmos.predict2_5.config import (
    CosmosPredict25ModelSection,
)
from vrl.models.families.echo.config import EchoModelSection
from vrl.models.families.flux.config import FluxModelSection
from vrl.models.families.janus_pro.config import JanusProModelSection
from vrl.models.families.llamagen.config import LlamaGenModelSection
from vrl.models.families.magi_1.config import Magi1ModelSection
from vrl.models.families.names import _FAMILY_BY_ALIAS
from vrl.models.families.nextstep_1.config import NextStep1ModelSection
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR,
    SHARED_MODEL_SECTION_CLS,
    get_model_family_entry,
)
from vrl.models.families.wan_2_1.config import WanModelSection
from vrl.models.interfaces.generation_memory import (
    GenerationMemoryPolicy,
    VaeDecodeMemory,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# This independent fixture pins the public capability contract for every
# canonical family. Production derives executor support from its binding and
# records memory support as target-section names; the test intentionally does
# not derive either expected value from those production fields.
_MODEL_RUNTIME_CAPABILITY_MATRIX = {
    "sd3_5": (True, True),
    "causvid": (False, False),
    "magi_1": (False, False),
    "flux": (True, True),
    "qwen_image": (True, True),
    "sana": (True, True),
    "lumina2": (True, True),
    "hunyuan_video": (True, True),
    "mochi": (True, True),
    "hunyuan_image": (True, True),
    "pixart_sigma": (True, True),
    "cogvideox": (True, True),
    "wan_2_1": (True, True),
    "wan_2_1_i2v": (False, True),
    "cosmos-predict2": (False, True),
    "cosmos-predict2.5": (False, True),
    "cosmos3": (False, True),
    "cosmos-predict2-anima": (True, True),
    "echo": (False, False),
    "janus_pro": (False, False),
    "janus_pro_r1": (False, False),
    "nextstep_1": (False, False),
    "emu3": (False, False),
    "glm_image": (False, False),
    "llamagen": (False, False),
}


def _literal_args(annotation) -> tuple[str, ...]:
    """Flatten a ``Literal[...]`` or ``Literal[...] | None`` annotation into its
    members, so the allow-list tests derive their cases from the schema's single
    source of truth instead of hand-copying the Literal members (a copy never
    sees a newly added member, leaving it silently untested)."""
    members = typing.get_args(annotation)
    # Optional[Literal[...]]: unwrap each non-None union member's Literal args.
    if any(m is type(None) for m in members):
        return tuple(a for m in members if m is not type(None) for a in typing.get_args(m))
    return members


def _minimal_grpo_cfg(**overrides):
    base = {
        "algorithm": {"kind": "grpo"},
        "data": {
            "loader": "prompt_manifest",
            "manifest": "datasets/ocr/train.txt",
            "preprocessing": {"format": "text"},
            "sampler": {"type": "random_without_replacement"},
        },
        "rollout": {"sde": {"type": "flow_grpo"}},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _kling_video_reward_kwargs(**overrides) -> dict:
    base = {
        "sleep_offload": True,
        "reward_name": "org/model@main",
        "score_key": "overall",
        "worker_config": {"model_path": "/tmp/model"},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("value", [None, 1, 8])
def test_sampling_scheduler_batch_size_accepts_null_or_positive_integer(
    value: int | None,
) -> None:
    sampling = ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})

    assert sampling.ar_scheduler_batch_size == value


@pytest.mark.parametrize("value", [True, 0])
def test_sampling_scheduler_batch_size_rejects_coercible_or_non_positive_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer or null"):
        ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})


# ── Algorithm kind discriminator ──────────────────────────────────────────────


def test_unknown_algorithm_kind_raises() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kind = "qpo"
    with pytest.raises(ValueError, match=r"unknown algorithm\.kind"):
        parse_config(cfg)


def test_unknown_algorithm_keys_are_rejected_together() -> None:
    """Removed keys, typos, and never-seen keys: one error naming all of them."""

    cfg = OmegaConf.create(
        {"algorithm": {"kind": "grpo", "adv_estimator": "dpo", "future_field": True}}
    )
    assert unknown_keys(cfg) == ["algorithm.adv_estimator", "algorithm.future_field"]


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        ("grpo", "flow_kl_use_dt", True),
        ("dance_grpo", "sft_weight", 0.1),
        ("flow_dppo", "add_kl_coefficient", False),
        ("grpo_guard", "clip_ratio", 0.2),
        ("token_grpo", "kl_estimator", "k1"),
        ("token_grpo_multisegment", "segment_weights", {"a": 1.0}),
        ("diffusion_dpo", "beta", 5000.0),
        ("diffusion_nft", "nft_beta", 0.1),
    ],
)
def test_algorithm_keys_derive_from_selected_runtime_config(
    kind: str,
    field: str,
    value: object,
) -> None:
    """The section accepts exactly the selected dataclass's fields, typed, and
    hands back the built dataclass (cross-field rules are the root's job)."""

    section = AlgorithmConfig.model_validate({"kind": kind, field: value})
    assert section.kind == kind
    assert getattr(section.hyperparameters, field) == value


@pytest.mark.parametrize(
    ("kind", "foreign_field"),
    [("grpo", "beta"), ("diffusion_dpo", "clip_ratio")],
)
def test_algorithm_keys_are_scoped_to_selected_kind(kind: str, foreign_field: str) -> None:
    with pytest.raises(ValueError, match=rf"unknown algorithm\.{foreign_field}"):
        AlgorithmConfig.model_validate({"kind": kind, foreign_field: 1})


def test_algorithm_dispatch_covers_schema_kind_vocabulary() -> None:
    from vrl.config.algorithm import algorithm_config_class

    kinds = _literal_args(AlgorithmConfig.model_fields["kind"].annotation)
    assert kinds
    assert all(algorithm_config_class(kind) for kind in kinds)


def test_positive_kl_reward_coef_is_accepted_for_diffusion_rollouts() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = 0.25

    assert parse_config(cfg).algorithm.kl_reward_coef == 0.25


@pytest.mark.parametrize("kind", ["token_grpo", "diffusion_dpo"])
def test_positive_kl_reward_coef_rejects_trajectories_without_step_kl(
    kind: str,
) -> None:
    cfg = OmegaConf.create(
        {"algorithm": {"kind": kind, "kl_reward_coef": 0.25}},
    )

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef > 0 requires a diffusion rollout trajectory",
    ):
        parse_config(cfg)


def test_zero_kl_reward_coef_remains_valid_for_token_rollouts() -> None:
    cfg = OmegaConf.create(
        {"algorithm": {"kind": "token_grpo", "kl_reward_coef": 0.0}},
    )

    assert parse_config(cfg).algorithm.kl_reward_coef == 0.0


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_kl_reward_coef_rejects_invalid_public_values(value: object) -> None:
    cfg = _minimal_grpo_cfg()
    cfg.algorithm.kl_reward_coef = value

    with pytest.raises(
        ValueError,
        match=r"algorithm\.kl_reward_coef must be a finite number >= 0",
    ):
        parse_config(cfg)


# ── rollout / sampling string-setting Literals ────────────────────────────────


@pytest.mark.parametrize("mode", ["native", "sde"])
def test_valid_denoise_modes_accepted(mode: str) -> None:
    """Both denoise modes validate; the Literal is the user-facing allow-list."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = mode
    assert parse_config(cfg).rollout.denoise_mode == mode


def test_unknown_denoise_mode_raises() -> None:
    """An out-of-set denoise_mode is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.denoise_mode = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.denoise_mode"):
        parse_config(cfg)


def test_shared_attention_backend_omission_stays_unset() -> None:
    """The runtime fallback owns the default; typed config stores no duplicate."""
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {}
    sampling = parse_config(cfg).sampling

    assert type(sampling) is JanusProSamplingSection
    assert sampling.attention_backend is None
    assert sampling.model_dump(exclude_unset=True) == {}


def test_shared_attention_family_accepts_explicit_backend() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "torch_native"}

    assert parse_config(cfg).sampling.attention_backend == "torch_native"


def test_unknown_attention_backend_raises() -> None:
    """An out-of-set attention_backend is rejected at parse with the dotted path."""
    cfg = _minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "bogus"}
    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


def test_native_cache_family_rejects_typed_attention_backend() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "llamagen"})
    cfg.sampling = {"attention_backend": "torch_native"}

    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


def test_llamagen_sampling_rejects_model_derived_topology() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "llamagen"}, sampling={"image_token_num": 256})

    with pytest.raises(ValueError, match=r"unknown sampling\.image_token_num"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "family",
    ["janus_pro", "nextstep_1", "emu3", "glm_image"],
)
def test_text_encoded_ar_families_keep_request_text_length(family: str) -> None:
    cfg = _minimal_grpo_cfg(
        model={"family": family},
        sampling={"max_text_length": 128},
    )

    sampling = parse_config(cfg).sampling

    assert sampling is not None
    assert sampling.max_text_length == 128


def test_sampling_schema_is_selected_from_model_family() -> None:
    cfg = _minimal_grpo_cfg(model={"family": "sana"})
    cfg.sampling = {"max_sequence_length": 300}

    sampling = parse_config(cfg).sampling

    assert type(sampling) is TextEncodedImageSamplingSection
    assert sampling.max_sequence_length == 300


def test_sampling_section_requires_model_family_for_schema_selection() -> None:
    cfg = _minimal_grpo_cfg(sampling={"num_steps": 10})

    with pytest.raises(ValueError, match=r"sampling requires model\.family"):
        parse_config(cfg)


@pytest.mark.parametrize(
    ("family", "expected_type"),
    [
        ("hunyuan_image", DenoiseImageSamplingSection),
        ("cosmos-predict2", VideoSamplingSection),
    ],
)
def test_family_without_text_length_rejects_max_sequence_length(
    family: str,
    expected_type: type,
) -> None:
    cfg = _minimal_grpo_cfg(model={"family": family})
    cfg.sampling = {"max_sequence_length": 512}

    with pytest.raises(ValueError, match=r"unknown sampling\.max_sequence_length"):
        parse_config(cfg)

    parsed = parse_config(
        _minimal_grpo_cfg(model={"family": family}, sampling={}),
    )
    assert type(parsed.sampling) is expected_type


def test_echo_accepts_only_baked_guidance_value() -> None:
    valid = _minimal_grpo_cfg(model={"family": "echo"})
    valid.sampling = {"guidance_scale": 1.0}
    assert parse_config(valid).sampling.guidance_scale == 1.0

    invalid = _minimal_grpo_cfg(model={"family": "echo"})
    invalid.sampling = {"guidance_scale": 4.5}
    with pytest.raises(ValueError, match=r"unknown sampling\.guidance_scale=4\.5"):
        parse_config(invalid)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("janus_pro", "max_reflect_len", 80),
        ("magi_1", "guidance_scale", 4.5),
    ],
)
def test_sampling_fields_are_rejected_outside_their_behavior_owner(
    family: str,
    field: str,
    value: object,
) -> None:
    cfg = _minimal_grpo_cfg(model={"family": family}, sampling={field: value})

    with pytest.raises(ValueError, match=rf"unknown sampling\.{field}"):
        parse_config(cfg)


def test_reflection_length_is_owned_only_by_janus_r1() -> None:
    cfg = _minimal_grpo_cfg(
        model={"family": "janus_pro_r1"},
        sampling={"max_reflect_len": 80},
    )
    cfg.algorithm.kind = "token_grpo_multisegment"
    cfg.rollout.final_image_policy = "always_generate"

    assert parse_config(cfg).sampling.max_reflect_len == 80


def test_unknown_final_image_policy_raises() -> None:
    """final_image_policy is Literal-typed regardless of algorithm kind."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.final_image_policy = "bogus"
    with pytest.raises(ValueError, match=r"unknown rollout\.final_image_policy"):
        parse_config(cfg)


# ── distributed.training strategy ─────────────────────────────────────────────


def test_unknown_training_strategy_raises() -> None:
    """An unimplemented/typo strategy is rejected at parse time, not silently run."""
    cfg = _minimal_grpo_cfg(distributed={"training": {"strategy": "deepspeed"}})
    with pytest.raises(ValueError, match=r"unknown distributed\.training\.strategy"):
        parse_config(cfg)


# ── model family scoped keys ──────────────────────────────────────────────────


def test_wan_model_keys_are_scoped_to_wan_family() -> None:
    """Wan trainable-topology/offload keys are accepted only for Wan families."""

    wan_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(wan_cfg) == []

    alias_cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(alias_cfg) == []

    sd3_cfg = OmegaConf.create(
        {
            "model": {
                "family": "sd3_5",
                "path": "stabilityai/stable-diffusion-3.5-medium",
                "boundary_ratio": 0.9,
                "trainable_transformers": ["transformer_2"],
                "offload_mode": "sequential",
            },
        },
    )
    assert unknown_keys(sd3_cfg) == [
        "model.boundary_ratio",
        "model.offload_mode",
        "model.trainable_transformers",
    ]


def test_wan_boundary_ratio_is_source_derived_not_public_config() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "boundary_ratio": 0.9,
            },
        },
    )

    assert unknown_keys(cfg) == ["model.boundary_ratio"]


@pytest.mark.parametrize("family", ["janus_pro", "janus_pro_r1"])
def test_janus_families_select_the_shared_family_section(family: str) -> None:
    cfg_data: dict[str, object] = {
        "model": {
            "family": family,
            "trust_remote_code": True,
            "vq_latent_channels": 8,
        },
    }
    if family == "janus_pro_r1":
        cfg_data.update(
            {
                "algorithm": {"kind": "token_grpo_multisegment"},
                "rollout": {"final_image_policy": "always_generate"},
            },
        )
    cfg = OmegaConf.create(cfg_data)

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, JanusProModelSection)
    assert parsed.model.trust_remote_code is True
    assert parsed.model.vq_latent_channels == 8


@pytest.mark.parametrize(
    ("family", "field"),
    [
        ("emu3", "vq_latent_channels"),
        ("sd3_5", "nft_previous_adapter"),
        ("cosmos-predict2", "skip_text_encoder"),
    ],
)
def test_family_owned_keys_are_unknown_for_sibling_families(family: str, field: str) -> None:
    """A key declared by one family's section is a typo for every other family."""

    cfg = OmegaConf.create({"model": {"family": family, field: True}})

    assert unknown_keys(cfg) == [f"model.{field}"]


def test_nextstep_keys_select_family_section() -> None:
    payload = {
        "freeze_vae": True,
        "vae_path": "stepfun-ai/NextStep-1-f8ch16-Tokenizer",
        "vae_revision": "immutable",
    }
    cfg = OmegaConf.create({"model": {"family": "nextstep_1", **payload}})

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, NextStep1ModelSection)
    assert parsed.model.freeze_vae is True
    assert parsed.model.vae_path == payload["vae_path"]
    assert parsed.model.vae_revision == payload["vae_revision"]


def test_unknown_wan_offload_mode_raises() -> None:
    """Wan offload mode is a typed three-state enum, not two independent bools."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "offload_mode": "stream",
            },
        },
    )
    with pytest.raises(ValueError, match=r"unknown model\.offload_mode"):
        parse_config(cfg)


def test_root_retains_selected_family_model_section_and_serializes_its_fields() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "wan_2_1_i2v",
                "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "expert_lifecycle_profiling": True,
                "offload_mode": "sequential",
            },
        },
    )

    parsed = parse_config(cfg)

    assert isinstance(parsed.model, WanModelSection)
    assert parsed.model.expert_lifecycle_profiling is True
    assert parsed.model_dump()["model"]["offload_mode"] == "sequential"


def test_cosmos_predict25_keys_select_family_section() -> None:
    cfg = OmegaConf.create(
        {"model": {"family": "cosmos-predict2.5", "skip_text_encoder": True}},
    )

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosPredict25ModelSection)
    assert parsed.model.skip_text_encoder is True


def test_cosmos_anima_keys_select_family_section() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2-anima",
                "qwen_tokenizer_path": "Qwen/Qwen2.5-0.5B",
                "scheduler_shift": 3.0,
                "transformer_file": "split_files/diffusion_models/anima.safetensors",
            },
        },
    )

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, CosmosAnimaModelSection)
    assert parsed.model.scheduler_shift == 3.0
    assert parsed.model.transformer_file == "split_files/diffusion_models/anima.safetensors"


def test_llamagen_keys_select_family_section() -> None:
    payload = {
        "gpt_ckpt": "custom-gpt.pt",
        "gpt_model": "GPT-XL",
        "image_token_num": 256,
        "t5_path": "org/t5",
        "t5_revision": "immutable",
        "vq_ckpt": "custom-vq.pt",
    }
    cfg = OmegaConf.create({"model": {"family": "llamagen", **payload}})

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert isinstance(parsed.model, LlamaGenModelSection)
    assert parsed.model is not None
    parsed_payload = parsed.model.model_dump()
    assert {key: parsed_payload[key] for key in payload} == payload


@pytest.mark.parametrize(
    ("family", "section_cls", "payload"),
    [
        ("flux", FluxModelSection, {"nft_previous_adapter": True}),
        (
            "echo",
            EchoModelSection,
            {
                "gemma_path": "google/gemma-3-12b-it",
                "gemma_revision": "revision",
                "video_height": 256,
                "video_width": 256,
            },
        ),
        (
            "causvid",
            CausVidModelSection,
            {
                "accept_noncommercial_license": True,
                "base_model_path": "Wan-AI/Wan2.1-T2V-1.3B",
                "checkpoint_file": "autoregressive_checkpoint/model.pt",
            },
        ),
        (
            "magi_1",
            Magi1ModelSection,
            {
                "python_executable": "third_party/MAGI-1/.venv/bin/python",
                "source_path": "third_party/MAGI-1",
                "timeout_seconds": 3600,
            },
        ),
    ],
)
def test_family_owned_denoise_keys_select_their_public_sections(
    family: str,
    section_cls: type[ModelSection],
    payload: dict[str, object],
) -> None:

    cfg = OmegaConf.create({"model": {"family": family, **payload}})

    assert unknown_keys(cfg) == []
    parsed = parse_config(cfg)
    assert type(parsed.model) is section_cls
    assert parsed.model is not None
    parsed_payload = parsed.model.model_dump()
    assert {key: parsed_payload[key] for key in payload} == payload


def test_model_family_aliases_select_their_canonical_section_classes() -> None:
    for alias, family in _FAMILY_BY_ALIAS.items():
        canonical = parse_config(
            OmegaConf.create({"model": {"family": family}}),
        )
        parsed_alias = parse_config(
            OmegaConf.create({"model": {"family": alias}}),
        )

        assert type(parsed_alias.model) is type(canonical.model)
        assert parsed_alias.model.family == alias


def test_model_runtime_capability_matrix_covers_every_registered_family() -> None:
    assert set(_MODEL_RUNTIME_CAPABILITY_MATRIX) == set(FAMILY_REGISTRY)

    for family, (supports_executor, supports_memory) in _MODEL_RUNTIME_CAPABILITY_MATRIX.items():
        entry = get_model_family_entry(family)

        assert (entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR) is supports_executor
        assert entry.runtime_capabilities.supported_model_memory_sections == (
            frozenset({"vae_decode"}) if supports_memory else frozenset()
        )


def test_shared_nested_model_sections_preserve_explicit_falsy_presence() -> None:
    raw_model = {
        "family": "flux",
        "lora": {
            "rank": 0,
            "alpha": 0,
            "path": None,
            "target_modules": [],
            "init_lora_weights": False,
            "dropout": 0.0,
            "init": None,
        },
        "memory": {
            "vae_decode": {
                "tiling": False,
                "slicing": None,
            },
        },
        "torch_compile": {
            "enable": False,
            "mode": None,
        },
        "executor": {
            "num_frames": 0,
            "max_sequence_length": 0,
            "fps": None,
            "batch_passthrough_keys": [],
        },
    }

    parsed = parse_config(OmegaConf.create({"model": raw_model}))

    assert isinstance(parsed.model, ModelSection)
    assert isinstance(parsed.model.lora, LoraSection)
    assert isinstance(parsed.model.memory, ModelMemorySection)
    assert isinstance(parsed.model.memory.vae_decode, VaeDecodeMemorySection)
    assert isinstance(parsed.model.torch_compile, TorchCompileSection)
    assert isinstance(parsed.model.executor, ModelExecutorSection)
    assert parsed.model.model_dump(exclude_unset=True) == raw_model


def test_generation_memory_schema_and_policy_share_one_field_vocabulary() -> None:
    assert tuple(ModelMemorySection.model_fields) == tuple(
        policy_field.name for policy_field in fields(GenerationMemoryPolicy)
    )
    assert tuple(VaeDecodeMemorySection.model_fields) == tuple(
        policy_field.name for policy_field in fields(VaeDecodeMemory)
    )


@pytest.mark.parametrize(
    ("model_fragment", "path"),
    [
        ({"lora": {"rnak": 16}}, "model.lora.rnak"),
        (
            {"memory": {"vae_decode": {"tileing": True}}},
            "model.memory.vae_decode.tileing",
        ),
    ],
)
def test_model_subtree_typos_are_named_with_complete_paths(
    model_fragment: dict[str, object],
    path: str,
) -> None:
    cfg = OmegaConf.create({"model": {"family": "flux", **model_fragment}})

    assert unknown_keys(cfg) == [path]


def _parse_error(model: dict[str, object]) -> str | None:
    try:
        parse_config(OmegaConf.create({"model": model}))
    except ValueError as error:
        return str(error)
    return None


def test_model_runtime_sections_follow_family_capabilities() -> None:
    """``model.executor`` / ``model.memory`` parse only for families that consume them."""

    for family, (supports_executor, supports_memory) in _MODEL_RUNTIME_CAPABILITY_MATRIX.items():
        executor_error = _parse_error(
            {"family": family, "executor": {"max_sequence_length": 123}},
        )
        memory_error = _parse_error(
            {"family": family, "memory": {"vae_decode": {"tiling": True}}},
        )
        if supports_executor:
            assert executor_error is None, family
        else:
            assert executor_error == f"model family {family!r} does not support model.executor"
        if supports_memory:
            assert memory_error is None, family
        else:
            assert memory_error == (
                f"model family {family!r} does not support model.memory section(s): vae_decode"
            )


@pytest.mark.parametrize("empty_value", [None, {}])
def test_empty_model_runtime_sections_are_valid_for_every_family(
    empty_value: object,
) -> None:
    for family in _MODEL_RUNTIME_CAPABILITY_MATRIX:
        parsed = parse_config(
            OmegaConf.create(
                {
                    "model": {
                        "family": family,
                        "executor": empty_value,
                        "memory": empty_value,
                    },
                },
            ),
        )

        assert parsed.model is not None


def test_shared_only_families_use_the_shared_model_section() -> None:
    shared_families = [
        entry.family
        for entry in FAMILY_REGISTRY.values()
        if entry.model_section_cls == SHARED_MODEL_SECTION_CLS
    ]
    assert shared_families

    for family in shared_families:
        parsed = parse_config(
            OmegaConf.create({"model": {"family": family, "path": f"org/{family}"}}),
        )

        assert type(parsed.model) is ModelSection
        assert parsed.model.path == f"org/{family}"


def test_unknown_model_family_fails_at_typed_parse() -> None:
    cfg = OmegaConf.create({"model": {"family": "not_a_family", "path": "org/model"}})

    with pytest.raises(ValueError, match=r"unsupported model family: 'not_a_family'"):
        parse_config(cfg)


def test_present_model_section_requires_a_family() -> None:
    cfg = OmegaConf.create({"model": {"path": "org/model"}})

    with pytest.raises(ValueError, match=r"config missing required field: model\.family"):
        parse_config(cfg)


# ── distributed.rollout knobs ─────────────────────────────────────────────────


def test_unknown_batch_placement_strategy_raises() -> None:
    """A typo batch placement strategy is rejected at parse time, not at launch."""
    cfg = _minimal_grpo_cfg(
        distributed={"rollout": {"batch_placement_strategy": "work_stealing"}},
    )
    with pytest.raises(
        ValueError,
        match=r"unknown distributed\.rollout\.batch_placement_strategy",
    ):
        parse_config(cfg)


def test_rollout_health_check_defaults_and_accepts_override() -> None:
    default = parse_config(_minimal_grpo_cfg(distributed={"rollout": {}}))
    assert default.distributed.rollout.health_check_interval_s == 30.0
    assert default.distributed.rollout.health_check_timeout_s == 30.0
    assert default.distributed.rollout.health_check_first_wait_s == 0.0
    assert default.distributed.rollout.worker_rpc_timeout_s == 600.0
    assert default.distributed.rollout.generation_stall_timeout_s == 3600.0

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 12.5,
                "health_check_timeout_s": 7.5,
                "health_check_first_wait_s": 2.5,
                "worker_rpc_timeout_s": 3600.0,
                "generation_stall_timeout_s": 1200.0,
            }
        },
    )
    rollout = parse_config(cfg).distributed.rollout
    assert rollout.health_check_interval_s == 12.5
    assert rollout.health_check_timeout_s == 7.5
    assert rollout.health_check_first_wait_s == 2.5
    assert rollout.worker_rpc_timeout_s == 3600.0
    assert rollout.generation_stall_timeout_s == 1200.0


def test_rollout_worker_section_mirrors_worker_runtime_config() -> None:
    """RolloutRuntimeSection (pydantic lint boundary) and RolloutWorkerConfig (the
    frozen runtime projection composed into RayGenerationConfig) must stay
    field-for-field identical. ``from_public_section`` builds the dataclass via
    ``cls(**section.model_dump())``, so a field on one but not the other silently
    breaks at runtime (TypeError / unfilled required field) instead of at parse.

    The two types are deliberately NOT merged (the pydantic schema is a lint-only
    boundary; the dataclass is the real runtime consumer), so this parity test is
    the guard against drift. Public defaults remain only on the section; the
    runtime dataclass has no fallback literals.
    """
    import dataclasses

    from vrl.generation.ray.config import RolloutWorkerConfig

    section_fields = set(RolloutRuntimeSection.model_fields)
    config_fields = {f.name for f in dataclasses.fields(RolloutWorkerConfig)}
    assert section_fields == config_fields
    assert "health_check_first_wait_s" in section_fields
    assert "worker_rpc_timeout_s" in section_fields
    assert "generation_stall_timeout_s" in section_fields

    # Per-field default parity: the section's declared defaults must survive the
    # projection unchanged (from_public_section adds no fallbacks or overrides), so
    # the section stays the single home of the default literals.
    projected = RolloutWorkerConfig.from_public_section(RolloutRuntimeSection())
    for name in section_fields:
        assert getattr(projected, name) == RolloutRuntimeSection.model_fields[name].default


def test_rollout_health_check_interval_le_zero_disables_probe() -> None:
    """A non-positive interval turns the probe off; the timeout is then unchecked."""

    cfg = _minimal_grpo_cfg(
        distributed={
            "rollout": {
                "health_check_interval_s": 0.0,
                "health_check_timeout_s": 0.0,
            }
        },
    )

    assert parse_config(cfg).distributed.rollout.health_check_interval_s == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("health_check_interval_s", float("nan"), r"health_check_interval_s must be finite"),
        ("health_check_timeout_s", 0.0, r"health_check_timeout_s must be finite and > 0"),
        ("health_check_first_wait_s", -1.0, r"health_check_first_wait_s must be finite and >= 0"),
        ("worker_rpc_timeout_s", float("inf"), r"worker_rpc_timeout_s must be finite and > 0"),
        ("generation_stall_timeout_s", 0.0, r"generation_stall_timeout_s must be finite and > 0"),
    ],
)
def test_rollout_worker_timeouts_must_be_finite_and_in_range(
    field: str,
    value: float,
    message: str,
) -> None:
    cfg = _minimal_grpo_cfg(distributed={"rollout": {field: value}})

    with pytest.raises(ValueError, match=message):
        parse_config(cfg)


# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize("loader", _literal_args(DataConfig.model_fields["loader"].annotation))
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    """Every loader in the DataConfig.loader Literal allow-list is accepted; the
    per-loader construction branches below stay as real behavior coverage."""
    if loader == "prompt_manifest":
        data = DataConfig(
            loader=loader,
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": "text"},
            sampler={"type": "random_without_replacement"},
        )
    elif loader == "prompt_image_manifest":
        data = DataConfig(
            loader=loader,
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            loader=loader,
            dataset_name="org/dataset",
            split="train",
            cache_dir="/tmp/cache",
            preprocessing={"resolution": 512, "random_crop": False, "horizontal_flip": True},
            sampler={"shuffle": True, "drop_last": True, "dataloader_num_workers": 4},
        )
    assert data.loader == loader


def test_unknown_data_loader_raises() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.data.loader = "s3_loader"
    with pytest.raises(ValueError, match=r"unknown data\.loader"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("image_caption_jsonl", "prompt_image_manifest"),
        ("jsonl", "prompt_manifest"),
        ("text", "prompt_manifest"),
    ],
)
def test_omitted_loader_derives_from_preprocessing_format(fmt: str, expected: str) -> None:
    """An omitted data.loader is derived from preprocessing.format for the prompt-* family."""
    if expected == "prompt_image_manifest":
        data = DataConfig(
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": fmt},
            sampler={"type": "random_without_replacement"},
        )
    assert data.loader == expected


@pytest.mark.parametrize(
    ("loader", "fmt", "message"),
    [
        (
            "prompt_manifest",
            "image_caption_jsonl",
            r"requires.*prompt_image_manifest",
        ),
        (
            "prompt_image_manifest",
            "text",
            r"requires.*image_caption_jsonl",
        ),
    ],
)
def test_explicit_data_loader_rejects_preprocessing_format_conflict(
    loader: str,
    fmt: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(
            loader=loader,
            manifest="train.jsonl",
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_requires_image_caption_fields() -> None:
    with pytest.raises(ValueError, match=r"data\.preprocessing\.caption_field"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest="x",
            eval_manifest="y",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_manifest_accepts_mixture_counts() -> None:
    """A {path: count} manifest is the recipe-level way to declare a prompt mix."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
        mix_seed=20260818,
        preprocessing={"format": "jsonl"},
        sampler={"type": "random_without_replacement"},
    )
    assert data.manifest == {"anatomy.jsonl": 6800, "safety.jsonl": 1200}


def test_prompt_manifest_mixture_requires_a_seed() -> None:
    """A mixture without a seed would draw a different prompt set on every rank."""
    with pytest.raises(ValueError, match=r"data\.mix_seed"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


@pytest.mark.parametrize("count", [0, "many"])
def test_prompt_manifest_rejects_non_positive_mixture_count(count: object) -> None:
    """A mixture count that cannot select prompts fails at config time."""
    with pytest.raises(ValueError, match=r"positive prompt count"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": count},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_rejects_mixture() -> None:
    """Image-conditioned runs pair one manifest with its reference tree."""
    with pytest.raises(ValueError, match=r"single data\.manifest path"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest={"a.jsonl": 10, "b.jsonl": 10},
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


# ── Sampler type literal ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sampler_type",
    ["random_without_replacement", "sequential_window"],
)
def test_valid_sampler_types_are_accepted(sampler_type: str) -> None:
    """Every sampler type in the registered set is accepted as-is."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest="x",
        preprocessing={"format": "text"},
        sampler={"type": sampler_type},
    )
    assert data.sampler.type == sampler_type


def test_unknown_sampler_type_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )


# ── Reward weight validation ──────────────────────────────────────────────────


def test_zero_weight_observation_component_is_valid() -> None:
    """Checks zero weight keeps a component valid for observation-only scoring."""
    cfg = RewardConfig.model_validate({"components": {"kling_video_reward": 0.0}, "kwargs": {}})
    assert cfg.components["kling_video_reward"] == 0.0


def test_non_numeric_reward_weight_raises() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        RewardConfig.model_validate({"components": {"aesthetic": "heavy"}, "kwargs": {}})


def test_reward_http_inference_config_is_typed_beside_open_component_kwargs() -> None:
    """Transport config is typed even though reward-specific kwargs are open."""

    cfg = RewardConfig.model_validate(
        {
            "components": {"videoscore2": 1.0},
            "kwargs": {"videoscore2": {"artifact_dir": "/shared/artifacts"}},
            "inference": {
                "videoscore2": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "expected_model": "videoscore2-v1",
                },
            },
        },
    )

    assert cfg.inference["videoscore2"].kind == "http"


def test_reward_inference_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match=r"unsupported reward\.inference\..* keys"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "videoscore2": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                        "service_url": "http://legacy",
                    },
                },
            },
        )


def test_reward_inference_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        RewardConfig.model_validate(
            {
                "components": {"videoscore2": 1.0},
                "inference": {
                    "typo_component": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                },
            },
        )


def test_grpo_requires_valid_sde_type() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "euler"
    with pytest.raises(ValueError, match=r"unknown rollout\.sde\.type"):
        parse_config(cfg)


def test_grpo_accepts_cps_sde_type() -> None:
    """``cps`` is a valid ``rollout.sde.type`` for GRPO."""
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "cps"
    parsed = parse_config(cfg)
    assert parsed.algorithm.kind == "grpo"


def test_token_grpo_multisegment_requires_explicit_janus_r1_family() -> None:
    """The algorithm cannot silently turn base Janus into the R1 protocol."""
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo_multisegment"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_pro"},
            "rollout": {"final_image_policy": "always_generate"},
        }
    )
    with pytest.raises(ValueError, match="janus_pro_r1"):
        parse_config(cfg)


def test_janus_r1_family_requires_multisegment_algorithm() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "token_grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
            },
            "model": {"family": "janus_r1"},
            "rollout": {},
        }
    )

    with pytest.raises(ValueError, match="token_grpo_multisegment"):
        parse_config(cfg)


def test_production_video_reward_structural_rules() -> None:
    """A production Kling video-reward config with sleep_offload, a hub reward name, mp4 video
    artifacts and a text-to-video task passes the production reward contract.
    """
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_manifest",
                "manifest": "x",
                "preprocessing": {},
                "sampler": {"type": "random_without_replacement"},
                "task_type": "text_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "sleep_offload": True,
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        }
    )
    from vrl.config.validation import validate_production_reward_contract

    validate_production_reward_contract(parse_config(cfg))


def test_production_video_reward_accepts_image_to_video_task_type() -> None:
    """The production contract also accepts ``image_to_video`` with an image-caption manifest and
    reference-image conditioning.
    """
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "grpo"},
            "data": {
                "loader": "prompt_image_manifest",
                "manifest": "x",
                "eval_manifest": "y",
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                    "conditioning": "reference_image",
                },
                "sampler": {"type": "random_without_replacement"},
                "task_type": "image_to_video",
            },
            "rollout": {"sde": {"type": "cps"}},
            "reward": {
                "components": {"kling_video_reward": 1.0},
                "kwargs": {
                    "kling_video_reward": {
                        "sleep_offload": True,
                        "reward_name": "org/model@main",
                        "score_key": "overall",
                        "media_type": "video",
                        "artifact_format": "mp4",
                        "worker_config": {},
                    }
                },
            },
            "production": {"kling_video_reward": {"enabled": True}},
        },
    )

    parsed = parse_config(cfg)

    assert parsed.data.task_type == "image_to_video"


def test_production_gate_defaults_to_disabled_and_accepts_enabled() -> None:
    disabled = parse_config(OmegaConf.create({"production": {}}))
    assert disabled.production.kling_video_reward.enabled is False

    enabled = parse_config(
        OmegaConf.create({"production": {"kling_video_reward": {"enabled": True}}}),
    )
    assert enabled.production.kling_video_reward.enabled is True


# ── Missing field mapping (??? → ValueError) ──────────────────────────────────


def test_missing_mandatory_value_produces_repo_standard_message() -> None:
    """An OmegaConf ``???`` marker surfaces as the repo-standard 'config missing required field'
    error.
    """
    cfg = _minimal_grpo_cfg()
    cfg.rollout.sde.type = "flow_grpo"
    # Inject an OmegaConf mandatory-missing marker
    OmegaConf.update(cfg, "algorithm.kind", "???")
    with pytest.raises(ValueError, match="config missing required field"):
        parse_config(cfg)


# ── extra="ignore" migration policy ──────────────────────────────────────────


def test_unknown_top_level_sections_are_rejected() -> None:
    """parse_config is the one gate: a section no consumer reads fails loud."""
    cfg = _minimal_grpo_cfg()
    OmegaConf.update(cfg, "some_future_section.foo", "bar")
    with pytest.raises(ValueError, match=r"unknown some_future_section"):
        parse_config(cfg)


# ── algorithm.sft_weight x data.sft_latents (regularizer data channel) ───────


def test_sft_weight_without_latents_shard_raises() -> None:
    """A weight without its data channel would be a silent no-op knob."""
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    with pytest.raises(ValueError, match=r"data\.sft_latents"):
        parse_config(cfg)


def test_sft_weight_with_latents_shard_parses() -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": 0.1})
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


def test_diffusion_dpo_sft_weight_does_not_require_online_latents_shard() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "diffusion_dpo", "sft_weight": 0.1},
    )
    del cfg.rollout
    parsed = parse_config(cfg)
    assert parsed.algorithm.hyperparameters.sft_weight == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("section", "payload", "field"),
    [
        ("actor", {"ema": {"enable": True}}, "actor.ema"),
        ("rollout", {"prompts_per_batch": 1}, "rollout.prompts_per_batch"),
    ],
)
def test_diffusion_dpo_rejects_online_only_config_fields(
    section: str,
    payload: dict,
    field: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            section: payload,
        },
    )

    with pytest.raises(ValueError, match=rf"{field}"):
        parse_config(cfg)


def test_diffusion_dpo_accepts_its_resume_and_optimizer_surface() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"kind": "diffusion_dpo"},
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "bf16"},
            },
            "actor": {
                "optim": {"lr": 1e-8},
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "max_norm": 1.0,
                "prediction_type": "flow_matching",
                "scale_lr": False,
                "train_batch_size": 1,
                "use_adafactor": False,
            },
            "trainer": {
                "checkpointing_steps": 10,
                "entrypoint": "pkg.module:train",
                "log_interval": 1,
                "max_train_steps": 20,
                "output_dir": "outputs/dpo",
                "resume_from": "",
                "resume_strict": True,
            },
        },
    )

    parsed = parse_config(cfg)

    assert parsed.algorithm.kind == "diffusion_dpo"


def test_latents_shard_without_weight_is_inert_and_allowed() -> None:
    cfg = _minimal_grpo_cfg()
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    parse_config(cfg)


@pytest.mark.parametrize("value", [-0.1, float("nan")])
def test_sft_weight_must_be_finite_and_nonnegative(value: float) -> None:
    cfg = _minimal_grpo_cfg(algorithm={"kind": "grpo", "sft_weight": value})
    with pytest.raises(ValueError, match="finite number >= 0"):
        parse_config(cfg)


def test_sft_weight_rejects_non_diffusion_grpo_kind() -> None:
    cfg = _minimal_grpo_cfg(
        algorithm={"kind": "flash_grpo", "sft_weight": 0.1},
    )
    cfg.data.sft_latents = "data/droid/sft_latents.pt"
    with pytest.raises(ValueError, match="only for diffusion"):
        parse_config(cfg)
