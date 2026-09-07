"""Tests for rollout runtime input construction."""

from __future__ import annotations

import pickle
import threading
from typing import Any

import pytest
from omegaconf import OmegaConf

from tests.generation.execution._helpers import launch_contract
from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.generation.bindings.full_sequence_denoise import DiffusionBatchGatherer
from vrl.generation.bindings.token_autoregressive.executor import ARDiscreteBatchGatherer
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationBatchExecutor
from vrl.generation.ray.launch_inputs import RayGenerationLaunchInputs
from vrl.models.families.janus_pro.runtime import JanusProR1GenerationBatchGatherer
from vrl.models.families.nextstep_1.runtime import NextStep1GenerationBatchGatherer
from vrl.models.families.registry import (
    FAMILY_REGISTRY,
    ModelFamilyEntry,
    get_model_family_entry,
)
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import (
    resolve_model,
    resolve_online_run,
)


class _TestGatherer:
    def gather_batches(self, *_args: Any) -> Any:
        raise AssertionError("test gatherer must not execute")


class _UnpickleableGatherer(_TestGatherer):
    def __init__(self) -> None:
        self.lock = threading.Lock()


def test_ray_launch_inputs_reject_unpickleable_gatherer_state() -> None:
    with pytest.raises(TypeError, match="must be pickle-serializable"):
        RayGenerationLaunchInputs(
            launch_contract=launch_contract(family="test"),
            gatherer=_UnpickleableGatherer(),
        )


def _capture_launch_inputs(
    cfg: Any,
    entry: ModelFamilyEntry,
) -> tuple[RayGenerationLaunchInputs, dict[str, Any]]:
    """Resolve the production launch payload without starting Ray actors."""

    run = resolve_online_run(cfg)
    assert run.family is entry
    replay_model = resolve_model(
        entry,
        run.built.root,
        run.device,
        precision=run.built.precision,
        for_rollout=False,
    )
    result = run.ray_launch_inputs(replay_model)
    return result, replay_model.identity


@pytest.mark.parametrize(
    ("family", "entry"),
    FAMILY_REGISTRY.items(),
    ids=FAMILY_REGISTRY,
)
def test_every_registry_entry_has_pickle_safe_ray_launch_inputs(
    family: str,
    entry: ModelFamilyEntry,
) -> None:
    """Every registry binding, including entries without presets, crosses Ray."""
    inputs = RayGenerationLaunchInputs(
        launch_contract=GenerationRuntimeLaunchContract(
            family=family,
            model_build={"device": "cpu"},
            expected_model_identity={"model_path": f"registry://{family}"},
            executor_kwargs={"registry_family": family},
            policy_version=7,
            torch_profiler={"enabled": False},
            sleep_offload=True,
            versioned_weight_sync=True,
        ),
        gatherer=entry.new_gatherer(),
    )

    restored = pickle.loads(pickle.dumps(inputs))

    assert isinstance(restored, RayGenerationLaunchInputs)
    assert restored.launch_contract == inputs.launch_contract
    assert (
        f"{type(restored.gatherer).__module__}:{type(restored.gatherer).__qualname__}"
        == entry.gatherer_cls
    )
    assert callable(restored.gatherer.gather_batches)
    assert not isinstance(restored.gatherer, GenerationBatchExecutor)


@pytest.mark.parametrize(
    ("experiment", "family", "expected_gatherer", "overrides"),
    [
        ("sd3_5/online_grpo_ocr", "sd3_5", DiffusionBatchGatherer, ()),
        (
            "wan_2_1/online_grpo_physics_i2v",
            "wan_2_1_i2v",
            DiffusionBatchGatherer,
            (),
        ),
        (
            "anima_preview3/online_grpo",
            "cosmos-predict2-anima",
            DiffusionBatchGatherer,
            (
                "+reward=aesthetic",
                "+dataset=drawbench_train_192",
                "actor.optim.lr=1.0e-5",
                "trainer.output_dir=outputs/test_anima_launch_inputs",
            ),
        ),
        (
            "janus_pro/online_grpo_ocr",
            "janus_pro",
            ARDiscreteBatchGatherer,
            (),
        ),
        (
            "janus_pro/online_r1_grpo_ocr",
            "janus_pro_r1",
            JanusProR1GenerationBatchGatherer,
            (),
        ),
        (
            "nextstep_1/online_grpo_ocr",
            "nextstep_1",
            NextStep1GenerationBatchGatherer,
            (),
        ),
    ],
)
def test_rollout_runtime_inputs_are_serializable_and_registry_backed(
    experiment: str,
    family: str,
    expected_gatherer: type,
    overrides: tuple[str, ...],
) -> None:
    """Launch inputs survive a pickle round trip; family lives only on the outer contract, batch
    width only in request sampling, and the gatherer is registry-built and is not itself an
    executor.
    """
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            *overrides,
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
            "distributed.rollout.cpus_per_worker=1",
            "rollout.samples_per_generation_batch=2",
        ],
    )
    entry = get_model_family_entry(family)

    inputs, expected_model_identity = _capture_launch_inputs(cfg, entry)

    assert isinstance(inputs, RayGenerationLaunchInputs)
    restored = pickle.loads(pickle.dumps(inputs))
    assert isinstance(restored, RayGenerationLaunchInputs)
    assert restored.launch_contract == inputs.launch_contract
    assert restored.launch_contract.family == family
    assert restored.launch_contract.expected_model_identity == expected_model_identity
    # Family identity lives once in the outer contract; worker-side executor
    # wiring comes from the registry, while this nested payload is per-run data.
    assert "family" not in restored.launch_contract.model_build
    assert restored.launch_contract.policy_version == 0
    # Batch width is per-request data (request.sampling), never executor wiring.
    assert "samples_per_generation_batch" not in restored.launch_contract.executor_kwargs
    assert isinstance(restored.gatherer, expected_gatherer)
    assert not isinstance(restored.gatherer, GenerationBatchExecutor)


def test_rollout_profiler_is_resolved_before_launch_contract_serialization() -> None:
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
        ],
    )
    cfg.rollout.torch_profiler = {
        "enabled": True,
        "activities": ["cuda"],
    }

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    profiler = inputs.launch_contract.torch_profiler
    assert profiler["enabled"] is True
    assert profiler["activities"] == ("cuda",)
    assert profiler["output_dir"] == str(cfg.trainer.output_dir)


def test_diffusion_launch_contract_uses_resolved_config_parameter_dtype() -> None:
    """The worker payload derives ordinary parameter dtype from rollout precision."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[0,1]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=1",
            "distributed.resources.rollout.num_engines=1",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert inputs.launch_contract.model_build["device"] == "cuda"
    assert inputs.launch_contract.model_build["parameter_dtype"] == "bfloat16"
    assert inputs.launch_contract.model_build["precision"] == {
        "dtype": "bf16",
        "float32_precision": "tf32",
        "quantization": None,
        "outer_autocast": True,
    }


def test_sana_launch_contract_carries_parameter_and_rollout_precision() -> None:
    """SANA's native FP16 policy and separate BF16 Gemma survive the Ray boundary."""
    cfg = load_config(
        "experiment/sana/online_grpo_aesthetic",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["precision"] == {
        "dtype": "fp16",
        "float32_precision": "ieee",
        "quantization": None,
        "outer_autocast": False,
    }
    assert model_build["rollout"] == {
        "prompt_encoder_dtype": "bfloat16",
        "base_weight_sync": False,
    }
    assert pickle.loads(pickle.dumps(model_build)) == model_build


def test_sana_fp8_rollout_keeps_native_policy_and_bf16_prompt_encoder() -> None:
    """FP8 swaps GEMMs without changing SANA or Gemma's base dtype policies."""
    cfg = load_config(
        "experiment/sana/online_grpo_aesthetic",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
        ],
    )
    cfg.precision.rollout.quantization = {"format": "fp8"}

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sana"),
    )

    model_build = inputs.launch_contract.model_build
    assert model_build["parameter_dtype"] == "float16"
    assert model_build["precision"] == {
        "dtype": "fp16",
        "float32_precision": "ieee",
        "quantization": {"format": "fp8", "recipe": "rowwise"},
        "outer_autocast": False,
    }
    assert model_build["rollout"]["prompt_encoder_dtype"] == "bfloat16"
    assert "quantization" not in model_build["rollout"]


def test_generation_chunk_auto_reaches_ray_runtime_without_executor_coercion() -> None:
    """Ray owns generation auto; the fixed executor fallback must not parse it."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            # This test only exercises input routing; resource validation has
            # dedicated coverage and the repository verify lane hides all GPUs.
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
            "rollout.samples_per_generation_batch=auto",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert "samples_per_generation_batch" not in inputs.launch_contract.executor_kwargs
    collector = RolloutCollectorConfig.from_root(parse_config(cfg))
    assert collector.samples_per_generation_batch == "auto"
    assert "samples_per_generation_batch" not in collector.request_sampling


@pytest.mark.parametrize(
    ("experiment", "family", "overrides"),
    [
        ("sd3_5/online_grpo_ocr", "sd3_5", ()),
        (
            "cosmos_predict2_5/online_nft_kling_video_reward",
            "cosmos-predict2.5",
            (),
        ),
    ],
)
def test_model_torch_compile_applies_to_all_diffusion_rollout_families(
    experiment: str,
    family: str,
    overrides: tuple[str, ...],
) -> None:
    """Checks model.torch_compile is the single compile source for rollout workers."""
    cfg = load_config(
        f"experiment/{experiment}",
        overrides=[
            *overrides,
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "distributed.resources.reward.device=cpu",
            "actor.gradient_checkpointing=off",
            "model.torch_compile.enable=true",
            "model.torch_compile.mode=default",
        ],
    )
    entry = get_model_family_entry(family)

    inputs, _ = _capture_launch_inputs(
        cfg,
        entry,
    )

    assert entry.policy_semantics.step_kind == "denoise"
    assert entry.policy_semantics.generation_regime == "full_sequence"
    model_config = inputs.launch_contract.model_build["model_config"]
    assert model_config["torch_compile"] == {
        "enable": True,
        "mode": "default",
    }


def test_executor_kwargs_use_configured_chunk_size() -> None:
    """The public config is the only batch-size input to the launch contract."""
    cfg = load_config(
        "experiment/sd3_5/online_grpo_ocr",
        overrides=[
            "distributed.resources.visible_devices=[]",
            "distributed.resources.trainer.num_gpus=0",
            "distributed.resources.rollout.num_gpus=0",
            "distributed.resources.rollout.num_engines=1",
            "rollout.samples_per_generation_batch=8",
        ],
    )

    inputs, _ = _capture_launch_inputs(
        cfg,
        get_model_family_entry("sd3_5"),
    )

    assert isinstance(inputs, RayGenerationLaunchInputs)
    assert "samples_per_generation_batch" not in inputs.launch_contract.executor_kwargs
    collector = RolloutCollectorConfig.from_root(parse_config(cfg))
    assert collector.samples_per_generation_batch == 8
    assert "samples_per_generation_batch" not in collector.request_sampling


def test_generic_executor_kwargs_project_the_complete_model_block() -> None:
    cfg = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "flux",
                    "executor": {
                        "num_frames": 17,
                        "max_sequence_length": 256,
                        "fps": 24,
                        "batch_passthrough_keys": ["text_ids"],
                    },
                    "memory": {"vae_decode": {"tiling": False}},
                },
                "rollout": {"samples_per_generation_batch": 3},
            },
        ),
    )

    assert get_model_family_entry("flux").executor_kwargs(cfg) == {
        "num_frames": 17,
        "max_sequence_length": 256,
        "fps": 24,
        "batch_passthrough_keys": ["text_ids"],
    }


def test_custom_executor_keeps_independent_supported_memory_config() -> None:
    cfg = parse_config(
        OmegaConf.create(
            {
                "model": {
                    "family": "wan_2_1_i2v",
                    "path": "unit-test",
                    "memory": {"vae_decode": {"tiling": True}},
                },
                "rollout": {"samples_per_generation_batch": 2},
            },
        ),
    )

    assert get_model_family_entry("wan_2_1_i2v").executor_kwargs(cfg) == {}
