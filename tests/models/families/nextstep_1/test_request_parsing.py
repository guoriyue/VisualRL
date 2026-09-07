"""NextStep AR request parsing tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import parse_config
from vrl.generation import GenerationRequest
from vrl.generation.bindings.token_autoregressive import ARRequestLayout
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.families.nextstep_1 import runtime as nextstep_runtime
from vrl.models.families.nextstep_1.config import (
    NextStep1Config,
)
from vrl.models.families.nextstep_1.runtime import (
    NextStep1ARBatchResult,
    NextStep1BatchExecutor,
    NextStep1GenerationBatchGatherer,
    nextstep_config_from_build,
)
from vrl.models.families.registry import get_model_family_entry


def _build_cfg(*, use_lora: bool, freeze_vae: bool):
    return OmegaConf.create(
        {
            "model": {
                "family": "nextstep_1",
                "use_lora": use_lora,
                "freeze_vae": freeze_vae,
            },
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {
                "guidance_scale": 4.5,
                "num_steps": 20,
                "image_token_num": 1024,
            },
        },
    )


@pytest.mark.parametrize(("use_lora", "freeze_vae"), [(True, False), (False, True)])
def test_config_projection_preserves_boolean_states(
    use_lora: bool,
    freeze_vae: bool,
) -> None:
    entry = get_model_family_entry("nextstep_1")
    cfg = _build_cfg(use_lora=use_lora, freeze_vae=freeze_vae)
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = entry.resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )

    projected = nextstep_config_from_build(build)
    resolved = NextStep1Config(**projected)

    assert projected["use_lora"] is use_lora
    assert projected["freeze_vae"] is freeze_vae
    assert "lora_target_modules" not in projected
    assert resolved.use_lora is use_lora
    assert resolved.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


def test_config_projection_ignores_sampling_fields_without_schema_producers() -> None:
    entry = get_model_family_entry("nextstep_1")
    cfg = _build_cfg(use_lora=False, freeze_vae=True)
    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = entry.resolve_model_build(
        root,
        device="cpu",
        precision=precision,
    )
    assert build.sampling_config is not None
    build.sampling_config.update({"noise_level": 0.25, "token_dim": 7})

    projected = nextstep_config_from_build(build)

    assert "noise_level" not in projected
    assert "token_dim" not in projected
    assert projected["image_token_num"] == 1024


def test_nextstep_layout_resolves_scheduler_batch_size_separately_from_sampling() -> None:
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
        sampling={
            "image_token_num": 8,
            "image_size": 256,
            "max_text_length": 16,
            "ar_scheduler_batch_size": 3,
        },
    )

    layout = ARRequestLayout()
    params = layout.parse_sampling_params(request)

    assert layout.resolve_scheduler_batch_size(request) == 3
    assert params.image_token_num == 8


def test_ar_layout_requires_shape_sampling_fields() -> None:
    """image_token_num, image_size and max_text_length are each required: dropping any one fails
    naming that key, never silently defaulting a geometry field.
    """
    for missing_key in ("image_token_num", "image_size", "max_text_length"):
        sampling = {
            "image_token_num": 8,
            "image_size": 256,
            "max_text_length": 16,
        }
        sampling.pop(missing_key)
        request = GenerationRequest(
            request_id="req",
            family="nextstep_1",
            task="ar_t2i",
            inputs=["draw text"],
            samples_per_prompt=1,
            sampling=sampling,
        )

        with pytest.raises(ValueError, match=f"request.sampling.{missing_key}"):
            ARRequestLayout().parse_sampling_params(request)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("off", False), ("full", True)],
)
def test_replay_build_resolves_gradient_checkpointing_mode(
    mode: str,
    expected: bool,
) -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": mode},
        },
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("nextstep_1").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=False,
    )

    assert build.family == "nextstep_1"
    assert build.model_config["gradient_checkpointing"] is expected


def test_replay_build_rejects_selective_gradient_checkpointing() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": "selective"},
        },
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    with pytest.raises(ValueError, match="does not support selective"):
        get_model_family_entry("nextstep_1").resolve_model_build(
            root,
            "cpu",
            precision=precision,
            for_rollout=False,
        )


def test_rollout_build_disables_gradient_checkpointing() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"family": "nextstep_1", "use_lora": False},
            "precision": {
                "float32_precision": "tf32",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "actor": {"gradient_checkpointing": "full"},
        },
    )

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    build = get_model_family_entry("nextstep_1").resolve_model_build(
        root,
        "cpu",
        precision=precision,
        for_rollout=True,
    )

    assert build.model_config["gradient_checkpointing"] is False


def test_nextstep_gather_uses_canonical_output_as_reward_source() -> None:
    """Decoded output stays outside replay state and remains the reward source."""
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
        sampling={"image_token_num": 2},
    )
    sample_rows = request.sample_rows()
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    batch = NextStep1ARBatchResult(
        batch=GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=2),
        output=images,
        tokens=torch.zeros(2, 2, 4),
        saved_noise=torch.zeros(2, 2, 4),
        log_probs=torch.zeros(2, 2),
        prompt_input_ids=torch.zeros(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={},
    )

    output = NextStep1GenerationBatchGatherer().gather_batches(
        request,
        sample_rows,
        [batch],
    )
    assert torch.equal(output.output, images)
    assert "decoded" not in output.trajectory.segments
    assert output.trajectory.reward_views["image"].tensor_refs == ()
    assert output.trajectory.reward_views["image"].metadata == {
        "output_ref": "GenerationOutput.output"
    }


def test_batch_context_keeps_only_flow_replay_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_kwargs: dict[str, Any] = {}

    class _FakeLoop:
        def __init__(self, **kwargs: Any) -> None:
            loop_kwargs.update(kwargs)

        def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tokens = torch.zeros(1, 4, 2)
            return tokens, torch.ones_like(tokens), torch.zeros(1, 4)

    decoded_sizes: list[int] = []

    def decode_image_tokens(tokens: torch.Tensor, *, image_size: int) -> torch.Tensor:
        assert tokens.shape == (1, 4, 2)
        decoded_sizes.append(image_size)
        return torch.zeros(1, 3, 2, 2)

    model = SimpleNamespace(
        processor=SimpleNamespace(pad_token_id=0),
        device=torch.device("cpu"),
        decode_image_tokens=decode_image_tokens,
    )
    executor = NextStep1BatchExecutor(model)
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(ids)
    monkeypatch.setattr(
        executor,
        "_tokenize_prompts",
        lambda prompts, *, max_text_length: (ids, mask),
    )
    monkeypatch.setattr(executor, "_embed", lambda token_ids: token_ids.unsqueeze(-1).float())
    monkeypatch.setattr(executor, "_ar_runner", lambda request: object())
    monkeypatch.setattr(nextstep_runtime, "TokenAutoregressiveLoop", _FakeLoop)
    request = GenerationRequest(
        request_id="req",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "guidance_scale": 2.5,
            "num_steps": 4,
            "image_token_num": 4,
            "image_size": 32,
            "max_text_length": 8,
            "ar_scheduler_batch_size": 3,
        },
        denoise=DenoiseRequestOptions(noise_level=0.8),
    )

    result = executor.forward_batch(
        request,
        GenerationSampleBatch(
            prompt_index=0,
            sample_start=0,
            sample_count=1,
        ),
    )

    assert result.context == {
        "guidance_scale": 2.5,
        "num_steps": 4,
        "noise_level": 0.8,
    }
    assert loop_kwargs["init_kwargs"]["image_token_num"] == 4
    assert loop_kwargs["scheduler_batch_size"] == 3
    assert decoded_sizes == [32]
