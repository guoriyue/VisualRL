"""SANA's ``from_build``: the shared loader plus its two deviations.

The pipeline load runs through ``DiffusersPipelineModelBase.from_build`` (frozen
Gemma encoder at the rollout prompt dtype, DC-AE in fp32); SANA then swaps the
shipped DPM-Solver for FlowMatchEuler carrying the checkpoint's ``flow_shift``,
and re-applies the fp16 linear-attention saturation clamp on any non-fp16 run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import RecordingModule, build_tiny_sana_transformer
from vrl.config.precision import RolePrecision
from vrl.models.families.sana.model import SanaModel
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

pytest.importorskip("diffusers")


def _build(parameter_dtype: torch.dtype) -> ModelBuild:
    return ModelBuild(
        model_name_or_path="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
        revision=None,
        device="cuda:0",
        parameter_dtype=parameter_dtype,
        family="sana",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16),
    )


def _load(monkeypatch: pytest.MonkeyPatch, parameter_dtype: torch.dtype) -> tuple[Any, Any, Any]:
    from diffusers import SanaPipeline

    pipeline = SimpleNamespace(
        transformer=build_tiny_sana_transformer(),
        vae=RecordingModule(),
        text_encoder=RecordingModule(),
        # The shipped scheduler config: DPM-Solver calls the shift ``flow_shift``.
        scheduler=SimpleNamespace(config={"num_train_timesteps": 1000, "flow_shift": 3.0}),
    )
    calls: list[dict[str, Any]] = []

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> Any:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return pipeline

    monkeypatch.setattr(SanaPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
    model = SanaModel.from_build(_build(parameter_dtype))
    return model, pipeline, calls


def _linear_attn_processor_names(transformer: Any) -> set[str]:
    return {
        type(processor).__name__
        for name, processor in transformer.attn_processors.items()
        if name.endswith("attn1.processor")
    }


def test_from_build_runs_the_shared_loader_then_swaps_in_flow_match(monkeypatch) -> None:
    """Frozen encoder at the rollout prompt dtype, VAE fp32, scheduler rebuilt with shift=3."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    model, pipeline, calls = _load(monkeypatch, torch.float16)

    assert model.pipeline is pipeline
    assert calls[0]["model_name_or_path"].startswith("Efficient-Large-Model/")
    assert pipeline.text_encoder.requires_grad_enabled is False
    assert pipeline.text_encoder.to_calls == [("cuda:0", torch.bfloat16)]
    assert pipeline.vae.requires_grad_enabled is False
    assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
    assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    assert pipeline.scheduler.config.shift == 3.0
    # Native fp16: diffusers' own dtype-conditional clip applies, no re-clamp.
    assert _linear_attn_processor_names(pipeline.transformer) == {"SanaLinearAttnProcessor2_0"}


def test_from_build_reapplies_the_fp16_saturation_clamp_off_fp16(monkeypatch) -> None:
    """A non-fp16 run gets the saturating linear-attention processor on every attn1."""
    _, pipeline, _ = _load(monkeypatch, torch.float32)

    assert _linear_attn_processor_names(pipeline.transformer) == {"_SaturatedLinearAttnProcessor"}
