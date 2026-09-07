from __future__ import annotations

from typing import Any

import torch

from tests.models.steps.denoise.fixtures import RecordingModule
from vrl.config.precision import RolePrecision
from vrl.models.families.sd3_5.model import SD3_5Model
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions


class _FakePipeline:
    def __init__(self) -> None:
        self.transformer = RecordingModule()
        self.vae = RecordingModule()
        self.text_encoder = RecordingModule()
        self.text_encoder_2 = RecordingModule()
        self.text_encoder_3 = RecordingModule()
        self.device = "cpu"


def test_sd3_fp32_runtime_loads_frozen_components_without_fp32_peak(monkeypatch) -> None:
    """fp32 training keeps the transformer and VAE in fp32 while the frozen text encoders load
    straight into the rollout prompt-encoder dtype through the per-component ``torch_dtype``
    mapping, so no fp32 encoder copy is ever materialized.
    """
    from diffusers import StableDiffusion3Pipeline

    calls: list[dict[str, Any]] = []
    pipeline = _FakePipeline()

    def fake_from_pretrained(model_name_or_path: str, **kwargs: Any) -> _FakePipeline:
        calls.append({"model_name_or_path": model_name_or_path, **kwargs})
        return pipeline

    monkeypatch.setattr(
        StableDiffusion3Pipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    build = ModelBuild(
        model_name_or_path="stabilityai/stable-diffusion-3.5-medium",
        revision=None,
        device="cuda:0",
        parameter_dtype=torch.float32,
        family="sd3_5",
        precision=RolePrecision("fp32", "tf32", outer_autocast=False),
        rollout=RolloutBuildOptions(
            prompt_encoder_dtype=torch.float16,
        ),
    )

    model = SD3_5Model.from_build(build)

    assert model.pipeline is pipeline
    assert calls == [
        {
            "model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
            "torch_dtype": {
                "transformer": torch.float32,
                "vae": torch.float32,
                "default": torch.float16,
            },
        },
    ]
    for encoder in (
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ):
        assert encoder.requires_grad_enabled is False
        assert encoder.to_calls == [("cuda:0", torch.float16)]
    assert pipeline.vae.requires_grad_enabled is False
    assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
