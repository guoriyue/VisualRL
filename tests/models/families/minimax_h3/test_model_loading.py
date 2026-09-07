"""MiniMax-H3's ``from_build`` and replay builder: what they ask diffusers for.

The modular pipeline is loaded for the ``t2va`` workflow only (the 33B
``transformer_ref`` partition stays on disk), the frozen conditioner follows the
rollout prompt dtype, both VAEs stay fp32, and the video scheduler is swapped
for the flow-convention subclass. The replay builder loads the transformer and
the two schedulers from their subfolders and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.steps.denoise.fixtures import RecordingModule, build_tiny_minimax_h3_transformer
from vrl.config.precision import RolePrecision
from vrl.models.families.minimax_h3.model import MiniMaxH3Model
from vrl.models.families.minimax_h3.runtime import (
    MiniMaxH3BatchExecutor,
    build_minimax_h3_replay_runtime_bundle,
)
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

_PATH = "MiniMaxAI/MiniMax-H3"


def _build(*, rollout: bool, num_steps: int | None = None, device: str = "cuda:0") -> ModelBuild:
    return ModelBuild(
        model_name_or_path=_PATH,
        revision="abc123",
        device=device,
        parameter_dtype=torch.bfloat16,
        family="minimax_h3",
        precision=RolePrecision("bf16", "tf32", outer_autocast=False),
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16) if rollout else None,
        sampling_config={"num_steps": num_steps} if num_steps is not None else None,
    )


class _FakeModularPipeline:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        from diffusers import MiniMaxH3Scheduler

        self.calls = calls
        self.transformer = build_tiny_minimax_h3_transformer()
        self.vae = RecordingModule()
        self.audio_vae = RecordingModule()
        self.text_encoder = RecordingModule()
        self.tokenizer = object()
        self.processor = object()
        self.scheduler = MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)

    def load_components(self, **kwargs: Any) -> None:
        self.calls.append({"load_components": kwargs})


def test_from_build_loads_the_t2va_workflow_and_freezes_the_generation_modules(
    monkeypatch,
) -> None:
    from diffusers import ModularPipeline

    calls: list[dict[str, Any]] = []
    fake = _FakeModularPipeline(calls)

    def fake_from_pretrained(path: str, **kwargs: Any) -> Any:
        calls.append({"from_pretrained": {"path": path, **kwargs}})
        return fake

    monkeypatch.setattr(ModularPipeline, "from_pretrained", staticmethod(fake_from_pretrained))

    model = MiniMaxH3Model.from_build(_build(rollout=True))

    assert calls[0]["from_pretrained"] == {"path": _PATH, "workflow": "t2va", "revision": "abc123"}
    assert calls[1]["load_components"] == {
        "workflow": "t2va",
        "revision": "abc123",
        "torch_dtype": torch.bfloat16,
    }
    assert model.transformer is fake.transformer
    assert fake.text_encoder.requires_grad_enabled is False
    assert fake.text_encoder.to_calls == [("cuda:0", torch.bfloat16)]
    for vae in (fake.vae, fake.audio_vae):
        assert vae.requires_grad_enabled is False
        assert vae.to_calls == [("cuda:0", torch.float32)]
    assert type(model.scheduler).__name__ == "MiniMaxH3FlowScheduler"
    assert model.scheduler.config.shift == 12.0
    assert model.audio_scheduler is fake.audio_scheduler
    # The shell exposes every frozen module for the offload discipline.
    assert set(model.pipeline.components) == {"transformer", "vae", "audio_vae", "text_encoder"}


def test_replay_builder_loads_transformer_and_both_schedulers_only(monkeypatch) -> None:
    import diffusers

    loads: list[dict[str, Any]] = []

    class _Scheduler(diffusers.MiniMaxH3Scheduler):
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            loads.append({"cls": cls.__name__, "path": path, **kwargs})
            return cls(shift=3.0 if kwargs.get("subfolder") == "audio_scheduler" else 12.0)

    def fake_transformer_loader(build: ModelBuild, class_name: str, **kwargs: Any) -> Any:
        loads.append({"cls": class_name, "path": build.model_name_or_path, **kwargs})
        return build_tiny_minimax_h3_transformer()

    monkeypatch.setattr(diffusers, "MiniMaxH3Scheduler", _Scheduler)
    monkeypatch.setattr(
        "vrl.models.families.minimax_h3.model.build_flow_scheduler_class",
        lambda: _Scheduler,
    )
    monkeypatch.setattr("vrl.models.loader.load_diffusers_transformer", fake_transformer_loader)
    monkeypatch.setattr(
        "vrl.models.steps.denoise.build.assemble_replay_bundle",
        lambda model, build: SimpleNamespace(model=model),
    )

    # CPU device: the builder eagerly sets both schedules on the build device.
    bundle = build_minimax_h3_replay_runtime_bundle(
        _build(rollout=False, num_steps=4, device="cpu")
    )

    assert [load["cls"] for load in loads] == [
        "MiniMaxH3Transformer3DModel",
        "_Scheduler",
        "_Scheduler",
    ]
    assert [load.get("subfolder") for load in loads[1:]] == ["scheduler", "audio_scheduler"]
    assert all(load["path"] == _PATH and load["revision"] == "abc123" for load in loads[1:])
    model = bundle.model
    assert model.scheduler.timesteps.numel() == 4
    assert model.audio_scheduler.timesteps.numel() == 4
    with pytest.raises(RuntimeError, match="cannot decode"):
        model.decode_latents(torch.zeros(1))


def test_batch_executor_pins_one_sample_and_carries_only_the_prompt() -> None:
    """Batch width is not a knob; the encoder is asked for the prompt with the
    request's text length, never a negative prompt."""
    encoded_calls: list[tuple[Any, ...]] = []

    class _Model:
        def encode_prompt(self, prompt: str, negative: Any, **kwargs: Any) -> dict[str, Any]:
            encoded_calls.append((prompt, negative, kwargs))
            return {"prompt_embeds": torch.zeros(1, 2, 3)}

    executor = MiniMaxH3BatchExecutor(_Model(), samples_per_generation_batch=8)

    assert executor.family == "minimax_h3"
    assert executor.default_num_frames == 124
    assert executor.default_fps == 24
    request = SimpleNamespace(inputs=[SimpleNamespace(prompt="a cat video")])
    params = SimpleNamespace(
        text_encode_kwargs=lambda: {"guidance_scale": 1.0, "max_sequence_length": 64}
    )
    encoded = executor.encode_prompt_for_batch(
        generation_request=request,
        video_request=SimpleNamespace(negative_prompt="blurry"),
        params=params,
        batch=SimpleNamespace(prompt_index=0, sample_count=1),
    )
    assert encoded_calls == [
        ("a cat video", None, {"guidance_scale": 1.0, "max_sequence_length": 64})
    ]
    passthrough = executor.build_batch_encoded(
        encoded=encoded,
        generation_request=request,
        video_request=None,
        params=params,
        batch=SimpleNamespace(prompt_index=0, sample_count=1),
    )
    assert passthrough["prompt_embeds"] is encoded["prompt_embeds"]
