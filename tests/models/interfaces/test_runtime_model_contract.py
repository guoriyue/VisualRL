"""Tests for runtime model contracts."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.models.interfaces import registered_runtime_model_classes
from vrl.config.precision import RolePrecision
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    RuntimeBundle,
    RuntimeModel,
)
from vrl.models.steps.denoise import DiffusionModelBase

# RuntimeModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_RUNTIME_MODEL_METHODS = tuple(sorted(RuntimeModel.__protocol_attrs__))


class _MinimalRuntimeModel:
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class _DiffusionModelBaseStub(DiffusionModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 1, bias=True)
        self.transformer.bias.requires_grad_(False)

    def encode_prompt(self, prompt, negative_prompt=None, **kwargs):
        del prompt, negative_prompt, kwargs
        return {}

    def prepare_sampling(self, request, encoded, **kwargs):
        del request, encoded, kwargs
        return None

    def forward_step(self, state, step_idx):
        del state, step_idx
        return {}

    def decode_latents(self, latents):
        return latents

    @classmethod
    def from_build(cls, build):
        del build
        return cls()

    def apply_lora(self, build):
        del build

    def apply_full_finetune(self, build):
        del build
        return None

    def torch_compile_transformer(self, mode: str):
        del mode

    def set_num_steps(self, n: int):
        del n

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return None

    @property
    def raw_handle(self) -> Any:
        return None

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        yield


@pytest.mark.parametrize(
    "family",
    sorted(registered_runtime_model_classes()),
)
def test_registered_family_runtime_model_satisfies_contract(family: str) -> None:
    """Every registered family's runtime-model class satisfies RuntimeModel.

    Runs over the family registry (not a hand-written list) so a newly
    registered family cannot silently skip the contract. The check is
    class-level — ``callable(getattr(cls, m))`` like ``_missing_callables`` —
    because instantiating a real family model needs weights/GPU.
    """
    runtime_cls = registered_runtime_model_classes()[family]
    missing = [m for m in _RUNTIME_MODEL_METHODS if not callable(getattr(runtime_cls, m, None))]
    assert not missing, f"{family}: {runtime_cls.__name__} missing RuntimeModel methods {missing}"


def test_runtime_bundle_exposes_model_contract() -> None:
    """``RuntimeBundle`` hands the very ``RolePrecision`` object it holds to the model, so bundle
    precision and ``model.precision`` cannot drift apart.
    """
    model = _MinimalRuntimeModel()
    bundle = RuntimeBundle(
        model=model,
        trainable_modules={},
        scheduler=None,
        raw_handle=None,
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
        loads_full_generation_modules=False,
    )

    assert bundle.model is model
    assert bundle.precision == RolePrecision("fp32", "ieee", outer_autocast=False)
    assert model.precision is bundle.precision


def test_diffusion_load_trainable_state_accepts_trainable_only_payload() -> None:
    """A payload holding exactly the trainable keys loads in place: the transformer weight takes
    the new values.
    """
    model = _DiffusionModelBaseStub()
    new_weight = torch.full_like(model.transformer.weight, 3.0)

    model.load_trainable_state({"transformer.weight": new_weight})

    assert torch.equal(model.transformer.weight, new_weight)


def test_diffusion_load_trainable_state_rejects_frozen_payload() -> None:
    model = _DiffusionModelBaseStub()

    with pytest.raises(ValueError, match="exactly trainable"):
        model.load_trainable_state(
            {
                "transformer.weight": torch.ones_like(model.transformer.weight),
                "transformer.bias": torch.ones_like(model.transformer.bias),
            },
        )
