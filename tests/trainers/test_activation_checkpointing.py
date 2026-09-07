from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.trainers.activation_checkpointing import (
    enable_transformer_gradient_checkpointing,
)


def _config(mode: str | bool):
    return OmegaConf.create(
        {
            "actor": {"gradient_checkpointing": mode},
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
        },
    )


def test_absent_gradient_checkpointing_defaults_to_off() -> None:
    class _Transformer:
        def enable_gradient_checkpointing(self) -> None:
            raise AssertionError("an absent public key must not enable checkpointing")

    cfg = OmegaConf.create(
        {
            "actor": {},
            "model": {"family": "sd3_5", "torch_compile": {"enable": False}},
        },
    )
    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(cfg))


@pytest.mark.parametrize("mode", ["off", False])
def test_gradient_checkpointing_off_skips_trainable_modules(mode: str | bool) -> None:
    class _Transformer:
        def enable_gradient_checkpointing(self) -> None:
            raise AssertionError("off must not enable gradient checkpointing")

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config(mode)))


@pytest.mark.parametrize("mode", ["full", True])
def test_full_gradient_checkpointing_uses_the_native_method(mode: str | bool) -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config(mode)))

    assert calls == [{}]


def test_selective_gradient_checkpointing_passes_a_custom_function() -> None:
    calls: list[dict[str, object]] = []

    class _Transformer:
        def enable_gradient_checkpointing(self, **kwargs: object) -> None:
            calls.append(kwargs)

    bundle = type("_Bundle", (), {"trainable_modules": {"transformer": _Transformer()}})()

    enable_transformer_gradient_checkpointing(bundle, parse_config(_config("selective")))

    assert len(calls) == 1
    assert callable(calls[0]["gradient_checkpointing_func"])


def test_selective_gradient_checkpointing_reports_legacy_full_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    class _LegacyTransformer:
        def enable_gradient_checkpointing(self) -> None:
            nonlocal calls
            calls += 1

    bundle = type(
        "_Bundle",
        (),
        {"trainable_modules": {"transformer": _LegacyTransformer()}},
    )()

    with caplog.at_level(logging.WARNING):
        enable_transformer_gradient_checkpointing(bundle, parse_config(_config("selective")))

    assert calls == 1
    assert "falling back to full checkpointing" in caplog.text


def test_checkpointing_rejects_replay_compile_but_accepts_rollout_scope() -> None:
    """The collision is with compiling the REPLAY policy; a rollout-scoped
    compile leaves the checkpointed trainer eager and must pass."""

    from vrl.trainers.activation_checkpointing import (
        validate_compile_checkpointing_compatible,
    )

    def _cfg(compile_block: dict):
        # The check reads the compile matrix off a parsed root, as the runtime
        # apply path hands it one.
        return parse_config(
            OmegaConf.create(
                {
                    "actor": {"gradient_checkpointing": "full"},
                    "model": {"family": "sd3_5", "torch_compile": compile_block},
                },
            )
        )

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        validate_compile_checkpointing_compatible(_cfg({"enable": True}))

    validate_compile_checkpointing_compatible(
        _cfg({"enable": True, "scope": "rollout"}),
    )
