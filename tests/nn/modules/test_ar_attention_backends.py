"""Tests for shared AR attention backend selection."""

from __future__ import annotations

from vrl.nn.modules import ar_attention_backends as backends


def test_unknown_backend_error_lists_available_names() -> None:
    """Checks unknown backend errors point to the supported backend names."""

    try:
        backends.resolve_attention_backend("janus_pro", "flashinfer", "model")
    except ValueError as exc:
        assert "unknown attention backend" in str(exc)
        assert "vllm_paged" in str(exc)
        assert "torch_native" in str(exc)
    else:
        raise AssertionError("expected unknown backend to raise")


def test_attention_backend_name_reads_explicit_name_with_paged_default() -> None:
    """``attention_backend_name`` reads the explicit sampling key and defaults to ``vllm_paged``."""

    assert backends.attention_backend_name({"attention_backend": "torch_native"}) == "torch_native"
    assert backends.attention_backend_name({"attention_backend": "vllm_paged"}) == "vllm_paged"
    assert backends.attention_backend_name({}) == "vllm_paged"
