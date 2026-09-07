"""CUDA integration tests for NextStep's vLLM paged-attention backend."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from vrl.generation.types import GenerationRequest
from vrl.models.families.nextstep_1.runner import (
    NextStep1ARModelRunner,
)
from vrl.models.families.nextstep_1.runtime import NextStep1BatchExecutor
from vrl.nn.layers.attention.paged import (
    ARAttentionPrefillInput,
    ARAttentionStepInput,
    ARAttentionUnavailable,
)
from vrl.nn.modules.ar_attention_backends import build_vllm_attention_backend

# Default cache dtype resolved from the source signature, so a default change
# (e.g. "auto" -> a concrete dtype) auto-flows into this assertion.
_DEFAULT_CACHE_DTYPE = (
    inspect.signature(build_vllm_attention_backend).parameters["cache_dtype"].default
)


@pytest.mark.gpu
def test_nextstep_vllm_paged_attention_matches_hf_qwen_one_step() -> None:
    """Same parity for NextStep's Qwen trunk: paged prefill plus step match HF's cached forward."""
    torch.manual_seed(0)
    trunk = _tiny_qwen_trunk()
    try:
        backend = build_vllm_attention_backend(_TinyNextStepLike(trunk), family="nextstep_1")
    except ARAttentionUnavailable as exc:
        pytest.skip(f"vLLM paged-attention internals are unavailable: {exc}")

    embeds = torch.randn(1, 4, 512, device="cuda", dtype=torch.float16)
    mask = torch.tensor([[0, 0, 1, 1]], device="cuda", dtype=torch.long)
    step_embeds = torch.randn(1, 1, 512, device="cuda", dtype=torch.float16)
    step_mask = torch.tensor([[0, 0, 1, 1, 1]], device="cuda", dtype=torch.long)

    with torch.no_grad():
        hf_prefill = trunk(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
        paged_prefill = backend.prefill(
            ARAttentionPrefillInput(
                inputs_embeds=embeds,
                attention_mask=mask,
                branch="cond",
                max_new_tokens=2,
            )
        )
        hf_step = trunk(
            inputs_embeds=step_embeds,
            attention_mask=step_mask,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        )
        paged_step = backend.step(
            ARAttentionStepInput(
                input_embeds=step_embeds,
                attention_mask=step_mask,
                sequence_states=paged_prefill.sequence_states,
            )
        )

    assert paged_prefill.last_hidden.shape == (1, 512)
    assert paged_step.last_hidden.shape == (1, 512)
    assert (
        hf_prefill.last_hidden_state[:, -1, :] - paged_prefill.last_hidden
    ).abs().max().item() <= 3e-3
    assert (
        hf_step.last_hidden_state[:, -1, :] - paged_step.last_hidden
    ).abs().max().item() <= 5e-3


def test_nextstep_runtime_uses_vllm_paged_attention_by_default(monkeypatch) -> None:
    """``ar_paged_block_size`` in sampling selects the vLLM paged backend for ``nextstep_1`` with
    that block size and the default cache dtype.
    """
    model = SimpleNamespace(
        config=SimpleNamespace(model_path="tiny-nextstep"),
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
    backend = object()

    def build_backend(
        passed_model: object,
        *,
        family: str,
        block_size: int,
        cache_dtype: str,
    ) -> object:
        assert passed_model is model
        assert family == "nextstep_1"
        assert block_size == 32
        assert cache_dtype == _DEFAULT_CACHE_DTYPE
        return backend

    monkeypatch.setattr(
        "vrl.nn.modules.ar_attention_backends.build_vllm_attention_backend",
        build_backend,
    )
    request = GenerationRequest(
        request_id="nextstep-vllm-paged",
        family="nextstep_1",
        task="ar_t2i",
        inputs=["test prompt"],
        samples_per_prompt=1,
        sampling={
            "ar_paged_block_size": 32,
        },
    )

    runner = NextStep1BatchExecutor(model)._ar_runner(request)

    assert isinstance(runner, NextStep1ARModelRunner)
    assert runner.attention_backend is backend


def _tiny_qwen_trunk() -> Qwen2Model:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )
    return Qwen2Model(config).eval().to(device="cuda", dtype=torch.float16)


class _TinyNextStepLike:
    config = SimpleNamespace(model_path="tiny-nextstep")
    device = torch.device("cuda")
    dtype = torch.float16

    def __init__(self, trunk: Qwen2Model) -> None:
        self._trunk = trunk

    def _lm_trunk(self) -> Qwen2Model:
        return self._trunk
