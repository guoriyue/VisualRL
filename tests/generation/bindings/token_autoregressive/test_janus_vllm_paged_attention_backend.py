"""CUDA integration tests for Janus' real vLLM paged-attention backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from vrl.nn.layers.attention.paged import (
    ARAttentionPrefillInput,
    ARAttentionStepInput,
    ARAttentionUnavailable,
)
from vrl.nn.modules.ar_attention_backends import build_vllm_attention_backend


@pytest.mark.gpu
def test_janus_vllm_paged_attention_matches_hf_llama_one_step() -> None:
    """One prefill plus one cached step through the vLLM paged backend reproduce HF Llama's hidden
    states for the same embeds and masks (real CUDA kernels).
    """
    torch.manual_seed(0)
    trunk = _tiny_llama_trunk()
    try:
        backend = build_vllm_attention_backend(_TinyJanusLike(trunk), family="janus_pro")
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
    ).abs().max().item() <= 3e-3


def _tiny_llama_trunk() -> LlamaModel:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )
    return LlamaModel(config).eval().to(device="cuda", dtype=torch.float16)


class _TinyJanusLike:
    config = SimpleNamespace(model_path="tiny-llama")
    device = torch.device("cuda")
    dtype = torch.float16

    def __init__(self, trunk: LlamaModel) -> None:
        self._trunk = trunk

    def _lm_trunk(self) -> LlamaModel:
        return self._trunk
