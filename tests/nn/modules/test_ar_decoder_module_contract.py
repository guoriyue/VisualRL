"""Tests for the AR decoder NN module contract."""

from __future__ import annotations

import pytest
import torch

from vrl.nn.layers.attention.paged import (
    ARAttentionPrefillInput,
    VllmPagedAttentionConfig,
)
from vrl.nn.modules.ar_decoder import VllmDecoderPagedAttentionBackend


def test_vllm_decoder_pack_prefill_compacts_left_padded_prompts() -> None:
    """Prefill packing drops left padding: embeds are compacted to the valid tokens, cache
    positions restart per sequence, position ids keep their original offsets, and each sequence
    state records its length and next position.
    """
    backend = _backend()
    embeds = torch.arange(10, dtype=torch.float32).view(2, 5, 1)
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    (
        packed_embeds,
        cache_positions,
        position_ids,
        query_start_loc,
        seq_lens,
        last_token_offsets,
        states,
    ) = backend._pack_prefill(
        ARAttentionPrefillInput(
            inputs_embeds=embeds,
            attention_mask=mask,
            branch="cond",
            max_new_tokens=2,
        )
    )

    assert torch.equal(packed_embeds[:, 0], torch.tensor([2, 3, 4, 5, 6, 7]))
    assert torch.equal(cache_positions, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.equal(position_ids, torch.tensor([2, 3, 4, 0, 1, 2]))
    assert torch.equal(query_start_loc, torch.tensor([0, 3, 6], dtype=torch.int32))
    assert torch.equal(seq_lens, torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(last_token_offsets, torch.tensor([2, 5]))
    assert [state.length for state in states] == [3, 3]
    assert [state.next_position_id for state in states] == [5, 3]


def test_vllm_decoder_pack_prefill_rejects_non_contiguous_prompt_mask() -> None:
    backend = _backend()
    embeds = torch.zeros(1, 5, 1)
    mask = torch.tensor([[1, 0, 1, 1, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="one contiguous valid-token span"):
        backend._pack_prefill(
            ARAttentionPrefillInput(
                inputs_embeds=embeds,
                attention_mask=mask,
                branch="cond",
            )
        )


def _backend() -> VllmDecoderPagedAttentionBackend:
    return VllmDecoderPagedAttentionBackend(
        trunk=object(),
        config=VllmPagedAttentionConfig(family="test"),
        kernels=object(),  # type: ignore[arg-type]
    )
