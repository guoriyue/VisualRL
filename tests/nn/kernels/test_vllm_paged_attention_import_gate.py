"""Tests for the vLLM paged-attention kernel import gate."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vrl.nn.kernels.attention.vllm_paged import VllmPagedAttentionKernels
from vrl.nn.layers.attention.paged import (
    ARAttentionUnavailable,
    VllmPagedAttentionConfig,
)


def test_vllm_paged_attention_kernels_report_abi_failure() -> None:
    """A vLLM whose internal ``block_table`` module fails to import is reported as
    ``ARAttentionUnavailable`` naming that module.
    """

    def import_module(name: str) -> object:
        if name == "vllm.v1.worker.block_table":
            raise ImportError("bad abi")
        return SimpleNamespace(__version__="test")

    with pytest.raises(ARAttentionUnavailable, match="block_table"):
        VllmPagedAttentionKernels(
            VllmPagedAttentionConfig(family="janus_pro"),
            import_module=import_module,
        )


def test_vllm_paged_attention_kernels_call_real_internal_api_boundary() -> None:
    """The kernels touch exactly the internal API boundary (block table, attention backend enum,
    flash-attn backend / impl / metadata), with the KV-cache shape and block size derived from
    the config.
    """

    def import_module(name: str) -> object:
        if name == "vllm.v1.worker.block_table":
            return SimpleNamespace(BlockTable=_FakeBlockTable)
        if name == "vllm.v1.attention.backend":
            return SimpleNamespace(AttentionType=SimpleNamespace(DECODER="decoder"))
        if name == "vllm.v1.attention.backends.flash_attn":
            return SimpleNamespace(
                FlashAttentionBackend=_FakeFlashAttentionBackend,
                FlashAttentionImpl=_FakeFlashAttentionImpl,
                FlashAttentionMetadata=_FakeFlashAttentionMetadata,
            )
        return SimpleNamespace(__name__=name, __version__="test")

    kernels = VllmPagedAttentionKernels(
        VllmPagedAttentionConfig(family="janus_pro"),
        import_module=import_module,
    )

    assert "vllm.v1.attention.backends.flash_attn" in kernels.modules
    assert kernels.get_kv_cache_shape(
        num_blocks=3,
        num_kv_heads=2,
        head_size=8,
    ) == (2, 3, 16, 2, 8)

    block_table = kernels.new_block_table(
        max_num_reqs=2,
        max_num_blocks_per_req=3,
        max_num_batched_tokens=4,
        device=torch.device("cpu"),
    )
    assert block_table.kwargs["block_size"] == 16

    key = torch.zeros(1, 2, 8)
    value = torch.ones(1, 2, 8)
    slot_mapping = torch.tensor([0])
    scale = torch.tensor(1.0)

    impl = kernels.make_flash_attention_impl(
        num_heads=2,
        head_size=8,
        scale=8**-0.5,
        num_kv_heads=2,
    )
    metadata = kernels.make_flash_attention_metadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1]),
        max_seq_len=1,
        seq_lens=torch.tensor([1]),
        block_table=torch.tensor([[0]]),
        slot_mapping=slot_mapping,
    )
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    kv_cache = torch.zeros(2, 3, 16, 2, 8)
    output = kernels.run_flash_attention(
        impl=impl,
        layer=layer,
        query=key,
        key=key,
        value=value,
        kv_cache=kv_cache,
        metadata=metadata,
    )
    assert output.shape == (1, 16)
    kernels.update_flash_kv_cache(
        impl=impl,
        layer=layer,
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )
    assert impl.forward_calls == 1
    assert impl.cache_update_calls == 1


class _FakeBlockTable:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        *,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str,
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _FakeFlashAttentionImpl:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.forward_calls = 0
        self.cache_update_calls = 0

    def forward(
        self,
        layer: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: object,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        del layer, key, value, kv_cache, metadata
        self.forward_calls += 1
        if output.ndim == 3:
            output.copy_(query)
        else:
            output.copy_(query.reshape(query.shape[0], -1))
        return output

    def do_kv_cache_update(
        self,
        layer: object,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        del layer, key, value, kv_cache, slot_mapping
        self.cache_update_calls += 1


@dataclass(slots=True)
class _FakeFlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0
    causal: bool = True
