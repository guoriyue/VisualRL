"""Real vLLM paged-attention integration test."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vrl.nn.kernels.attention.vllm_paged import VllmPagedAttentionKernels
from vrl.nn.layers.attention.paged import VllmPagedAttentionConfig


@pytest.mark.gpu
def test_vllm_paged_attention_writes_real_cuda_kv_cache() -> None:
    """Real vLLM kernels write a KV cache on CUDA through the block table and slot mapping. Only a
    missing vLLM is a capability gap worth skipping: an installed vLLM whose internal API does
    not import (``pip install --no-deps`` leaves its declared dependencies out) is a broken
    environment, and the only real kernel test in the repo must say so instead of silently
    skipping.
    """
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed")
    kernels = VllmPagedAttentionKernels(VllmPagedAttentionConfig(family="janus_pro"))

    assert kernels.get_kv_cache_shape(
        num_blocks=1,
        num_kv_heads=2,
        head_size=8,
    ) == (2, 1, 16, 2, 8)

    block_table = kernels.new_block_table(
        max_num_reqs=1,
        max_num_blocks_per_req=1,
        max_num_batched_tokens=1,
        device=torch.device("cuda"),
    )
    block_table.add_row([0], row_idx=0)
    block_table.commit_block_table(1)
    slot_mapping = kernels.compute_slot_mapping(
        block_table=block_table,
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        positions=torch.tensor([0], device="cuda", dtype=torch.int64),
    )

    kv_cache = torch.zeros((2, 1, 16, 2, 8), device="cuda", dtype=torch.float16)
    key = torch.arange(16, device="cuda", dtype=torch.float16).reshape(1, 2, 8)
    value = torch.arange(16, 32, device="cuda", dtype=torch.float16).reshape(1, 2, 8)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    impl = kernels.make_flash_attention_impl(
        num_heads=2,
        head_size=8,
        scale=8**-0.5,
        num_kv_heads=2,
    )
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    kernels.update_flash_kv_cache(
        impl=impl,
        layer=layer,
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
    )
    torch.cuda.synchronize()

    assert kv_cache[0].sum().item() == pytest.approx(120.0)
    assert kv_cache[1].sum().item() == pytest.approx(376.0)
