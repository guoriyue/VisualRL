"""PyTorch-wrapped ops with autograd support.

Each op wraps one or more Triton kernels and provides:
  - Tensor-in, tensor-out interface (no raw pointers)
  - Automatic autograd backward
  - Shape validation and dtype handling
  - Grid/block size selection

Usage:
    from wm_infra.ops import topk_route, permute_tokens, grouped_gemm, fused_moe
"""

from wm_infra.ops.routing import topk_route, update_expert_bias
from wm_infra.ops.permute import permute_tokens, unpermute_tokens
from wm_infra.ops.group_gemm import grouped_gemm, grouped_gemm_fp8
from wm_infra.ops.activation import fused_swiglu
from wm_infra.ops.fused_moe import fused_moe
from wm_infra.ops.quantize import (
    quantize_per_tensor, dequantize_per_tensor, quantize_per_expert,
)
from wm_infra.ops.expert_cache import ExpertCache
from wm_infra.ops.rmsnorm import rms_norm, rms_norm_naive
from wm_infra.ops.rope import apply_rope, apply_rope_naive, precompute_rope_freqs
from wm_infra.ops.attention import (
    flash_attention,
    flash_attention_decode,
    flash_attention_naive,
)
from wm_infra.ops.kv_cache import KVCache, MLAKVCache
from wm_infra.ops.matvec import (
    batched_matvec, batched_matvec_varying,
    indexed_dual_matvec, indexed_matvec_varying,
)
