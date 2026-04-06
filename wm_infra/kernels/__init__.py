"""Triton-native GPU kernels for wm-infra runtime components.

Reused from moemoekit with updated imports. These are the lowest-level
building blocks — @triton.jit functions operating on raw pointers.
The ops/ layer wraps these with PyTorch tensors and autograd.
"""

from wm_infra.kernels.routing_kernel import topk_softmax_kernel, compute_expert_counts_kernel
from wm_infra.kernels.permute_kernel import (
    moe_align_sort_kernel,
    permute_tokens_kernel,
    unpermute_tokens_kernel,
)
from wm_infra.kernels.group_gemm_kernel import grouped_gemm_kernel, grouped_gemm_fp8_kernel
from wm_infra.kernels.activation_kernel import (
    swiglu_fwd_kernel,
    swiglu_bwd_kernel,
)
from wm_infra.kernels.fused_moe_kernel import (
    fused_moe_kernel,
    fused_gate_up_kernel,
    fused_down_kernel,
)
from wm_infra.kernels.rmsnorm_kernel import rmsnorm_fwd_kernel, rmsnorm_bwd_kernel
from wm_infra.kernels.rope_kernel import rope_fwd_kernel, rope_bwd_kernel
from wm_infra.kernels.flash_attn_kernel import (
    flash_attn_fwd_kernel,
    flash_attn_bwd_preprocess,
    flash_attn_bwd_kernel,
)
from wm_infra.kernels.kv_cache_kernel import kv_cache_append_kernel
from wm_infra.kernels.matvec_kernel import (
    batched_matvec_kernel,
    batched_matvec_varying_kernel,
    indexed_dual_matvec_kernel,
    indexed_matvec_varying_kernel,
)
