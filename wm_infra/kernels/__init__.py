"""Triton kernels retained for repo-specific runtime bottlenecks.

This package now focuses on layout transforms, routing, KV/state movement,
and MoE-specific compute paths. Generic attention dispatch has moved to
vendor/framework implementations such as FlashAttention and PyTorch SDPA.
The ops/ layer wraps these kernels with PyTorch tensors and autograd.
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
from wm_infra.kernels.fused_moe_kernel import fused_gate_up_kernel, fused_down_kernel
from wm_infra.kernels.rmsnorm_kernel import rmsnorm_fwd_kernel
from wm_infra.kernels.kv_cache_kernel import kv_cache_append_kernel
from wm_infra.kernels.matvec_kernel import (
    batched_matvec_kernel,
    batched_matvec_varying_kernel,
    indexed_dual_matvec_kernel,
    indexed_matvec_varying_kernel,
)
