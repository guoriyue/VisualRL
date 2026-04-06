"""Batched GEMV ops for M=1 decode: single-token matrix-vector multiply.

These wrap the Triton GEMV kernels with PyTorch tensor interfaces.
Only used at decode time (M=1) -- no backward pass needed.

Two flavors:
  - batched_matvec: shared input x across experts (gate/up)
  - batched_matvec_varying: per-expert input (down)
"""

import torch
import triton

from wm_infra.kernels.matvec_kernel import (
    batched_matvec_kernel,
    batched_matvec_varying_kernel,
    indexed_dual_matvec_kernel,
    indexed_matvec_varying_kernel,
)


def batched_matvec(
    x: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Batched GEMV: out[e, :] = x @ W[e, :, :] for selected experts.

    Args:
        x: [K] -- single input vector (1D, shared across all experts)
        W: [num_experts_active, K, N] -- selected expert weight matrices
        out: optional pre-allocated [num_experts_active, N] output buffer

    Returns:
        out: [num_experts_active, N]
    """
    num_experts_active, K, N = W.shape
    assert x.shape == (K,), f"Expected x shape ({K},), got {x.shape}"

    if out is None:
        out = torch.empty(num_experts_active, N, dtype=x.dtype, device=x.device)

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    batched_matvec_kernel[grid](
        x, W, out,
        N, K, num_experts_active,
        W.stride(0), W.stride(1), W.stride(2),
        out.stride(0), out.stride(1),
    )

    return out


def batched_matvec_varying(
    x: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Batched GEMV with per-expert input: out[e, :] = x[e, :] @ W[e, :, :].

    Args:
        x: [num_experts_active, K] -- per-expert input vectors
        W: [num_experts_active, K, N] -- selected expert weight matrices
        out: optional pre-allocated [num_experts_active, N] output buffer

    Returns:
        out: [num_experts_active, N]
    """
    num_experts_active, K, N = W.shape
    assert x.shape == (num_experts_active, K), \
        f"Expected x shape ({num_experts_active}, {K}), got {x.shape}"

    if out is None:
        out = torch.empty(num_experts_active, N, dtype=x.dtype, device=x.device)

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    batched_matvec_varying_kernel[grid](
        x, W, out,
        N, K, num_experts_active,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1), W.stride(2),
        out.stride(0), out.stride(1),
    )

    return out


def indexed_dual_matvec(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    expert_ids: torch.Tensor,
    out_gate: torch.Tensor,
    out_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared-input dual GEMV for gate/up projections without weight gathering."""
    num_experts_active = expert_ids.shape[0]
    _, K, N = w_gate.shape
    assert x.shape == (K,), f"Expected x shape ({K},), got {x.shape}"
    assert w_up.shape == w_gate.shape, f"Expected w_up shape {w_gate.shape}, got {w_up.shape}"
    assert out_gate.shape == (num_experts_active, N)
    assert out_up.shape == (num_experts_active, N)

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    indexed_dual_matvec_kernel[grid](
        x, w_gate, w_up, expert_ids, out_gate, out_up,
        N, K, num_experts_active,
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        out_gate.stride(0), out_gate.stride(1),
    )
    return out_gate, out_up


def indexed_matvec_varying(
    x: torch.Tensor,
    W: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Per-expert GEMV for down projection without weight gathering."""
    num_experts_active, K = x.shape
    _, K_w, N = W.shape
    assert K == K_w, f"Expected K={K_w}, got {K}"
    assert expert_ids.shape == (num_experts_active,)
    assert out.shape == (num_experts_active, N)

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    indexed_matvec_varying_kernel[grid](
        x, W, expert_ids, out,
        N, K, num_experts_active,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1), W.stride(2),
        out.stride(0), out.stride(1),
    )
    return out


def indexed_dual_matvec_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    expert_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Fused gate+up GEMV with inline SwiGLU — outputs activated [top_k, I].

    Replaces: indexed_dual_matvec + swiglu_fused_gate_up (2 kernels -> 1).
    """
    from wm_infra.kernels.matvec_kernel import indexed_dual_matvec_swiglu_kernel

    num_experts_active = expert_ids.shape[0]
    _, K, N = w_gate.shape
    assert x.shape == (K,), f"Expected x shape ({K},), got {x.shape}"
    assert out.shape == (num_experts_active, N)

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    indexed_dual_matvec_swiglu_kernel[grid](
        x, w_gate, w_up, expert_ids, out,
        N, K, num_experts_active,
        w_gate.stride(0), w_gate.stride(1), w_gate.stride(2),
        w_up.stride(0), w_up.stride(1), w_up.stride(2),
        out.stride(0), out.stride(1),
    )
    return out


def indexed_matvec_varying_weighted(
    x: torch.Tensor,
    W: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Fused down GEMV + routing weight + sum — outputs [1, hidden_dim].

    Replaces: indexed_matvec_varying + unsqueeze + mul + sum (4 ops -> 1 kernel).
    The output buffer must be pre-zeroed (kernel uses atomic_add).
    """
    from wm_infra.kernels.matvec_kernel import indexed_matvec_varying_weighted_kernel

    num_experts_active, K = x.shape
    _, K_w, N = W.shape
    assert K == K_w, f"Expected K={K_w}, got {K}"
    assert expert_ids.shape == (num_experts_active,)
    assert weights.shape == (num_experts_active,)
    assert out.shape[0] == 1 and out.shape[1] == N

    def grid(META):
        return (num_experts_active, triton.cdiv(N, META["BLOCK_N"]))

    indexed_matvec_varying_weighted_kernel[grid](
        x, W, expert_ids, weights, out,
        N, K, num_experts_active,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1), W.stride(2),
        out.stride(1),
    )
    return out
