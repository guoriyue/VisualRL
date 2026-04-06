"""Decode-specific GEMV kernels for M=1 (single-token) matrix-vector multiplication.

At decode time (M=1), the standard grouped GEMM kernel wastes >93% of compute
on zero-padded rows (BLOCK_M=16, but only 1 real row). These GEMV kernels are
designed for memory-bandwidth-bound M=1 workloads:

  - Each thread block computes a chunk of the N output dimension
  - The single input vector is loaded into registers/shared memory once
  - Weight rows are streamed through and dotted with the input

Variants:
  - batched_matvec_kernel: shared input across experts (gate/up proj)
  - batched_matvec_varying_kernel: per-expert input (down proj)
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_kernel(
    # ── Data pointers ──
    x_ptr,                  # [K] — single input vector (shared across experts)
    W_ptr,                  # [num_experts_active, K, N] — selected expert weights
    out_ptr,                # [num_experts_active, N] — output per expert
    # ── Dimensions ──
    N,                      # output dim
    K,                      # input dim (reduction)
    num_experts_active,     # number of active experts (top_k)
    # ── Layout: W strides ──
    stride_we,              # stride for expert dim
    stride_wk,              # stride for K dim
    stride_wn,              # stride for N dim
    # ── Layout: out strides ──
    stride_oe,              # stride for expert dim in output
    stride_on,              # stride for N dim in output
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched GEMV: out[e, :] = x @ W[e, :, :] for all active experts.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Each program handles one expert and one chunk of N output elements.

    This is a pure GEMV: M=1, so no tl.dot needed. We use vector loads
    and element-wise multiply-accumulate, which maximizes memory bandwidth
    utilization for the memory-bound M=1 regime.
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in tiles, loading x and W
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load x vector chunk: [BLOCK_K]
        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_offsets = (expert_idx * stride_we +
                     k_range[:, None] * stride_wk +
                     n_range[None, :] * stride_wn)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        # Dot: acc += sum_k(x[k] * W[k, n]) — broadcast x over N
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # Store output
    out_offsets = expert_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


# ─── Per-expert input variants (for down projection) ───
# These are like the shared-x variants above, but each expert has its own
# input vector x[e, :] instead of a single shared x[:].

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def batched_matvec_varying_kernel(
    # ── Data pointers ──
    x_ptr,                  # [num_experts_active, K] — per-expert input vectors
    W_ptr,                  # [num_experts_active, K, N] — selected expert weights
    out_ptr,                # [num_experts_active, N] — output per expert
    # ── Dimensions ──
    N,
    K,
    num_experts_active,
    # ── Layout: x strides ──
    stride_xe,
    stride_xk,
    # ── Layout: W strides ──
    stride_we,
    stride_wk,
    stride_wn,
    # ── Layout: out strides ──
    stride_oe,
    stride_on,
    # ── Tuning ──
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Batched GEMV with per-expert input: out[e, :] = x[e, :] @ W[e, :, :].

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Used for down projection where each expert has a different activated input.
    """
    expert_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if expert_idx >= num_experts_active:
        return

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load per-expert x vector chunk: [BLOCK_K]
        x_offsets = expert_idx * stride_xe + k_range * stride_xk
        x_vals = tl.load(x_ptr + x_offsets, mask=k_mask, other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_offsets = (expert_idx * stride_we +
                     k_range[:, None] * stride_wk +
                     n_range[None, :] * stride_wn)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    out_offsets = expert_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_dual_matvec_kernel(
    x_ptr,                  # [K] — shared input vector
    W_gate_ptr,             # [num_experts, K, N]
    W_up_ptr,               # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_gate_ptr,           # [num_experts_active, N]
    out_up_ptr,             # [num_experts_active, N]
    N,
    K,
    num_experts_active,
    stride_wge,
    stride_wgk,
    stride_wgn,
    stride_wue,
    stride_wuk,
    stride_wun,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Shared-input dual GEMV for gate/up projections using expert id indirection.

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Avoids per-step weight gathering by reading directly from the source expert tensors.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        gate_offsets = (
            expert_idx * stride_wge
            + k_range[:, None] * stride_wgk
            + n_range[None, :] * stride_wgn
        )
        up_offsets = (
            expert_idx * stride_wue
            + k_range[:, None] * stride_wuk
            + n_range[None, :] * stride_wun
        )
        w_mask = k_mask[:, None] & n_mask[None, :]

        gate_vals = tl.load(W_gate_ptr + gate_offsets, mask=w_mask, other=0.0).to(tl.float32)
        up_vals = tl.load(W_up_ptr + up_offsets, mask=w_mask, other=0.0).to(tl.float32)

        prod = x_vals[:, None]
        acc_gate += tl.sum(prod * gate_vals, axis=0)
        acc_up += tl.sum(prod * up_vals, axis=0)

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_gate_ptr + out_offsets, acc_gate.to(out_gate_ptr.dtype.element_ty), mask=n_mask)
    tl.store(out_up_ptr + out_offsets, acc_up.to(out_up_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_matvec_varying_kernel(
    x_ptr,                  # [num_experts_active, K]
    W_ptr,                  # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_ptr,                # [num_experts_active, N]
    N,
    K,
    num_experts_active,
    stride_xe,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Per-expert-input GEMV using expert id indirection for down projection."""
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(
            x_ptr + row_idx * stride_xe + k_range * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_offsets = (
            expert_idx * stride_we
            + k_range[:, None] * stride_wk
            + n_range[None, :] * stride_wn
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


# ─── Fused kernels for reduced kernel count ───

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
)
@triton.jit
def indexed_dual_matvec_swiglu_kernel(
    x_ptr,                  # [K] — shared input vector
    W_gate_ptr,             # [num_experts, K, N]
    W_up_ptr,               # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    out_ptr,                # [num_experts_active, N] — SwiGLU output
    N,
    K,
    num_experts_active,
    stride_wge,
    stride_wgk,
    stride_wgn,
    stride_wue,
    stride_wuk,
    stride_wun,
    stride_oe,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Fused gate+up GEMV with inline SwiGLU activation.

    Computes: out[e, :] = SiLU(x @ W_gate[expert_ids[e]]) * (x @ W_up[expert_ids[e]])

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Saves one kernel launch per MoE layer by fusing the SwiGLU activation
    into the matvec output stage.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(x_ptr + k_range, mask=k_mask, other=0.0).to(tl.float32)

        gate_offsets = (
            expert_idx * stride_wge
            + k_range[:, None] * stride_wgk
            + n_range[None, :] * stride_wgn
        )
        up_offsets = (
            expert_idx * stride_wue
            + k_range[:, None] * stride_wuk
            + n_range[None, :] * stride_wun
        )
        w_mask = k_mask[:, None] & n_mask[None, :]

        gate_vals = tl.load(W_gate_ptr + gate_offsets, mask=w_mask, other=0.0).to(tl.float32)
        up_vals = tl.load(W_up_ptr + up_offsets, mask=w_mask, other=0.0).to(tl.float32)

        prod = x_vals[:, None]
        acc_gate += tl.sum(prod * gate_vals, axis=0)
        acc_up += tl.sum(prod * up_vals, axis=0)

    # Fused SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up
    # Already in fp32 — sigmoid needs fp32 anyway
    sigmoid_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sigmoid_gate
    result = silu_gate * acc_up

    out_offsets = row_idx * stride_oe + n_range * stride_on
    tl.store(out_ptr + out_offsets, result.to(out_ptr.dtype.element_ty), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
    ],
    key=["N", "K"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def indexed_matvec_varying_weighted_kernel(
    x_ptr,                  # [num_experts_active, K]
    W_ptr,                  # [num_experts, K, N]
    expert_ids_ptr,         # [num_experts_active]
    weights_ptr,            # [num_experts_active] — routing weights
    out_ptr,                # [1, N] — weighted sum output (accumulated via atomic_add)
    N,
    K,
    num_experts_active,
    stride_xe,
    stride_xk,
    stride_we,
    stride_wk,
    stride_wn,
    stride_on,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 128,
):
    """Per-expert GEMV with routing weight applied + atomic accumulation.

    Computes: out[0, :] = sum_e(weights[e] * x[e] @ W[expert_ids[e]])

    Grid: (num_experts_active, ceil(N / BLOCK_N))
    Fuses the down projection, weight multiplication, and expert reduction
    into a single kernel, eliminating 3 separate kernel launches (matvec,
    unsqueeze+mul, sum) per MoE layer.
    """
    row_idx = tl.program_id(0)
    pid_n = tl.program_id(1)

    if row_idx >= num_experts_active:
        return

    expert_idx = tl.load(expert_ids_ptr + row_idx)
    weight = tl.load(weights_ptr + row_idx).to(tl.float32)
    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        x_vals = tl.load(
            x_ptr + row_idx * stride_xe + k_range * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        w_offsets = (
            expert_idx * stride_we
            + k_range[:, None] * stride_wk
            + n_range[None, :] * stride_wn
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # Apply routing weight and accumulate into shared output via atomic_add
    weighted = acc * weight
    out_offsets = n_range * stride_on
    tl.atomic_add(out_ptr + out_offsets, weighted.to(out_ptr.dtype.element_ty), mask=n_mask)
