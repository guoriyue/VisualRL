"""Fused MoE kernels: 2-stage design for high performance.

Stage 1: fused_gate_up_kernel — gather input + gate/up GEMMs + SwiGLU
Stage 2: fused_down_kernel   — down GEMM + weighted scatter (atomic_add)

This 2-stage approach distributes work across many thread blocks (same as
composable mode), unlike the earlier single-kernel design where each block
looped over the entire intermediate dimension.
"""

import triton
import triton.language as tl


# ─── Stage 1: Gate+Up GEMM + SwiGLU ───

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_INTER": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_INTER": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_INTER": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_INTER": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_INTER": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_INTER": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_INTER": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_INTER": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=4),
    ],
    key=["hidden_dim", "intermediate_dim"],
)
@triton.jit
def fused_gate_up_kernel(
    # ── Input ──
    hidden_ptr,             # [num_tokens, hidden_dim]
    # ── Expert weights ──
    w_gate_ptr,             # [num_experts, hidden_dim, intermediate_dim]
    w_up_ptr,               # [num_experts, hidden_dim, intermediate_dim]
    # ── Routing info ──
    sorted_token_ids_ptr,   # [num_tokens * top_k]
    # ── Tile mapping ──
    tile_expert_ids_ptr,    # [total_m_tiles]
    tile_m_offsets_ptr,     # [total_m_tiles]
    tile_m_ends_ptr,        # [total_m_tiles]
    # ── Output ──
    intermediate_ptr,       # [total_sorted, intermediate_dim] — SwiGLU output
    # ── Shapes ──
    hidden_dim: tl.constexpr,
    intermediate_dim,
    top_k: tl.constexpr,
    total_m_tiles,
    # ── Strides ──
    stride_hm, stride_hd,
    stride_ge, stride_gk, stride_gn,
    stride_ue, stride_uk, stride_un,
    stride_im, stride_in,            # intermediate strides
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_INTER: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
):
    """Stage 1: Fused gather + gate/up GEMM + SwiGLU.

    Grid: (total_m_tiles * ceil(intermediate_dim / BLOCK_INTER),)
    Each block computes one (BLOCK_M, BLOCK_INTER) tile — only loops over hidden_dim.
    """
    pid = tl.program_id(0)

    # ── Map pid to (pid_m, pid_n) with GROUP_M swizzle ──
    num_inter_tiles = tl.cdiv(intermediate_dim, BLOCK_INTER)
    num_pid_in_group = GROUP_M * num_inter_tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(total_m_tiles - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    # ── O(1) tile→expert lookup ──
    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end

    # ── Gather input (fused permute) ──
    sorted_ids = tl.load(sorted_token_ids_ptr + m_range, mask=m_mask, other=0)
    original_token_ids = sorted_ids // top_k

    # ── Output tile boundaries ──
    n_start = pid_n * BLOCK_INTER
    n_range = n_start + tl.arange(0, BLOCK_INTER)
    n_mask = n_range < intermediate_dim

    # ── Tiled gate+up GEMM: hidden @ W_gate, hidden @ W_up ──
    gate_acc = tl.zeros((BLOCK_M, BLOCK_INTER), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_INTER), dtype=tl.float32)

    for k_start in range(0, hidden_dim, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < hidden_dim

        # Load input tile: [BLOCK_M, BLOCK_K]
        inp_offsets = original_token_ids[:, None] * stride_hm + k_range[None, :] * stride_hd
        inp_mask = m_mask[:, None] & k_mask[None, :]
        inp = tl.load(hidden_ptr + inp_offsets, mask=inp_mask, other=0.0)

        # Load W_gate tile: [BLOCK_K, BLOCK_INTER]
        wg_offsets = (expert_id * stride_ge +
                     k_range[:, None] * stride_gk +
                     n_range[None, :] * stride_gn)
        wg_mask = k_mask[:, None] & n_mask[None, :]
        wg = tl.load(w_gate_ptr + wg_offsets, mask=wg_mask, other=0.0)

        # Load W_up tile: [BLOCK_K, BLOCK_INTER]
        wu_offsets = (expert_id * stride_ue +
                     k_range[:, None] * stride_uk +
                     n_range[None, :] * stride_un)
        wu = tl.load(w_up_ptr + wu_offsets, mask=wg_mask, other=0.0)

        gate_acc += tl.dot(inp, wg)
        up_acc += tl.dot(inp, wu)

    # ── SwiGLU in registers ──
    sigmoid_gate = tl.sigmoid(gate_acc)
    activated = (gate_acc * sigmoid_gate) * up_acc

    # ── Store to intermediate buffer ──
    out_offsets = m_range[:, None] * stride_im + n_range[None, :] * stride_in
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(intermediate_ptr + out_offsets, activated.to(intermediate_ptr.dtype.element_ty), mask=out_mask)


# ─── Stage 2: Down GEMM + Weighted Scatter ───

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HIDDEN": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_HIDDEN": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_HIDDEN": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_HIDDEN": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_HIDDEN": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_HIDDEN": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_HIDDEN": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_HIDDEN": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=2, num_warps=4),
    ],
    key=["hidden_dim", "intermediate_dim"],
    # Must re-zero output between autotune trials since kernel uses atomic_add
    reset_to_zero=["output_ptr"],
)
@triton.jit
def fused_down_kernel(
    # ── Input (from Stage 1) ──
    intermediate_ptr,       # [total_sorted, intermediate_dim]
    # ── Expert weights ──
    w_down_ptr,             # [num_experts, intermediate_dim, hidden_dim]
    # ── Routing info ──
    sorted_token_ids_ptr,   # [num_tokens * top_k]
    topk_weights_ptr,       # [num_tokens, top_k]
    # ── Tile mapping ──
    tile_expert_ids_ptr,
    tile_m_offsets_ptr,
    tile_m_ends_ptr,
    # ── Output ──
    output_ptr,             # [num_tokens, hidden_dim] — atomic add for combine
    # ── Shapes ──
    hidden_dim,
    intermediate_dim: tl.constexpr,
    top_k: tl.constexpr,
    total_m_tiles,
    # ── Strides ──
    stride_im, stride_in,
    stride_de, stride_dk, stride_dn,
    stride_om, stride_od,
    # ── Tuning ──
    BLOCK_M: tl.constexpr = 128,
    BLOCK_HIDDEN: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
):
    """Stage 2: Down GEMM + weighted scatter.

    Grid: (total_m_tiles * ceil(hidden_dim / BLOCK_HIDDEN),)
    Each block computes one (BLOCK_M, BLOCK_HIDDEN) tile — loops over intermediate_dim.
    Applies routing weights and scatters via atomic_add.
    """
    pid = tl.program_id(0)

    # ── Map pid to (pid_m, pid_n) with GROUP_M swizzle ──
    num_hidden_tiles = tl.cdiv(hidden_dim, BLOCK_HIDDEN)
    num_pid_in_group = GROUP_M * num_hidden_tiles
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(total_m_tiles - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size
    pid_n = (pid % num_pid_in_group) // group_size

    # ── O(1) tile→expert lookup ──
    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_offsets_ptr + pid_m)
    e_end = tl.load(tile_m_ends_ptr + pid_m)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < e_end

    # ── Token IDs for scatter ──
    sorted_ids = tl.load(sorted_token_ids_ptr + m_range, mask=m_mask, other=0)
    original_token_ids = sorted_ids // top_k
    k_indices = sorted_ids % top_k

    # ── Output tile boundaries ──
    h_start = pid_n * BLOCK_HIDDEN
    h_range = h_start + tl.arange(0, BLOCK_HIDDEN)
    h_mask = h_range < hidden_dim

    # ── Tiled down GEMM: intermediate @ W_down ──
    acc = tl.zeros((BLOCK_M, BLOCK_HIDDEN), dtype=tl.float32)

    for k_start in range(0, intermediate_dim, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < intermediate_dim

        # Load intermediate tile: [BLOCK_M, BLOCK_K]
        i_offsets = m_range[:, None] * stride_im + k_range[None, :] * stride_in
        i_mask = m_mask[:, None] & k_mask[None, :]
        inter = tl.load(intermediate_ptr + i_offsets, mask=i_mask, other=0.0)

        # Load W_down tile: [BLOCK_K, BLOCK_HIDDEN]
        wd_offsets = (expert_id * stride_de +
                     k_range[:, None] * stride_dk +
                     h_range[None, :] * stride_dn)
        wd_mask = k_mask[:, None] & h_mask[None, :]
        wd = tl.load(w_down_ptr + wd_offsets, mask=wd_mask, other=0.0)

        acc += tl.dot(inter, wd)

    # ── Apply routing weights ──
    weight_offsets = original_token_ids * top_k + k_indices
    weights = tl.load(topk_weights_ptr + weight_offsets, mask=m_mask, other=0.0)
    acc = acc * weights[:, None]

    # ── Scatter output with atomic_add ──
    out_offsets = original_token_ids[:, None] * stride_om + h_range[None, :] * stride_od
    out_mask = m_mask[:, None] & h_mask[None, :]
    tl.atomic_add(output_ptr + out_offsets, acc.to(output_ptr.dtype.element_ty), mask=out_mask)
