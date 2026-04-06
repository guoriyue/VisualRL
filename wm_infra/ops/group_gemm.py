"""Grouped GEMM ops: single-kernel multi-expert matmul with autograd."""

import torch
import triton

from wm_infra.kernels.group_gemm_kernel import (
    grouped_gemm_kernel,
    grouped_gemm_bwd_dA_kernel,
    grouped_gemm_bwd_dB_kernel,
    grouped_gemm_fp8_kernel,
)

def _select_block_m(expert_offsets, num_experts, decode_mode=False):
    """Select optimal BLOCK_M based on max expert load.

    Uses smaller blocks when experts have few tokens (decode regime)
    to avoid wasted compute on zero-padded rows within each tile.

    At decode with E=64, top-2, M=128: max_load ≈ 9, so BLOCK_M=16
    saves 8x compute vs BLOCK_M=128 (only 16 rows padded instead of 128).

    Args:
        decode_mode: If True, skip GPU→CPU sync and return BLOCK_M=16
            immediately. Safe when num_tokens is small (S=1 decode).
    """
    if decode_mode:
        return 16
    loads = expert_offsets[1:num_experts + 1] - expert_offsets[:num_experts]
    max_load = int(loads.max().item())
    if max_load <= 16:
        return 16
    if max_load <= 32:
        return 32
    if max_load <= 64:
        return 64
    return 128


def _build_tile_mapping(expert_offsets, num_experts, block_m, device,
                        decode_mode=False):
    """Precompute tile→expert mapping arrays entirely on GPU.

    Uses vectorized torch ops (no Python loops, no .cpu() calls) so it's
    compatible with torch.compile.

    Args:
        decode_mode: If True, avoid GPU→CPU sync by using an upper-bound
            grid size. Arrays are padded with sentinel values (m_end=0)
            so extra kernel threads produce no output. Safe because all
            kernels mask writes with `m_range < e_end`.

    Returns:
        tile_expert_ids: [total_m_tiles] int32 — expert id per M-tile
        tile_m_offsets:  [total_m_tiles] int32 — token start offset per M-tile
        tile_m_ends:     [total_m_tiles] int32 — token end offset per M-tile
        total_m_tiles:   int
    """
    # Compute per-expert token counts and tile counts on GPU
    starts = expert_offsets[:num_experts]
    ends = expert_offsets[1:num_experts + 1]
    loads = ends - starts
    tile_counts = (loads + block_m - 1) // block_m  # cdiv on GPU

    if decode_mode:
        # At decode (S=1) with BLOCK_M=16, each expert has at most 1 tile.
        # We assign exactly 1 tile per expert (num_experts tiles total).
        # Experts with zero load get start==end, so the kernel's m_mask is
        # all-False and writes nothing. This avoids torch.repeat_interleave
        # (which is synchronizing) and all GPU→CPU syncs.
        tile_expert_ids = torch.arange(num_experts, device=device, dtype=torch.int32)
        tile_m_offsets = starts.to(torch.int32)
        tile_m_ends = ends.to(torch.int32)
        return tile_expert_ids, tile_m_offsets, tile_m_ends, num_experts

    total_m_tiles = int(tile_counts.sum().item())  # one GPU→CPU sync
    if total_m_tiles == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            0,
        )

    # Expert IDs: repeat each expert ID by its tile count
    expert_arange = torch.arange(num_experts, device=device, dtype=torch.int32)
    tile_expert_ids = torch.repeat_interleave(expert_arange, tile_counts.int())

    # Tile starts (exclusive prefix sum of tile_counts)
    tile_starts = torch.cumsum(tile_counts, dim=0) - tile_counts  # [num_experts]

    # Within-expert tile index: [0, 1, ..., tiles_e-1] for each expert
    global_idx = torch.arange(total_m_tiles, device=device, dtype=torch.int64)
    expert_tile_start = torch.repeat_interleave(tile_starts, tile_counts.int())
    within_tile = global_idx - expert_tile_start

    # M-offsets: expert_start + within_tile * block_m
    expert_starts = torch.repeat_interleave(starts, tile_counts.int())
    tile_m_offsets = (expert_starts + within_tile * block_m).to(torch.int32)

    # M-ends: expert_end for each tile
    tile_m_ends = torch.repeat_interleave(ends, tile_counts.int()).to(torch.int32)

    return tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles


class GroupedGEMMFunction(torch.autograd.Function):
    """Grouped GEMM with forward and backward.

    Forward:  C[e] = A[start_e:end_e] @ B[e]  for all experts
    Backward: dA[e] = dC[e] @ B[e]^T          (input gradient)
              dB[e] = A[e]^T @ dC[e]           (weight gradient)
    """

    @staticmethod
    def forward(ctx, A, B, expert_offsets, num_experts, decode_mode=False):
        total_tokens, K = A.shape
        _, _, N = B.shape

        C = torch.empty(total_tokens, N, dtype=A.dtype, device=A.device)

        BLOCK_M = _select_block_m(expert_offsets, num_experts,
                                  decode_mode=decode_mode)

        tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles = \
            _build_tile_mapping(expert_offsets, num_experts, BLOCK_M, A.device,
                                decode_mode=decode_mode)

        if total_m_tiles == 0:
            return C

        def grid(META):
            return (total_m_tiles * triton.cdiv(N, META["BLOCK_N"]),)

        grouped_gemm_kernel[grid](
            A, B, C,
            tile_expert_ids, tile_m_offsets, tile_m_ends,
            N, K, total_m_tiles,
            A.stride(1), A.stride(0),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
        )

        ctx.save_for_backward(A, B, expert_offsets)
        ctx.num_experts = num_experts
        return C

    @staticmethod
    def backward(ctx, dC):
        A, B, expert_offsets = ctx.saved_tensors
        num_experts = ctx.num_experts
        total_tokens, K = A.shape
        _, _, N = B.shape

        dC = dC.contiguous()

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles = \
            _build_tile_mapping(expert_offsets, num_experts, BLOCK_M, A.device)

        dA = None
        dB = None

        # ── dA = dC @ B^T (input gradient) ──
        if ctx.needs_input_grad[0]:
            dA = torch.empty_like(A)
            if total_m_tiles > 0:
                num_k_tiles = triton.cdiv(K, BLOCK_N)
                grid_dA = (total_m_tiles * num_k_tiles,)
                grouped_gemm_bwd_dA_kernel[grid_dA](
                    dC, B, dA,
                    tile_expert_ids, tile_m_offsets, tile_m_ends,
                    N, K,
                    dC.stride(0), dC.stride(1),
                    B.stride(0), B.stride(1), B.stride(2),
                    dA.stride(0), dA.stride(1),
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K,
                )

        # ── dB = A^T @ dC (weight gradient) ──
        if ctx.needs_input_grad[1]:
            dB = torch.zeros_like(B)
            # Use smaller block sizes for dB to fit in shared memory
            DB_BLOCK_M = 64   # reduction dim (tokens)
            DB_BLOCK_K = 32   # output K dim
            DB_BLOCK_N = 64   # output N dim
            num_k_tiles = triton.cdiv(K, DB_BLOCK_K)
            num_n_tiles = triton.cdiv(N, DB_BLOCK_N)
            grid_dB = (num_experts * num_k_tiles * num_n_tiles,)
            grouped_gemm_bwd_dB_kernel[grid_dB](
                A, dC, dB,
                expert_offsets,
                N, K, num_experts,
                A.stride(0), A.stride(1),
                dC.stride(0), dC.stride(1),
                dB.stride(0), dB.stride(1), dB.stride(2),
                BLOCK_M=DB_BLOCK_M,
                BLOCK_N=DB_BLOCK_N,
                BLOCK_K=DB_BLOCK_K,
            )

        return dA, dB, None, None, None


def grouped_gemm(A, B, expert_offsets, num_experts, decode_mode=False):
    """Grouped GEMM: C[e] = A_e @ B[e] for all experts in one kernel.

    Args:
        A: [total_tokens, K] — input sorted by expert
        B: [num_experts, K, N] — expert weight matrices
        expert_offsets: [num_experts + 1] — start index per expert
        num_experts: int
        decode_mode: skip GPU→CPU syncs (safe when num_tokens is small)

    Returns:
        C: [total_tokens, N]
    """
    return GroupedGEMMFunction.apply(A, B, expert_offsets, num_experts,
                                    decode_mode)


def grouped_gemm_naive(A, B, expert_offsets, num_experts):
    """Naive reference implementation: loop over experts."""
    total_tokens, K = A.shape
    N = B.shape[2]
    C = torch.empty(total_tokens, N, dtype=A.dtype, device=A.device)

    for e in range(num_experts):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end > start:
            C[start:end] = A[start:end] @ B[e]

    return C


# ─── FP8 Grouped GEMM ───

class GroupedGEMMFP8Function(torch.autograd.Function):
    """FP8 Grouped GEMM with forward in FP8, backward in FP16.

    Forward:  C[e] = (A_fp8[start_e:end_e] @ B_fp8[e]) * a_scale * b_scale[e]
    Backward: Dequantizes to FP16 and uses standard backward kernels.
    """

    @staticmethod
    def forward(ctx, A_fp8, A_scale, B_fp8, B_scales, expert_offsets, num_experts,
                output_dtype, decode_mode=False):
        total_tokens, K = A_fp8.shape
        _, _, N = B_fp8.shape

        C = torch.empty(total_tokens, N, dtype=output_dtype, device=A_fp8.device)

        BLOCK_M = _select_block_m(expert_offsets, num_experts,
                                  decode_mode=decode_mode)

        tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles = \
            _build_tile_mapping(expert_offsets, num_experts, BLOCK_M, A_fp8.device,
                                decode_mode=decode_mode)

        if total_m_tiles == 0:
            return C

        def grid(META):
            return (total_m_tiles * triton.cdiv(N, META["BLOCK_N"]),)

        grouped_gemm_fp8_kernel[grid](
            A_fp8, B_fp8, C,
            A_scale, B_scales,
            tile_expert_ids, tile_m_offsets, tile_m_ends,
            N, K, total_m_tiles,
            A_fp8.stride(1), A_fp8.stride(0),
            B_fp8.stride(0), B_fp8.stride(1), B_fp8.stride(2),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
        )

        ctx.save_for_backward(A_fp8, A_scale, B_fp8, B_scales, expert_offsets)
        ctx.num_experts = num_experts
        ctx.output_dtype = output_dtype
        return C

    @staticmethod
    def backward(ctx, dC):
        A_fp8, A_scale, B_fp8, B_scales, expert_offsets = ctx.saved_tensors
        num_experts = ctx.num_experts
        output_dtype = ctx.output_dtype

        # Dequantize to FP16/BF16 for backward
        from wm_infra.ops.quantize import dequantize_per_tensor

        A = dequantize_per_tensor(A_fp8, A_scale, output_dtype)
        # Dequantize B per-expert
        B = B_fp8.to(output_dtype) * B_scales[:, None, None].to(output_dtype)

        dC = dC.contiguous()

        total_tokens, K = A.shape
        _, _, N = B.shape

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        tile_expert_ids, tile_m_offsets, tile_m_ends, total_m_tiles = \
            _build_tile_mapping(expert_offsets, num_experts, BLOCK_M, A.device)

        dA = None
        dB = None

        if ctx.needs_input_grad[0]:
            dA = torch.empty_like(A)
            if total_m_tiles > 0:
                num_k_tiles = triton.cdiv(K, BLOCK_N)
                grid_dA = (total_m_tiles * num_k_tiles,)
                grouped_gemm_bwd_dA_kernel[grid_dA](
                    dC, B, dA,
                    tile_expert_ids, tile_m_offsets, tile_m_ends,
                    N, K,
                    dC.stride(0), dC.stride(1),
                    B.stride(0), B.stride(1), B.stride(2),
                    dA.stride(0), dA.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )

        if ctx.needs_input_grad[2]:
            dB = torch.zeros_like(B)
            DB_BLOCK_M = 64
            DB_BLOCK_K = 32
            DB_BLOCK_N = 64
            num_k_tiles = triton.cdiv(K, DB_BLOCK_K)
            num_n_tiles = triton.cdiv(N, DB_BLOCK_N)
            grid_dB = (num_experts * num_k_tiles * num_n_tiles,)
            grouped_gemm_bwd_dB_kernel[grid_dB](
                A, dC, dB,
                expert_offsets,
                N, K, num_experts,
                A.stride(0), A.stride(1),
                dC.stride(0), dC.stride(1),
                dB.stride(0), dB.stride(1), dB.stride(2),
                BLOCK_M=DB_BLOCK_M, BLOCK_N=DB_BLOCK_N, BLOCK_K=DB_BLOCK_K,
            )

        return dA, None, dB, None, None, None, None, None


def grouped_gemm_fp8(A_fp8, A_scale, B_fp8, B_scales, expert_offsets, num_experts,
                     output_dtype=torch.float16, decode_mode=False):
    """FP8 Grouped GEMM: C[e] = (A_fp8_e @ B_fp8[e]) * a_scale * b_scale[e].

    Args:
        A_fp8: [total_tokens, K] — FP8 activations
        A_scale: scalar — per-tensor activation scale (fp32)
        B_fp8: [num_experts, K, N] — FP8 weights
        B_scales: [num_experts] — per-expert weight scales (fp32)
        expert_offsets: [num_experts + 1]
        num_experts: int
        output_dtype: output precision
        decode_mode: skip GPU→CPU syncs (safe when num_tokens is small)

    Returns:
        C: [total_tokens, N] in output_dtype
    """
    return GroupedGEMMFP8Function.apply(
        A_fp8, A_scale, B_fp8, B_scales, expert_offsets, num_experts,
        output_dtype, decode_mode)
