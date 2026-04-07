"""Attention Layer: Multi-Head Attention with GQA, RoPE, and optional KV cache.

Also includes MLAAttentionLayer for DeepSeek-V2/V3 Multi-head Latent Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from wm_infra.config import TransformerConfig
from wm_infra.ops.attention import (
    flash_attention,
    flash_attention_decode,
    resolve_attention_backend,
)
from wm_infra.ops.rope import precompute_rope_freqs, apply_rope
from wm_infra.ops.rmsnorm import rms_norm
from wm_infra.ops.kv_cache import KVCache, MLAKVCache


def _apply_rope_inline(x, cos_row, sin_row):
    """Apply RoPE to a single decode position using plain torch tensor ops.

    Args:
        x: [B, 1, H, D] or [B, H, 1, D] input tensor
        cos_row: [D//2] cos values for this position
        sin_row: [D//2] sin values for this position

    Returns:
        Rotated tensor, same shape as x.
    """
    half_d = x.shape[-1] // 2
    x0 = x[..., :half_d]
    x1 = x[..., half_d:]
    y0 = x0 * cos_row - x1 * sin_row
    y1 = x0 * sin_row + x1 * cos_row
    return torch.cat([y0, y1], dim=-1)


class AttentionLayer(nn.Module):
    """Multi-Head Attention with Grouped-Query Attention and Rotary Position Embeddings.

    Llama-style: no bias in Q/K/V/O projections, pre-norm applied externally.

    Input:  [B, S, hidden_dim]
    Output: [B, S, hidden_dim]

    Usage:
        cfg = TransformerConfig(num_heads=32, num_kv_heads=8, head_dim=128)
        attn = AttentionLayer(cfg).cuda().half()

        x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
        pos = torch.arange(128, device="cuda")
        out = attn(x, pos)  # training

        # Inference with KV cache
        cache = KVCache(2, 8, 4096, 128, dtype=torch.float16, device="cuda")
        out = attn(x[:, :1, :], pos[:1], kv_cache=cache)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Q/K/V/O projections
        qkv_bias = getattr(config, 'qkv_bias', False)
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)

        # Fused QKV projection for decode: single matmul instead of 3 separate ones.
        # Lazily initialized on first forward_decode call via _init_fused_qkv().
        self._qkv_proj: Optional[nn.Linear] = None
        self._qkv_q_dim = self.num_heads * self.head_dim
        self._qkv_k_dim = self.num_kv_heads * self.head_dim
        self._qkv_v_dim = self.num_kv_heads * self.head_dim

        # GQA group count
        self._gqa_groups = self.num_heads // self.num_kv_heads

        # Attention backend
        self._attention_backend = getattr(config, 'attention_backend', 'auto')
        self._resolved_attention_backend = resolve_attention_backend(
            self._attention_backend
        )
        self._compile_attention = getattr(config, 'compile_attention', False)
        self._compiled_decode = None  # lazily compiled
        self._decode_position_buffer = None
        self._rope_cache = {}

        # Precompute RoPE cos/sin as persistent buffers
        cos, sin = precompute_rope_freqs(
            self.head_dim, config.max_seq_len, config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_fused_qkv(self):
        """Build the fused QKV weight matrix from the existing separate projections.

        This is called lazily on the first forward_decode call. The fused weight
        is stored as a parameter-less buffer so it doesn't interfere with
        optimizer state or checkpoint loading (which uses q_proj/k_proj/v_proj).
        """
        q_w = self.q_proj.weight.data  # [Hq*D, hidden]
        k_w = self.k_proj.weight.data  # [Hkv*D, hidden]
        v_w = self.v_proj.weight.data  # [Hkv*D, hidden]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)  # [(Hq+2*Hkv)*D, hidden]

        qkv_bias = getattr(self.config, 'qkv_bias', False)
        if qkv_bias:
            q_b = self.q_proj.bias.data
            k_b = self.k_proj.bias.data
            v_b = self.v_proj.bias.data
            qkv_b = torch.cat([q_b, k_b, v_b], dim=0)
        else:
            qkv_b = None

        total_out = qkv_w.shape[0]
        self._qkv_proj = nn.Linear(self.hidden_dim, total_out, bias=qkv_bias,
                                    device=qkv_w.device, dtype=qkv_w.dtype)
        self._qkv_proj.weight = nn.Parameter(qkv_w, requires_grad=False)
        if qkv_b is not None:
            self._qkv_proj.bias = nn.Parameter(qkv_b, requires_grad=False)

    def _split_qkv(self, qkv: torch.Tensor, B: int):
        """Split fused QKV output into separate Q, K, V tensors.

        Args:
            qkv: [B, 1, Hq*D + Hkv*D + Hkv*D]
            B: batch size

        Returns:
            q: [B, self.num_heads, 1, D]
            k: [B, self.num_kv_heads, 1, D]
            v: [B, self.num_kv_heads, 1, D]
        """
        q_end = self._qkv_q_dim
        k_end = q_end + self._qkv_k_dim
        # All three are contiguous slices of a contiguous tensor, so .view() is free
        q = qkv[:, :, :q_end].view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[:, :, q_end:k_end].view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = qkv[:, :, k_end:].view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_dim]
            positions: [S] position indices (shared across batch)
            kv_cache: optional KVCache for inference

        Returns:
            output: [B, S, hidden_dim]
        """
        B, S, _ = x.shape

        # Project Q/K/V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)      # [B, S, Hq, D]
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)   # [B, S, Hkv, D]
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)   # [B, S, Hkv, D]

        # Apply RoPE to Q and K
        cos, sin = self._get_rope_tables(x.device, x.dtype)
        q = apply_rope(q, cos, sin, positions)   # [B, S, Hq, D]
        k = apply_rope(k, cos, sin, positions)   # [B, S, Hkv, D]

        # Transpose to [B, H, S, D] for attention
        q = q.transpose(1, 2)   # [B, Hq, S, D]
        k = k.transpose(1, 2)   # [B, Hkv, S, D]
        v = v.transpose(1, 2)   # [B, Hkv, S, D]

        # KV cache update (inference)
        if kv_cache is not None:
            k_cont = k.contiguous()
            v_cont = v.contiguous()
            k, v = kv_cache.update(k_cont, v_cont, positions)
            # Prefill (S_q == S_kv): need causal mask so tokens don't see future
            # Decode (S_q == 1 < S_kv): no mask needed, cache is already causal
            causal = (S == k.shape[2])
        else:
            causal = True

        # Keep layouts explicit before dispatching to FlashAttention or SDPA.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Flash Attention with backend dispatch
        attn_out = flash_attention(q, k, v, causal=causal,
                                   backend=self._resolved_attention_backend)  # [B, Hq, S, D]

        # Reshape back and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, hidden_dim]
        return self.o_proj(attn_out)

    @property
    def resolved_attention_backend(self) -> str:
        return self._resolved_attention_backend

    def _get_decode_positions(self, device: torch.device, position: int) -> torch.Tensor:
        """Reuse a 1-element positions tensor during decode to avoid allocations."""
        if (
            self._decode_position_buffer is None
            or self._decode_position_buffer.device != device
        ):
            self._decode_position_buffer = torch.empty(
                1, device=device, dtype=torch.int64
            )
        self._decode_position_buffer[0] = position
        return self._decode_position_buffer

    def _get_rope_tables(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = (
                self.rope_cos.to(device=device, dtype=dtype),
                self.rope_sin.to(device=device, dtype=dtype),
            )
            self._rope_cache[key] = cached
        return cached

    def forward_decode(
        self,
        x: torch.Tensor,
        position: int,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Optimized single-token decode path.

        Compared to the generic forward(), this path:
        1. Fused QKV projection: single matmul instead of 3 separate ones
        2. Inline RoPE: uses plain torch tensor ops for a single position
        3. Minimal reshapes: no unnecessary .contiguous() calls
        4. SDPA decode: uses F.scaled_dot_product_attention (fast for S_q=1)
        """
        B, S, _ = x.shape
        assert S == 1, f"forward_decode expects S=1, got {S}"

        # ── Lazy init fused QKV weight ──
        if self._qkv_proj is None:
            self._init_fused_qkv()

        # ── Fused QKV projection: 1 matmul instead of 3 ──
        qkv = self._qkv_proj(x)  # [B, 1, (Hq + 2*Hkv) * D]
        q, k, v = self._split_qkv(qkv, B)  # each: [B, H, 1, D]

        # ── Inline RoPE for single position (plain torch tensor ops) ──
        cos_full, sin_full = self._get_rope_tables(x.device, x.dtype)
        cos_row = cos_full[position]  # [D//2]
        sin_row = sin_full[position]  # [D//2]
        q = _apply_rope_inline(q, cos_row, sin_row)
        k = _apply_rope_inline(k, cos_row, sin_row)

        # ── KV cache update ──
        # k, v are [B, Hkv, 1, D] — contiguous after the view+transpose above
        # (transpose of [B, 1, H, D] with dim1=1 gives [B, H, 1, D] which is
        # contiguous since size-1 dims don't affect memory layout)
        if kv_cache is not None:
            k, v = kv_cache.update_decode(k.contiguous(), v.contiguous())

        # ── Attention (backend-dispatched for decode) ──
        backend = self._resolved_attention_backend
        if backend == "flashinfer":
            # FlashInfer has specialized decode kernels; handles GQA natively
            attn_out = flash_attention_decode(
                q, k, v,
                backend="flashinfer",
            )  # [B, Hq, 1, D]
        else:
            # SDPA with native GQA support (PyTorch 2.6+).
            # enable_gqa=True lets SDPA handle Hq != Hkv internally,
            # avoiding the expensive K/V expansion + .contiguous() copies.
            # q is already contiguous after inline RoPE (torch.cat creates new tensor).
            # k, v are cache slices — non-contiguous but SDPA handles that fine.
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False,
                enable_gqa=(self._gqa_groups > 1),
            )  # [B, Hq, 1, D]

        # ── Output projection ──
        # attn_out is [B, Hq, 1, D], reshape to [B, 1, Hq*D] for o_proj
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(attn_out)

    def forward_decode_cudagraph(
        self,
        x: torch.Tensor,
        pos_index: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CUDA graph-compatible single-token decode path.

        All operations use static tensor shapes/addresses. Position is a GPU
        tensor (not Python int) so RoPE indexing stays on-device. KV cache
        returns full pre-allocated buffers; attention masking hides unused
        positions.

        Args:
            x: [B, 1, hidden_dim] input hidden state
            pos_index: [1] int64 GPU tensor with the current position
            kv_cache: KVCache instance (required)
            attn_mask: [1, 1, 1, max_seq_len] float mask (-inf for padding, 0 for valid)

        Returns:
            output: [B, 1, hidden_dim]
        """
        B = x.shape[0]

        # Lazy init fused QKV weight
        if self._qkv_proj is None:
            self._init_fused_qkv()

        # Fused QKV projection
        qkv = self._qkv_proj(x)
        q, k, v = self._split_qkv(qkv, B)

        # Inline RoPE using GPU-side index_select (graph-safe, no CPU sync).
        # cos_full: [max_seq, D//2] -> index_select(0, pos_index) -> [1, D//2]
        cos_full, sin_full = self._get_rope_tables(x.device, x.dtype)
        cos_row = cos_full.index_select(0, pos_index).squeeze(0)  # [D//2]
        sin_row = sin_full.index_select(0, pos_index).squeeze(0)  # [D//2]
        q = _apply_rope_inline(q, cos_row, sin_row)
        k = _apply_rope_inline(k, cos_row, sin_row)

        # KV cache update — returns FULL buffers [B, Hkv, max_seq, D]
        k_full, v_full = kv_cache.update_decode_cudagraph(
            k.contiguous(), v.contiguous(), pos_index
        )

        # GQA expansion (static shapes for graph compatibility)
        if self._gqa_groups > 1:
            Sk = k_full.shape[2]
            k_full = k_full.unsqueeze(2).expand(
                B, self.num_kv_heads, self._gqa_groups, Sk, self.head_dim
            ).reshape(B, self.num_heads, Sk, self.head_dim)
            v_full = v_full.unsqueeze(2).expand(
                B, self.num_kv_heads, self._gqa_groups, Sk, self.head_dim
            ).reshape(B, self.num_heads, Sk, self.head_dim)

        # SDPA with attention mask (static shapes, graph-compatible)
        attn_out = F.scaled_dot_product_attention(
            q.contiguous(), k_full.contiguous(), v_full.contiguous(),
            attn_mask=attn_mask, is_causal=False,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(attn_out)


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm wrapping the Triton op (avoids circular import with transformer_block)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class MLAAttentionLayer(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V2/V3.

    Compresses KV cache via low-rank latent representations with decoupled RoPE.
    Achieves ~93% KV cache reduction compared to standard MHA.

    Data flow:
        Q path:  hidden → q_a_proj → q_a_layernorm → q_b_proj → split(q_nope, q_rope) → RoPE
        KV path: hidden → kv_a_proj → split(kv_latent, k_rope) → layernorm → RoPE
                 kv_latent → kv_b_proj → split(k_nope, v)
                 k = concat(k_nope, k_rope_expanded)
                 q = concat(q_nope, q_rope)
        Attention: SDPA(q, k, v) where d_qk != d_v
        KV cache: stores (kv_latent, k_rope_after_RoPE) — NOT full K/V

    Input:  [B, S, hidden_dim]
    Output: [B, S, hidden_dim]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim  # nope + rope
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank

        # Q path
        if self.q_lora_rank is not None:
            # Compressed Q: hidden → q_a → layernorm → q_b
            self.q_a_proj = nn.Linear(self.hidden_dim, self.q_lora_rank, bias=False)
            self.q_a_layernorm = _RMSNorm(self.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            # Direct Q projection (e.g., DeepSeek-V2-Lite)
            self.q_proj = nn.Linear(
                self.hidden_dim, self.num_heads * self.qk_head_dim, bias=False
            )

        # KV path: hidden → kv_a_proj → [kv_latent, k_rope]
        self.kv_a_proj = nn.Linear(
            self.hidden_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = _RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        # Reconstruct K/V from latent: latent → [k_nope, v] per head
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_dim, bias=False
        )

        # RoPE for rope dim only
        cos, sin = precompute_rope_freqs(
            self.qk_rope_head_dim, config.max_seq_len, config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Scale factor for attention
        self.scale = self.qk_head_dim ** -0.5
        self._decode_position_buffer = None

    @property
    def resolved_attention_backend(self) -> str:
        return "sdpa"

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[MLAKVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_dim]
            positions: [S] position indices
            kv_cache: optional MLAKVCache for inference

        Returns:
            output: [B, S, hidden_dim]
        """
        B, S, _ = x.shape

        # ─── Q path ───
        if self.q_lora_rank is not None:
            q = self.q_a_proj(x)                         # [B, S, q_lora_rank]
            q = self.q_a_layernorm(q)                    # [B, S, q_lora_rank]
            q = self.q_b_proj(q)                         # [B, S, H * qk_head_dim]
        else:
            q = self.q_proj(x)                           # [B, S, H * qk_head_dim]

        q = q.view(B, S, self.num_heads, self.qk_head_dim)  # [B, S, H, qk_head_dim]
        q_nope = q[..., :self.qk_nope_head_dim]              # [B, S, H, qk_nope]
        q_rope = q[..., self.qk_nope_head_dim:].contiguous() # [B, S, H, qk_rope]

        # Apply RoPE to q_rope
        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        q_rope = apply_rope(q_rope, cos, sin, positions)     # [B, S, H, qk_rope]

        # ─── KV path ───
        kv_a = self.kv_a_proj(x)                              # [B, S, kv_lora_rank + qk_rope]
        kv_latent = kv_a[..., :self.kv_lora_rank]             # [B, S, kv_lora_rank]
        k_rope = kv_a[..., self.kv_lora_rank:]                # [B, S, qk_rope]

        # Apply layernorm to latent
        kv_latent = self.kv_a_layernorm(kv_latent)            # [B, S, kv_lora_rank]

        # Apply RoPE to k_rope (treat as [B, S, 1, qk_rope] for apply_rope)
        k_rope_4d = k_rope.unsqueeze(2).contiguous()           # [B, S, 1, qk_rope]
        k_rope_4d = apply_rope(k_rope_4d, cos, sin, positions)  # [B, S, 1, qk_rope]
        k_rope = k_rope_4d.squeeze(2)                         # [B, S, qk_rope]

        # ─── KV cache ───
        if kv_cache is not None:
            # Store compressed (latent, k_rope) in cache
            kv_latent, k_rope = kv_cache.update(kv_latent, k_rope, positions)
            # Prefill (S_q == S_kv): need causal mask so tokens don't see future
            # Decode (S_q == 1 < S_kv): no mask needed, cache is already causal
            causal = (S == kv_latent.shape[1])
        else:
            causal = True

        # ─── Reconstruct K/V from latent ───
        S_kv = kv_latent.shape[1]
        kv_b = self.kv_b_proj(kv_latent)  # [B, S_kv, H * (qk_nope + v_head_dim)]
        kv_b = kv_b.view(B, S_kv, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        k_nope = kv_b[..., :self.qk_nope_head_dim]    # [B, S_kv, H, qk_nope]
        v = kv_b[..., self.qk_nope_head_dim:]          # [B, S_kv, H, v_head_dim]

        # Expand k_rope to all heads: [B, S_kv, qk_rope] → [B, S_kv, H, qk_rope]
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # Assemble full K and Q
        k = torch.cat([k_nope, k_rope_expanded], dim=-1)  # [B, S_kv, H, qk_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1)           # [B, S, H, qk_head_dim]

        # Transpose to [B, H, S, D] for SDPA
        q = q.transpose(1, 2).contiguous()   # [B, H, S, qk_head_dim]
        k = k.transpose(1, 2).contiguous()   # [B, H, S_kv, qk_head_dim]
        v = v.transpose(1, 2).contiguous()   # [B, H, S_kv, v_head_dim]

        # Use PyTorch SDPA (handles d_qk != d_v, unlike our Triton flash_attention)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=self.scale,
        )  # [B, H, S, v_head_dim]

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, H * v_head_dim]
        return self.o_proj(attn_out)

    def _get_decode_positions(self, device: torch.device, position: int) -> torch.Tensor:
        """Reuse a 1-element positions tensor during decode to avoid allocations."""
        if (
            self._decode_position_buffer is None
            or self._decode_position_buffer.device != device
        ):
            self._decode_position_buffer = torch.empty(
                1, device=device, dtype=torch.int64
            )
        self._decode_position_buffer[0] = position
        return self._decode_position_buffer

    def forward_decode(
        self,
        x: torch.Tensor,
        position: int,
        kv_cache: Optional[MLAKVCache] = None,
    ) -> torch.Tensor:
        """Optimized single-token decode path for MLA.

        Uses inline RoPE and minimizes reshapes compared to generic forward().
        """
        B, S, _ = x.shape
        assert S == 1, f"forward_decode expects S=1, got {S}"

        if self.q_lora_rank is not None:
            q = self.q_a_proj(x)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(x)

        q = q.view(B, 1, self.num_heads, self.qk_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim:].contiguous()

        # Inline RoPE for single position (plain torch tensor ops)
        cos_full = self.rope_cos.to(x.dtype)
        sin_full = self.rope_sin.to(x.dtype)
        cos_row = cos_full[position]  # [qk_rope_head_dim // 2]
        sin_row = sin_full[position]
        q_rope = _apply_rope_inline(q_rope, cos_row, sin_row)

        kv_a = self.kv_a_proj(x)
        kv_latent = self.kv_a_layernorm(kv_a[..., :self.kv_lora_rank])
        k_rope = kv_a[..., self.kv_lora_rank:]
        # Inline RoPE for k_rope: [B, 1, qk_rope] -> apply rotation
        k_rope = _apply_rope_inline(k_rope.unsqueeze(2), cos_row, sin_row).squeeze(2)

        causal = True
        if kv_cache is not None:
            kv_latent, k_rope = kv_cache.update_decode(kv_latent, k_rope)
            causal = False

        S_kv = kv_latent.shape[1]
        kv_b = self.kv_b_proj(kv_latent)
        kv_b = kv_b.view(
            B, S_kv, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv_b[..., :self.qk_nope_head_dim]
        v = kv_b[..., self.qk_nope_head_dim:]
        k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        k = torch.cat([k_nope, k_rope_expanded], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            is_causal=causal,
            scale=self.scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(attn_out)

    def forward_decode_cudagraph(
        self,
        x: torch.Tensor,
        pos_index: torch.Tensor,
        kv_cache: Optional[MLAKVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CUDA graph-compatible single-token decode path for MLA.

        Uses GPU tensor position for RoPE. Returns full KV cache buffers with
        attention masking to hide padding positions.

        Args:
            x: [B, 1, hidden_dim]
            pos_index: [1] int64 GPU tensor with the current position
            kv_cache: MLAKVCache instance (required)
            attn_mask: [1, 1, 1, max_seq_len] float mask (-inf for padding, 0 for valid)

        Returns:
            output: [B, 1, hidden_dim]
        """
        B = x.shape[0]

        if self.q_lora_rank is not None:
            q = self.q_a_proj(x)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(x)

        q = q.view(B, 1, self.num_heads, self.qk_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim:].contiguous()

        # Inline RoPE using GPU-side index_select (graph-safe, no CPU sync)
        cos_full = self.rope_cos.to(x.dtype)
        sin_full = self.rope_sin.to(x.dtype)
        cos_row = cos_full.index_select(0, pos_index).squeeze(0)
        sin_row = sin_full.index_select(0, pos_index).squeeze(0)
        q_rope = _apply_rope_inline(q_rope, cos_row, sin_row)

        kv_a = self.kv_a_proj(x)
        kv_latent = self.kv_a_layernorm(kv_a[..., :self.kv_lora_rank])
        k_rope = kv_a[..., self.kv_lora_rank:]
        k_rope = _apply_rope_inline(k_rope.unsqueeze(2), cos_row, sin_row).squeeze(2)

        # Update cache — returns FULL buffers [B, max_seq, D]
        kv_latent_full, k_rope_full = kv_cache.update_decode_cudagraph(
            kv_latent, k_rope, pos_index
        )

        # Reconstruct K/V from full latent cache
        S_kv = kv_latent_full.shape[1]
        kv_b = self.kv_b_proj(kv_latent_full)
        kv_b = kv_b.view(
            B, S_kv, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv_b[..., :self.qk_nope_head_dim]
        v = kv_b[..., self.qk_nope_head_dim:]
        k_rope_expanded = k_rope_full.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        k = torch.cat([k_nope, k_rope_expanded], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
        return self.o_proj(attn_out)
