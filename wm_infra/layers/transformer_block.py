"""Transformer Block: pre-norm attention + MoE/Dense FFN.

Pre-norm architecture (Llama/DeepSeek style):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

Returns (output, aux_loss) following MoELayer convention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from wm_infra.config import TransformerConfig
from wm_infra.layers.attention_layer import AttentionLayer, MLAAttentionLayer
from wm_infra.layers.moe_layer import MoELayer
from wm_infra.ops.rmsnorm import rms_norm, rms_norm_into
from wm_infra.ops.kv_cache import KVCache, MLAKVCache


class RMSNorm(nn.Module):
    """RMSNorm as nn.Module, using torch by default and Triton for decode buffers."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class DenseFFN(nn.Module):
    """Standard SwiGLU FFN for non-MoE layers.

    Used in models like DeepSeek-V3 where the first few layers are dense.
    Returns (output, None) to match MoELayer's (output, aux_loss) interface.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        ), None


class TransformerBlock(nn.Module):
    """Single Transformer block: RMSNorm + Attention + RMSNorm + FFN.

    The FFN can be either MoE (MoELayer) or dense (DenseFFN), controlled
    by the ``use_moe`` parameter.

    Usage:
        cfg = TransformerConfig(num_heads=8, num_kv_heads=2, head_dim=64,
                                moe=MoEConfig(num_experts=4, top_k=2,
                                              hidden_dim=512, intermediate_dim=1024))
        block = TransformerBlock(cfg).cuda().half()

        x = torch.randn(2, 128, 512, device="cuda", dtype=torch.float16)
        pos = torch.arange(128, device="cuda")

        # Training
        out, aux_loss = block(x, pos)
        (out.sum() + aux_loss).backward()

        # Inference with KV cache
        block.eval()
        cache = KVCache(2, 2, 4096, 64, dtype=torch.float16, device="cuda")
        out, _ = block(x[:, :1, :], pos[:1], kv_cache=cache)
    """

    def __init__(
        self,
        config: TransformerConfig,
        use_moe: bool = True,
        dense_intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        self.config = config

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        # Attention: dispatch based on attention_type
        if getattr(config, 'attention_type', 'mha') == 'mla':
            self.attention = MLAAttentionLayer(config)
        else:
            self.attention = AttentionLayer(config)
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        # FFN: MoE or Dense
        if use_moe:
            self.ffn = MoELayer(config.moe)
        else:
            inter_dim = dense_intermediate_dim or config.moe.intermediate_dim
            self.ffn = DenseFFN(config.hidden_dim, inter_dim)

        # Cache the MoE check for fast decode path dispatch
        self._ffn_is_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, S, hidden_dim]
            positions: [S] position indices
            kv_cache: optional KVCache for inference

        Returns:
            (output, aux_loss): output same shape as x, aux_loss scalar or None
        """
        # Attention sublayer with residual
        h = self.attn_norm(x)
        h = self.attention(h, positions, kv_cache)
        x = x + h

        # FFN sublayer with residual
        h = self.ffn_norm(x)
        h, aux_loss = self.ffn(h)
        x = x + h

        return x, aux_loss

    def forward_decode(
        self,
        x: torch.Tensor,
        position: int,
        kv_cache: Optional[object] = None,
        decode_bufs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single-token decode path that avoids per-step position allocations.

        Args:
            decode_bufs: Optional pre-allocated buffers from TransformerModel.
                Contains 'norm_out', 'norm_rstd', 'gate_up_out', 'activated', 'down_out'.
                When provided, RMSNorm and MoE matvec write into these buffers
                instead of allocating new tensors, eliminating ~7 allocations/layer.
        """
        if decode_bufs is not None:
            h = rms_norm_into(x, self.attn_norm.weight,
                              decode_bufs['norm_out'], decode_bufs['norm_rstd'],
                              self.attn_norm.eps)
        else:
            h = self.attn_norm(x)
        h = self.attention.forward_decode(h, position, kv_cache)
        x = x + h

        if decode_bufs is not None:
            h = rms_norm_into(x, self.ffn_norm.weight,
                              decode_bufs['norm_out'], decode_bufs['norm_rstd'],
                              self.ffn_norm.eps)
        else:
            h = self.ffn_norm(x)

        if self._ffn_is_moe and decode_bufs is not None:
            h, aux_loss = self.ffn.forward_decode_buffered(h, decode_bufs)
        else:
            h, aux_loss = self.ffn(h)
        x = x + h

        return x, aux_loss

    def forward_decode_cudagraph(
        self,
        x: torch.Tensor,
        pos_index: torch.Tensor,
        kv_cache: Optional[object] = None,
        decode_bufs: Optional[dict] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """CUDA graph-compatible single-token decode path.

        Uses GPU tensor position and full KV cache buffers with attention masking
        so all tensor shapes/addresses are static across graph replays.

        Args:
            x: [B, 1, hidden_dim]
            pos_index: [1] int64 GPU tensor with the current position
            kv_cache: KVCache or MLAKVCache instance
            decode_bufs: Pre-allocated buffers dict from TransformerModel
            attn_mask: [1, 1, 1, max_seq_len] float mask for attention
        """
        if decode_bufs is not None:
            h = rms_norm_into(x, self.attn_norm.weight,
                              decode_bufs['norm_out'], decode_bufs['norm_rstd'],
                              self.attn_norm.eps)
        else:
            h = self.attn_norm(x)
        h = self.attention.forward_decode_cudagraph(
            h, pos_index, kv_cache, attn_mask=attn_mask
        )
        x = x + h

        if decode_bufs is not None:
            h = rms_norm_into(x, self.ffn_norm.weight,
                              decode_bufs['norm_out'], decode_bufs['norm_rstd'],
                              self.ffn_norm.eps)
        else:
            h = self.ffn_norm(x)

        if self._ffn_is_moe and decode_bufs is not None:
            h, aux_loss = self.ffn.forward_decode_buffered(h, decode_bufs)
        else:
            h, aux_loss = self.ffn(h)
        x = x + h

        return x, aux_loss
