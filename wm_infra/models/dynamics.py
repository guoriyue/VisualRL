"""Latent dynamics model: block-causal transformer for world state prediction.

The dynamics model operates on interleaved sequences of latent tokens and action tokens:
    [z_0_tokens, a_0, z_1_tokens, a_1, z_2_tokens, ...]

Each z_t is a set of N spatial latent tokens. Each a_t is a single action token
(projected from the action vector). The transformer uses causal attention so that
predictions at time t only depend on states and actions at times <= t.

This model uses vendor/framework attention backends for the core attention
computation, with RMSNorm and RoPE for normalization and position encoding.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm_infra.config import DynamicsConfig
from wm_infra.models.base import WorldModel, RolloutInput, RolloutOutput
from wm_infra.ops.rmsnorm import rms_norm, rms_norm_naive
from wm_infra.ops.activation import fused_swiglu


class ActionProjection(nn.Module):
    """Project action vectors into the latent token space."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B, A] -> [B, 1, D]
        return self.proj(action).unsqueeze(1)


class DynamicsBlock(nn.Module):
    """Single transformer block for dynamics model.

    Uses RMSNorm + causal self-attention + SwiGLU FFN.
    Uses framework attention dispatch plus Triton fast paths where the repo has
    workload-specific fused operators.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Pre-norm
        self.norm1_weight = nn.Parameter(torch.ones(hidden_dim))
        self.norm2_weight = nn.Parameter(torch.ones(hidden_dim))

        # Attention
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # SwiGLU FFN
        ffn_dim = int(hidden_dim * 8 / 3)
        ffn_dim = ((ffn_dim + 63) // 64) * 64  # round to multiple of 64
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Pre-norm attention (use naive impl on CPU since fused RMSNorm requires CUDA)
        _norm = rms_norm if x.is_cuda else rms_norm_naive
        normed = _norm(x, self.norm1_weight)

        qkv = self.qkv_proj(normed)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch SDPA (which dispatches to flash attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)

        # Pre-norm FFN with SwiGLU
        normed = _norm(x, self.norm2_weight)
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)

        # Use Triton fused SwiGLU when on CUDA
        if gate.is_cuda:
            ffn_out = fused_swiglu(gate.contiguous(), up.contiguous())
        else:
            ffn_out = F.silu(gate) * up

        ffn_out = self.down_proj(ffn_out)
        x = x + self.dropout(ffn_out)

        return x


class LatentDynamicsModel(nn.Module, WorldModel):
    """Block-causal transformer that predicts next latent states.

    Input sequence: [z_0, a_0, z_1, a_1, ...] where each z_t is N tokens.
    The model predicts the next state's tokens autoregressively.
    """

    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config

        # Token type embeddings (latent vs action)
        self.latent_embed = nn.Linear(config.latent_token_dim, config.hidden_dim, bias=False)
        self.action_proj = ActionProjection(config.action_dim, config.hidden_dim)

        # Learnable position encoding for spatial tokens within a frame
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 256, config.hidden_dim) * 0.02  # up to 256 spatial tokens
        )

        # Temporal position encoding (step index)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, config.max_rollout_steps, config.hidden_dim) * 0.02
        )

        # Token type embedding (0=latent, 1=action)
        self.type_embed = nn.Embedding(2, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DynamicsBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Output projection: predict next latent tokens
        self.norm_out = nn.Parameter(torch.ones(config.hidden_dim))
        self.head = nn.Linear(config.hidden_dim, config.latent_token_dim, bias=False)

    def _build_sequence(
        self,
        latent_states: list[torch.Tensor],
        actions: list[torch.Tensor],
    ) -> torch.Tensor:
        """Build interleaved sequence of latent and action tokens.

        Args:
            latent_states: list of [B, N, D_latent] tensors
            actions: list of [B, A] tensors (one fewer than latent_states, or equal)

        Returns:
            sequence: [B, total_seq_len, hidden_dim]
        """
        tokens = []
        B = latent_states[0].shape[0]
        N = latent_states[0].shape[1]

        for t, z in enumerate(latent_states):
            # Embed latent tokens
            z_emb = self.latent_embed(z)  # [B, N, D]
            # Add spatial position
            z_emb = z_emb + self.spatial_pos_embed[:, :N, :]
            # Add temporal position
            if t < self.temporal_pos_embed.shape[1]:
                z_emb = z_emb + self.temporal_pos_embed[:, t:t+1, :]
            # Add type embedding
            z_emb = z_emb + self.type_embed(torch.zeros(1, dtype=torch.long, device=z.device))
            tokens.append(z_emb)

            # Add action token if available
            if t < len(actions):
                a_emb = self.action_proj(actions[t])  # [B, 1, D]
                if t < self.temporal_pos_embed.shape[1]:
                    a_emb = a_emb + self.temporal_pos_embed[:, t:t+1, :]
                a_emb = a_emb + self.type_embed(torch.ones(1, dtype=torch.long, device=z.device))
                tokens.append(a_emb)

        return torch.cat(tokens, dim=1)  # [B, total_len, D]

    def forward(
        self,
        latent_states: list[torch.Tensor],
        actions: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through dynamics model.

        Args:
            latent_states: list of [B, N, D_latent] — observed latent states
            actions: list of [B, A] — actions taken

        Returns:
            predictions: [B, S, D_latent] — predicted token values for the full sequence
        """
        x = self._build_sequence(latent_states, actions)

        for block in self.blocks:
            x = block(x)

        _norm = rms_norm if x.is_cuda else rms_norm_naive
        x = _norm(x, self.norm_out)
        return self.head(x)

    # ─── WorldModel interface ───

    def predict_next(
        self,
        latent_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next state given current state and action."""
        N = latent_state.shape[1]
        predictions = self.forward([latent_state], [action])
        # The prediction for the next state starts after the latent + action tokens
        # Sequence is: [z_0 (N tokens), a_0 (1 token)] — next state predicted at positions N+1 onwards
        # We take the last N token predictions
        return predictions[:, -N:, :]

    @torch.inference_mode()
    def rollout(self, input: RolloutInput) -> RolloutOutput:
        """Autoregressive multi-step rollout."""
        B, N, D = input.latent_state.shape
        T = input.num_steps

        states = [input.latent_state]
        predictions = []
        current_state = input.latent_state

        for t in range(T):
            action = input.actions[:, t, :]  # [B, A]
            next_state = self.predict_next(current_state, action)
            predictions.append(next_state)
            current_state = next_state
            states.append(current_state)

        predicted_states = torch.stack(predictions, dim=1)  # [B, T, N, D]
        return RolloutOutput(predicted_states=predicted_states)

    def get_initial_state(self, observation: torch.Tensor) -> torch.Tensor:
        """Placeholder — actual implementation requires tokenizer."""
        raise NotImplementedError("Use VideoTokenizer.encode() to get initial state")
