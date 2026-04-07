"""Rotary Position Embedding helpers built on plain torch ops."""

import torch


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Returns:
        cos: [max_seq_len, head_dim // 2]
        sin: [max_seq_len, head_dim // 2]
    """
    half_d = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_d, device=device).float() / half_d))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [max_seq_len, half_d]
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: [B, S, H, D] input tensor
        cos: [max_seq_len, D//2] precomputed cos table
        sin: [max_seq_len, D//2] precomputed sin table
        positions: [S] position indices

    Returns:
        Rotated tensor, same shape as x
    """
    return apply_rope_naive(x, cos, sin, positions)


def apply_rope_naive(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Reference RoPE for testing."""
    B, S, H, D = x.shape
    half_d = D // 2

    # Gather cos/sin for these positions: [S, D//2] -> [1, S, 1, D//2]
    cos_pos = cos[positions].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]
    sin_pos = sin[positions].unsqueeze(0).unsqueeze(2)  # [1, S, 1, D//2]

    x0 = x[..., :half_d]
    x1 = x[..., half_d:]

    y0 = x0 * cos_pos - x1 * sin_pos
    y1 = x0 * sin_pos + x1 * cos_pos

    return torch.cat([y0, y1], dim=-1)
