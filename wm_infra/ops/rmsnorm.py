"""RMSNorm helpers with torch-native training path."""

import torch
import torch.nn.functional as F
import triton

from wm_infra.kernels.rmsnorm_kernel import rmsnorm_fwd_kernel


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """PyTorch RMSNorm for the general training path.

    Args:
        x: [*, N] input
        weight: [N] learnable scale
        eps: epsilon for numerical stability

    Returns:
        Normalized tensor, same shape as x
    """
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def rms_norm_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    rstd_buf: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Triton RMSNorm that writes into pre-allocated buffers.

    Used in the decode fast path to eliminate per-call tensor allocations.
    Does NOT save tensors for backward (inference only).

    Args:
        x: [*, N] input (will be reshaped to [M, N] internally)
        weight: [N] learnable scale
        out: [*, N] pre-allocated output buffer (same shape as x)
        rstd_buf: [M] pre-allocated buffer for rstd (M = product of leading dims)
        eps: epsilon for numerical stability

    Returns:
        out (same tensor passed in, now filled with normalized values)
    """
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out_2d = out.view(-1, x.shape[-1])
    M, N = x_2d.shape
    BLOCK_N = triton.next_power_of_2(N)

    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out_2d, rstd_buf,
        N, eps=eps, BLOCK_N=BLOCK_N,
    )

    return out.view(orig_shape)


def rms_norm_naive(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm for testing."""
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)
