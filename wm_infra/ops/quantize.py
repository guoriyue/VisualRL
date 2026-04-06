"""Quantization utilities: FP8 per-tensor and per-expert."""

import torch
from typing import Tuple


def quantize_per_tensor(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 with per-tensor scale.

    Args:
        x: input tensor (any shape)
        dtype: target FP8 dtype

    Returns:
        x_fp8: quantized tensor
        scale: scalar scale factor (fp32)
    """
    fp8_max = torch.finfo(dtype).max
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / fp8_max).float()
    x_scaled = x.float() / scale
    x_fp8 = x_scaled.to(dtype)
    return x_fp8, scale


def dequantize_per_tensor(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to higher precision.

    Args:
        x_fp8: FP8 tensor
        scale: per-tensor scale factor
        output_dtype: desired output dtype

    Returns:
        x: dequantized tensor
    """
    return x_fp8.to(output_dtype) * scale.to(output_dtype)


def quantize_per_expert(
    weights: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert weight matrices with per-expert scales.

    Args:
        weights: [num_experts, K, N] weight tensor

    Returns:
        weights_fp8: [num_experts, K, N] quantized weights
        scales: [num_experts] per-expert scale factors (fp32)
    """
    num_experts = weights.shape[0]
    fp8_max = torch.finfo(dtype).max

    # Compute per-expert absmax
    flat = weights.view(num_experts, -1)
    amax = flat.abs().max(dim=1).values.clamp(min=1e-12)  # [num_experts]
    scales = (amax / fp8_max).float()  # [num_experts]

    # Scale and quantize
    weights_scaled = weights.float() / scales[:, None, None]
    weights_fp8 = weights_scaled.to(dtype)

    return weights_fp8, scales
