"""Attention dispatch built on vendor and framework implementations.

Priority order:
1. FlashAttention when available
2. PyTorch SDPA
3. Explicit FlashInfer decode path when requested

The custom Triton attention kernels have been removed from the default path.
"""

import torch
from typing import Optional

# ─── Backend detection ───
_HAS_FA2 = False
_HAS_FLASHINFER = False

try:
    from flash_attn import flash_attn_func as _fa2_func
    _HAS_FA2 = True
except ImportError:
    pass

try:
    from flashinfer import single_prefill_with_kv_cache as _flashinfer_prefill
    from flashinfer import single_decode_with_kv_cache as _flashinfer_decode
    # FlashInfer is importable but uses JIT compilation that may fail on
    # unsupported GPU architectures (e.g., Blackwell/sm_120a with nvcc < 12.8).
    # Verify the GPU's compute capability is supported before declaring available.
    _fi_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    # FlashInfer 0.6.x supports up to sm_90a (Hopper). Blackwell (sm_120a) is not yet supported.
    _HAS_FLASHINFER = _fi_cap[0] <= 9
except ImportError:
    pass


def resolve_attention_backend(backend: str = "auto") -> str:
    """Resolve the requested backend to the concrete runtime implementation."""
    if backend == "auto":
        if _HAS_FA2:
            return "flash_attn"
        return "sdpa"

    if backend == "flash_attn":
        if not _HAS_FA2:
            raise ImportError("flash_attn not installed. pip install flash-attn")
        return backend

    if backend == "flashinfer":
        if not _HAS_FLASHINFER:
            raise ImportError("flashinfer not installed. pip install flashinfer")
        return backend

    if backend == "triton":
        # Preserve the legacy config value, but route it to the maintained
        # framework backend instead of carrying a custom attention kernel.
        return "sdpa"

    if backend != "sdpa":
        raise ValueError(f"Unknown attention backend: {backend}")

    return backend


def _flash_attention_fa2(Q, K, V, causal):
    """FlashAttention-2 backend. Expects [B, H, S, D], converts to [B, S, H, D]."""
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape

    # FA2 expects [B, S, H, D]
    q = Q.transpose(1, 2).contiguous()
    k = K.transpose(1, 2).contiguous()
    v = V.transpose(1, 2).contiguous()

    # FA2 handles GQA natively when num_heads_q != num_heads_k
    out = _fa2_func(q, k, v, causal=causal)  # [B, S, H, D]
    return out.transpose(1, 2)  # [B, H, S, D]


def _flash_attention_sdpa(Q, K, V, causal):
    """PyTorch SDPA backend with native GQA support (PyTorch 2.6+).

    Uses enable_gqa=True to let SDPA handle Hq != Hkv internally,
    avoiding the expensive K/V unsqueeze+expand+reshape+contiguous copies.
    """
    import torch.nn.functional as F
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape
    return F.scaled_dot_product_attention(
        Q, K, V, is_causal=causal, enable_gqa=(Hq != Hkv),
    )


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    """Flash Attention with GQA support and multiple backend dispatch.

    Args:
        Q: [B, Hq, Sq, D] queries
        K: [B, Hkv, Sk, D] keys
        V: [B, Hkv, Sk, D] values
        causal: whether to apply causal masking
        backend: "auto" | "flash_attn" | "flashinfer" | "sdpa" | legacy "triton"

    Returns:
        Output: [B, Hq, Sq, D]
    """
    backend = resolve_attention_backend(backend)

    if backend == "flash_attn":
        return _flash_attention_fa2(Q, K, V, causal)
    if backend in {"flashinfer", "sdpa"}:
        # FlashInfer remains decode-specialized; the maintained prefill path is SDPA.
        return _flash_attention_sdpa(Q, K, V, causal)
    raise ValueError(f"Unsupported resolved attention backend: {backend}")


def _flash_attention_flashinfer_decode(Q, K, V):
    """FlashInfer decode backend for Sq=1 with GQA support.

    FlashInfer's single_decode_with_kv_cache uses specialized decode kernels
    that are significantly faster than SDPA for single-query attention.

    Args:
        Q: [B, Hq, 1, D]
        K: [B, Hkv, Sk, D]
        V: [B, Hkv, Sk, D]

    Returns:
        Output: [B, Hq, 1, D]
    """
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, Dv = V.shape
    assert Sq == 1, f"FlashInfer decode expects Sq=1, got {Sq}"

    # Process each batch element separately since flashinfer.single_decode
    # expects unbatched inputs: q=[Hq, D], k=[Sk, Hkv, D], v=[Sk, Hkv, Dv]
    outputs = []
    for b in range(B):
        q_b = Q[b, :, 0, :]        # [Hq, D]
        k_b = K[b].transpose(0, 1)  # [Sk, Hkv, D]  (HND -> NHD)
        v_b = V[b].transpose(0, 1)  # [Sk, Hkv, Dv] (HND -> NHD)
        # single_decode_with_kv_cache returns [Hq, Dv] with kv_layout="NHD"
        o_b = _flashinfer_decode(
            q_b.contiguous(),
            k_b.contiguous(),
            v_b.contiguous(),
            kv_layout="NHD",
        )  # [Hq, Dv]
        outputs.append(o_b)
    # Stack: [B, Hq, Dv] -> [B, Hq, 1, Dv]
    return torch.stack(outputs, dim=0).unsqueeze(2)


def flash_attention_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Decode-only attention path for Sq=1 autoregressive inference."""
    backend = resolve_attention_backend(backend)

    if backend == "flashinfer":
        return _flash_attention_flashinfer_decode(Q, K, V)
    if backend == "flash_attn":
        return _flash_attention_fa2(Q, K, V, causal=False)
    if backend == "sdpa":
        return _flash_attention_sdpa(Q, K, V, causal=False)
    raise ValueError(f"Unsupported resolved attention backend: {backend}")


def flash_attention_naive(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Reference attention using PyTorch SDPA, for testing."""
    import torch.nn.functional as F
    # SDPA expects [B, H, S, D] and handles GQA via broadcasting
    B, Hq, Sq, D = Q.shape
    _, Hkv, Sk, _ = K.shape
    groups = Hq // Hkv
    # Expand K/V to match Q heads
    K_exp = K.unsqueeze(2).expand(B, Hkv, groups, Sk, D).reshape(B, Hq, Sk, D)
    V_exp = V.unsqueeze(2).expand(B, Hkv, groups, Sk, D).reshape(B, Hq, Sk, D)
    return F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=causal)
