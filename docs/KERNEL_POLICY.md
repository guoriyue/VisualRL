# Kernel Policy

`wm-infra` should prefer maintained vendor and framework implementations for
generic transformer primitives.

Current policy:

- Attention defaults to `FlashAttention` when available.
- Attention otherwise falls back to PyTorch `scaled_dot_product_attention`.
- Regular matmul should prefer PyTorch and vendor libraries instead of custom kernels.
- Generic RoPE and the general training RMSNorm path should prefer plain torch ops.

Triton kernels are still appropriate when the bottleneck is specific to this
repository's workloads and execution model, especially:

- state pack and unpack paths
- episode or session locality gather and scatter
- checkpoint delta apply
- MoE routing and non-standard layout transforms
- world-model or video-model state movement hotspots

The repository should avoid carrying custom Triton kernels for generic attention
when maintained implementations already cover that path better.
