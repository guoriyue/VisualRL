"""Full Transformer model: embedding + N blocks + LM head + generation.

Designed for single-GPU MoE inference with optional expert offloading.

Usage:
    from moemoekit import TransformerModel

    # From pretrained HF checkpoint
    model = TransformerModel.from_pretrained(
        "deepseek-ai/DeepSeek-V3", max_experts_in_gpu=8
    )
    output_ids = model.generate(input_ids, max_new_tokens=100)

    # From config (random weights)
    from wm_infra.config import ModelConfig, MIXTRAL_8x7B_MODEL
    model = TransformerModel(MIXTRAL_8x7B_MODEL).cuda().half()
"""

import torch
import torch.nn as nn
from typing import List, Optional

from wm_infra.config import ModelConfig
from wm_infra.layers.transformer_block import TransformerBlock, RMSNorm, DenseFFN
from wm_infra.ops.kv_cache import KVCache, MLAKVCache
from wm_infra.ops.rmsnorm import rms_norm_into


class TransformerModel(nn.Module):
    """Full transformer with N blocks, embedding, LM head, and offloading.

    Supports mixed MoE/dense layers (e.g., DeepSeek-V3's first 3 layers
    are dense, the rest are MoE).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList()
        moe_set = set(config.moe_layer_indices)
        for i in range(config.num_layers):
            use_moe = i in moe_set
            self.layers.append(TransformerBlock(
                config.block,
                use_moe=use_moe,
                dense_intermediate_dim=config.intermediate_dim_dense,
            ))

        # Final norm
        self.norm = RMSNorm(config.hidden_dim, config.block.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Decode buffers (created lazily on first _forward_decode call)
        self._decode_bufs = None

        # Cache the layer list for faster iteration in decode loop
        # (avoids ModuleList.__iter__ overhead per step)
        self._cached_layers = list(self.layers)

        # CUDA graph state (initialized by enable_cuda_graphs())
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._cg_enabled: bool = False
        self._cg_input_ids: Optional[torch.Tensor] = None     # [B, 1] static buffer
        self._cg_pos_index: Optional[torch.Tensor] = None     # [1] int64 static buffer
        self._cg_attn_mask: Optional[torch.Tensor] = None     # [1, 1, 1, max_seq]
        self._cg_logits: Optional[torch.Tensor] = None        # [B, 1, V] output buffer
        self._cg_kv_caches: Optional[List] = None             # reference to KV caches
        self._cg_max_seq: int = 0
        self._cg_seq_len: int = 0                             # tracks current seq pos
        self._cg_has_offloading: bool = config.block.moe.max_experts_in_gpu is not None

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        max_experts_in_gpu: Optional[int] = None,
        weight_dtype: Optional[str] = None,
        attention_backend: Optional[str] = None,
        compile_decode: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "TransformerModel":
        """Load a full model from a HuggingFace checkpoint.

        Args:
            model_name_or_path: Local path or HF Hub model ID.
            max_experts_in_gpu: If set, enables expert offloading with this
                many experts cached on GPU per MoE layer.
            weight_dtype: MoE expert weight dtype. None keeps original.
            compile_decode: If True, apply torch.compile to decode path.
            dtype: Target compute dtype (default: float16).
            device: Target device (default: cuda).

        Returns:
            TransformerModel with pretrained weights loaded.
        """
        from wm_infra.utils.hf_compat import (
            _resolve_model_path, _load_config_json, _detect_architecture,
            _transformer_config_from_hf, load_full_model_weights,
        )
        from pathlib import Path

        model_path = _resolve_model_path(model_name_or_path)

        # GGUF file support
        if isinstance(model_path, Path) and model_path.suffix == ".gguf":
            return cls._from_gguf(
                model_path,
                max_experts_in_gpu=max_experts_in_gpu,
                attention_backend=attention_backend,
                dtype=dtype,
                device=device,
            )

        config_dict = _load_config_json(model_path)
        arch, weight_map = _detect_architecture(config_dict)

        # Build ModelConfig from HF config
        model_config = _transformer_config_from_hf(config_dict, arch)

        # Apply offloading config
        if max_experts_in_gpu is not None:
            model_config.block.moe.max_experts_in_gpu = max_experts_in_gpu

        # Apply weight quantization
        if weight_dtype is not None:
            model_config.block.moe.weight_dtype = weight_dtype
        if attention_backend is not None:
            model_config.block.attention_backend = attention_backend

        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model directly in target dtype to avoid fp32 intermediate allocation.
        # For large models like Mixtral-8x7B (46.7B params), fp32 init would need
        # ~187GB while fp16 init needs ~93GB — the difference between OOM and success.
        # set_default_dtype affects torch.empty() inside _init_offloaded_weights,
        # nn.Linear, nn.Embedding, etc., so ALL weights are created in target dtype.
        #
        # Also skip kaiming init for offloaded expert weights — they'll be overwritten
        # by pretrained weights, and skipping avoids committing ~87GB of physical pages
        # on Linux (torch.empty doesn't allocate until written to).
        from wm_infra.layers.moe_layer import MoELayer
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        MoELayer._skip_expert_init = True
        try:
            model = cls(model_config)
        finally:
            torch.set_default_dtype(old_default)
            MoELayer._skip_expert_init = False

        # Load weights (copy_() auto-converts from checkpoint dtype to model dtype)
        load_full_model_weights(model, model_path, weight_map, config_dict, arch)

        # Move non-offloaded parameters to device (offloaded _w_*_cpu stay on CPU)
        model = model.to(device=device)

        # Link MoE layers for speculative prefetching if offloading
        if max_experts_in_gpu is not None:
            from wm_infra.ops.speculative_prefetch import link_moe_layers
            from wm_infra.layers.moe_layer import MoELayer
            moe_layers = [
                block.ffn for block in model.layers
                if isinstance(block.ffn, MoELayer)
            ]
            if len(moe_layers) > 1:
                link_moe_layers(moe_layers)

        model.eval()

        # Apply torch.compile to decode path if requested
        if compile_decode:
            model.enable_compile()

        return model

    @classmethod
    def _from_gguf(
        cls,
        gguf_path,
        max_experts_in_gpu=None,
        attention_backend=None,
        dtype=None,
        device=None,
    ) -> "TransformerModel":
        """Load model from a GGUF file."""
        from wm_infra.utils.gguf_loader import load_gguf_tensors, _config_from_gguf
        from wm_infra.utils.hf_compat import (
            _detect_architecture, _transformer_config_from_hf,
        )

        metadata, tensors = load_gguf_tensors(str(gguf_path))
        config_dict = _config_from_gguf(metadata)

        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model config
        arch, weight_map = _detect_architecture(config_dict)
        model_config = _transformer_config_from_hf(config_dict, arch)

        if max_experts_in_gpu is not None:
            model_config.block.moe.max_experts_in_gpu = max_experts_in_gpu
        if attention_backend is not None:
            model_config.block.attention_backend = attention_backend

        from wm_infra.layers.moe_layer import MoELayer
        old_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        MoELayer._skip_expert_init = True
        try:
            model = cls(model_config)
        finally:
            torch.set_default_dtype(old_default)
            MoELayer._skip_expert_init = False

        # Assign tensors by matching HF-style names
        state_dict = model.state_dict()
        for name, tensor in tensors.items():
            if name in state_dict:
                state_dict[name].copy_(tensor.to(dtype))

        model = model.to(device=device)
        model.eval()
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, S] token IDs
            positions: [S] position indices
            kv_caches: optional list of KVCache (one per layer)

        Returns:
            logits: [B, S, vocab_size]
        """
        h = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, _ = layer(h, positions, kv_cache=cache)

        h = self.norm(h)
        return self.lm_head(h)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        use_cuda_graphs: Optional[bool] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache.

        Args:
            input_ids: [B, S] prompt token IDs (on the model's device)
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = no change, <1 = sharper).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.
            use_cuda_graphs: Whether to enable CUDA graph capture for decode.
                ``None`` auto-enables it for eligible single-token CUDA decode.
                Not compatible with expert offloading.

        Returns:
            generated_ids: [B, S + max_new_tokens] full sequence including prompt.
        """
        B, S = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        cfg = self.config.block

        # Create KV caches for all layers
        max_seq = S + max_new_tokens
        if getattr(cfg, 'attention_type', 'mha') == 'mla':
            kv_caches = [
                MLAKVCache(B, max_seq, cfg.kv_lora_rank, cfg.qk_rope_head_dim,
                           dtype=dtype, device=device)
                for _ in range(self.config.num_layers)
            ]
        else:
            kv_caches = [
                KVCache(B, cfg.num_kv_heads, max_seq, cfg.head_dim,
                        dtype=dtype, device=device)
                for _ in range(self.config.num_layers)
            ]

        # Prefill: process all prompt tokens at once
        positions = torch.arange(S, device=device)
        logits = self.forward(input_ids, positions, kv_caches)  # [B, S, V]

        # Sample first new token from last position
        next_token = self._sample(logits[:, -1, :], temperature, top_p, top_k)
        generated = [next_token]  # list of [B, 1]

        use_graphs = self._resolve_cuda_graph_usage(
            use_cuda_graphs=use_cuda_graphs,
            batch_size=B,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        enabled_graphs_here = False

        # Enable CUDA graphs after prefill (all lazy state is initialized)
        if use_graphs:
            # Run one decode step WITHOUT graphs to ensure lazy init completes
            # (fused QKV weights, fused gate+up weights, etc.)
            logits = self._forward_decode(next_token, S, kv_caches)
            next_token = self._sample(logits[:, 0, :], temperature, top_p, top_k)
            generated.append(next_token)

            # Now enable CUDA graphs for remaining steps
            self.enable_cuda_graphs(
                batch_size=B,
                max_seq_len=max_seq,
                kv_caches=kv_caches,
                prefill_len=S + 1,  # prefill + 1 warmup decode step
                device=device,
                dtype=dtype,
            )
            enabled_graphs_here = True
            start_step = 2  # already did step 0 (prefill) and step 1 (warmup)
        else:
            start_step = 1

        # Decode loop: one token at a time
        for step in range(start_step, max_new_tokens):
            logits = self._forward_decode(next_token, S + step - 1, kv_caches)
            next_token = self._sample(logits[:, 0, :], temperature, top_p, top_k)
            generated.append(next_token)

        # Clean up CUDA graph state
        if enabled_graphs_here:
            self.disable_cuda_graphs()

        # Concatenate prompt + generated
        return torch.cat([input_ids] + generated, dim=1)

    def _resolve_cuda_graph_usage(
        self,
        use_cuda_graphs: Optional[bool],
        batch_size: int,
        max_new_tokens: int,
        device: torch.device,
    ) -> bool:
        """Decide whether decode should use CUDA graphs."""
        if use_cuda_graphs is not None:
            return bool(use_cuda_graphs) and not self._cg_has_offloading

        return (
            device.type == "cuda"
            and batch_size == 1
            and max_new_tokens >= 4
            and not self._cg_has_offloading
        )

    def _get_decode_bufs(self, device, dtype):
        """Lazily create model-level pre-allocated buffers for S=1 decode.

        These buffers are reused across ALL layers and decode steps, eliminating
        ~170 tensor allocations per decode token. Buffers include:
          - RMSNorm output and rstd buffers (shared across all norm calls)
          - MoE matvec gate/up/down output buffers (shared across all MoE layers)
        """
        if self._decode_bufs is not None:
            return self._decode_bufs

        cfg = self.config
        H = cfg.hidden_dim
        top_k = cfg.block.moe.top_k
        I = cfg.block.moe.intermediate_dim

        self._decode_bufs = {
            # RMSNorm buffers (M=1 for [1, H] decode input)
            'norm_out': torch.empty(1, H, device=device, dtype=dtype),
            'norm_rstd': torch.empty(1, device=device, dtype=torch.float32),
            # MoE matvec buffers — fused gate_up+swiglu output and weighted down output
            'gate_up_out': torch.empty(top_k, 2 * I, device=device, dtype=dtype),
            'activated': torch.empty(top_k, I, device=device, dtype=dtype),
            'down_out': torch.empty(top_k, H, device=device, dtype=dtype),
            'weighted_out': torch.zeros(1, H, device=device, dtype=dtype),
        }
        return self._decode_bufs

    def _forward_decode(
        self,
        input_ids: torch.Tensor,
        position: int,
        kv_caches: List,
    ) -> torch.Tensor:
        """Optimized forward for S=1 decode step.

        Uses per-layer decode fast paths to avoid per-step positions tensor
        allocations and to route KV cache writes through update_decode().
        Pre-allocates and reuses buffers across layers to minimize allocations.

        Automatically dispatches to the CUDA graph path when enabled.
        """
        # CUDA graph fast path: replay captured graph instead of re-launching kernels
        if self._cg_enabled and self._cuda_graph is not None:
            return self._forward_decode_cudagraph(input_ids, position)

        h = self.embed_tokens(input_ids)
        bufs = self._get_decode_bufs(h.device, h.dtype)

        # Use cached list for faster iteration (avoids ModuleList overhead)
        layers = self._cached_layers
        num_layers = len(layers)
        for i in range(num_layers):
            h, _ = layers[i].forward_decode(h, position,
                                            kv_cache=kv_caches[i],
                                            decode_bufs=bufs)

        h = rms_norm_into(h, self.norm.weight,
                          bufs['norm_out'], bufs['norm_rstd'],
                          self.norm.eps)
        return self.lm_head(h)

    def get_last_decode_moe_state(self) -> dict[int, dict[str, object]]:
        """Return per-layer MoE state from the most recent decode step.

        Each entry is keyed by the transformer layer index and may contain:
          - ``expert_ids``: the actual routed experts for the last decode token
          - ``cached_experts``: experts currently resident on GPU for that layer
        Dense FFN layers are omitted.
        """
        from wm_infra.layers.moe_layer import MoELayer

        state: dict[int, dict[str, object]] = {}
        num_experts = self.config.block.moe.num_experts
        all_experts = list(range(num_experts))

        for layer_idx, block in enumerate(self._cached_layers):
            ffn = block.ffn
            if not isinstance(ffn, MoELayer):
                continue

            layer_state: dict[str, object] = {}
            if ffn.last_routed_expert_ids is not None:
                layer_state["expert_ids"] = ffn.last_routed_expert_ids

            cached_experts = ffn.cached_expert_ids
            if cached_experts is None:
                layer_state["cached_experts"] = all_experts
            else:
                layer_state["cached_experts"] = list(cached_experts)

            state[layer_idx] = layer_state

        return state

    # ────────────────────────────────────────────────────────────────────
    #  CUDA Graph-accelerated decode
    # ────────────────────────────────────────────────────────────────────

    def enable_cuda_graphs(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_caches: List,
        prefill_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Enable CUDA graph capture for the decode path.

        Must be called AFTER prefill so the KV caches are populated and all
        lazy-init state (fused QKV weights, fused gate+up weights, etc.) is
        ready. Expert offloading is NOT compatible with CUDA graphs because
        cache.ensure_loaded() triggers host-side control flow.

        Args:
            batch_size: Batch size (must be 1 for decode).
            max_seq_len: Maximum sequence length (prompt + max_new_tokens).
            kv_caches: List of KVCache/MLAKVCache objects (one per layer).
            prefill_len: Number of prompt tokens already in the KV cache.
            device: CUDA device (defaults to model device).
            dtype: Compute dtype (defaults to model dtype).
        """
        if self._cg_has_offloading:
            raise RuntimeError(
                "CUDA graphs are not compatible with expert offloading. "
                "Offloading uses dynamic host-side control flow (ensure_loaded) "
                "that cannot be captured in a graph."
            )

        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        B = batch_size
        V = self.config.vocab_size
        H = self.config.hidden_dim

        # Static input/output buffers
        self._cg_input_ids = torch.zeros(B, 1, device=device, dtype=torch.long)
        # Initialize pos_index to prefill_len so warmup iterations during graph
        # capture write to the next decode position (not position 0 which holds
        # the first prompt token's KV and must not be corrupted).
        self._cg_pos_index = torch.tensor([prefill_len], device=device, dtype=torch.long)
        self._cg_logits = torch.zeros(B, 1, V, device=device, dtype=dtype)

        # Attention mask: [1, 1, 1, max_seq_len]
        # -inf for padding (positions beyond current seq_len), 0.0 for valid
        self._cg_attn_mask = torch.full(
            (1, 1, 1, max_seq_len), float("-inf"), device=device, dtype=dtype,
        )
        # Unmask positions already filled by prefill
        self._cg_attn_mask[:, :, :, :prefill_len] = 0.0

        self._cg_kv_caches = kv_caches
        self._cg_max_seq = max_seq_len
        self._cg_seq_len = prefill_len
        self._cg_enabled = True

        # Ensure all lazy state is initialized before capture
        self._get_decode_bufs(device, dtype)
        for layer in self._cached_layers:
            attn = layer.attention
            if hasattr(attn, '_qkv_proj') and attn._qkv_proj is None:
                attn._init_fused_qkv()
            ffn = layer.ffn
            if hasattr(ffn, '_ensure_fused_gate_up') and hasattr(ffn, '_fused_gate_up'):
                ffn._ensure_fused_gate_up()

        # Capture the graph
        self._capture_decode_graph()

    def _capture_decode_graph(self) -> None:
        """Capture the decode step as a CUDA graph.

        Runs the full decode step once on the current stream (with a warmup),
        then captures it. The captured graph is replayed on subsequent calls.

        Uses a private CUDA memory pool for graph capture so that temporary
        allocations during the captured run come from the graph's own pool
        rather than the default allocator (which would break capture).
        """
        device = self._cg_input_ids.device

        # Warmup runs: trigger all JIT compilation, autotuning, and lazy
        # initialization before we start capturing. We run several iterations
        # to ensure Triton kernels are fully compiled and cached.
        for _ in range(3):
            self._run_decode_cudagraph_body()

        torch.cuda.synchronize(device)

        # Allocate a private memory pool for graph capture. This avoids the
        # "operation not permitted when stream is capturing" error that occurs
        # when operations try to allocate from the default CUDA memory pool.
        self._cg_pool = torch.cuda.graph_pool_handle()

        # Capture the graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, pool=self._cg_pool):
            self._cg_logits = self._run_decode_cudagraph_body()

        torch.cuda.synchronize(device)

    def _run_decode_cudagraph_body(self) -> torch.Tensor:
        """Execute the decode body that will be captured by CUDA graph.

        Uses cudagraph-compatible paths throughout:
        - GPU tensor position for RoPE indexing
        - Full KV cache buffers with attention masking
        - Pre-allocated decode buffers for MoE

        Returns:
            logits: [B, 1, V]
        """
        h = self.embed_tokens(self._cg_input_ids)
        bufs = self._decode_bufs

        layers = self._cached_layers
        pos_index = self._cg_pos_index
        attn_mask = self._cg_attn_mask
        num_layers = len(layers)

        for i in range(num_layers):
            h, _ = layers[i].forward_decode_cudagraph(
                h, pos_index,
                kv_cache=self._cg_kv_caches[i],
                decode_bufs=bufs,
                attn_mask=attn_mask,
            )

        h = rms_norm_into(h, self.norm.weight,
                          bufs['norm_out'], bufs['norm_rstd'],
                          self.norm.eps)
        return self.lm_head(h)

    def _forward_decode_cudagraph(
        self,
        input_ids: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Replay the captured CUDA graph for one decode step.

        Updates the static input buffers, unmarks the next attention mask
        position, then replays the graph.

        Args:
            input_ids: [B, 1] token IDs for this step
            position: int, the sequence position for this token

        Returns:
            logits: [B, 1, V] (clone of the static output buffer)
        """
        # Update static input buffers (these are copy_ into the same addresses
        # the graph was captured with, so the graph sees the new values)
        self._cg_input_ids.copy_(input_ids)
        self._cg_pos_index.fill_(position)

        # Unmask this position in the attention mask so SDPA can attend to it.
        # This happens BEFORE graph replay — it modifies the mask tensor that
        # the graph will read.
        self._cg_attn_mask[0, 0, 0, position] = 0.0

        # Also update KV cache seq_len counters (Python-side bookkeeping
        # needed by non-graph paths; the graph itself uses pos_index)
        for cache in self._cg_kv_caches:
            cache.seq_len = position + 1

        # Replay the captured graph
        self._cuda_graph.replay()

        return self._cg_logits

    def disable_cuda_graphs(self) -> None:
        """Disable CUDA graphs and free the captured graph."""
        self._cg_enabled = False
        if self._cuda_graph is not None:
            del self._cuda_graph
            self._cuda_graph = None
        self._cg_input_ids = None
        self._cg_pos_index = None
        self._cg_attn_mask = None
        self._cg_logits = None
        self._cg_kv_caches = None

    def get_attention_backends(self) -> List[str]:
        """Return the resolved attention backend used by each block."""
        return [
            getattr(layer.attention, "resolved_attention_backend", "sdpa")
            if hasattr(layer.attention, "resolved_attention_backend")
            else "sdpa"
            for layer in self.layers
        ]

    def configure_speculative_prefetch(self, speculative_top_k: Optional[int]) -> None:
        """Reconfigure speculative prefetch across MoE layers.

        Set ``speculative_top_k=0`` to disable prefetch entirely.
        """
        from wm_infra.layers.moe_layer import MoELayer
        from wm_infra.ops.speculative_prefetch import link_moe_layers

        moe_layers = [
            block.ffn for block in self.layers
            if isinstance(block.ffn, MoELayer)
        ]
        for layer in moe_layers:
            layer.config.speculative_top_k = speculative_top_k
            layer._speculator = None

        if self.config.block.moe.max_experts_in_gpu is not None and len(moe_layers) > 1:
            link_moe_layers(moe_layers)

    def enable_compile(self, mode: str = "reduce-overhead") -> None:
        """Apply torch.compile to the decode path for reduced overhead.

        Compiles each TransformerBlock's forward_decode method individually.
        Block-level compilation avoids dynamic shape issues from the model-level
        loop while still capturing key fusions (norm + attention, norm + MoE,
        residual adds, SwiGLU activation).

        For the MoE decode path (bmm-based), torch.compile can fuse:
          - RMSNorm elementwise ops
          - SwiGLU activation (silu * up)
          - Residual additions
          - Routing softmax + topk

        Must be called AFTER model is on the target device and in eval mode.

        Args:
            mode: torch.compile mode. "reduce-overhead" uses CUDA graphs
                  internally for minimal launch overhead. "max-autotune"
                  additionally tunes kernel configs. Default: "reduce-overhead".
        """
        import logging
        logger = logging.getLogger(__name__)

        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available (requires PyTorch >= 2.0)")
            return

        compile_kwargs = dict(
            mode=mode,
            fullgraph=False,  # allow graph breaks for Triton/custom ops
            dynamic=False,    # shapes are static at S=1 decode
        )

        num_compiled = 0
        for i, layer in enumerate(self._cached_layers):
            try:
                layer.forward_decode = torch.compile(
                    layer.forward_decode, **compile_kwargs
                )
                num_compiled += 1
            except Exception as e:
                logger.warning(f"Failed to compile layer {i}: {e}")

        logger.info(
            f"torch.compile applied to {num_compiled}/{len(self._cached_layers)} "
            f"blocks (mode={mode})"
        )

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample a single token from logits with temperature + top-k + top-p.

        Args:
            logits: [B, V] raw logits for one position.

        Returns:
            token_ids: [B, 1] sampled token IDs.
        """
        if temperature <= 0:
            # Greedy
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < logits.shape[-1]:
            top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = top_k_vals[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # Remove tokens with cumulative probability above threshold
            mask = cumulative_probs - probs > top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            # Scatter back to original indices
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
