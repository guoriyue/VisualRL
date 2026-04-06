"""MoE Layer: drop-in replacement for a transformer FFN block."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from wm_infra.config import MoEConfig
from wm_infra.ops.fused_moe import fused_moe
from wm_infra.ops.matvec import (
    indexed_dual_matvec, indexed_matvec_varying,
    indexed_dual_matvec_swiglu, indexed_matvec_varying_weighted,
)


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Drop-in replacement for a standard FFN block in a transformer.

    Input:  [batch, seq_len, hidden_dim] or [num_tokens, hidden_dim]
    Output: [same shape as input]

    Usage:
        config = MoEConfig(num_experts=8, top_k=2, hidden_dim=4096, intermediate_dim=14336)
        moe = MoELayer(config)
        output, aux_loss = moe(hidden_states)

    FP8 usage:
        config = MoEConfig(..., weight_dtype="float8_e4m3")
        moe = MoELayer(config)  # weights stored as FP8 with per-expert scales

    Expert offloading usage:
        config = MoEConfig(num_experts=256, ..., max_experts_in_gpu=32)
        moe = MoELayer(config)
        # Only 32 experts on GPU at a time; LRU cache handles paging
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self._use_fp8 = config.weight_dtype == "float8_e4m3"
        self._use_offloading = config.max_experts_in_gpu is not None
        self._expert_cache = None

        # Router (gate)
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

        if self._use_offloading:
            self._init_offloaded_weights(config)
        elif self._use_fp8:
            self._init_fp8_weights(config)
        else:
            self.w_gate = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.intermediate_dim))
            self.w_up = nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.intermediate_dim))
            self.w_down = nn.Parameter(
                torch.empty(config.num_experts, config.intermediate_dim, config.hidden_dim))

        # Shared expert (DeepSeek-V3 style)
        self.shared_expert = None
        if config.has_shared_experts:
            self.shared_expert = SharedExpert(config)

        # Bias-based load balancing (DeepSeek-V3 style)
        # Registered as a buffer (not a parameter) — updated by statistics, not gradient
        if config.use_expert_bias:
            self.register_buffer("expert_bias",
                                 torch.zeros(config.num_experts))
        else:
            self.expert_bias = None

        # CUDA stream for overlapping shared expert with routed experts (created lazily)
        self._shared_stream: Optional[torch.cuda.Stream] = None

        # Speculative prefetcher for next layer (set by link_moe_layers())
        self._speculator = None

        # Pre-allocated decode buffers (created lazily on first decode call)
        self._decode_buffers: Optional[dict] = None
        self._last_routed_expert_ids: Optional[torch.Tensor] = None

        self._init_weights()

    def _should_use_direct_decode_fast_path(self, hidden_2d: torch.Tensor) -> bool:
        """Use a direct top-k matvec path for true single-token fp16 decode."""
        return (
            not self.training
            and hidden_2d.shape[0] == 1
            and not self._use_fp8
        )

    def forward_decode_buffered(
        self,
        hidden_states: torch.Tensor,
        decode_bufs: dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode fast path using pre-allocated buffers to eliminate allocations.

        Dispatches to the appropriate internal path based on weight type and
        offloading config. Only called for S=1 (single-token decode) in eval mode.

        Args:
            hidden_states: [B, 1, hidden_dim] or [1, hidden_dim]
            decode_bufs: Pre-allocated buffers dict from TransformerModel.
        """
        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]
        hidden_2d = hidden_states.view(-1, hidden_dim)

        if hidden_2d.shape[0] != 1 or self.training:
            return self.forward(hidden_states)

        if not self._use_fp8:
            if self._use_offloading:
                output, aux_loss = self._forward_decode_direct_offloaded_buffered(
                    hidden_2d, decode_bufs)
            else:
                output, aux_loss = self._forward_decode_direct_fp16_buffered(
                    hidden_2d, decode_bufs)
        else:
            output, aux_loss = self._forward_decode_direct_fp16(hidden_2d)

        # Shared expert (use optimized decode path)
        if self.shared_expert is not None:
            shared_out = self.shared_expert.forward_decode(hidden_2d)
            output = output + shared_out

        # Expert bias update
        if self.expert_bias is not None:
            self._update_expert_bias(hidden_2d)

        output = output.view(original_shape)
        return output, aux_loss

    def _forward_decode_direct_fp16_buffered(
        self,
        hidden_2d: torch.Tensor,
        decode_bufs: dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Direct single-token decode using fused Triton matvec kernels.

        Uses two fused kernels for the entire MoE computation:
        1. indexed_dual_matvec_swiglu: gate+up GEMV + SwiGLU activation (1 kernel)
        2. indexed_matvec_varying_weighted: down GEMV + weight + sum (1 kernel)
        """
        topk_weights, topk_ids = self._route_single_token(hidden_2d)
        weight_ids = topk_ids[0]
        hidden = hidden_2d[0]
        I = self.config.intermediate_dim

        # Fused gate+up GEMV + SwiGLU: outputs [top_k, I] directly
        activated = decode_bufs['activated']  # [top_k, I]
        indexed_dual_matvec_swiglu(
            hidden,
            self.w_gate.data,
            self.w_up.data,
            weight_ids,
            activated,
        )

        # Fused down GEMV + routing weight + sum: outputs [1, hidden_dim]
        output = decode_bufs.get('weighted_out')
        if output is None:
            output = torch.zeros(1, self.config.hidden_dim,
                                 device=hidden_2d.device, dtype=hidden_2d.dtype)
        else:
            output.zero_()
        indexed_matvec_varying_weighted(
            activated,
            self.w_down.data,
            weight_ids,
            topk_weights[0],
            output,
        )

        return output, None

    def _forward_decode_direct_offloaded_buffered(
        self,
        hidden_2d: torch.Tensor,
        decode_bufs: dict,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Direct single-token decode for offloaded experts via fused indexed matvec."""
        topk_weights, topk_ids = self._route_single_token(hidden_2d)
        cache = self._get_expert_cache(hidden_2d.device)
        needed_expert_ids = topk_ids[0]

        if self._speculator is not None:
            self._speculator.predict_and_prefetch(hidden_2d)

        cache.ensure_loaded(needed_expert_ids)
        slot_ids = cache.remap_expert_ids(topk_ids)[0]

        hidden = hidden_2d[0]
        I = self.config.intermediate_dim

        # Fused gate+up GEMV + SwiGLU
        activated = decode_bufs['activated']  # [top_k, I]
        indexed_dual_matvec_swiglu(
            hidden,
            cache.w_gate_gpu,
            cache.w_up_gpu,
            slot_ids,
            activated,
        )

        # Fused down GEMV + routing weight + sum
        output = decode_bufs.get('weighted_out')
        if output is None:
            output = torch.zeros(1, self.config.hidden_dim,
                                 device=hidden_2d.device, dtype=hidden_2d.dtype)
        else:
            output.zero_()
        indexed_matvec_varying_weighted(
            activated,
            cache.w_down_gpu,
            slot_ids,
            topk_weights[0],
            output,
        )

        return output, None

    def _route_single_token(self, hidden_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token routing without the generic Triton top-k helper."""
        gate_weight = self.gate.weight
        if gate_weight.dtype != hidden_2d.dtype:
            gate_weight = gate_weight.to(hidden_2d.dtype)

        logits = hidden_2d @ gate_weight.T
        select_logits = logits
        if self.expert_bias is not None:
            select_logits = logits + self.expert_bias.unsqueeze(0).to(logits.dtype)

        topk_vals, topk_ids = torch.topk(select_logits, self.config.top_k, dim=-1)
        if self.expert_bias is None:
            topk_weights = torch.softmax(topk_vals, dim=-1)
        else:
            topk_weights = torch.softmax(logits.gather(1, topk_ids), dim=-1)

        # Expose the actual routing decision so the serving scheduler can
        # feed real decode-time expert usage back into MoE-aware scheduling.
        self._last_routed_expert_ids = topk_ids[0].detach().clone()
        return topk_weights, topk_ids

    @property
    def last_routed_expert_ids(self) -> Optional[torch.Tensor]:
        """Most recent single-token routing decision for this layer."""
        return self._last_routed_expert_ids

    @property
    def cached_expert_ids(self) -> Optional[list[int]]:
        """Experts currently resident on GPU for offloaded weights.

        Returns ``None`` when expert offloading is disabled, meaning all
        experts are effectively resident.
        """
        if self._expert_cache is None:
            return None
        return self._expert_cache.cached_experts

    def _direct_moe_matvec(
        self,
        hidden_2d: torch.Tensor,
        topk_weights: torch.Tensor,
        weight_ids: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        """Run the selected experts via fused indexed decode matvec path."""
        hidden = hidden_2d[0]  # [K]
        top_k = weight_ids.shape[0]
        I = w_gate.shape[-1]  # intermediate_dim

        # Fused gate+up GEMV + SwiGLU
        activated = torch.empty(
            top_k, I, device=hidden.device, dtype=hidden.dtype)
        indexed_dual_matvec_swiglu(
            hidden, w_gate, w_up, weight_ids, activated,
        )

        # Fused down GEMV + routing weight + sum
        output = torch.zeros(
            1, w_down.shape[-1], device=hidden.device, dtype=hidden.dtype)
        indexed_matvec_varying_weighted(
            activated, w_down, weight_ids, topk_weights[0], output,
        )

        return output

    def _forward_decode_direct_fp16(
        self,
        hidden_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Direct single-token decode for resident fp16/bf16 expert weights."""
        topk_weights, topk_ids = self._route_single_token(hidden_2d)
        output = self._direct_moe_matvec(
            hidden_2d,
            topk_weights,
            topk_ids[0],
            self.w_gate,
            self.w_up,
            self.w_down,
        )
        return output, None

    def _forward_decode_direct_offloaded(
        self,
        hidden_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Direct single-token decode for fp16/bf16 offloaded experts."""
        topk_weights, topk_ids = self._route_single_token(hidden_2d)
        cache = self._get_expert_cache(hidden_2d.device)
        needed_expert_ids = topk_ids[0]

        if self._speculator is not None:
            self._speculator.predict_and_prefetch(hidden_2d)

        cache.ensure_loaded(needed_expert_ids)
        slot_ids = cache.remap_expert_ids(topk_ids)[0]
        output = self._direct_moe_matvec(
            hidden_2d,
            topk_weights,
            slot_ids,
            cache.w_gate_gpu,
            cache.w_up_gpu,
            cache.w_down_gpu,
        )
        return output, None

    def _init_fp8_weights(self, config):
        """Initialize FP8 weight storage with per-expert scale buffers."""
        from wm_infra.ops.quantize import quantize_per_expert

        # Create temporary FP32 weights, quantize, and store
        w_gate_fp32 = torch.empty(
            config.num_experts, config.hidden_dim, config.intermediate_dim)
        w_up_fp32 = torch.empty(
            config.num_experts, config.hidden_dim, config.intermediate_dim)
        w_down_fp32 = torch.empty(
            config.num_experts, config.intermediate_dim, config.hidden_dim)

        nn.init.kaiming_uniform_(w_gate_fp32, a=5**0.5)
        nn.init.kaiming_uniform_(w_up_fp32, a=5**0.5)
        nn.init.kaiming_uniform_(w_down_fp32, a=5**0.5)

        w_gate_fp8, gate_scales = quantize_per_expert(w_gate_fp32)
        w_up_fp8, up_scales = quantize_per_expert(w_up_fp32)
        w_down_fp8, down_scales = quantize_per_expert(w_down_fp32)

        # FP8 weights are Parameters (for .to() device movement, but no gradient through them)
        self.w_gate = nn.Parameter(w_gate_fp8, requires_grad=False)
        self.w_up = nn.Parameter(w_up_fp8, requires_grad=False)
        self.w_down = nn.Parameter(w_down_fp8, requires_grad=False)

        # Scale factors as buffers
        self.register_buffer("w_gate_scale", gate_scales)
        self.register_buffer("w_up_scale", up_scales)
        self.register_buffer("w_down_scale", down_scales)

    # Class-level flag: set True by from_pretrained() to skip kaiming init
    # for offloaded weights (saves ~87GB physical memory for Mixtral-scale models)
    _skip_expert_init: bool = False

    def _init_offloaded_weights(self, config: MoEConfig):
        """Initialize expert weights on CPU pinned memory with an LRU GPU cache.

        Weights are created on CPU, initialized with Kaiming uniform, pinned,
        and handed to ExpertCache. The cache pre-allocates GPU buffers for
        max_experts_in_gpu experts and handles on-demand paging.
        """
        from wm_infra.ops.expert_cache import ExpertCache

        w_gate_cpu = torch.empty(
            config.num_experts, config.hidden_dim, config.intermediate_dim)
        w_up_cpu = torch.empty(
            config.num_experts, config.hidden_dim, config.intermediate_dim)
        w_down_cpu = torch.empty(
            config.num_experts, config.intermediate_dim, config.hidden_dim)

        if not MoELayer._skip_expert_init:
            nn.init.kaiming_uniform_(w_gate_cpu, a=5**0.5)
            nn.init.kaiming_uniform_(w_up_cpu, a=5**0.5)
            nn.init.kaiming_uniform_(w_down_cpu, a=5**0.5)

        # Store as plain attributes (NOT buffers/parameters) so .to(device)
        # doesn't move them — they must stay on CPU for offloading to work.
        self._w_gate_cpu = w_gate_cpu
        self._w_up_cpu = w_up_cpu
        self._w_down_cpu = w_down_cpu

        # ExpertCache is created lazily in forward (needs to know the device)
        self._expert_cache = None

    def _get_expert_cache(self, device):
        """Lazily create the ExpertCache on first forward pass."""
        if self._expert_cache is None:
            from wm_infra.ops.expert_cache import ExpertCache
            self._expert_cache = ExpertCache(
                w_gate_all=self._w_gate_cpu,
                w_up_all=self._w_up_cpu,
                w_down_all=self._w_down_cpu,
                max_experts_in_gpu=self.config.max_experts_in_gpu,
                device=device,
            )
        return self._expert_cache

    def _get_decode_buffers(self, device, dtype):
        """Lazily create pre-allocated buffers for decode (S=1) forward passes.

        These buffers are reused across decode steps to eliminate ~11 tensor
        allocations per MoE layer per step (saves ~200+ kernel launches total
        across all layers in the model).
        """
        if self._decode_buffers is not None:
            return self._decode_buffers

        E = self.config.num_experts
        K = self.config.top_k
        D = self.config.hidden_dim
        I = self.config.intermediate_dim

        self._decode_buffers = {
            # Tile mapping (static: arange + offsets)
            'tile_expert_ids': torch.arange(E, device=device, dtype=torch.int32),
            'tile_m_offsets': torch.empty(E, device=device, dtype=torch.int32),
            'tile_m_ends': torch.empty(E, device=device, dtype=torch.int32),
            # Output + intermediate buffers
            'output': torch.empty(1, D, device=device, dtype=dtype),
            'intermediate': torch.empty(K, I, device=device, dtype=dtype),
        }
        return self._decode_buffers

    def _init_weights(self):
        """Kaiming uniform initialization, same as standard linear layers."""
        if not self._use_fp8 and not self._use_offloading:
            for w in [self.w_gate, self.w_up, self.w_down]:
                nn.init.kaiming_uniform_(w, a=5**0.5)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MoELayer":
        """Load a single MoE layer from a HuggingFace checkpoint.

        Args:
            model_name_or_path: Local path or HF Hub model ID
                (e.g. "mistralai/Mixtral-8x7B-v0.1").
            layer_idx: Which transformer layer's MoE block to load.
            device: Target device (default: cpu).
            dtype: Target dtype (default: keep original).

        Returns:
            MoELayer with pretrained weights loaded.
        """
        from wm_infra.utils.hf_compat import _from_pretrained_impl
        return _from_pretrained_impl(model_name_or_path, layer_idx, device, dtype)

    def _load_pretrained_weights(self, weights: dict) -> None:
        """Load converted weights from a HF checkpoint into this layer.

        Args:
            weights: Dict with keys "w_gate", "w_up", "w_down", "router",
                     and optionally "shared_gate_proj", "shared_up_proj",
                     "shared_down_proj", "expert_bias".
        """
        # Router weight
        if "router" in weights:
            self.gate.weight.data.copy_(weights["router"])

        # Expert weights
        if self._use_fp8:
            from wm_infra.ops.quantize import quantize_per_expert
            for name in ("w_gate", "w_up", "w_down"):
                if name in weights:
                    w_fp8, scales = quantize_per_expert(weights[name])
                    getattr(self, name).data.copy_(w_fp8)
                    getattr(self, f"{name}_scale").copy_(scales)
        elif self._use_offloading:
            for name, buf_name in [
                ("w_gate", "_w_gate_cpu"),
                ("w_up", "_w_up_cpu"),
                ("w_down", "_w_down_cpu"),
            ]:
                if name in weights:
                    getattr(self, buf_name).copy_(weights[name])
            self._expert_cache = None
        else:
            for name in ("w_gate", "w_up", "w_down"):
                if name in weights:
                    getattr(self, name).data.copy_(weights[name])

        # Shared expert weights
        if self.shared_expert is not None:
            for proj_name, key in [
                ("gate_proj", "shared_gate_proj"),
                ("up_proj", "shared_up_proj"),
                ("down_proj", "shared_down_proj"),
            ]:
                if key in weights:
                    getattr(self.shared_expert, proj_name).weight.data.copy_(
                        weights[key]
                    )

        # Expert bias
        if "expert_bias" in weights and self.expert_bias is not None:
            self.expert_bias.copy_(weights["expert_bias"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [*, hidden_dim] -- any number of leading dims
            mode: override execution mode ("naive", "composable", "fused")
                  Default: "composable" during training, "fused" during inference

        Returns:
            output: same shape as hidden_states
            aux_loss: scalar auxiliary loss for training, or None
        """
        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]

        # Flatten to [num_tokens, hidden_dim]
        hidden_2d = hidden_states.view(-1, hidden_dim)

        # Choose mode
        if mode is None:
            mode = "composable" if self.training else "fused"

        # Determine aux_loss_weight
        aux_loss_weight = self.config.aux_loss_weight if self.training else 0.0

        # Decide whether to overlap shared expert on a separate CUDA stream.
        # Only overlap during inference (eval mode) to avoid autograd complications.
        use_overlap = (
            self.shared_expert is not None
            and self.config.use_stream_overlap
            and not self.training
            and hidden_states.is_cuda
        )

        # Launch shared expert on a separate stream (before routed experts)
        shared_out = None
        if use_overlap:
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream(device=hidden_states.device)
            # Record the current stream so the shared stream waits for hidden_2d to be ready
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_expert(hidden_2d)

        # Decode mode: skip GPU→CPU syncs when processing single tokens
        num_tokens = hidden_2d.shape[0]
        decode_mode = not self.training and num_tokens <= self.config.top_k

        if self._should_use_direct_decode_fast_path(hidden_2d):
            if self._use_offloading:
                output, aux_loss = self._forward_decode_direct_offloaded(hidden_2d)
            else:
                output, aux_loss = self._forward_decode_direct_fp16(hidden_2d)
        elif self._use_offloading:
            output, aux_loss = self._forward_offloaded(hidden_2d, aux_loss_weight,
                                                       mode, decode_mode)
        elif self._use_fp8 and mode == "composable":
            output, aux_loss = self._forward_fp8(hidden_2d, aux_loss_weight,
                                                  decode_mode)
        else:
            # For FP8 + naive/fused, dequantize weights first
            w_gate = self.w_gate
            w_up = self.w_up
            w_down = self.w_down
            if self._use_fp8:
                compute_dtype = hidden_states.dtype
                w_gate = (self.w_gate.to(compute_dtype) *
                          self.w_gate_scale[:, None, None].to(compute_dtype))
                w_up = (self.w_up.to(compute_dtype) *
                        self.w_up_scale[:, None, None].to(compute_dtype))
                w_down = (self.w_down.to(compute_dtype) *
                          self.w_down_scale[:, None, None].to(compute_dtype))

            # Pass pre-allocated decode buffers for fused mode to reduce allocations
            # Only for true single-token decode (num_tokens==1) where buffer sizes are fixed
            db = None
            if decode_mode and mode == "fused" and num_tokens == 1:
                db = self._get_decode_buffers(hidden_2d.device, hidden_2d.dtype)

            output, aux_loss = fused_moe(
                hidden_states=hidden_2d,
                gate_weight=self.gate.weight,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                top_k=self.config.top_k,
                renormalize=self.config.renormalize,
                aux_loss_weight=aux_loss_weight,
                mode=mode,
                expert_bias=self.expert_bias,
                decode_mode=decode_mode,
                decode_buffers=db,
            )

        # Update expert bias based on load statistics (DeepSeek-V3 style)
        if self.expert_bias is not None:
            self._update_expert_bias(hidden_2d)

        # Add shared expert output
        if use_overlap:
            # Wait for the shared stream to finish before combining
            torch.cuda.current_stream().wait_stream(self._shared_stream)
            output = output + shared_out
        elif self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_2d)
            output = output + shared_out

        # Restore original shape
        output = output.view(original_shape)

        return output, aux_loss

    def _forward_offloaded(self, hidden_2d, aux_loss_weight, mode,
                           decode_mode=False):
        """Forward pass with expert offloading via LRU cache.

        Uses pipelined loading: starts async H2D for cache misses while
        computing GEMMs for cache hits, then waits and computes misses.
        """
        from wm_infra.ops.routing import topk_route, compute_expert_offsets
        from wm_infra.ops.permute import permute_tokens, unpermute_tokens
        from wm_infra.ops.group_gemm import grouped_gemm
        from wm_infra.ops.activation import fused_swiglu

        num_tokens, hidden_dim = hidden_2d.shape
        num_experts = self.config.num_experts
        top_k = self.config.top_k

        cache = self._get_expert_cache(hidden_2d.device)

        # Step 1: Route to determine which experts are needed
        topk_weights, topk_ids, aux_loss = topk_route(
            hidden_2d, self.gate.weight, top_k, self.config.renormalize,
            aux_loss_weight=aux_loss_weight,
            expert_bias=self.expert_bias)

        # Step 2: Partition into hits/misses and start async loading
        needed_expert_ids = topk_ids.unique()
        hits, misses = cache.partition_experts(needed_expert_ids)

        # Touch hits to refresh LRU
        for eid in hits:
            cache._lru.move_to_end(eid, last=True)

        # Start async loads for misses (non-blocking)
        cache.start_async_loads(misses)

        # Step 2.5: Speculative prefetch for next layer (non-blocking)
        if self._speculator is not None:
            self._speculator.predict_and_prefetch(hidden_2d)

        # Wait for loads to complete before remapping
        if misses:
            cache.wait_for_loads()

        # Access GPU weight buffers directly (no redundant ensure_loaded call)
        w_gate_gpu = cache.w_gate_gpu
        w_up_gpu = cache.w_up_gpu
        w_down_gpu = cache.w_down_gpu

        # Step 3: Remap expert IDs to GPU slot indices (uses persistent GPU table)
        remapped_ids = cache.remap_expert_ids(topk_ids)

        # Step 4: Compute expert offsets using remapped (slot) IDs.
        num_cached = self.config.max_experts_in_gpu
        sorted_token_ids, expert_offsets, expert_counts = compute_expert_offsets(
            remapped_ids, num_cached)

        # Step 5: Permute hidden states into expert order
        permuted = permute_tokens(hidden_2d, sorted_token_ids, top_k)

        if mode == "naive":
            # Naive loop using remapped weights for correctness testing
            output = torch.zeros_like(hidden_2d)
            for t in range(num_tokens):
                for k in range(top_k):
                    slot = remapped_ids[t, k].item()
                    weight = topk_weights[t, k]
                    gate_out = hidden_2d[t] @ w_gate_gpu[slot]
                    up_out = hidden_2d[t] @ w_up_gpu[slot]
                    activated = torch.nn.functional.silu(gate_out) * up_out
                    expert_out = activated @ w_down_gpu[slot]
                    output[t] += weight * expert_out
            return output, aux_loss

        # Step 6-9: GEMMs
        gate_out = grouped_gemm(permuted, w_gate_gpu, expert_offsets, num_cached,
                                decode_mode=decode_mode)
        up_out = grouped_gemm(permuted, w_up_gpu, expert_offsets, num_cached,
                              decode_mode=decode_mode)
        activated = fused_swiglu(gate_out, up_out)
        expert_out = grouped_gemm(activated, w_down_gpu, expert_offsets, num_cached,
                                  decode_mode=decode_mode)

        # Step 10: Unpermute + weighted combine
        output = unpermute_tokens(expert_out, sorted_token_ids, topk_weights,
                                  num_tokens, top_k)

        return output, aux_loss

    def _forward_fp8(self, hidden_2d, aux_loss_weight, decode_mode=False):
        """FP8 composable forward: quantize activations, use FP8 GEMM kernels."""
        from wm_infra.ops.routing import topk_route, compute_expert_offsets
        from wm_infra.ops.permute import permute_tokens, unpermute_tokens
        from wm_infra.ops.group_gemm import grouped_gemm_fp8
        from wm_infra.ops.activation import fused_swiglu
        from wm_infra.ops.quantize import quantize_per_tensor

        num_tokens, hidden_dim = hidden_2d.shape
        num_experts = self.config.num_experts
        top_k = self.config.top_k
        output_dtype = hidden_2d.dtype

        # Step 1: Route
        topk_weights, topk_ids, aux_loss = topk_route(
            hidden_2d, self.gate.weight, top_k, self.config.renormalize,
            aux_loss_weight=aux_loss_weight,
            expert_bias=self.expert_bias)

        # Step 2: Compute expert offsets
        sorted_token_ids, expert_offsets, expert_counts = compute_expert_offsets(
            topk_ids, num_experts)

        # Step 3: Permute
        permuted = permute_tokens(hidden_2d, sorted_token_ids, top_k)

        # Step 4: Quantize activations to FP8
        permuted_fp8, a_scale = quantize_per_tensor(permuted.detach())

        # Step 5: FP8 Grouped GEMM -- gate projection
        gate_out = grouped_gemm_fp8(
            permuted_fp8, a_scale, self.w_gate, self.w_gate_scale,
            expert_offsets, num_experts, output_dtype,
            decode_mode=decode_mode)

        # Step 6: FP8 Grouped GEMM -- up projection
        up_out = grouped_gemm_fp8(
            permuted_fp8, a_scale, self.w_up, self.w_up_scale,
            expert_offsets, num_experts, output_dtype,
            decode_mode=decode_mode)

        # Step 7: SwiGLU activation
        activated = fused_swiglu(gate_out, up_out)

        # Step 8: Quantize activated for down projection
        activated_fp8, act_scale = quantize_per_tensor(activated.detach())

        # Step 9: FP8 Grouped GEMM -- down projection
        expert_out = grouped_gemm_fp8(
            activated_fp8, act_scale, self.w_down, self.w_down_scale,
            expert_offsets, num_experts, output_dtype,
            decode_mode=decode_mode)

        # Step 10: Unpermute + weighted combine
        output = unpermute_tokens(expert_out, sorted_token_ids, topk_weights,
                                  num_tokens, top_k)

        return output, aux_loss

    @torch.no_grad()
    def _update_expert_bias(self, hidden_2d: torch.Tensor) -> None:
        """Recompute routing with current bias and update bias based on load.

        This is a lightweight operation: just a matmul for logits + torch.topk
        + bincount. No GEMMs involved. The cost is negligible relative to the
        expert FFN computation.
        """
        from wm_infra.ops.routing import update_expert_bias

        num_experts = self.config.num_experts
        top_k = self.config.top_k

        # Compute biased logits and select experts (same as what topk_route did)
        gate_weight = self.gate.weight
        if gate_weight.dtype != hidden_2d.dtype:
            gate_weight = gate_weight.to(hidden_2d.dtype)
        logits = hidden_2d @ gate_weight.T
        biased_logits = logits + self.expert_bias.unsqueeze(0)
        _, topk_ids = torch.topk(biased_logits, top_k, dim=-1)

        # Compute per-expert token counts via bincount
        expert_counts = torch.bincount(
            topk_ids.view(-1), minlength=num_experts)

        # Update bias in-place
        update_expert_bias(
            self.expert_bias, expert_counts, num_experts,
            lr=self.config.expert_bias_lr)

    def extra_repr(self):
        c = self.config
        dtype_str = "fp8" if self._use_fp8 else c.dtype
        parts = [
            f"num_experts={c.num_experts}", f"top_k={c.top_k}",
            f"hidden_dim={c.hidden_dim}", f"intermediate_dim={c.intermediate_dim}",
            f"activation={c.activation}", f"dtype={dtype_str}",
        ]
        if self._use_offloading:
            parts.append(f"max_experts_in_gpu={c.max_experts_in_gpu}")
        return ", ".join(parts)


class SharedExpert(nn.Module):
    """Shared expert that processes all tokens (DeepSeek-V3 / Qwen-MoE style).

    Always active regardless of routing. Output is added to MoE output.
    Optionally has a sigmoid gate (Qwen2-MoE style).
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        inter_dim = config.shared_expert_intermediate_dim or config.intermediate_dim
        self.gate_proj = nn.Linear(config.hidden_dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, config.hidden_dim, bias=False)
        self._inter_dim = inter_dim
        # Optional sigmoid gate (Qwen2-MoE)
        self.shared_expert_gate = None
        if config.use_shared_expert_gate:
            self.shared_expert_gate = nn.Linear(config.hidden_dim, 1, bias=False)
        # Fused gate+up weight (lazily initialized)
        self._fused_gate_up: Optional[nn.Linear] = None

    def _init_fused_gate_up(self):
        """Build fused gate+up weight matrix from separate projections."""
        gate_w = self.gate_proj.weight.data  # [I, H]
        up_w = self.up_proj.weight.data      # [I, H]
        fused_w = torch.cat([gate_w, up_w], dim=0)  # [2*I, H]
        self._fused_gate_up = nn.Linear(
            fused_w.shape[1], fused_w.shape[0], bias=False,
            device=fused_w.device, dtype=fused_w.dtype,
        )
        self._fused_gate_up.weight = nn.Parameter(fused_w, requires_grad=False)

    def forward(self, x):
        out = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        if self.shared_expert_gate is not None:
            out = torch.sigmoid(self.shared_expert_gate(x)) * out
        return out

    def forward_decode(self, x):
        """Optimized decode path: fused gate+up matmul + fused SwiGLU.

        Saves 1 cuBLAS GEMV (fused gate+up) + 2 elementwise (fused SwiGLU)
        per layer compared to the generic forward().
        """
        from wm_infra.ops.activation import swiglu_fused_gate_up

        if self._fused_gate_up is None:
            self._init_fused_gate_up()

        I = self._inter_dim
        gate_up = self._fused_gate_up(x)  # [B, 1, 2*I] — single matmul

        # Need contiguous [*, 2*I] for the fused SwiGLU kernel
        shape = gate_up.shape
        gate_up_2d = gate_up.view(-1, 2 * I)
        activated = torch.empty(gate_up_2d.shape[0], I,
                                device=x.device, dtype=x.dtype)
        swiglu_fused_gate_up(gate_up_2d, activated)

        out = self.down_proj(activated.view(*shape[:-1], I))

        if self.shared_expert_gate is not None:
            out = torch.sigmoid(self.shared_expert_gate(x)) * out
        return out
