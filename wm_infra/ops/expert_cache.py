"""LRU cache for expert weight offloading between GPU and CPU pinned memory.

For large MoE models (e.g., DeepSeek-V3 with 256 experts), not all expert
weights fit in GPU VRAM. This module keeps frequently-used experts on GPU
and pages the rest to/from CPU pinned memory using async CUDA streams.
"""

import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Union


class ExpertCache:
    """LRU cache for expert weights. Keeps hot experts on GPU, cold experts on CPU.

    Pre-allocates a fixed GPU buffer of shape [max_experts_in_gpu, ...] for each
    weight matrix. Maintains a mapping from expert_id to gpu_slot_index so that
    kernels can index into the compacted GPU buffers.

    Async transfers use a dedicated CUDA stream so that host-to-device copies
    can overlap with compute on the default stream (caller must synchronize
    before using the returned tensors).

    Usage:
        cache = ExpertCache(
            w_gate_all=w_gate_cpu,   # [E, K, N] on CPU pinned memory
            w_up_all=w_up_cpu,       # [E, K, N] on CPU pinned memory
            w_down_all=w_down_cpu,   # [E, N_inter, K] on CPU pinned memory
            max_experts_in_gpu=32,
            device='cuda',
        )

        # Before each forward pass
        gpu_weights, expert_to_slot = cache.ensure_loaded(needed_expert_ids)
        w_gate_gpu, w_up_gpu, w_down_gpu = gpu_weights
        # expert_to_slot: dict mapping expert_id -> gpu_slot_index
    """

    def __init__(
        self,
        w_gate_all: torch.Tensor,
        w_up_all: torch.Tensor,
        w_down_all: torch.Tensor,
        max_experts_in_gpu: int,
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize the expert cache.

        Args:
            w_gate_all: [num_experts, hidden_dim, intermediate_dim] on CPU (pinned).
            w_up_all: [num_experts, hidden_dim, intermediate_dim] on CPU (pinned).
            w_down_all: [num_experts, intermediate_dim, hidden_dim] on CPU (pinned).
            max_experts_in_gpu: Maximum number of experts to keep on GPU at once.
            device: GPU device to cache experts on.
        """
        self.device = torch.device(device)
        self.num_experts = w_gate_all.shape[0]
        self.max_experts_in_gpu = min(max_experts_in_gpu, self.num_experts)

        # Ensure CPU tensors are in pinned memory for fast async transfers
        self.w_gate_cpu = self._ensure_pinned(w_gate_all)
        self.w_up_cpu = self._ensure_pinned(w_up_all)
        self.w_down_cpu = self._ensure_pinned(w_down_all)

        # Pre-allocate GPU buffers: [max_experts_in_gpu, ...]
        self.w_gate_gpu = torch.empty(
            self.max_experts_in_gpu, *w_gate_all.shape[1:],
            dtype=w_gate_all.dtype, device=self.device,
        )
        self.w_up_gpu = torch.empty(
            self.max_experts_in_gpu, *w_up_all.shape[1:],
            dtype=w_up_all.dtype, device=self.device,
        )
        self.w_down_gpu = torch.empty(
            self.max_experts_in_gpu, *w_down_all.shape[1:],
            dtype=w_down_all.dtype, device=self.device,
        )

        # LRU tracking: OrderedDict preserves insertion order; most-recently-used
        # items are moved to the end. Keys are expert_id, values are gpu_slot_index.
        self._lru: OrderedDict[int, int] = OrderedDict()

        # Free GPU slots (stack). Initially all slots are free.
        self._free_slots: List[int] = list(range(self.max_experts_in_gpu - 1, -1, -1))

        # Dedicated CUDA stream for async H2D copies
        self._transfer_stream = torch.cuda.Stream(device=self.device)

        # Persistent GPU remap table: expert_id → gpu_slot_index.
        # Updated incrementally on load/evict. Avoids rebuilding from Python dict
        # on every remap_expert_ids() call (eliminates CPU→GPU sync).
        self._remap_table = torch.full(
            (self.num_experts,), -1, dtype=torch.int64, device=self.device)

    @staticmethod
    def _ensure_pinned(t: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU pinned memory if not already there."""
        if t.is_cuda:
            t = t.cpu()
        if not t.is_pinned():
            return t.pin_memory()
        return t

    def _evict(self, count: int) -> List[int]:
        """Evict the `count` least-recently-used experts and return freed slot indices.

        LRU order: the front of the OrderedDict is the least-recently-used.
        """
        freed = []
        for _ in range(count):
            expert_id, slot = self._lru.popitem(last=False)
            self._remap_table[expert_id] = -1
            freed.append(slot)
        return freed

    def _load_expert_to_slot(self, expert_id: int, slot: int):
        """Async copy one expert's weights from CPU pinned memory to a GPU slot.

        Must be called within the transfer stream context.
        """
        self.w_gate_gpu[slot].copy_(self.w_gate_cpu[expert_id], non_blocking=True)
        self.w_up_gpu[slot].copy_(self.w_up_cpu[expert_id], non_blocking=True)
        self.w_down_gpu[slot].copy_(self.w_down_cpu[expert_id], non_blocking=True)

    def ensure_loaded(
        self,
        expert_ids: Union[torch.Tensor, List[int]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[int, int]]:
        """Ensure all requested experts are resident on GPU.

        Loads missing experts (evicting LRU ones if necessary), then returns
        the GPU weight buffers and a mapping from expert_id to gpu_slot_index.

        Args:
            expert_ids: Expert IDs needed for the current forward pass.
                Can be a 1-D tensor or a list of ints. Duplicates are ignored.

        Returns:
            gpu_weights: (w_gate_gpu, w_up_gpu, w_down_gpu) each
                [max_experts_in_gpu, ...]. Only slots in expert_to_slot are valid.
            expert_to_slot: dict mapping each requested expert_id (int) to its
                gpu_slot_index (int).
        """
        if isinstance(expert_ids, torch.Tensor):
            unique_ids = expert_ids.unique().cpu().tolist()
        else:
            unique_ids = list(set(expert_ids))

        unique_ids = [int(eid) for eid in unique_ids]

        if len(unique_ids) > self.max_experts_in_gpu:
            raise ValueError(
                f"Requested {len(unique_ids)} unique experts but cache only has "
                f"{self.max_experts_in_gpu} GPU slots. Increase max_experts_in_gpu "
                f"or reduce the number of active experts per batch."
            )

        # Partition into already-cached and missing
        hits = []
        misses = []
        for eid in unique_ids:
            if eid in self._lru:
                hits.append(eid)
            else:
                misses.append(eid)

        # Touch all hits to refresh their LRU position
        for eid in hits:
            self._lru.move_to_end(eid, last=True)

        if misses:
            # Determine how many evictions are needed
            need_slots = len(misses) - len(self._free_slots)
            if need_slots > 0:
                freed = self._evict(need_slots)
                self._free_slots.extend(freed)

            # Async load missing experts
            with torch.cuda.stream(self._transfer_stream):
                for eid in misses:
                    slot = self._free_slots.pop()
                    self._load_expert_to_slot(eid, slot)
                    self._lru[eid] = slot
                    self._remap_table[eid] = slot

            # Synchronize transfer stream so weights are ready before compute
            torch.cuda.current_stream(self.device).wait_stream(self._transfer_stream)

        # Build the expert_id -> slot mapping for the requested experts
        expert_to_slot = {eid: self._lru[eid] for eid in unique_ids}

        gpu_weights = (self.w_gate_gpu, self.w_up_gpu, self.w_down_gpu)
        return gpu_weights, expert_to_slot

    def remap_expert_ids(
        self,
        topk_ids: torch.Tensor,
        expert_to_slot: Dict[int, int] = None,
    ) -> torch.Tensor:
        """Remap original expert IDs in topk_ids to GPU slot indices.

        Uses a persistent GPU remap table that is updated incrementally
        on load/evict, avoiding CPU→GPU sync per call.

        Args:
            topk_ids: [num_tokens, top_k] — original expert IDs from the router.
            expert_to_slot: deprecated, ignored. Kept for backward compatibility.

        Returns:
            remapped_ids: [num_tokens, top_k] — slot indices, same shape/device.
        """
        return self._remap_table[topk_ids]

    def partition_experts(
        self,
        expert_ids: Union[torch.Tensor, List[int]],
    ) -> Tuple[List[int], List[int]]:
        """Partition requested experts into cache hits and misses.

        Args:
            expert_ids: Needed expert IDs (1-D tensor or list).

        Returns:
            (hits, misses): Lists of expert IDs that are/aren't cached.
        """
        if isinstance(expert_ids, torch.Tensor):
            unique_ids = expert_ids.unique().cpu().tolist()
        else:
            unique_ids = list(set(expert_ids))
        unique_ids = [int(eid) for eid in unique_ids]

        hits = [eid for eid in unique_ids if eid in self._lru]
        misses = [eid for eid in unique_ids if eid not in self._lru]
        return hits, misses

    def start_async_loads(self, misses: List[int]) -> None:
        """Begin async H2D copies for missing experts without blocking.

        Touch hits' LRU positions, evict if needed, start transfers.
        Does NOT synchronize — caller must call wait_for_loads() before
        using the loaded weights.
        """
        if not misses:
            return

        if len(misses) > self.max_experts_in_gpu:
            raise ValueError(
                f"Need {len(misses)} slots but cache only has {self.max_experts_in_gpu}"
            )

        # Evict if needed
        need_slots = len(misses) - len(self._free_slots)
        if need_slots > 0:
            freed = self._evict(need_slots)
            self._free_slots.extend(freed)

        # Start async loads (do NOT sync)
        with torch.cuda.stream(self._transfer_stream):
            for eid in misses:
                slot = self._free_slots.pop()
                self._load_expert_to_slot(eid, slot)
                self._lru[eid] = slot
                self._remap_table[eid] = slot

    def wait_for_loads(self) -> None:
        """Block until all pending async loads complete."""
        torch.cuda.current_stream(self.device).wait_stream(self._transfer_stream)

    def speculative_prefetch(self, expert_ids: List[int]):
        """Speculatively prefetch experts using only free cache slots.

        Unlike ensure_loaded(), this method NEVER evicts
        cached experts. It only loads experts into currently-free GPU slots.
        If there are no free slots, this is a no-op.

        This is safe for speculative predictions: mispredictions only
        "waste" free slots that will be reclaimed by the real ensure_loaded()
        call when the next layer runs.

        Args:
            expert_ids: Expert IDs predicted to be needed by the next layer.
        """
        # Filter to experts not already cached
        misses = [eid for eid in expert_ids if eid not in self._lru]
        if not misses:
            return

        # Only load as many as we have free slots (no eviction)
        loadable = misses[:len(self._free_slots)]
        if not loadable:
            return

        with torch.cuda.stream(self._transfer_stream):
            for eid in loadable:
                slot = self._free_slots.pop()
                self._load_expert_to_slot(eid, slot)
                self._lru[eid] = slot
                self._remap_table[eid] = slot

    @property
    def cached_experts(self) -> List[int]:
        """Return the list of expert IDs currently resident on GPU, in LRU order."""
        return list(self._lru.keys())

    @property
    def num_cached(self) -> int:
        """Number of experts currently on GPU."""
        return len(self._lru)

    @property
    def num_free_slots(self) -> int:
        """Number of available GPU slots."""
        return len(self._free_slots)

