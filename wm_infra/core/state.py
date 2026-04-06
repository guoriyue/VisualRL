"""Latent state manager for world model rollouts.

Like vLLM's KV cache but for world model latent states. Manages temporal
state across rollout steps with memory pooling and eviction.

Each active rollout has a state buffer that accumulates latent tokens
across prediction steps. The manager handles allocation, pooling, and
cleanup of these buffers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(slots=True)
class RolloutState:
    """State for a single active rollout."""

    rollout_id: str
    latent_states: list[torch.Tensor] = field(default_factory=list)  # [N, D] per step
    actions: list[torch.Tensor] = field(default_factory=list)  # [A] per step
    current_step: int = 0
    max_steps: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.current_step >= self.max_steps

    @property
    def num_latent_tokens(self) -> int:
        if not self.latent_states:
            return 0
        return self.latent_states[0].shape[0]

    @property
    def memory_bytes(self) -> int:
        total = 0
        for s in self.latent_states:
            total += s.element_size() * s.nelement()
        for a in self.actions:
            total += a.element_size() * a.nelement()
        return total


class LatentStateManager:
    """Manages latent state buffers for concurrent world model rollouts.

    Key differences from KV cache:
    - Stores full latent state tensors (not just K/V projections)
    - State grows with each prediction step (append-only within a rollout)
    - Supports branching: fork a rollout state to explore multiple action sequences
    - LRU eviction when memory budget is exceeded
    """

    def __init__(
        self,
        max_concurrent: int = 64,
        max_memory_gb: float = 4.0,
        device: torch.device | str = "cuda",
        fork_mode: str = "copy_on_write",
    ):
        if fork_mode not in {"copy_on_write", "deep_copy"}:
            raise ValueError(f"Unsupported fork_mode: {fork_mode}")
        self.max_concurrent = max_concurrent
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.fork_mode = fork_mode

        self._states: dict[str, RolloutState] = {}
        self._current_memory = 0
        self._tensor_refs: dict[tuple[int, int], int] = {}
        self._stats = {
            "fork_mode": fork_mode,
            "fork_count": 0,
            "tensor_materializations": 0,
            "tensor_reuse_hits": 0,
            "bytes_saved_via_sharing": 0,
        }

    @property
    def num_active(self) -> int:
        return len(self._states)

    @property
    def memory_used_gb(self) -> float:
        return self._current_memory / (1024**3)

    def stats_snapshot(self) -> dict[str, float | int | str]:
        denominator = self._stats["tensor_reuse_hits"] + self._stats["tensor_materializations"]
        reuse_hit_rate = (self._stats["tensor_reuse_hits"] / denominator) if denominator else 0.0
        logical_bytes = sum(state.memory_bytes for state in self._states.values())
        return {
            **self._stats,
            "reuse_hit_rate": reuse_hit_rate,
            "physical_bytes": self._current_memory,
            "logical_bytes": logical_bytes,
            "memory_used_bytes": self._current_memory,
            "num_active": self.num_active,
        }

    def _tensor_key(self, tensor: torch.Tensor) -> tuple[int, int]:
        return (int(tensor.data_ptr()), int(tensor.element_size() * tensor.nelement()))

    def _required_physical_bytes(self, tensors: list[torch.Tensor]) -> int:
        required = 0
        for tensor in tensors:
            key = self._tensor_key(tensor)
            if key not in self._tensor_refs:
                required += key[1]
        return required

    def _register_tensor(self, tensor: torch.Tensor) -> None:
        key = self._tensor_key(tensor)
        if key in self._tensor_refs:
            self._tensor_refs[key] += 1
            return
        self._tensor_refs[key] = 1
        self._current_memory += key[1]
        self._stats["tensor_materializations"] += 1

    def _share_tensor(self, tensor: torch.Tensor) -> None:
        key = self._tensor_key(tensor)
        if key not in self._tensor_refs:
            self._register_tensor(tensor)
            return
        self._tensor_refs[key] += 1
        self._stats["tensor_reuse_hits"] += 1
        self._stats["bytes_saved_via_sharing"] += key[1]

    def _release_tensor(self, tensor: torch.Tensor) -> None:
        key = self._tensor_key(tensor)
        if key not in self._tensor_refs:
            return
        self._tensor_refs[key] -= 1
        if self._tensor_refs[key] <= 0:
            self._tensor_refs.pop(key, None)
            self._current_memory -= key[1]

    def _ensure_capacity(self, required_bytes: int, *, protected_ids: set[str] | None = None) -> None:
        protected_ids = protected_ids or set()
        if required_bytes > self.max_memory_bytes:
            raise MemoryError(
                f"Required state allocation {required_bytes} bytes exceeds budget {self.max_memory_bytes} bytes"
            )

        while self._current_memory + required_bytes > self.max_memory_bytes:
            evictable = [
                (rid, state)
                for rid, state in self._states.items()
                if rid not in protected_ids
            ]
            if not evictable:
                raise MemoryError(
                    f"Latent state budget exceeded: need {required_bytes} more bytes with {self._current_memory} bytes already tracked"
                )
            oldest_id, _ = min(evictable, key=lambda item: item[1].last_accessed)
            self.remove(oldest_id)

    def create(
        self,
        rollout_id: str,
        initial_state: torch.Tensor,
        max_steps: int,
    ) -> RolloutState:
        """Create a new rollout state with initial latent state.

        Args:
            rollout_id: unique identifier
            initial_state: [N, D] initial latent tokens (or [B, N, D] for batch)
            max_steps: maximum prediction steps

        Returns:
            The created RolloutState
        """
        if rollout_id in self._states:
            raise ValueError(f"Rollout {rollout_id} already exists")

        if self.num_active >= self.max_concurrent:
            self._evict_lru()

        initial_state = initial_state.to(self.device)
        initial_bytes = self._required_physical_bytes([initial_state])
        self._ensure_capacity(initial_bytes)

        now = time.monotonic()
        state = RolloutState(
            rollout_id=rollout_id,
            latent_states=[initial_state],
            current_step=0,
            max_steps=max_steps,
            created_at=now,
            last_accessed=now,
        )
        self._states[rollout_id] = state
        self._register_tensor(initial_state)
        return state

    def append_step(
        self,
        rollout_id: str,
        action: torch.Tensor,
        predicted_state: torch.Tensor,
    ) -> RolloutState:
        """Append a prediction step to an active rollout.

        Args:
            rollout_id: which rollout to update
            action: [A] or [B, A] action taken
            predicted_state: [N, D] or [B, N, D] predicted next latent state

        Returns:
            Updated RolloutState
        """
        state = self._states[rollout_id]
        action = action.to(self.device)
        predicted_state = predicted_state.to(self.device)
        required_bytes = self._required_physical_bytes([action, predicted_state])
        self._ensure_capacity(required_bytes, protected_ids={rollout_id})
        state.actions.append(action)
        state.latent_states.append(predicted_state)
        state.current_step += 1
        state.last_accessed = time.monotonic()
        self._register_tensor(action)
        self._register_tensor(predicted_state)
        return state

    def get(self, rollout_id: str) -> RolloutState:
        """Get a rollout state, updating access time."""
        state = self._states[rollout_id]
        state.last_accessed = time.monotonic()
        return state

    def fork(self, source_id: str, new_id: str, max_steps: Optional[int] = None) -> RolloutState:
        """Fork a rollout state to explore alternative action sequences.

        Creates a deep copy of the source rollout's state history.
        """
        source = self._states[source_id]
        if new_id in self._states:
            raise ValueError(f"Rollout {new_id} already exists")
        self._stats["fork_count"] += 1

        if self.num_active >= self.max_concurrent:
            self._evict_lru(protected_ids={source_id})

        now = time.monotonic()
        if self.fork_mode == "deep_copy":
            latent_states = [s.clone() for s in source.latent_states]
            actions = [a.clone() for a in source.actions]
            required_bytes = self._required_physical_bytes([*latent_states, *actions])
            self._ensure_capacity(required_bytes, protected_ids={source_id})
            forked = RolloutState(
                rollout_id=new_id,
                latent_states=latent_states,
                actions=actions,
                current_step=source.current_step,
                max_steps=max_steps or source.max_steps,
                created_at=now,
                last_accessed=now,
            )
            self._states[new_id] = forked
            for tensor in [*latent_states, *actions]:
                self._register_tensor(tensor)
            return forked

        forked = RolloutState(
            rollout_id=new_id,
            latent_states=list(source.latent_states),
            actions=list(source.actions),
            current_step=source.current_step,
            max_steps=max_steps or source.max_steps,
            created_at=now,
            last_accessed=now,
        )
        self._states[new_id] = forked
        for tensor in [*forked.latent_states, *forked.actions]:
            self._share_tensor(tensor)
        return forked

    def remove(self, rollout_id: str) -> None:
        """Remove a rollout and free its memory."""
        if rollout_id in self._states:
            state = self._states.pop(rollout_id)
            for tensor in [*state.latent_states, *state.actions]:
                self._release_tensor(tensor)

    def _evict_lru(self, protected_ids: set[str] | None = None) -> None:
        """Evict the least recently used rollout."""
        if not self._states:
            return
        protected_ids = protected_ids or set()
        candidates = [rid for rid in self._states if rid not in protected_ids]
        if not candidates:
            return
        oldest_id = min(candidates, key=lambda k: self._states[k].last_accessed)
        self.remove(oldest_id)

    def cleanup_completed(self) -> list[str]:
        """Remove all completed rollouts, return their IDs."""
        completed = [rid for rid, s in self._states.items() if s.is_complete]
        for rid in completed:
            self.remove(rid)
        return completed
