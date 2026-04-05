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
    ):
        self.max_concurrent = max_concurrent
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.device = torch.device(device) if isinstance(device, str) else device

        self._states: dict[str, RolloutState] = {}
        self._current_memory = 0

    @property
    def num_active(self) -> int:
        return len(self._states)

    @property
    def memory_used_gb(self) -> float:
        return self._current_memory / (1024**3)

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

        initial_bytes = initial_state.element_size() * initial_state.nelement()
        self._ensure_capacity(initial_bytes)

        now = time.monotonic()
        state = RolloutState(
            rollout_id=rollout_id,
            latent_states=[initial_state.to(self.device)],
            current_step=0,
            max_steps=max_steps,
            created_at=now,
            last_accessed=now,
        )
        self._states[rollout_id] = state
        self._current_memory += state.memory_bytes
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
        required_bytes = action.element_size() * action.nelement() + predicted_state.element_size() * predicted_state.nelement()
        self._ensure_capacity(required_bytes, protected_ids={rollout_id})
        state.actions.append(action.to(self.device))
        state.latent_states.append(predicted_state.to(self.device))
        state.current_step += 1
        state.last_accessed = time.monotonic()
        self._current_memory += required_bytes
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

        now = time.monotonic()
        forked = RolloutState(
            rollout_id=new_id,
            latent_states=[s.clone() for s in source.latent_states],
            actions=[a.clone() for a in source.actions],
            current_step=source.current_step,
            max_steps=max_steps or source.max_steps,
            created_at=now,
            last_accessed=now,
        )
        self._ensure_capacity(forked.memory_bytes, protected_ids={source_id})
        self._states[new_id] = forked
        self._current_memory += forked.memory_bytes
        return forked

    def remove(self, rollout_id: str) -> None:
        """Remove a rollout and free its memory."""
        if rollout_id in self._states:
            state = self._states.pop(rollout_id)
            self._current_memory -= state.memory_bytes

    def _evict_lru(self) -> None:
        """Evict the least recently used rollout."""
        if not self._states:
            return
        oldest_id = min(self._states, key=lambda k: self._states[k].last_accessed)
        self.remove(oldest_id)

    def cleanup_completed(self) -> list[str]:
        """Remove all completed rollouts, return their IDs."""
        completed = [rid for rid, s in self._states.items() if s.is_complete]
        for rid in completed:
            self.remove(rid)
        return completed
