"""Backward-compatible wrappers mapping old ``rollout_engine`` names.

This module preserves the public API that was previously exported by
``wm_infra.rollout_engine`` so that consumers can migrate to
``wm_infra.engine.compat_rollout`` (or ``wm_infra.engine``) without
large-scale rewrites.

The implementations are inlined here rather than thin wrappers because the
old rollout-engine classes (``LatentStateManager``, ``RolloutScheduler``,
``WorldModelEngine``, ``AsyncWorldModelEngine``) have substantially different
semantics from the new engine module (``PagedLatentPool``,
``ContinuousBatchingScheduler``, ``EngineLoop``).  A pure delegation layer
would need to emulate too many details, so we keep the proven code intact.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Latent state manager (previously rollout_engine/state.py)
# ---------------------------------------------------------------------------

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

    def ensure_capacity_for_tensors(
        self,
        tensors: list[torch.Tensor],
        *,
        protected_ids: set[str] | None = None,
    ) -> None:
        """Validate that the given tensors can be tracked without evicting protected rollouts."""
        required_bytes = self._required_physical_bytes(tensors)
        self._ensure_capacity(required_bytes, protected_ids=protected_ids)

    def create(
        self,
        rollout_id: str,
        initial_state: torch.Tensor,
        max_steps: int,
        *,
        protected_ids: set[str] | None = None,
    ) -> RolloutState:
        """Create a new rollout state with initial latent state."""
        if rollout_id in self._states:
            raise ValueError(f"Rollout {rollout_id} already exists")

        if self.num_active >= self.max_concurrent:
            self._evict_lru(protected_ids=protected_ids)
        if self.num_active >= self.max_concurrent:
            raise MemoryError(
                f"Concurrent rollout budget exceeded: limit {self.max_concurrent} active rollouts"
            )

        initial_state = initial_state.to(self.device)
        initial_bytes = self._required_physical_bytes([initial_state])
        self._ensure_capacity(initial_bytes, protected_ids=protected_ids)

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
        *,
        protected_ids: set[str] | None = None,
    ) -> RolloutState:
        """Append a prediction step to an active rollout."""
        state = self._states[rollout_id]
        action = action.to(self.device)
        predicted_state = predicted_state.to(self.device)
        required_bytes = self._required_physical_bytes([action, predicted_state])
        protected = set(protected_ids or set())
        protected.add(rollout_id)
        self._ensure_capacity(required_bytes, protected_ids=protected)
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
        """Fork a rollout state to explore alternative action sequences."""
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


# ---------------------------------------------------------------------------
# Rollout scheduler (previously rollout_engine/scheduler.py)
# ---------------------------------------------------------------------------

from collections import deque
from typing import Deque

from wm_infra.config import SchedulerConfig, SchedulerPolicy
from wm_infra.engine._types import RolloutTaskConfig, VideoMemoryProfile

LOW_VRAM_MEMORY_MULTIPLIER = 0.65
HIGH_QUALITY_MEMORY_MULTIPLIER = 1.25
DEFAULT_FRAME_COUNT = 1
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_RESOURCE_UNITS_PER_GB = 3.0


@dataclass(slots=True)
class RolloutRequest:
    """A pending rollout request."""

    request_id: str
    num_steps: int
    priority: float = 0.0
    created_at: float = field(default_factory=time.monotonic)
    deadline: Optional[float] = None
    frame_count: int = DEFAULT_FRAME_COUNT
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    memory_profile: Optional[str] = None
    estimated_resource_units: Optional[float] = None

    def estimate_resource_units(self) -> float:
        if self.estimated_resource_units is not None:
            return self.estimated_resource_units
        megapixels = (self.width * self.height) / 1_000_000
        frame_pressure = max(self.frame_count, 1) * max(self.num_steps, 1)
        multiplier = 1.0
        if self.memory_profile == VideoMemoryProfile.LOW_VRAM.value:
            multiplier = LOW_VRAM_MEMORY_MULTIPLIER
        elif self.memory_profile == VideoMemoryProfile.HIGH_QUALITY.value:
            multiplier = HIGH_QUALITY_MEMORY_MULTIPLIER
        return max(frame_pressure * max(megapixels, 0.1) * multiplier, 0.1)

    @classmethod
    def from_task_config(cls, request_id: str, task_config: Optional[RolloutTaskConfig], *, priority: float = 0.0, deadline: Optional[float] = None) -> "RolloutRequest":
        task_config = task_config or RolloutTaskConfig()
        memory_profile = task_config.memory_profile.value if task_config.memory_profile else None
        return cls(
            request_id=request_id,
            num_steps=task_config.num_steps,
            priority=priority,
            deadline=deadline,
            frame_count=task_config.frame_count or DEFAULT_FRAME_COUNT,
            width=task_config.width or DEFAULT_WIDTH,
            height=task_config.height or DEFAULT_HEIGHT,
            memory_profile=memory_profile,
        )


@dataclass(slots=True)
class ScheduledBatch:
    request_ids: list[str]
    step_indices: list[int]
    actions: list

    @property
    def size(self) -> int:
        return len(self.request_ids)


class RolloutScheduler:
    """Schedules world model rollout steps across concurrent requests."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._pending: Deque[RolloutRequest] = deque()
        self._active: dict[str, RolloutRequest] = {}
        self._step_counts: dict[str, int] = {}
        self._waiting_since: dict[str, float] = {}

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    @property
    def num_active(self) -> int:
        return len(self._active)

    def submit(self, request: RolloutRequest) -> None:
        self._pending.append(request)
        self._waiting_since[request.request_id] = time.monotonic()

    def admit(self) -> list[str]:
        admitted = []
        while self._pending and self.num_active < self.config.max_concurrent_rollouts:
            req = self._pending.popleft()
            self._active[req.request_id] = req
            self._step_counts[req.request_id] = 0
            admitted.append(req.request_id)
        return admitted

    def schedule_batch(self) -> ScheduledBatch:
        self.admit()
        if not self._active:
            return ScheduledBatch(request_ids=[], step_indices=[], actions=[])

        candidates = list(self._active.values())
        if self.config.policy == SchedulerPolicy.SJF:
            candidates.sort(key=lambda r: r.num_steps - self._step_counts.get(r.request_id, 0))
        elif self.config.policy == SchedulerPolicy.DEADLINE:
            candidates.sort(key=lambda r: r.deadline or float("inf"))
        elif self.config.policy == SchedulerPolicy.MEMORY_AWARE:
            candidates.sort(key=lambda r: (r.estimate_resource_units(), -r.priority))

        now = time.monotonic()
        urgent = [r for r in candidates if (now - self._waiting_since.get(r.request_id, now)) * 1000 > self.config.max_waiting_time_ms]
        if urgent:
            candidates = urgent + [c for c in candidates if c not in urgent]

        selected = []
        consumed_units = 0.0
        for candidate in candidates:
            if len(selected) >= self.config.max_batch_size:
                break
            units = candidate.estimate_resource_units()
            if self.config.max_batch_resource_units is not None and selected and consumed_units + units > self.config.max_batch_resource_units:
                continue
            selected.append(candidate)
            consumed_units += units

        if not selected:
            selected = candidates[:1]

        return ScheduledBatch(
            request_ids=[r.request_id for r in selected],
            step_indices=[self._step_counts.get(r.request_id, 0) for r in selected],
            actions=[],
        )

    def step_completed(self, request_id: str) -> bool:
        self._step_counts[request_id] = self._step_counts.get(request_id, 0) + 1
        req = self._active.get(request_id)
        return bool(req and self._step_counts[request_id] >= req.num_steps)

    def complete(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        self._active.pop(request_id, None)
        self._step_counts.pop(request_id, None)
        self._waiting_since.pop(request_id, None)
        self._pending = deque(r for r in self._pending if r.request_id != request_id)

    def has_work(self) -> bool:
        return bool(self._pending or self._active)


# ---------------------------------------------------------------------------
# Engine classes (previously rollout_engine/engine.py)
# ---------------------------------------------------------------------------

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Callable

import torch.nn as nn

from wm_infra.config import EngineConfig
from wm_infra.execution import (
    BatchSignature,
    ExecutionBatchPolicy,
    ExecutionChunk,
    ExecutionEntity,
    ExecutionStats,
    ExecutionWorkItem,
    HomogeneousChunkScheduler,
)
from wm_infra.models.base import WorldModel, RolloutInput, RolloutOutput
from wm_infra.tokenizer.video_tokenizer import VideoTokenizer

logger = logging.getLogger("wm_infra.engine")


@dataclass(slots=True)
class RolloutJob:
    """A user-facing rollout job."""

    job_id: str
    initial_observation: Optional[torch.Tensor] = None  # [C, H, W] or [T, C, H, W]
    initial_latent: Optional[torch.Tensor] = None  # [N, D] pre-encoded
    actions: Optional[torch.Tensor] = None  # [T, A]
    num_steps: int = 1
    return_frames: bool = True
    return_latents: bool = False
    stream: bool = False
    created_at: float = field(default_factory=time.monotonic)
    # Optional per-step callback for streaming: fn(job_id, step_idx, latent_state)
    step_callback: Optional[Callable[[str, int, torch.Tensor], None]] = None


@dataclass(slots=True)
class RolloutResult:
    """Result of a completed rollout."""

    job_id: str
    predicted_frames: Optional[torch.Tensor] = None  # [T, C, H, W]
    predicted_latents: Optional[torch.Tensor] = None  # [T, N, D]
    elapsed_ms: float = 0.0
    steps_completed: int = 0


class WorldModelEngine:
    """Main inference engine for world model serving.

    Orchestrates:
    - VideoTokenizer: observation -> latent tokens
    - LatentDynamicsModel: latent + action -> next latent
    - LatentStateManager: temporal state across rollout steps
    - RolloutScheduler: batching concurrent rollouts
    """

    def __init__(
        self,
        config: EngineConfig,
        dynamics_model: nn.Module,
        tokenizer: Optional[VideoTokenizer] = None,
        execution_mode: str = "chunked",
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.tokenizer = tokenizer
        if execution_mode != "chunked":
            raise ValueError(f"Unsupported execution_mode: {execution_mode}")
        self.execution_mode = execution_mode

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(config.dtype, torch.float16)
        device_str = config.device.value if hasattr(config.device, 'value') else str(config.device)
        self.device = torch.device(device_str)

        # Move model to device
        self.dynamics_model = self.dynamics_model.to(self.device, self.dtype)
        self.dynamics_model.eval()
        if self.tokenizer is not None:
            self.tokenizer = self.tokenizer.to(self.device, self.dtype)
            self.tokenizer.eval()

        # State management
        self.state_manager = LatentStateManager(
            max_concurrent=config.scheduler.max_concurrent_rollouts,
            max_memory_gb=config.state_cache.pool_size_gb,
            device=self.device,
        )

        # Scheduling
        self.scheduler = RolloutScheduler(config.scheduler)

        # Job tracking
        self._jobs: dict[str, RolloutJob] = {}
        self._results: dict[str, RolloutResult] = {}
        self._execution_stats = ExecutionStats(mode=execution_mode)

    def submit_job(self, job: RolloutJob) -> str:
        """Submit a rollout job. Returns job_id."""
        if not job.job_id:
            job.job_id = str(uuid.uuid4())

        self._jobs[job.job_id] = job
        self.scheduler.submit(RolloutRequest(
            request_id=job.job_id,
            num_steps=job.num_steps,
        ))
        return job.job_id

    @torch.inference_mode()
    def step(self) -> list[str]:
        """Run one engine step: schedule + execute one batch of predictions."""
        from wm_infra.engine.metrics import BATCH_SIZE, STEP_DURATION

        # 1. Admit pending jobs and encode initial states
        admitted = self.scheduler.admit()
        admitted_set = set(admitted)
        prepared_initial_states = {
            job_id: self._prepare_initial_state(job_id)
            for job_id in admitted
        }
        if prepared_initial_states:
            self.state_manager.ensure_capacity_for_tensors(
                list(prepared_initial_states.values()),
                protected_ids=admitted_set,
            )
        for job_id, initial_state in prepared_initial_states.items():
            self._initialize_rollout(job_id, initial_state, protected_ids=admitted_set)

        # 2. Schedule a batch
        batch = self.scheduler.schedule_batch()
        if batch.size == 0:
            return []

        BATCH_SIZE.observe(batch.size)

        # 3. Execute predictions for the batch
        t0 = time.monotonic()
        completed_ids = self._execute_batch(batch)
        STEP_DURATION.observe(time.monotonic() - t0)

        # 4. Finalize completed jobs
        for job_id in completed_ids:
            self._finalize_job(job_id)

        return completed_ids

    def run_until_done(self) -> list[RolloutResult]:
        """Run engine until all submitted jobs are complete."""
        all_completed = []
        while self.scheduler.has_work():
            completed = self.step()
            for job_id in completed:
                if job_id in self._results:
                    all_completed.append(self._results[job_id])
        return all_completed

    def get_result(self, job_id: str) -> Optional[RolloutResult]:
        return self._results.get(job_id)

    def has_pending_work(self) -> bool:
        return self.scheduler.has_work()

    def execution_stats_snapshot(self) -> dict[str, Any]:
        return self._execution_stats.snapshot()

    def reset_execution_stats(self) -> None:
        self._execution_stats = ExecutionStats(mode=self.execution_mode)

    # --- Internal ---

    def _prepare_initial_state(self, job_id: str) -> torch.Tensor:
        """Encode or materialize the initial state for one rollout."""
        job = self._jobs[job_id]

        if job.initial_latent is not None:
            initial_state = job.initial_latent.to(self.device, self.dtype)
        elif job.initial_observation is not None and self.tokenizer is not None:
            obs = job.initial_observation.to(self.device, self.dtype)
            if obs.ndim == 3:  # [C, H, W] single frame
                obs = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            elif obs.ndim == 4:  # [T, C, H, W]
                obs = obs.unsqueeze(0)  # [1, T, C, H, W]
            z_q, _ = self.tokenizer.encode(obs)
            initial_state = z_q.squeeze(0)[-1]  # last frame's tokens: [N, D]
        else:
            raise ValueError(f"Job {job_id}: must provide initial_observation or initial_latent")

        if initial_state.ndim == 2:
            initial_state = initial_state.unsqueeze(0)  # [1, N, D]
        return initial_state

    def _initialize_rollout(self, job_id: str, initial_state: torch.Tensor, *, protected_ids: set[str] | None = None) -> None:
        """Create rollout state after the initial tensor has been prepared."""
        job = self._jobs[job_id]
        self.state_manager.create(job_id, initial_state, max_steps=job.num_steps, protected_ids=protected_ids)

    def _execute_batch(self, batch: ScheduledBatch) -> list[str]:
        """Execute one prediction step for a batch of rollouts."""
        return self._execute_batch_chunked(batch)

    def _transition_signature(self, current_state: torch.Tensor, action: torch.Tensor) -> BatchSignature:
        state_shape = tuple(current_state.shape[-2:])
        action_dim = int(action.shape[-1])
        return BatchSignature(
            stage="transition",
            latent_shape=state_shape,
            action_dim=action_dim,
            dtype=str(self.dtype).replace("torch.", ""),
            device=str(self.device),
            needs_decode=False,
        )

    def _normalize_current_state(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if state_tensor.ndim == 3 and state_tensor.shape[0] == 1:
            return state_tensor.squeeze(0)
        if state_tensor.ndim != 2:
            raise ValueError(f"Expected rollout state to be [N, D] or [1, N, D], got {tuple(state_tensor.shape)}")
        return state_tensor

    def _transition_batch_policy(self, logical_batch_size: int) -> ExecutionBatchPolicy:
        return ExecutionBatchPolicy(
            mode="sync",
            max_chunk_size=max(1, logical_batch_size),
            min_ready_size=1,
            return_when_ready_count=max(1, logical_batch_size),
            allow_partial_batch=True,
        )

    def _build_transition_chunks(self, batch: ScheduledBatch) -> list[ExecutionChunk]:
        work_items: list[ExecutionWorkItem] = []
        for i, job_id in enumerate(batch.request_ids):
            step_idx = batch.step_indices[i]
            job = self._jobs[job_id]
            state = self.state_manager.get(job_id)

            if job.actions is not None and step_idx < job.actions.shape[0]:
                action = job.actions[step_idx].to(self.device, self.dtype)
            else:
                action = torch.zeros(self.config.dynamics.action_dim, device=self.device, dtype=self.dtype)

            current_state = self._normalize_current_state(state.latent_states[-1]).to(self.device, self.dtype)
            signature = self._transition_signature(current_state, action)
            entity = ExecutionEntity(
                entity_id=f"{job_id}:transition:{step_idx}",
                rollout_id=job_id,
                stage="transition",
                step_idx=step_idx,
                batch_signature=signature,
            )
            work_items.append(
                ExecutionWorkItem(
                    entity=entity,
                    latent_item=current_state,
                    action_item=action,
                )
            )
        chunks, _ = HomogeneousChunkScheduler().schedule(
            work_items=work_items,
            policy=self._transition_batch_policy(batch.size),
            chunk_id_prefix="transition",
            latent_join=lambda items: torch.stack(items, dim=0),
            action_join=lambda items: torch.stack(items, dim=0),
        )
        return chunks

    def _record_transition_chunk(self, size: int, logical_batch_size: int) -> None:
        from wm_infra.engine.metrics import BATCH_FILL_RATIO, EXECUTION_CHUNK_SIZE, EXECUTION_CHUNK_TOTAL

        EXECUTION_CHUNK_TOTAL.labels(stage="transition", mode=self.execution_mode).inc()
        EXECUTION_CHUNK_SIZE.labels(stage="transition", mode=self.execution_mode).observe(size)
        if logical_batch_size > 0:
            BATCH_FILL_RATIO.observe(size / logical_batch_size)
        self._execution_stats.record_transition_chunk(size)

    def _execute_batch_chunked(self, batch: ScheduledBatch) -> list[str]:
        """Chunked execution path that batches homogeneous transition work."""
        completed: list[str] = []
        protected_ids = set(batch.request_ids)
        transition_updates: list[tuple[ExecutionEntity, torch.Tensor, torch.Tensor]] = []
        for chunk in self._build_transition_chunks(batch):
            self._record_transition_chunk(chunk.size, batch.size)
            next_states = self.dynamics_model.predict_next(chunk.latent_batch, chunk.action_batch)
            for entity, next_state, action in zip(chunk.entities, next_states, chunk.action_batch):
                transition_updates.append((entity, action, next_state))

        tensors_to_track = []
        for _, action, next_state in transition_updates:
            tensors_to_track.extend([action, next_state])
        if tensors_to_track:
            self.state_manager.ensure_capacity_for_tensors(tensors_to_track, protected_ids=protected_ids)

        for entity, action, next_state in transition_updates:
            job_id = entity.rollout_id
            job = self._jobs[job_id]
            self.state_manager.append_step(
                job_id,
                action,
                next_state,
                protected_ids=protected_ids,
            )

            if job.step_callback is not None:
                try:
                    job.step_callback(job_id, entity.step_idx, next_state)
                except Exception:
                    logger.exception("Step callback failed for job %s step %d", job_id, entity.step_idx)

            if self.scheduler.step_completed(job_id):
                self.scheduler.complete(job_id)
                completed.append(job_id)

        return completed

    def _finalize_job(self, job_id: str) -> None:
        """Build result for a completed job."""
        job = self._jobs[job_id]
        state = self.state_manager.get(job_id)
        start_time = job.created_at

        # Stack predicted states (skip initial state)
        predicted_latents = torch.stack(state.latent_states[1:], dim=0)  # [T, N, D] or [T, B, N, D]

        result = RolloutResult(
            job_id=job_id,
            steps_completed=state.current_step,
            elapsed_ms=(time.monotonic() - start_time) * 1000,
        )

        if job.return_latents:
            result.predicted_latents = predicted_latents

        if job.return_frames and self.tokenizer is not None:
            # Decode latents back to frames
            if predicted_latents.ndim == 3:
                predicted_latents = predicted_latents.unsqueeze(0)  # [1, T, N, D]
            frames = self.tokenizer.decode(predicted_latents)
            result.predicted_frames = frames.squeeze(0)  # [T, C, H, W]

        self._results[job_id] = result

        # Cleanup state
        self.state_manager.remove(job_id)
        self._jobs.pop(job_id, None)


class AsyncWorldModelEngine:
    """Async wrapper around WorldModelEngine for non-blocking serving."""

    def __init__(
        self,
        config: EngineConfig,
        dynamics_model: nn.Module,
        tokenizer: Optional[VideoTokenizer] = None,
        execution_mode: str = "chunked",
    ):
        self.engine = WorldModelEngine(config, dynamics_model, tokenizer, execution_mode=execution_mode)
        self._queue: asyncio.Queue[tuple[RolloutJob, asyncio.Future]] = asyncio.Queue()
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._loop_task: Optional[asyncio.Task] = None
        self._shutdown = False

    def start(self) -> None:
        """Start the background engine loop. Must be called from a running event loop."""
        if self._loop_task is not None:
            return
        self._shutdown = False
        self._loop_task = asyncio.get_event_loop().create_task(self._engine_loop())
        logger.info("Async engine loop started")

    async def stop(self) -> None:
        """Stop the background engine loop gracefully."""
        self._shutdown = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        # Cancel any pending futures
        for fut in self._pending_futures.values():
            if not fut.done():
                fut.cancel()
        self._pending_futures.clear()
        logger.info("Async engine loop stopped")

    async def submit(self, job: RolloutJob) -> RolloutResult:
        """Submit a job and await its completion."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future[RolloutResult] = loop.create_future()
        await self._queue.put((job, future))
        return await future

    async def submit_stream(self, job: RolloutJob) -> AsyncIterator[tuple[int, torch.Tensor]]:
        """Submit a job and yield (step_idx, latent_state) as each step completes."""
        step_queue: asyncio.Queue[tuple[int, torch.Tensor] | None] = asyncio.Queue()

        def _on_step(job_id: str, step_idx: int, latent: torch.Tensor) -> None:
            step_queue.put_nowait((step_idx, latent))

        job.step_callback = _on_step

        loop = asyncio.get_event_loop()
        future: asyncio.Future[RolloutResult] = loop.create_future()
        await self._queue.put((job, future))

        # Yield steps as they arrive
        steps_yielded = 0
        while steps_yielded < job.num_steps:
            item = await step_queue.get()
            if item is None:
                break
            steps_yielded += 1
            yield item

        # Ensure the future is resolved (it should already be by now)
        await future

    @property
    def num_active(self) -> int:
        """Number of active rollouts in the engine."""
        return self.engine.state_manager.num_active

    @property
    def num_queued(self) -> int:
        """Number of jobs waiting in the submission queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    async def _engine_loop(self) -> None:
        """Background loop: drain queue, step engine, resolve futures."""
        from wm_infra.engine.metrics import QUEUE_DEPTH, ACTIVE_ROLLOUTS, VRAM_USED_BYTES

        logger.info("Engine loop running")
        try:
            while not self._shutdown:
                # 1. Drain the submission queue
                drained = 0
                while not self._queue.empty():
                    try:
                        job, future = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not job.job_id:
                        job.job_id = str(uuid.uuid4())
                    self.engine.submit_job(job)
                    self._pending_futures[job.job_id] = future
                    drained += 1

                # Update gauges
                QUEUE_DEPTH.set(self._queue.qsize())
                ACTIVE_ROLLOUTS.set(self.engine.state_manager.num_active)
                VRAM_USED_BYTES.set(self.engine.state_manager._current_memory)

                # 2. If no work, sleep briefly to avoid busy-spin
                if not self.engine.has_pending_work() and self._queue.empty():
                    await asyncio.sleep(0.001)
                    continue

                # 3. Run one step of the sync engine (GPU work)
                completed_ids = self.engine.step()

                # 4. Resolve futures for completed jobs
                for job_id in completed_ids:
                    result = self.engine.get_result(job_id)
                    future = self._pending_futures.pop(job_id, None)
                    if future is not None and not future.done():
                        if result is not None:
                            future.set_result(result)
                        else:
                            future.set_exception(
                                RuntimeError(f"Rollout {job_id} completed but no result")
                            )

                # 5. Yield to event loop so HTTP handlers can run
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("Engine loop cancelled")
        except Exception:
            logger.exception("Engine loop crashed")
            # Fail all pending futures
            for fut in self._pending_futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("Engine loop crashed"))
            self._pending_futures.clear()


# ---------------------------------------------------------------------------
# Public re-export list
# ---------------------------------------------------------------------------

__all__ = [
    "AsyncWorldModelEngine",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_HEIGHT",
    "DEFAULT_RESOURCE_UNITS_PER_GB",
    "DEFAULT_WIDTH",
    "HIGH_QUALITY_MEMORY_MULTIPLIER",
    "LOW_VRAM_MEMORY_MULTIPLIER",
    "LatentStateManager",
    "RolloutJob",
    "RolloutRequest",
    "RolloutResult",
    "RolloutScheduler",
    "RolloutState",
    "ScheduledBatch",
    "WorldModelEngine",
]
