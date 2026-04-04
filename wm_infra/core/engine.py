"""WorldModelEngine: orchestrates tokenization, prediction, and decoding.

The engine manages the full rollout lifecycle:
  1. Accept rollout requests (observation + action sequence)
  2. Tokenize observation into latent state
  3. Schedule prediction steps across concurrent rollouts
  4. Execute dynamics model predictions (batched)
  5. Optionally decode latent states back to pixel frames
  6. Stream results back to the client

AsyncWorldModelEngine wraps the sync engine with an asyncio-based
background loop for non-blocking concurrent request handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import torch
import torch.nn as nn

from wm_infra.config import EngineConfig
from wm_infra.core.state import LatentStateManager
from wm_infra.core.scheduler import RolloutScheduler, RolloutRequest, ScheduledBatch
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
    ):
        self.config = config
        self.dynamics_model = dynamics_model
        self.tokenizer = tokenizer

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
            max_concurrent=config.state_cache.max_batch_size,
            max_memory_gb=config.state_cache.pool_size_gb,
            device=self.device,
        )

        # Scheduling
        self.scheduler = RolloutScheduler(config.scheduler)

        # Job tracking
        self._jobs: dict[str, RolloutJob] = {}
        self._results: dict[str, RolloutResult] = {}

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
        """Run one engine step: schedule + execute one batch of predictions.

        Returns:
            List of job IDs that completed during this step
        """
        # 1. Admit pending jobs and encode initial states
        admitted = self.scheduler.admit()
        for job_id in admitted:
            self._initialize_rollout(job_id)

        # 2. Schedule a batch
        batch = self.scheduler.schedule_batch()
        if batch.size == 0:
            return []

        # 3. Execute predictions for the batch
        completed_ids = self._execute_batch(batch)

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

    # ─── Internal ───

    def _initialize_rollout(self, job_id: str) -> None:
        """Encode initial observation and create rollout state."""
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

        self.state_manager.create(job_id, initial_state, max_steps=job.num_steps)

    def _execute_batch(self, batch: ScheduledBatch) -> list[str]:
        """Execute one prediction step for a batch of rollouts."""
        completed = []

        for i, job_id in enumerate(batch.request_ids):
            step_idx = batch.step_indices[i]
            job = self._jobs[job_id]
            state = self.state_manager.get(job_id)

            # Get action for this step
            if job.actions is not None and step_idx < job.actions.shape[0]:
                action = job.actions[step_idx].unsqueeze(0).to(self.device, self.dtype)
            else:
                # Zero action if not provided
                action_dim = self.config.dynamics.action_dim
                action = torch.zeros(1, action_dim, device=self.device, dtype=self.dtype)

            # Get current latent state
            current_state = state.latent_states[-1]
            if current_state.ndim == 2:
                current_state = current_state.unsqueeze(0)

            # Predict next state
            next_state = self.dynamics_model.predict_next(current_state, action)

            # Update state
            self.state_manager.append_step(job_id, action.squeeze(0), next_state.squeeze(0))

            # Per-step callback for streaming
            if job.step_callback is not None:
                try:
                    job.step_callback(job_id, step_idx, next_state.squeeze(0))
                except Exception:
                    logger.exception("Step callback failed for job %s step %d", job_id, step_idx)

            # Check completion
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
    """Async wrapper around WorldModelEngine for non-blocking serving.

    Runs a background asyncio task that continuously:
      1. Drains the submission queue for new jobs
      2. Calls the sync engine's step() for one batch of GPU work
      3. Resolves futures for completed jobs
      4. Yields back to the event loop

    This allows multiple concurrent HTTP requests to submit jobs and
    await results without blocking each other or the event loop.
    """

    def __init__(
        self,
        config: EngineConfig,
        dynamics_model: nn.Module,
        tokenizer: Optional[VideoTokenizer] = None,
    ):
        self.engine = WorldModelEngine(config, dynamics_model, tokenizer)
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
        """Submit a job and await its completion.

        Returns:
            RolloutResult when the job finishes.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[RolloutResult] = loop.create_future()
        await self._queue.put((job, future))
        return await future

    async def submit_stream(self, job: RolloutJob) -> AsyncIterator[tuple[int, torch.Tensor]]:
        """Submit a job and yield (step_idx, latent_state) as each step completes.

        The final RolloutResult is still available via engine.get_result() after
        the iterator is exhausted.
        """
        step_queue: asyncio.Queue[tuple[int, torch.Tensor] | None] = asyncio.Queue()

        def _on_step(job_id: str, step_idx: int, latent: torch.Tensor) -> None:
            # Called from the sync engine step — safe because engine loop is
            # the only writer and it runs in the same thread as the event loop.
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
        logger.info("Engine loop running")
        try:
            while not self._shutdown:
                # 1. Drain the submission queue — admit new jobs immediately
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
