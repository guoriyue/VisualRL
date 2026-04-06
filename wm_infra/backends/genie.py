"""State-aware rollout backend for Genie (STMaskGIT) temporal episodes."""

from __future__ import annotations

import base64
import hashlib
import io
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.backends.genie_batcher import GenieTransitionBatcher
from wm_infra.backends.genie_checkpoint import build_checkpoint_delta, checkpoint_due, persist_checkpoint_delta
from wm_infra.backends.genie_persist import build_runtime_status_history, write_json, write_log
from wm_infra.backends.genie_runner import GenieRunResult, GenieRunner
from wm_infra.backends.genie_runtime import (
    GENIE_STAGE_GRAPH,
    GenieExecutionEntity,
    GeniePromptStateCache,
    GenieQueueLane,
    GenieResidencyTier,
    GenieRuntimeState,
    build_transition_entities,
    make_stage_signature,
    prompt_cache_key,
)
from wm_infra.backends.genie_scheduler import GenieScheduler
from wm_infra.controlplane import (
    ArtifactKind,
    ArtifactRecord,
    CheckpointCreate,
    GenieTaskConfig,
    ProduceSampleRequest,
    RolloutCreate,
    RolloutTaskConfig,
    SampleRecord,
    SampleStatus,
    StateHandleCreate,
    StateHandleKind,
    TaskType,
    TemporalRefs,
    TemporalStatus,
    TemporalStore,
    TokenInputSource,
    TokenizerFamily,
    TokenizerKind,
    estimate_rollout_request,
)


class GenieRolloutBackend(ProduceSampleBackend):
    """Genie rollout backend with stage-aware execution and persisted artifacts."""

    def __init__(
        self,
        temporal_store: TemporalStore,
        output_root: str | Path | None = None,
        backend_name: str = "genie-rollout",
        runner: GenieRunner | None = None,
    ) -> None:
        self.temporal_store = temporal_store
        self.backend_name = backend_name
        self._runner = runner or GenieRunner()
        self._job_queue = None
        self._prompt_state_cache = GeniePromptStateCache()
        self._transition_batcher = GenieTransitionBatcher(self._runner)

        if output_root is not None:
            self.output_root: Path | None = Path(output_root)
            self.output_root.mkdir(parents=True, exist_ok=True)
        else:
            self.output_root = None

    def ensure_runner_loaded(self) -> str:
        return self._runner.load()

    @property
    def runner_mode(self) -> str:
        return self._runner.mode

    @property
    def runner(self) -> GenieRunner:
        return self._runner

    def _sample_dir(self, sample_id: str) -> Path:
        if self.output_root is not None:
            path = self.output_root / sample_id
        else:
            import tempfile

            path = Path(tempfile.gettempdir()) / "wm_infra_genie" / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _file_details(path: Path) -> tuple[int | None, str | None]:
        if not path.exists() or not path.is_file():
            return None, None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return path.stat().st_size, digest.hexdigest()

    def _artifact_record(
        self,
        *,
        artifact_id: str,
        kind: ArtifactKind,
        path: Path,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        bytes_size, sha256 = self._file_details(path)
        payload: dict[str, Any] = {"exists": path.exists()}
        if metadata:
            payload.update(metadata)
        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            uri=f"file://{path}",
            mime_type=mime_type,
            bytes=bytes_size,
            sha256=sha256,
            metadata=payload,
        )

    def _effective_task_config(self, request: ProduceSampleRequest) -> RolloutTaskConfig:
        task_config = request.task_config.model_copy(deep=True) if request.task_config is not None else RolloutTaskConfig()
        genie_config = request.genie_config
        if genie_config is not None:
            task_config.frame_count = genie_config.num_frames
        return task_config

    def _effective_genie_config(self, request: ProduceSampleRequest, task_config: RolloutTaskConfig) -> GenieTaskConfig:
        genie_config = request.genie_config.model_copy(deep=True) if request.genie_config is not None else GenieTaskConfig()
        if request.genie_config is None:
            genie_config.num_frames = task_config.frame_count or genie_config.num_frames
        return genie_config

    def _resolve_genie_config_tokens(
        self,
        genie_config: GenieTaskConfig,
        sample_dir: Path,
    ) -> tuple[np.ndarray, dict[str, Any], list[ArtifactRecord]]:
        try:
            encoded = base64.b64decode(genie_config.input_tokens_b64.encode("utf-8"))
        except Exception as exc:
            raise ValueError("genie_config.input_tokens_b64 must be valid base64") from exc

        try:
            raw = np.load(io.BytesIO(encoded), allow_pickle=False).astype(np.uint32)
        except Exception as exc:
            raise ValueError("genie_config.input_tokens_b64 must decode to a .npy uint32 tensor") from exc

        if raw.ndim != 3:
            raise ValueError("genie_config.input_tokens_b64 must decode to a 3D [T,H,W] uint32 tensor")

        input_path = sample_dir / "input_tokens.npy"
        np.save(str(input_path), raw)
        artifact = self._artifact_record(
            artifact_id=f"{sample_dir.name}:input-tokens",
            kind=ArtifactKind.LATENT,
            path=input_path,
            mime_type="application/octet-stream",
            metadata={
                "role": "input_tokens",
                "format": "numpy",
                "shape": list(raw.shape),
                "dtype": "uint32",
                "source": "genie_config_b64",
                "tokenizer_kind": genie_config.tokenizer_kind.value,
            },
        )
        scaffold = {
            "token_input_mode": "genie_config_b64",
            "resolved_shape": list(raw.shape),
            "resolved_path": str(input_path),
            "tokenizer_kind": genie_config.tokenizer_kind.value,
        }
        return raw, scaffold, [artifact]

    def _resolve_input_tokens(self, request: ProduceSampleRequest, sample_dir: Path) -> tuple[np.ndarray | None, dict[str, Any], list[ArtifactRecord]]:
        token_input = request.token_input
        if token_input is None:
            genie_config = request.genie_config
            if genie_config is not None and genie_config.input_tokens_b64:
                return self._resolve_genie_config_tokens(genie_config, sample_dir)
            return None, {"token_input_mode": "none"}, []

        scaffold = {
            "tokenizer_family": token_input.tokenizer_family.value,
            "tokenizer_name": token_input.tokenizer_name,
            "source": token_input.source.value,
            "layout": token_input.layout,
            "dtype": token_input.dtype,
            "shape": token_input.shape,
            "metadata": token_input.metadata,
        }
        artifacts: list[ArtifactRecord] = []

        if token_input.dtype != "uint32":
            raise ValueError("Genie token_input currently only supports dtype=uint32")

        input_path = sample_dir / "input_tokens.npy"
        if token_input.source == TokenInputSource.INLINE:
            if not token_input.inline_tokens:
                raise ValueError("token_input.inline_tokens is required for source=inline")
            raw = np.asarray(token_input.inline_tokens, dtype=np.uint32)
        else:
            if not token_input.uri:
                raise ValueError("token_input.uri is required for source=uri")
            if not token_input.uri.startswith("file://"):
                raise ValueError("Genie token_input currently only supports file:// URIs")
            raw = np.load(token_input.uri[7:]).astype(np.uint32)

        if token_input.layout == "flat":
            if len(token_input.shape) != 3:
                raise ValueError("Flat token_input requires shape=[T,H,W]")
            raw = raw.reshape(token_input.shape)
        elif raw.ndim != 3:
            if len(token_input.shape) == 3:
                raw = raw.reshape(token_input.shape)
            else:
                raise ValueError("Genie token_input must resolve to a 3D [T,H,W] token tensor")

        np.save(str(input_path), raw)
        artifacts.append(
            self._artifact_record(
                artifact_id=f"{sample_dir.name}:input-tokens",
                kind=ArtifactKind.LATENT,
                path=input_path,
                mime_type="application/octet-stream",
                metadata={
                    "role": "input_tokens",
                    "format": "numpy",
                    "shape": list(raw.shape),
                    "dtype": "uint32",
                    "tokenizer_family": token_input.tokenizer_family.value,
                },
            )
        )
        scaffold.update(
            {
                "token_input_mode": "raw_tokens",
                "resolved_shape": list(raw.shape),
                "resolved_path": str(input_path),
                "magvit2_scaffold": token_input.tokenizer_family == TokenizerFamily.MAGVIT2,
            }
        )
        return raw, scaffold, artifacts

    def _validate_real_mode_request(
        self,
        *,
        request: ProduceSampleRequest,
        genie_config: GenieTaskConfig,
        input_tokens: np.ndarray | None,
    ) -> None:
        model = self.runner._model
        if model is None:
            raise ValueError("Genie real mode requested but model is not loaded")

        max_frames = int(model.config.T)
        expected_hw = (int(model.h), int(model.w))
        max_token_id = int(model.config.image_vocab_size) - 1

        if genie_config.tokenizer_kind != TokenizerKind.GENIE_STMASKGIT:
            raise ValueError("Genie real mode currently only supports genie_config.tokenizer_kind=genie_stmaskgit")

        if genie_config.num_frames > max_frames:
            raise ValueError(f"Genie real mode only supports num_frames <= {max_frames}; got {genie_config.num_frames}")

        if genie_config.num_prompt_frames >= genie_config.num_frames:
            raise ValueError("Genie real mode requires num_prompt_frames < num_frames")

        if request.token_input is not None and request.token_input.tokenizer_family != TokenizerFamily.RAW:
            raise ValueError(
                "Genie real mode currently only supports token_input.tokenizer_family=raw; tokenizer_family=magvit2 remains scaffold-only"
            )

        if input_tokens is None:
            return

        if input_tokens.ndim != 3:
            raise ValueError("Genie real mode token input must resolve to a 3D [T,H,W] tensor")
        if tuple(input_tokens.shape[1:]) != expected_hw:
            raise ValueError(
                f"Genie real mode token input must have spatial shape [T,{expected_hw[0]},{expected_hw[1]}], got {list(input_tokens.shape)}"
            )
        if input_tokens.shape[0] > max_frames:
            raise ValueError(f"Genie real mode token input only supports T <= {max_frames}; got {input_tokens.shape[0]}")

        max_seen = int(input_tokens.max())
        min_seen = int(input_tokens.min())
        if min_seen < 0 or max_seen > max_token_id:
            raise ValueError(f"Genie real mode token ids must be in [0,{max_token_id}], got [{min_seen},{max_seen}]")

    async def execute_job(self, request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        if request.task_type not in {TaskType.GENIE_ROLLOUT, TaskType.WORLD_MODEL_ROLLOUT}:
            raise ValueError(f"Backend {self.backend_name} only supports rollout-style temporal tasks")

        from wm_infra.api.metrics import (
            GENIE_CHECKPOINT_BUILD_SECONDS,
            GENIE_CHECKPOINT_DELTA_BYTES,
            GENIE_CHUNK_FILL_RATIO,
            GENIE_CHUNK_SIZE,
            GENIE_GPU_OCCUPANCY_ESTIMATE,
            GENIE_PERSIST_BACKLOG,
            GENIE_PROMPT_REUSE_EVENTS,
            GENIE_RESIDENCY_EVENTS,
            GENIE_STAGE_DURATION,
            GENIE_STATE_MATERIALIZE_BYTES,
            GENIE_STATE_MATERIALIZE_SECONDS,
            GENIE_TRANSITION_FRAMES_TOTAL,
            GENIE_TRANSITION_TOKENS_TOTAL,
        )

        total_start = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        stage_history: list[dict[str, Any]] = []

        def finish_stage(stage: str, started_at: float, lane: str, runner_mode: str, extra: dict[str, Any] | None = None) -> float:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
            stage_timings_ms[f"{stage}_ms"] = elapsed_ms
            GENIE_STAGE_DURATION.labels(stage=stage, lane=lane, runner_mode=runner_mode).observe(elapsed_ms / 1000.0)
            entry = {"stage": stage, "elapsed_ms": elapsed_ms, "queue_lane": lane, "runner_mode": runner_mode}
            if extra:
                entry.update(extra)
            stage_history.append(entry)
            return elapsed_ms

        temporal = request.temporal
        if temporal is None or not temporal.episode_id:
            raise ValueError("genie-rollout requests require temporal.episode_id")

        episode = self.temporal_store.episodes.get(temporal.episode_id)
        if episode is None:
            raise ValueError(f"Unknown episode_id: {temporal.episode_id}")

        runner_load_start = time.perf_counter()
        self.ensure_runner_loaded()
        stage_timings_ms["runner_load_ms"] = round((time.perf_counter() - runner_load_start) * 1000.0, 3)

        task_config = self._effective_task_config(request)
        genie_config = self._effective_genie_config(request, task_config)
        estimate = estimate_rollout_request(task_config)
        step_count = task_config.num_steps if task_config is not None else 1
        sample_dir = self._sample_dir(sample_id)
        request_path = sample_dir / "request.json"
        log_path = sample_dir / "runner.log"
        runtime_path = sample_dir / "runtime.json"
        checkpoint_path = sample_dir / "checkpoint.json"
        recovery_path = sample_dir / "recovery.json"
        seed = request.sample_spec.seed or 42
        queue_lane = GenieQueueLane.HOT_CONTINUATION.value if temporal.state_handle_id else GenieQueueLane.COLD_MATERIALIZE.value

        admission_start = time.perf_counter()
        finish_stage(
            "admission",
            admission_start,
            GenieQueueLane.COLD_MATERIALIZE.value,
            self.runner_mode,
            extra={"sample_id": sample_id, "episode_id": episode.episode_id},
        )

        state_materialize_start = time.perf_counter()
        input_tokens, token_input_runtime, input_artifacts = self._resolve_input_tokens(request, sample_dir)
        cache_key = prompt_cache_key(
            input_state_handle_id=temporal.state_handle_id,
            input_tokens=input_tokens,
            prompt=request.sample_spec.prompt or "",
            seed=seed,
        )
        cached_prompt_state = self._prompt_state_cache.get(cache_key)
        prompt_reuse_hit = cached_prompt_state is not None
        if prompt_reuse_hit and input_tokens is None and cached_prompt_state is not None:
            input_tokens = cached_prompt_state.tokens.copy()
            token_input_runtime = {
                **token_input_runtime,
                "token_input_mode": "prompt_cache",
                "cache_key": cache_key,
                "source_state_handle_id": cached_prompt_state.source_state_handle_id,
            }
        resident_tier = (
            cached_prompt_state.resident_tier
            if cached_prompt_state is not None
            else (GenieResidencyTier.WARM_PINNED_CPU if input_tokens is not None else GenieResidencyTier.COLD_FILE)
        )
        materialized_bytes = int(input_tokens.nbytes) if input_tokens is not None else 0
        GENIE_STATE_MATERIALIZE_BYTES.observe(materialized_bytes)
        GENIE_STATE_MATERIALIZE_SECONDS.observe(time.perf_counter() - state_materialize_start)
        GENIE_PROMPT_REUSE_EVENTS.labels(outcome="hit" if prompt_reuse_hit else "miss").inc()
        GENIE_RESIDENCY_EVENTS.labels(tier=resident_tier.value).inc()
        finish_stage(
            "state_materialize",
            state_materialize_start,
            GenieQueueLane.COLD_MATERIALIZE.value,
            self.runner_mode,
            extra={"materialized_bytes": materialized_bytes, "resident_tier": resident_tier.value},
        )

        prompt_prepare_start = time.perf_counter()
        if self.runner_mode == "real":
            self._validate_real_mode_request(
                request=request,
                genie_config=genie_config,
                input_tokens=input_tokens,
            )
        prepared = self.runner.prepare_inputs(
            prompt=request.sample_spec.prompt or "",
            seed=seed,
            num_frames=genie_config.num_frames,
            input_tokens=input_tokens,
            num_prompt_frames=genie_config.num_prompt_frames,
            maskgit_steps=genie_config.maskgit_steps,
            temperature=genie_config.temperature,
        )
        if prepared.mode == "real":
            resident_tier = GenieResidencyTier.HOT_GPU
            GENIE_RESIDENCY_EVENTS.labels(tier=resident_tier.value).inc()
        prompt_tokens = prepared.current_tokens_numpy()[: prepared.prompt_frames] if prepared.prompt_frames > 0 else prepared.current_tokens_numpy()[:1]
        self._prompt_state_cache.put(
            cache_key,
            prompt_tokens,
            source_state_handle_id=temporal.state_handle_id,
            resident_tier=resident_tier,
        )
        finish_stage(
            "prompt_prepare",
            prompt_prepare_start,
            queue_lane,
            prepared.mode,
            extra={"prompt_frames": prepared.prompt_frames, "total_frames": prepared.total_frames},
        )
        stage_timings_ms["state_token_prep_ms"] = round(
            stage_timings_ms["state_materialize_ms"] + stage_timings_ms["prompt_prepare_ms"],
            3,
        )

        request_payload = {
            "sample_id": sample_id,
            "task_type": request.task_type.value,
            "backend": request.backend,
            "model": request.model,
            "model_revision": request.model_revision,
            "sample_spec": request.sample_spec.model_dump(mode="json"),
            "temporal": temporal.model_dump(mode="json"),
            "token_input": request.token_input.model_dump(mode="json") if request.token_input else None,
            "task_config": task_config.model_dump(mode="json") if task_config else None,
            "genie_config": genie_config.model_dump(mode="json"),
            "runner_mode": prepared.mode,
            "stage_graph": GENIE_STAGE_GRAPH,
        }
        write_json(request_path, request_payload)

        started_at = time.time()
        rollout = self.temporal_store.create_rollout(
            RolloutCreate(
                episode_id=episode.episode_id,
                branch_id=temporal.branch_id,
                backend=self.backend_name,
                model=request.model,
                sample_id=sample_id,
                request_id=sample_id,
                input_state_handle_id=temporal.state_handle_id,
                step_count=step_count,
                priority=request.priority,
                metadata={
                    "runner_mode": prepared.mode,
                    "prompt": request.sample_spec.prompt,
                    "controls": request.sample_spec.controls,
                    "token_input": token_input_runtime,
                    "genie_config": genie_config.model_dump(mode="json"),
                    "stage_graph": GENIE_STAGE_GRAPH,
                },
            ),
            status=TemporalStatus.ACTIVE,
        )

        runtime_state = GenieRuntimeState(
            rollout_id=rollout.rollout_id,
            prompt_tokens_ref=cache_key,
            generated_tokens_ref=None,
            last_completed_frame=prepared.prompt_frames,
            resident_tier=resident_tier,
            ancestor_state_ref=temporal.state_handle_id,
            checkpoint_delta_ref=None,
            materialized_bytes=materialized_bytes,
            dirty_since_checkpoint=False,
            prompt_reuse_hit=prompt_reuse_hit,
            source_cache_key=cache_key,
            reuse_hits=1 if prompt_reuse_hit else 0,
            reuse_misses=0 if prompt_reuse_hit else 1,
        )

        root_signature = make_stage_signature(
            backend=self.backend_name,
            model_name=request.model,
            stage="transition",
            device=prepared.device,
            dtype=prepared.dtype,
            tokenizer_kind=genie_config.tokenizer_kind.value,
            spatial_h=prepared.spatial_h,
            spatial_w=prepared.spatial_w,
            window_num_frames=max(prepared.total_frames - prepared.prompt_frames, 0),
            num_prompt_frames=prepared.prompt_frames,
            maskgit_steps=genie_config.maskgit_steps,
            temperature=genie_config.temperature,
            checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
            runner_mode=prepared.mode,
            needs_persist=False,
        )
        root_entity = GenieExecutionEntity(
            entity_id=f"{sample_id}:root",
            rollout_id=rollout.rollout_id,
            episode_id=episode.episode_id,
            branch_id=temporal.branch_id,
            sample_id=sample_id,
            input_state_handle_id=temporal.state_handle_id,
            current_stage="transition",
            next_stage="artifact_persist",
            window_start_frame=prepared.prompt_frames,
            window_num_frames=max(prepared.total_frames - prepared.prompt_frames, 0),
            total_frames=prepared.total_frames,
            num_prompt_frames=prepared.prompt_frames,
            checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
            priority=request.priority,
            deadline_s=None,
            batch_signature=root_signature,
            queue_lane=queue_lane,
        )

        transition_entities = build_transition_entities(root_entity)
        scheduler = GenieScheduler(max_chunk_size=max(1, len(transition_entities) or 1))
        decisions = scheduler.schedule(
            transition_entities,
            persist_backlog=0,
            prompt_state_hot=runtime_state.resident_tier == GenieResidencyTier.HOT_GPU,
            estimated_transfer_bytes=runtime_state.materialized_bytes,
        )
        chunks = [decision.chunk for decision in decisions]
        scheduler_inputs_history = [
            {**decision.scheduler_inputs, "chunk_id": decision.chunk.chunk_id, "frame_range": list(decision.chunk.frame_range)}
            for decision in decisions
        ]

        GENIE_PERSIST_BACKLOG.set(0)
        checkpoint_deltas: list[Any] = []
        intermediate_checkpoints = []
        transition_total_ms = 0.0
        checkpoint_total_ms = 0.0
        observed_cross_request_batch_sizes: list[int] = []
        for decision in decisions:
            chunk = decision.chunk
            transition_outcome = await self._transition_batcher.run_transition(
                sample_id=sample_id,
                prepared=prepared,
                chunk=chunk,
            )
            window_result = transition_outcome.window_result
            transition_elapsed_ms = transition_outcome.elapsed_ms
            transition_total_ms += transition_elapsed_ms
            observed_cross_request_batch_sizes.append(transition_outcome.batch_size)
            cross_request_fill_ratio = min(
                max(transition_outcome.batch_size / max(self._transition_batcher.max_batch_size, 1), 0.0),
                1.0,
            )
            GENIE_CHUNK_SIZE.labels(stage=chunk.runnable_stage, lane=chunk.queue_lane).observe(transition_outcome.batch_size)
            GENIE_CHUNK_FILL_RATIO.labels(stage=chunk.runnable_stage, lane=chunk.queue_lane).observe(cross_request_fill_ratio)
            GENIE_GPU_OCCUPANCY_ESTIMATE.set(max(chunk.expected_occupancy, cross_request_fill_ratio))
            if window_result.frames_generated > 0:
                GENIE_TRANSITION_FRAMES_TOTAL.labels(runner_mode=prepared.mode).inc(window_result.frames_generated)
                GENIE_TRANSITION_TOKENS_TOTAL.labels(runner_mode=prepared.mode).inc(
                    window_result.frames_generated * prepared.spatial_h * prepared.spatial_w
                )
            runtime_state.generated_tokens_ref = chunk.chunk_id
            runtime_state.last_completed_frame = max(runtime_state.last_completed_frame, window_result.frame_end)
            runtime_state.dirty_since_checkpoint = True
            stage_history.append(
                {
                    "stage": "transition",
                    "elapsed_ms": transition_elapsed_ms,
                    "queue_lane": chunk.queue_lane,
                    "runner_mode": prepared.mode,
                    "chunk_id": chunk.chunk_id,
                    "chunk_size": transition_outcome.batch_size,
                    "frame_range": list(chunk.frame_range),
                    "expected_occupancy": max(chunk.expected_occupancy, cross_request_fill_ratio),
                    "cross_request_batch_id": transition_outcome.batch_id,
                    "cross_request_sample_ids": transition_outcome.sample_ids,
                    "scheduler_inputs": decision.scheduler_inputs,
                }
            )

            if checkpoint_due(
                frame_end=window_result.frame_end,
                total_frames=prepared.total_frames,
                checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
            ):
                checkpoint_start = time.perf_counter()
                delta = build_checkpoint_delta(
                    rollout_id=rollout.rollout_id,
                    sample_id=sample_id,
                    parent_state_handle_id=temporal.state_handle_id,
                    all_tokens=prepared.current_tokens_numpy(),
                    start_frame=window_result.frame_start,
                    end_frame=window_result.frame_end,
                    checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
                    runner_mode=prepared.mode,
                )
                persist_checkpoint_delta(sample_dir, delta)
                checkpoint_deltas.append(delta)
                runtime_state.checkpoint_delta_ref = delta.artifact_id
                runtime_state.dirty_since_checkpoint = False
                GENIE_CHECKPOINT_DELTA_BYTES.observe(delta.bytes_size)
                GENIE_CHECKPOINT_BUILD_SECONDS.observe(time.perf_counter() - checkpoint_start)
                checkpoint_elapsed_ms = round((time.perf_counter() - checkpoint_start) * 1000.0, 3)
                checkpoint_total_ms += checkpoint_elapsed_ms
                stage_history.append(
                    {
                        "stage": "checkpoint",
                        "elapsed_ms": checkpoint_elapsed_ms,
                        "queue_lane": GenieQueueLane.CHECKPOINT_HEAVY.value,
                        "runner_mode": prepared.mode,
                        "artifact_id": delta.artifact_id,
                        "frame_end": delta.end_frame,
                    }
                )
                intermediate_checkpoints.append(
                    self.temporal_store.create_checkpoint(
                        CheckpointCreate(
                            episode_id=episode.episode_id,
                            rollout_id=rollout.rollout_id,
                            branch_id=temporal.branch_id,
                            artifact_ids=[delta.artifact_id] if delta.artifact_id else [],
                            step_index=max(delta.end_frame - 1, 0),
                            tag=f"frame-{delta.end_frame}",
                            metadata={**delta.metadata, "kind": "delta"},
                        )
                    )
                )

        stage_timings_ms["runner_exec_ms"] = round(transition_total_ms, 3)
        stage_timings_ms["transition_ms"] = round(transition_total_ms, 3)
        stage_timings_ms["checkpoint_ms"] = round(checkpoint_total_ms, 3)

        persist_start = time.perf_counter()
        run_result: GenieRunResult = self.runner.persist_outputs(prepared, output_dir=sample_dir)
        mode = run_result.mode
        request_payload["runner_mode"] = mode
        if run_result.extra.get("fallback_from"):
            request_payload["fallback_from"] = run_result.extra["fallback_from"]
            request_payload["fallback_error"] = run_result.extra.get("fallback_error")
        write_json(request_path, request_payload)

        status = SampleStatus.SUCCEEDED
        error_info: str | None = None
        if run_result.error:
            status = SampleStatus.FAILED
            error_info = run_result.error

        log_lines = [
            f"Genie stage runtime: {' -> '.join(GENIE_STAGE_GRAPH)}",
            f"Genie runner mode: {mode}",
            f"Model: {run_result.model_name}",
            f"Device: {run_result.device}",
            f"Requested frames: {genie_config.num_frames}",
            f"Prompt frames: {genie_config.num_prompt_frames}",
            f"MaskGIT steps: {genie_config.maskgit_steps}",
            f"Temperature: {genie_config.temperature}",
            f"Frames generated: {run_result.frames_generated}/{run_result.total_frames}",
            f"Tokens generated: {run_result.tokens_generated}",
            f"Elapsed: {run_result.elapsed_s:.3f}s",
            f"Token input: {token_input_runtime.get('token_input_mode')}",
            f"Transition chunks: {len(chunks)}",
        ]
        if token_input_runtime.get("magvit2_scaffold"):
            log_lines.append(
                "MAGVIT2 scaffold present: raw token path accepted; native MAGVIT2 tokenization is not implemented yet."
            )
        if run_result.error:
            log_lines.append(f"ERROR: {run_result.error}")
        if run_result.extra.get("fallback_from"):
            log_lines.append(f"FALLBACK: {run_result.extra['fallback_from']} -> {mode}")
        write_log(log_path, log_lines)

        state_uri = f"file://{run_result.tokens_path}" if run_result.tokens_path else None
        output_state = self.temporal_store.create_state_handle(
            StateHandleCreate(
                episode_id=episode.episode_id,
                branch_id=temporal.branch_id,
                rollout_id=rollout.rollout_id,
                kind=StateHandleKind.VIDEO_LATENT,
                uri=state_uri,
                shape=[run_result.total_frames, run_result.spatial_h, run_result.spatial_w],
                dtype="uint32",
                artifact_ids=[
                    f"{sample_id}:tokens",
                    f"{sample_id}:state",
                    f"{sample_id}:checkpoint",
                    f"{sample_id}:recovery",
                    *[delta.artifact_id for delta in checkpoint_deltas if delta.artifact_id],
                ],
                metadata={
                    "runner_mode": mode,
                    "source_backend": self.backend_name,
                    "prompt": request.sample_spec.prompt,
                    "tokens_generated": run_result.tokens_generated,
                    "frames_generated": run_result.frames_generated,
                    "sample_id": sample_id,
                    "token_input": token_input_runtime,
                    "genie_config": genie_config.model_dump(mode="json"),
                    "stage_graph": GENIE_STAGE_GRAPH,
                },
            )
        )

        completed_at = time.time()
        checkpoint_payload = {
            "sample_id": sample_id,
            "episode_id": episode.episode_id,
            "branch_id": temporal.branch_id,
            "rollout_id": rollout.rollout_id,
            "parent_state_handle_id": temporal.state_handle_id,
            "output_state_handle_id": output_state.state_handle_id,
            "tokens_path": run_result.tokens_path,
            "state_path": run_result.state_path,
            "log_path": str(log_path),
            "request_path": str(request_path),
            "runner_mode": mode,
            "status": status.value,
            "token_input": token_input_runtime,
            "genie_config": genie_config.model_dump(mode="json"),
            "checkpoint_deltas": [delta.metadata for delta in checkpoint_deltas],
            "timestamps": {"started_at": started_at, "completed_at": completed_at},
        }
        write_json(checkpoint_path, checkpoint_payload)
        write_json(
            recovery_path,
            {
                **checkpoint_payload,
                "recovery_hint": "Reload tokens_path into token_input or fork from checkpoint/state_handle for retry.",
            },
        )

        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=episode.episode_id,
                rollout_id=rollout.rollout_id,
                branch_id=temporal.branch_id,
                state_handle_id=output_state.state_handle_id,
                artifact_ids=[
                    f"{sample_id}:checkpoint",
                    f"{sample_id}:recovery",
                    f"{sample_id}:tokens",
                    f"{sample_id}:state",
                    *[delta.artifact_id for delta in checkpoint_deltas if delta.artifact_id],
                ],
                step_index=max(step_count - 1, 0),
                tag="terminal",
                metadata={
                    "runner_mode": mode,
                    "frames_generated": run_result.frames_generated,
                    "checkpoint_path": str(checkpoint_path),
                    "recovery_path": str(recovery_path),
                    "sample_id": sample_id,
                    "genie_config": genie_config.model_dump(mode="json"),
                    "checkpoint_deltas": [delta.metadata for delta in checkpoint_deltas],
                },
            )
        )

        output_state.checkpoint_id = checkpoint.checkpoint_id
        output_state.artifact_ids = list(
            dict.fromkeys(
                [
                    *output_state.artifact_ids,
                    f"{sample_id}:request",
                    f"{sample_id}:runtime",
                    f"{sample_id}:log",
                ]
            )
        )
        output_state.metadata["checkpoint_id"] = checkpoint.checkpoint_id
        output_state.metadata["checkpoint_path"] = str(checkpoint_path)
        output_state.metadata["recovery_path"] = str(recovery_path)
        self.temporal_store.state_handles.put(output_state)
        stage_timings_ms["artifact_persist_ms"] = round((time.perf_counter() - persist_start) * 1000.0, 3)
        stage_history.append(
            {
                "stage": "artifact_persist",
                "elapsed_ms": stage_timings_ms["artifact_persist_ms"],
                "queue_lane": GenieQueueLane.PERSIST_ONLY.value,
                "runner_mode": mode,
                "checkpoint_delta_count": len(checkpoint_deltas),
            }
        )

        temporal_persist_start = time.perf_counter()
        episode.updated_at = time.time()
        self.temporal_store.episodes.put(episode)
        stage_timings_ms["temporal_persist_ms"] = round((time.perf_counter() - temporal_persist_start) * 1000.0, 3)
        stage_timings_ms["controlplane_commit_ms"] = stage_timings_ms["temporal_persist_ms"]
        stage_history.append(
            {
                "stage": "controlplane_commit",
                "elapsed_ms": stage_timings_ms["controlplane_commit_ms"],
                "queue_lane": GenieQueueLane.PERSIST_ONLY.value,
                "runner_mode": mode,
                "checkpoint_count": len(intermediate_checkpoints) + 1,
            }
        )

        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 3)
        stage_timings_ms["total_elapsed_ms"] = total_elapsed_ms

        runtime: dict[str, Any] = {
            "runner": f"genie-{mode}",
            "runner_mode": mode,
            "temporal": True,
            "async": False,
            "rollout_id": rollout.rollout_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "state_handle_id": output_state.state_handle_id,
            "model": run_result.model_name,
            "device": run_result.device,
            "frames_generated": run_result.frames_generated,
            "prompt_frames": run_result.prompt_frames,
            "total_frames": run_result.total_frames,
            "tokens_generated": run_result.tokens_generated,
            "elapsed_s": run_result.elapsed_s,
            "request_path": str(request_path),
            "log_path": str(log_path),
            "runtime_path": str(runtime_path),
            "checkpoint_path": str(checkpoint_path),
            "recovery_path": str(recovery_path),
            "tokens_path": run_result.tokens_path,
            "state_path": run_result.state_path,
            "token_input": token_input_runtime,
            "genie_config": genie_config.model_dump(mode="json"),
            "status_history": build_runtime_status_history(
                started_at=started_at,
                completed_at=completed_at,
                terminal_status=status.value,
            ),
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_ms": round((completed_at - started_at) * 1000, 2),
            "stage_timings_ms": stage_timings_ms,
            "stage_graph": GENIE_STAGE_GRAPH,
            "stage_history": stage_history,
            "scheduler": {
                "transition_entities": len(transition_entities),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "queue_lane": chunk.queue_lane,
                        "frame_range": list(chunk.frame_range),
                        "chunk_size": chunk.size,
                        "expected_occupancy": chunk.expected_occupancy,
                    }
                    for chunk in chunks
                ],
                "scheduler_inputs": scheduler_inputs_history,
                "cross_request_batcher": self._transition_batcher.snapshot(),
                "observed_batch_sizes": observed_cross_request_batch_sizes,
            },
            "queue_lane": queue_lane,
            "runtime_state": {
                "resident_tier": runtime_state.resident_tier.value,
                "materialized_bytes": runtime_state.materialized_bytes,
                "reuse_hits": runtime_state.reuse_hits,
                "reuse_misses": runtime_state.reuse_misses,
                "last_completed_frame": runtime_state.last_completed_frame,
                "checkpoint_delta_ref": runtime_state.checkpoint_delta_ref,
            },
            "checkpoint_deltas": [{**delta.metadata, "path": delta.path} for delta in checkpoint_deltas],
        }
        if run_result.extra.get("fallback_from"):
            runtime["fallback_from"] = run_result.extra["fallback_from"]
            runtime["fallback_error"] = run_result.extra.get("fallback_error")
        if error_info:
            runtime["error"] = error_info
        write_json(runtime_path, runtime)

        artifacts = [
            self._artifact_record(
                artifact_id=f"{sample_id}:request",
                kind=ArtifactKind.METADATA,
                path=request_path,
                mime_type="application/json",
                metadata={"role": "request"},
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:log",
                kind=ArtifactKind.LOG,
                path=log_path,
                mime_type="text/plain",
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:runtime",
                kind=ArtifactKind.METADATA,
                path=runtime_path,
                mime_type="application/json",
                metadata={"role": "runtime"},
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:checkpoint",
                kind=ArtifactKind.METADATA,
                path=checkpoint_path,
                mime_type="application/json",
                metadata={"role": "checkpoint"},
            ),
            self._artifact_record(
                artifact_id=f"{sample_id}:recovery",
                kind=ArtifactKind.METADATA,
                path=recovery_path,
                mime_type="application/json",
                metadata={"role": "recovery"},
            ),
            *input_artifacts,
        ]
        for delta in checkpoint_deltas:
            if delta.artifact_id and delta.path:
                artifacts.append(
                    self._artifact_record(
                        artifact_id=delta.artifact_id,
                        kind=ArtifactKind.METADATA,
                        path=Path(delta.path),
                        mime_type="application/json",
                        metadata={"role": "checkpoint_delta", **delta.metadata},
                    )
                )
        if run_result.tokens_path and Path(run_result.tokens_path).exists():
            artifacts.append(
                self._artifact_record(
                    artifact_id=f"{sample_id}:tokens",
                    kind=ArtifactKind.LATENT,
                    path=Path(run_result.tokens_path),
                    mime_type="application/octet-stream",
                    metadata={
                        "format": "numpy",
                        "shape": [run_result.total_frames, run_result.spatial_h, run_result.spatial_w],
                        "dtype": "uint32",
                    },
                )
            )
        if run_result.state_path and Path(run_result.state_path).exists():
            artifacts.append(
                self._artifact_record(
                    artifact_id=f"{sample_id}:state",
                    kind=ArtifactKind.METADATA,
                    path=Path(run_result.state_path),
                    mime_type="application/json",
                    metadata={"role": "state"},
                )
            )
        artifacts.append(
            ArtifactRecord(
                artifact_id=f"{sample_id}:temporal-rollout",
                kind=ArtifactKind.METADATA,
                uri=f"temporal://rollouts/{rollout.rollout_id}",
                metadata={
                    "episode_id": episode.episode_id,
                    "rollout_id": rollout.rollout_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "state_handle_id": output_state.state_handle_id,
                    "runner_mode": mode,
                },
            )
        )

        rollout.output_state_handle_id = output_state.state_handle_id
        rollout.status = TemporalStatus.SUCCEEDED if status == SampleStatus.SUCCEEDED else TemporalStatus.FAILED
        rollout.started_at = rollout.started_at or started_at
        rollout.completed_at = completed_at
        rollout.checkpoint_ids = list(
            dict.fromkeys([*rollout.checkpoint_ids, *[cp.checkpoint_id for cp in intermediate_checkpoints], checkpoint.checkpoint_id])
        )
        rollout.artifact_ids = [artifact.artifact_id for artifact in artifacts]
        rollout.metadata["checkpoint_id"] = checkpoint.checkpoint_id
        rollout.metadata["checkpoint_path"] = str(checkpoint_path)
        rollout.metadata["recovery_path"] = str(recovery_path)
        rollout.metadata["stage_graph"] = GENIE_STAGE_GRAPH
        rollout.metadata["stage_timings_ms"] = stage_timings_ms
        rollout.metadata["stage_history"] = stage_history
        rollout.metadata["scheduler"] = {
            "transition_entities": len(transition_entities),
            "chunk_count": len(chunks),
            "queue_lanes": [chunk.queue_lane for chunk in chunks],
            "scheduler_inputs": scheduler_inputs_history,
            "cross_request_batcher": self._transition_batcher.snapshot(),
            "observed_batch_sizes": observed_cross_request_batch_sizes,
        }
        rollout.metrics = {
            "steps": float(step_count),
            "estimated_units": estimate.estimated_units,
            "frames_generated": float(run_result.frames_generated),
            "tokens_generated": float(run_result.tokens_generated),
            "elapsed_s": run_result.elapsed_s,
            "runner_load_ms": stage_timings_ms["runner_load_ms"],
            "state_token_prep_ms": stage_timings_ms["state_token_prep_ms"],
            "runner_exec_ms": stage_timings_ms["runner_exec_ms"],
            "transition_ms": stage_timings_ms["transition_ms"],
            "checkpoint_ms": stage_timings_ms["checkpoint_ms"],
            "artifact_persist_ms": stage_timings_ms["artifact_persist_ms"],
            "temporal_persist_ms": stage_timings_ms["temporal_persist_ms"],
            "checkpoint_delta_count": float(len(checkpoint_deltas)),
            "max_cross_request_batch_size": float(max(observed_cross_request_batch_sizes) if observed_cross_request_batch_sizes else 1),
            "total_elapsed_ms": total_elapsed_ms,
        }
        self.temporal_store.update_rollout(rollout)

        temporal_refs = TemporalRefs(
            episode_id=episode.episode_id,
            rollout_id=rollout.rollout_id,
            branch_id=temporal.branch_id,
            checkpoint_id=checkpoint.checkpoint_id,
            state_handle_id=output_state.state_handle_id,
            parent_state_handle_id=temporal.state_handle_id,
        )

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=self.backend_name,
            model=request.model,
            model_revision=request.model_revision,
            status=status,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            temporal=temporal_refs,
            token_input=request.token_input,
            task_config=task_config,
            genie_config=genie_config,
            resource_estimate=estimate,
            artifacts=artifacts,
            runtime=runtime,
            metadata={
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "runner_mode": mode,
                "fallback_from": run_result.extra.get("fallback_from"),
                "stubbed": mode == "stub",
                "async": False,
                "genie_config_applied": True,
                "stage_timings_ms": stage_timings_ms,
                "stage_graph": GENIE_STAGE_GRAPH,
                "notes": (
                    f"Genie rollout executed via {mode} runner. "
                    f"{run_result.frames_generated} frames generated, {run_result.tokens_generated} tokens produced."
                ),
            },
        )

    def queue_batch_key(self, request: ProduceSampleRequest) -> tuple[Any, ...]:
        """Best-effort queue key for grouping compatible Genie requests."""

        task_config = self._effective_task_config(request)
        genie_config = self._effective_genie_config(request, task_config)
        token_shape = tuple(request.token_input.shape) if request.token_input and request.token_input.shape else None
        return (
            request.backend,
            request.model,
            request.task_type.value,
            genie_config.num_frames,
            genie_config.num_prompt_frames,
            genie_config.maskgit_steps,
            genie_config.temperature,
            genie_config.tokenizer_kind.value,
            genie_config.checkpoint_every_n_frames,
            token_shape,
            bool(request.genie_config and request.genie_config.input_tokens_b64),
            bool(request.temporal and request.temporal.state_handle_id),
        )

    async def execute_job_batch(self, items: list[tuple[ProduceSampleRequest, str]]) -> list[SampleRecord]:
        """Execute multiple compatible Genie jobs with cross-request transition batching."""

        if not items:
            return []
        if len(items) == 1:
            request, sample_id = items[0]
            return [await self.execute_job(request, sample_id)]

        from wm_infra.api.metrics import (
            GENIE_CHECKPOINT_BUILD_SECONDS,
            GENIE_CHECKPOINT_DELTA_BYTES,
            GENIE_CHUNK_FILL_RATIO,
            GENIE_CHUNK_SIZE,
            GENIE_GPU_OCCUPANCY_ESTIMATE,
            GENIE_PERSIST_BACKLOG,
            GENIE_PROMPT_REUSE_EVENTS,
            GENIE_RESIDENCY_EVENTS,
            GENIE_STAGE_DURATION,
            GENIE_STATE_MATERIALIZE_BYTES,
            GENIE_STATE_MATERIALIZE_SECONDS,
            GENIE_TRANSITION_FRAMES_TOTAL,
            GENIE_TRANSITION_TOKENS_TOTAL,
        )

        def finish_stage(
            ctx: dict[str, Any],
            stage: str,
            started_at: float,
            lane: str,
            runner_mode: str,
            extra: dict[str, Any] | None = None,
        ) -> float:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
            ctx["stage_timings_ms"][f"{stage}_ms"] = elapsed_ms
            GENIE_STAGE_DURATION.labels(stage=stage, lane=lane, runner_mode=runner_mode).observe(elapsed_ms / 1000.0)
            entry = {"stage": stage, "elapsed_ms": elapsed_ms, "queue_lane": lane, "runner_mode": runner_mode}
            if extra:
                entry.update(extra)
            ctx["stage_history"].append(entry)
            return elapsed_ms

        runner_load_start = time.perf_counter()
        self.ensure_runner_loaded()
        runner_load_ms = round((time.perf_counter() - runner_load_start) * 1000.0, 3)

        contexts: list[dict[str, Any]] = []
        entity_to_context: dict[str, dict[str, Any]] = {}
        total_materialized_bytes = 0
        for request, sample_id in items:
            if request.task_type not in {TaskType.GENIE_ROLLOUT, TaskType.WORLD_MODEL_ROLLOUT}:
                raise ValueError(f"Backend {self.backend_name} only supports rollout-style temporal tasks")
            temporal = request.temporal
            if temporal is None or not temporal.episode_id:
                raise ValueError("genie-rollout requests require temporal.episode_id")
            episode = self.temporal_store.episodes.get(temporal.episode_id)
            if episode is None:
                raise ValueError(f"Unknown episode_id: {temporal.episode_id}")

            task_config = self._effective_task_config(request)
            genie_config = self._effective_genie_config(request, task_config)
            estimate = estimate_rollout_request(task_config)
            step_count = task_config.num_steps if task_config is not None else 1
            seed = request.sample_spec.seed or 42
            queue_lane = GenieQueueLane.HOT_CONTINUATION.value if temporal.state_handle_id else GenieQueueLane.COLD_MATERIALIZE.value
            sample_dir = self._sample_dir(sample_id)
            request_path = sample_dir / "request.json"
            log_path = sample_dir / "runner.log"
            runtime_path = sample_dir / "runtime.json"
            checkpoint_path = sample_dir / "checkpoint.json"
            recovery_path = sample_dir / "recovery.json"

            ctx: dict[str, Any] = {
                "request": request,
                "sample_id": sample_id,
                "episode": episode,
                "temporal": temporal,
                "task_config": task_config,
                "genie_config": genie_config,
                "estimate": estimate,
                "step_count": step_count,
                "seed": seed,
                "queue_lane": queue_lane,
                "sample_dir": sample_dir,
                "request_path": request_path,
                "log_path": log_path,
                "runtime_path": runtime_path,
                "checkpoint_path": checkpoint_path,
                "recovery_path": recovery_path,
                "started_at": time.time(),
                "stage_timings_ms": {"runner_load_ms": runner_load_ms},
                "stage_history": [],
                "checkpoint_deltas": [],
                "intermediate_checkpoints": [],
                "chunk_history": [],
                "scheduler_inputs_history": [],
            }

            admission_start = time.perf_counter()
            finish_stage(
                ctx,
                "admission",
                admission_start,
                GenieQueueLane.COLD_MATERIALIZE.value,
                self.runner_mode,
                extra={"sample_id": sample_id, "episode_id": episode.episode_id},
            )

            state_materialize_start = time.perf_counter()
            input_tokens, token_input_runtime, input_artifacts = self._resolve_input_tokens(request, sample_dir)
            cache_key = prompt_cache_key(
                input_state_handle_id=temporal.state_handle_id,
                input_tokens=input_tokens,
                prompt=request.sample_spec.prompt or "",
                seed=seed,
            )
            cached_prompt_state = self._prompt_state_cache.get(cache_key)
            prompt_reuse_hit = cached_prompt_state is not None
            if prompt_reuse_hit and input_tokens is None and cached_prompt_state is not None:
                input_tokens = cached_prompt_state.tokens.copy()
                token_input_runtime = {
                    **token_input_runtime,
                    "token_input_mode": "prompt_cache",
                    "cache_key": cache_key,
                    "source_state_handle_id": cached_prompt_state.source_state_handle_id,
                }
            resident_tier = (
                cached_prompt_state.resident_tier
                if cached_prompt_state is not None
                else (GenieResidencyTier.WARM_PINNED_CPU if input_tokens is not None else GenieResidencyTier.COLD_FILE)
            )
            materialized_bytes = int(input_tokens.nbytes) if input_tokens is not None else 0
            total_materialized_bytes += materialized_bytes
            GENIE_STATE_MATERIALIZE_BYTES.observe(materialized_bytes)
            GENIE_STATE_MATERIALIZE_SECONDS.observe(time.perf_counter() - state_materialize_start)
            GENIE_PROMPT_REUSE_EVENTS.labels(outcome="hit" if prompt_reuse_hit else "miss").inc()
            GENIE_RESIDENCY_EVENTS.labels(tier=resident_tier.value).inc()
            finish_stage(
                ctx,
                "state_materialize",
                state_materialize_start,
                GenieQueueLane.COLD_MATERIALIZE.value,
                self.runner_mode,
                extra={"materialized_bytes": materialized_bytes, "resident_tier": resident_tier.value},
            )

            prompt_prepare_start = time.perf_counter()
            if self.runner_mode == "real":
                self._validate_real_mode_request(
                    request=request,
                    genie_config=genie_config,
                    input_tokens=input_tokens,
                )
            prepared = self.runner.prepare_inputs(
                prompt=request.sample_spec.prompt or "",
                seed=seed,
                num_frames=genie_config.num_frames,
                input_tokens=input_tokens,
                num_prompt_frames=genie_config.num_prompt_frames,
                maskgit_steps=genie_config.maskgit_steps,
                temperature=genie_config.temperature,
            )
            if prepared.mode == "real":
                resident_tier = GenieResidencyTier.HOT_GPU
                GENIE_RESIDENCY_EVENTS.labels(tier=resident_tier.value).inc()
            prompt_tokens = prepared.current_tokens_numpy()[: prepared.prompt_frames] if prepared.prompt_frames > 0 else prepared.current_tokens_numpy()[:1]
            self._prompt_state_cache.put(
                cache_key,
                prompt_tokens,
                source_state_handle_id=temporal.state_handle_id,
                resident_tier=resident_tier,
            )
            finish_stage(
                ctx,
                "prompt_prepare",
                prompt_prepare_start,
                queue_lane,
                prepared.mode,
                extra={"prompt_frames": prepared.prompt_frames, "total_frames": prepared.total_frames},
            )
            ctx["stage_timings_ms"]["state_token_prep_ms"] = round(
                ctx["stage_timings_ms"]["state_materialize_ms"] + ctx["stage_timings_ms"]["prompt_prepare_ms"],
                3,
            )

            request_payload = {
                "sample_id": sample_id,
                "task_type": request.task_type.value,
                "backend": request.backend,
                "model": request.model,
                "model_revision": request.model_revision,
                "sample_spec": request.sample_spec.model_dump(mode="json"),
                "temporal": temporal.model_dump(mode="json"),
                "token_input": request.token_input.model_dump(mode="json") if request.token_input else None,
                "task_config": task_config.model_dump(mode="json") if task_config else None,
                "genie_config": genie_config.model_dump(mode="json"),
                "runner_mode": prepared.mode,
                "stage_graph": GENIE_STAGE_GRAPH,
            }
            write_json(request_path, request_payload)
            rollout = self.temporal_store.create_rollout(
                RolloutCreate(
                    episode_id=episode.episode_id,
                    branch_id=temporal.branch_id,
                    backend=self.backend_name,
                    model=request.model,
                    sample_id=sample_id,
                    request_id=sample_id,
                    input_state_handle_id=temporal.state_handle_id,
                    step_count=step_count,
                    priority=request.priority,
                    metadata={
                        "runner_mode": prepared.mode,
                        "prompt": request.sample_spec.prompt,
                        "controls": request.sample_spec.controls,
                        "token_input": token_input_runtime,
                        "genie_config": genie_config.model_dump(mode="json"),
                        "stage_graph": GENIE_STAGE_GRAPH,
                    },
                ),
                status=TemporalStatus.ACTIVE,
            )

            runtime_state = GenieRuntimeState(
                rollout_id=rollout.rollout_id,
                prompt_tokens_ref=cache_key,
                generated_tokens_ref=None,
                last_completed_frame=prepared.prompt_frames,
                resident_tier=resident_tier,
                ancestor_state_ref=temporal.state_handle_id,
                checkpoint_delta_ref=None,
                materialized_bytes=materialized_bytes,
                dirty_since_checkpoint=False,
                prompt_reuse_hit=prompt_reuse_hit,
                source_cache_key=cache_key,
                reuse_hits=1 if prompt_reuse_hit else 0,
                reuse_misses=0 if prompt_reuse_hit else 1,
            )
            root_signature = make_stage_signature(
                backend=self.backend_name,
                model_name=request.model,
                stage="transition",
                device=prepared.device,
                dtype=prepared.dtype,
                tokenizer_kind=genie_config.tokenizer_kind.value,
                spatial_h=prepared.spatial_h,
                spatial_w=prepared.spatial_w,
                window_num_frames=max(prepared.total_frames - prepared.prompt_frames, 0),
                num_prompt_frames=prepared.prompt_frames,
                maskgit_steps=genie_config.maskgit_steps,
                temperature=genie_config.temperature,
                checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
                runner_mode=prepared.mode,
                needs_persist=False,
            )
            root_entity = GenieExecutionEntity(
                entity_id=f"{sample_id}:root",
                rollout_id=rollout.rollout_id,
                episode_id=episode.episode_id,
                branch_id=temporal.branch_id,
                sample_id=sample_id,
                input_state_handle_id=temporal.state_handle_id,
                current_stage="transition",
                next_stage="artifact_persist",
                window_start_frame=prepared.prompt_frames,
                window_num_frames=max(prepared.total_frames - prepared.prompt_frames, 0),
                total_frames=prepared.total_frames,
                num_prompt_frames=prepared.prompt_frames,
                checkpoint_every_n_frames=genie_config.checkpoint_every_n_frames,
                priority=request.priority,
                deadline_s=None,
                batch_signature=root_signature,
                queue_lane=queue_lane,
            )
            transition_entities = build_transition_entities(root_entity)
            for entity in transition_entities:
                entity_to_context[entity.entity_id] = ctx

            ctx.update(
                {
                    "token_input_runtime": token_input_runtime,
                    "input_artifacts": input_artifacts,
                    "prepared": prepared,
                    "request_payload": request_payload,
                    "rollout": rollout,
                    "runtime_state": runtime_state,
                    "transition_entities": transition_entities,
                    "cache_key": cache_key,
                }
            )
            contexts.append(ctx)

        all_entities = [entity for ctx in contexts for entity in ctx["transition_entities"]]
        scheduler = GenieScheduler(max_chunk_size=max(1, len(contexts)))
        decisions = scheduler.schedule(
            all_entities,
            persist_backlog=0,
            prompt_state_hot=any(ctx["runtime_state"].resident_tier == GenieResidencyTier.HOT_GPU for ctx in contexts),
            estimated_transfer_bytes=total_materialized_bytes,
        )
        GENIE_PERSIST_BACKLOG.set(0)

        for decision in decisions:
            chunk = decision.chunk
            GENIE_CHUNK_SIZE.labels(stage=chunk.runnable_stage, lane=chunk.queue_lane).observe(chunk.size)
            GENIE_CHUNK_FILL_RATIO.labels(stage=chunk.runnable_stage, lane=chunk.queue_lane).observe(chunk.fill_ratio)
            GENIE_GPU_OCCUPANCY_ESTIMATE.set(chunk.expected_occupancy)

            chunk_contexts = [entity_to_context[entity_id] for entity_id in chunk.entity_ids]
            prepared_runs = [ctx["prepared"] for ctx in chunk_contexts]
            transition_start = time.perf_counter()
            results = self.runner.run_window_batch(
                prepared_runs,
                frame_start=chunk.frame_range[0],
                frame_end=chunk.frame_range[1],
            )
            transition_elapsed_ms = round((time.perf_counter() - transition_start) * 1000.0, 3)
            for ctx, result in zip(chunk_contexts, results):
                ctx["stage_timings_ms"]["runner_exec_ms"] = round(
                    ctx["stage_timings_ms"].get("runner_exec_ms", 0.0) + transition_elapsed_ms,
                    3,
                )
                ctx["stage_timings_ms"]["transition_ms"] = round(
                    ctx["stage_timings_ms"].get("transition_ms", 0.0) + transition_elapsed_ms,
                    3,
                )
                if result.frames_generated > 0:
                    GENIE_TRANSITION_FRAMES_TOTAL.labels(runner_mode=ctx["prepared"].mode).inc(result.frames_generated)
                    GENIE_TRANSITION_TOKENS_TOTAL.labels(runner_mode=ctx["prepared"].mode).inc(
                        result.frames_generated * ctx["prepared"].spatial_h * ctx["prepared"].spatial_w
                    )
                ctx["runtime_state"].generated_tokens_ref = chunk.chunk_id
                ctx["runtime_state"].last_completed_frame = max(ctx["runtime_state"].last_completed_frame, result.frame_end)
                ctx["runtime_state"].dirty_since_checkpoint = True
                ctx["chunk_history"].append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "queue_lane": chunk.queue_lane,
                        "frame_range": list(chunk.frame_range),
                        "chunk_size": chunk.size,
                        "expected_occupancy": chunk.expected_occupancy,
                    }
                )
                ctx["scheduler_inputs_history"].append(
                    {
                        **decision.scheduler_inputs,
                        "chunk_id": chunk.chunk_id,
                        "frame_range": list(chunk.frame_range),
                    }
                )
                ctx["stage_history"].append(
                    {
                        "stage": "transition",
                        "elapsed_ms": transition_elapsed_ms,
                        "queue_lane": chunk.queue_lane,
                        "runner_mode": ctx["prepared"].mode,
                        "chunk_id": chunk.chunk_id,
                        "chunk_size": chunk.size,
                        "frame_range": list(chunk.frame_range),
                        "expected_occupancy": chunk.expected_occupancy,
                        "batched_across_requests": True,
                    }
                )

                if checkpoint_due(
                    frame_end=result.frame_end,
                    total_frames=ctx["prepared"].total_frames,
                    checkpoint_every_n_frames=ctx["genie_config"].checkpoint_every_n_frames,
                ):
                    checkpoint_start = time.perf_counter()
                    delta = build_checkpoint_delta(
                        rollout_id=ctx["rollout"].rollout_id,
                        sample_id=ctx["sample_id"],
                        parent_state_handle_id=ctx["temporal"].state_handle_id,
                        all_tokens=ctx["prepared"].current_tokens_numpy(),
                        start_frame=result.frame_start,
                        end_frame=result.frame_end,
                        checkpoint_every_n_frames=ctx["genie_config"].checkpoint_every_n_frames,
                        runner_mode=ctx["prepared"].mode,
                    )
                    persist_checkpoint_delta(ctx["sample_dir"], delta)
                    ctx["checkpoint_deltas"].append(delta)
                    ctx["runtime_state"].checkpoint_delta_ref = delta.artifact_id
                    ctx["runtime_state"].dirty_since_checkpoint = False
                    GENIE_CHECKPOINT_DELTA_BYTES.observe(delta.bytes_size)
                    GENIE_CHECKPOINT_BUILD_SECONDS.observe(time.perf_counter() - checkpoint_start)
                    checkpoint_elapsed_ms = round((time.perf_counter() - checkpoint_start) * 1000.0, 3)
                    ctx["stage_timings_ms"]["checkpoint_ms"] = round(
                        ctx["stage_timings_ms"].get("checkpoint_ms", 0.0) + checkpoint_elapsed_ms,
                        3,
                    )
                    ctx["stage_history"].append(
                        {
                            "stage": "checkpoint",
                            "elapsed_ms": checkpoint_elapsed_ms,
                            "queue_lane": GenieQueueLane.CHECKPOINT_HEAVY.value,
                            "runner_mode": ctx["prepared"].mode,
                            "artifact_id": delta.artifact_id,
                            "frame_end": delta.end_frame,
                        }
                    )
                    ctx["intermediate_checkpoints"].append(
                        self.temporal_store.create_checkpoint(
                            CheckpointCreate(
                                episode_id=ctx["episode"].episode_id,
                                rollout_id=ctx["rollout"].rollout_id,
                                branch_id=ctx["temporal"].branch_id,
                                artifact_ids=[delta.artifact_id] if delta.artifact_id else [],
                                step_index=max(delta.end_frame - 1, 0),
                                tag=f"frame-{delta.end_frame}",
                                metadata={**delta.metadata, "kind": "delta"},
                            )
                        )
                    )

        records: list[SampleRecord] = []
        for ctx in contexts:
            persist_start = time.perf_counter()
            run_result = self.runner.persist_outputs(ctx["prepared"], output_dir=ctx["sample_dir"])
            mode = run_result.mode
            request_payload = ctx["request_payload"]
            request_payload["runner_mode"] = mode
            if run_result.extra.get("fallback_from"):
                request_payload["fallback_from"] = run_result.extra["fallback_from"]
                request_payload["fallback_error"] = run_result.extra.get("fallback_error")
            write_json(ctx["request_path"], request_payload)

            status = SampleStatus.SUCCEEDED
            error_info: str | None = None
            if run_result.error:
                status = SampleStatus.FAILED
                error_info = run_result.error

            log_lines = [
                f"Genie stage runtime: {' -> '.join(GENIE_STAGE_GRAPH)}",
                f"Genie runner mode: {mode}",
                f"Model: {run_result.model_name}",
                f"Device: {run_result.device}",
                f"Requested frames: {ctx['genie_config'].num_frames}",
                f"Prompt frames: {ctx['genie_config'].num_prompt_frames}",
                f"MaskGIT steps: {ctx['genie_config'].maskgit_steps}",
                f"Temperature: {ctx['genie_config'].temperature}",
                f"Frames generated: {run_result.frames_generated}/{run_result.total_frames}",
                f"Tokens generated: {run_result.tokens_generated}",
                f"Elapsed: {run_result.elapsed_s:.3f}s",
                f"Token input: {ctx['token_input_runtime'].get('token_input_mode')}",
                f"Batched transition chunks: {len(ctx['chunk_history'])}",
            ]
            if error_info:
                log_lines.append(f"ERROR: {error_info}")
            if run_result.extra.get("fallback_from"):
                log_lines.append(f"FALLBACK: {run_result.extra['fallback_from']} -> {mode}")
            write_log(ctx["log_path"], log_lines)

            state_uri = f"file://{run_result.tokens_path}" if run_result.tokens_path else None
            output_state = self.temporal_store.create_state_handle(
                StateHandleCreate(
                    episode_id=ctx["episode"].episode_id,
                    branch_id=ctx["temporal"].branch_id,
                    rollout_id=ctx["rollout"].rollout_id,
                    kind=StateHandleKind.VIDEO_LATENT,
                    uri=state_uri,
                    shape=[run_result.total_frames, run_result.spatial_h, run_result.spatial_w],
                    dtype="uint32",
                    artifact_ids=[
                        f"{ctx['sample_id']}:tokens",
                        f"{ctx['sample_id']}:state",
                        f"{ctx['sample_id']}:checkpoint",
                        f"{ctx['sample_id']}:recovery",
                        *[delta.artifact_id for delta in ctx["checkpoint_deltas"] if delta.artifact_id],
                    ],
                    metadata={
                        "runner_mode": mode,
                        "source_backend": self.backend_name,
                        "prompt": ctx["request"].sample_spec.prompt,
                        "tokens_generated": run_result.tokens_generated,
                        "frames_generated": run_result.frames_generated,
                        "sample_id": ctx["sample_id"],
                        "token_input": ctx["token_input_runtime"],
                        "genie_config": ctx["genie_config"].model_dump(mode="json"),
                        "stage_graph": GENIE_STAGE_GRAPH,
                    },
                )
            )

            completed_at = time.time()
            checkpoint_payload = {
                "sample_id": ctx["sample_id"],
                "episode_id": ctx["episode"].episode_id,
                "branch_id": ctx["temporal"].branch_id,
                "rollout_id": ctx["rollout"].rollout_id,
                "parent_state_handle_id": ctx["temporal"].state_handle_id,
                "output_state_handle_id": output_state.state_handle_id,
                "tokens_path": run_result.tokens_path,
                "state_path": run_result.state_path,
                "log_path": str(ctx["log_path"]),
                "request_path": str(ctx["request_path"]),
                "runner_mode": mode,
                "status": status.value,
                "token_input": ctx["token_input_runtime"],
                "genie_config": ctx["genie_config"].model_dump(mode="json"),
                "checkpoint_deltas": [delta.metadata for delta in ctx["checkpoint_deltas"]],
                "timestamps": {"started_at": ctx["started_at"], "completed_at": completed_at},
            }
            write_json(ctx["checkpoint_path"], checkpoint_payload)
            write_json(
                ctx["recovery_path"],
                {
                    **checkpoint_payload,
                    "recovery_hint": "Reload tokens_path into token_input or fork from checkpoint/state_handle for retry.",
                },
            )

            checkpoint = self.temporal_store.create_checkpoint(
                CheckpointCreate(
                    episode_id=ctx["episode"].episode_id,
                    rollout_id=ctx["rollout"].rollout_id,
                    branch_id=ctx["temporal"].branch_id,
                    state_handle_id=output_state.state_handle_id,
                    artifact_ids=[
                        f"{ctx['sample_id']}:checkpoint",
                        f"{ctx['sample_id']}:recovery",
                        f"{ctx['sample_id']}:tokens",
                        f"{ctx['sample_id']}:state",
                        *[delta.artifact_id for delta in ctx["checkpoint_deltas"] if delta.artifact_id],
                    ],
                    step_index=max(ctx["step_count"] - 1, 0),
                    tag="terminal",
                    metadata={
                        "runner_mode": mode,
                        "frames_generated": run_result.frames_generated,
                        "checkpoint_path": str(ctx["checkpoint_path"]),
                        "recovery_path": str(ctx["recovery_path"]),
                        "sample_id": ctx["sample_id"],
                        "genie_config": ctx["genie_config"].model_dump(mode="json"),
                        "checkpoint_deltas": [delta.metadata for delta in ctx["checkpoint_deltas"]],
                    },
                )
            )

            output_state.checkpoint_id = checkpoint.checkpoint_id
            output_state.artifact_ids = list(
                dict.fromkeys(
                    [
                        *output_state.artifact_ids,
                        f"{ctx['sample_id']}:request",
                        f"{ctx['sample_id']}:runtime",
                        f"{ctx['sample_id']}:log",
                    ]
                )
            )
            output_state.metadata["checkpoint_id"] = checkpoint.checkpoint_id
            output_state.metadata["checkpoint_path"] = str(ctx["checkpoint_path"])
            output_state.metadata["recovery_path"] = str(ctx["recovery_path"])
            self.temporal_store.state_handles.put(output_state)
            ctx["stage_timings_ms"]["artifact_persist_ms"] = round((time.perf_counter() - persist_start) * 1000.0, 3)
            ctx["stage_history"].append(
                {
                    "stage": "artifact_persist",
                    "elapsed_ms": ctx["stage_timings_ms"]["artifact_persist_ms"],
                    "queue_lane": GenieQueueLane.PERSIST_ONLY.value,
                    "runner_mode": mode,
                    "checkpoint_delta_count": len(ctx["checkpoint_deltas"]),
                }
            )

            temporal_persist_start = time.perf_counter()
            ctx["episode"].updated_at = time.time()
            self.temporal_store.episodes.put(ctx["episode"])
            ctx["stage_timings_ms"]["temporal_persist_ms"] = round((time.perf_counter() - temporal_persist_start) * 1000.0, 3)
            ctx["stage_timings_ms"]["controlplane_commit_ms"] = ctx["stage_timings_ms"]["temporal_persist_ms"]
            ctx["stage_history"].append(
                {
                    "stage": "controlplane_commit",
                    "elapsed_ms": ctx["stage_timings_ms"]["controlplane_commit_ms"],
                    "queue_lane": GenieQueueLane.PERSIST_ONLY.value,
                    "runner_mode": mode,
                    "checkpoint_count": len(ctx["intermediate_checkpoints"]) + 1,
                }
            )
            ctx["stage_timings_ms"]["total_elapsed_ms"] = round((time.perf_counter() - ctx["started_at"]) * 1000.0, 3)

            runtime: dict[str, Any] = {
                "runner": f"genie-{mode}",
                "runner_mode": mode,
                "temporal": True,
                "async": False,
                "rollout_id": ctx["rollout"].rollout_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "state_handle_id": output_state.state_handle_id,
                "model": run_result.model_name,
                "device": run_result.device,
                "frames_generated": run_result.frames_generated,
                "prompt_frames": run_result.prompt_frames,
                "total_frames": run_result.total_frames,
                "tokens_generated": run_result.tokens_generated,
                "elapsed_s": run_result.elapsed_s,
                "request_path": str(ctx["request_path"]),
                "log_path": str(ctx["log_path"]),
                "runtime_path": str(ctx["runtime_path"]),
                "checkpoint_path": str(ctx["checkpoint_path"]),
                "recovery_path": str(ctx["recovery_path"]),
                "tokens_path": run_result.tokens_path,
                "state_path": run_result.state_path,
                "token_input": ctx["token_input_runtime"],
                "genie_config": ctx["genie_config"].model_dump(mode="json"),
                "status_history": build_runtime_status_history(
                    started_at=ctx["started_at"],
                    completed_at=completed_at,
                    terminal_status=status.value,
                ),
                "started_at": ctx["started_at"],
                "completed_at": completed_at,
                "elapsed_ms": round((completed_at - ctx["started_at"]) * 1000, 2),
                "stage_timings_ms": ctx["stage_timings_ms"],
                "stage_graph": GENIE_STAGE_GRAPH,
                "stage_history": ctx["stage_history"],
                "scheduler": {
                    "transition_entities": len(ctx["transition_entities"]),
                    "chunks": ctx["chunk_history"],
                    "scheduler_inputs": ctx["scheduler_inputs_history"],
                    "batched_across_requests": True,
                },
                "queue_lane": ctx["queue_lane"],
                "runtime_state": {
                    "resident_tier": ctx["runtime_state"].resident_tier.value,
                    "materialized_bytes": ctx["runtime_state"].materialized_bytes,
                    "reuse_hits": ctx["runtime_state"].reuse_hits,
                    "reuse_misses": ctx["runtime_state"].reuse_misses,
                    "last_completed_frame": ctx["runtime_state"].last_completed_frame,
                    "checkpoint_delta_ref": ctx["runtime_state"].checkpoint_delta_ref,
                },
                "checkpoint_deltas": [{**delta.metadata, "path": delta.path} for delta in ctx["checkpoint_deltas"]],
                "batched_transition": True,
            }
            if run_result.extra.get("fallback_from"):
                runtime["fallback_from"] = run_result.extra["fallback_from"]
                runtime["fallback_error"] = run_result.extra.get("fallback_error")
            if error_info:
                runtime["error"] = error_info
            write_json(ctx["runtime_path"], runtime)

            artifacts = [
                self._artifact_record(
                    artifact_id=f"{ctx['sample_id']}:request",
                    kind=ArtifactKind.METADATA,
                    path=ctx["request_path"],
                    mime_type="application/json",
                    metadata={"role": "request"},
                ),
                self._artifact_record(
                    artifact_id=f"{ctx['sample_id']}:log",
                    kind=ArtifactKind.LOG,
                    path=ctx["log_path"],
                    mime_type="text/plain",
                ),
                self._artifact_record(
                    artifact_id=f"{ctx['sample_id']}:runtime",
                    kind=ArtifactKind.METADATA,
                    path=ctx["runtime_path"],
                    mime_type="application/json",
                    metadata={"role": "runtime"},
                ),
                self._artifact_record(
                    artifact_id=f"{ctx['sample_id']}:checkpoint",
                    kind=ArtifactKind.METADATA,
                    path=ctx["checkpoint_path"],
                    mime_type="application/json",
                    metadata={"role": "checkpoint"},
                ),
                self._artifact_record(
                    artifact_id=f"{ctx['sample_id']}:recovery",
                    kind=ArtifactKind.METADATA,
                    path=ctx["recovery_path"],
                    mime_type="application/json",
                    metadata={"role": "recovery"},
                ),
                *ctx["input_artifacts"],
            ]
            for delta in ctx["checkpoint_deltas"]:
                if delta.artifact_id and delta.path:
                    artifacts.append(
                        self._artifact_record(
                            artifact_id=delta.artifact_id,
                            kind=ArtifactKind.METADATA,
                            path=Path(delta.path),
                            mime_type="application/json",
                            metadata={"role": "checkpoint_delta", **delta.metadata},
                        )
                    )
            if run_result.tokens_path and Path(run_result.tokens_path).exists():
                artifacts.append(
                    self._artifact_record(
                        artifact_id=f"{ctx['sample_id']}:tokens",
                        kind=ArtifactKind.LATENT,
                        path=Path(run_result.tokens_path),
                        mime_type="application/octet-stream",
                        metadata={
                            "format": "numpy",
                            "shape": [run_result.total_frames, run_result.spatial_h, run_result.spatial_w],
                            "dtype": "uint32",
                        },
                    )
                )
            if run_result.state_path and Path(run_result.state_path).exists():
                artifacts.append(
                    self._artifact_record(
                        artifact_id=f"{ctx['sample_id']}:state",
                        kind=ArtifactKind.METADATA,
                        path=Path(run_result.state_path),
                        mime_type="application/json",
                        metadata={"role": "state"},
                    )
                )
            artifacts.append(
                ArtifactRecord(
                    artifact_id=f"{ctx['sample_id']}:temporal-rollout",
                    kind=ArtifactKind.METADATA,
                    uri=f"temporal://rollouts/{ctx['rollout'].rollout_id}",
                    metadata={
                        "episode_id": ctx["episode"].episode_id,
                        "rollout_id": ctx["rollout"].rollout_id,
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "state_handle_id": output_state.state_handle_id,
                        "runner_mode": mode,
                    },
                )
            )

            ctx["rollout"].output_state_handle_id = output_state.state_handle_id
            ctx["rollout"].status = TemporalStatus.SUCCEEDED if status == SampleStatus.SUCCEEDED else TemporalStatus.FAILED
            ctx["rollout"].started_at = ctx["rollout"].started_at or ctx["started_at"]
            ctx["rollout"].completed_at = completed_at
            ctx["rollout"].checkpoint_ids = list(
                dict.fromkeys([*ctx["rollout"].checkpoint_ids, *[cp.checkpoint_id for cp in ctx["intermediate_checkpoints"]], checkpoint.checkpoint_id])
            )
            ctx["rollout"].artifact_ids = [artifact.artifact_id for artifact in artifacts]
            ctx["rollout"].metadata["checkpoint_id"] = checkpoint.checkpoint_id
            ctx["rollout"].metadata["checkpoint_path"] = str(ctx["checkpoint_path"])
            ctx["rollout"].metadata["recovery_path"] = str(ctx["recovery_path"])
            ctx["rollout"].metadata["stage_graph"] = GENIE_STAGE_GRAPH
            ctx["rollout"].metadata["stage_timings_ms"] = ctx["stage_timings_ms"]
            ctx["rollout"].metadata["stage_history"] = ctx["stage_history"]
            ctx["rollout"].metadata["scheduler"] = {
                "transition_entities": len(ctx["transition_entities"]),
                "chunk_count": len(ctx["chunk_history"]),
                "queue_lanes": [chunk["queue_lane"] for chunk in ctx["chunk_history"]],
                "scheduler_inputs": ctx["scheduler_inputs_history"],
                "batched_across_requests": True,
            }
            ctx["rollout"].metrics = {
                "steps": float(ctx["step_count"]),
                "estimated_units": ctx["estimate"].estimated_units,
                "frames_generated": float(run_result.frames_generated),
                "tokens_generated": float(run_result.tokens_generated),
                "elapsed_s": run_result.elapsed_s,
                "runner_load_ms": ctx["stage_timings_ms"]["runner_load_ms"],
                "state_token_prep_ms": ctx["stage_timings_ms"]["state_token_prep_ms"],
                "runner_exec_ms": ctx["stage_timings_ms"].get("runner_exec_ms", 0.0),
                "transition_ms": ctx["stage_timings_ms"].get("transition_ms", 0.0),
                "checkpoint_ms": ctx["stage_timings_ms"].get("checkpoint_ms", 0.0),
                "artifact_persist_ms": ctx["stage_timings_ms"]["artifact_persist_ms"],
                "temporal_persist_ms": ctx["stage_timings_ms"]["temporal_persist_ms"],
                "checkpoint_delta_count": float(len(ctx["checkpoint_deltas"])),
                "total_elapsed_ms": ctx["stage_timings_ms"]["total_elapsed_ms"],
            }
            self.temporal_store.update_rollout(ctx["rollout"])

            temporal_refs = TemporalRefs(
                episode_id=ctx["episode"].episode_id,
                rollout_id=ctx["rollout"].rollout_id,
                branch_id=ctx["temporal"].branch_id,
                checkpoint_id=checkpoint.checkpoint_id,
                state_handle_id=output_state.state_handle_id,
                parent_state_handle_id=ctx["temporal"].state_handle_id,
            )
            records.append(
                SampleRecord(
                    sample_id=ctx["sample_id"],
                    task_type=ctx["request"].task_type,
                    backend=self.backend_name,
                    model=ctx["request"].model,
                    model_revision=ctx["request"].model_revision,
                    status=status,
                    experiment=ctx["request"].experiment,
                    sample_spec=ctx["request"].sample_spec,
                    temporal=temporal_refs,
                    token_input=ctx["request"].token_input,
                    task_config=ctx["task_config"],
                    genie_config=ctx["genie_config"],
                    resource_estimate=ctx["estimate"],
                    artifacts=artifacts,
                    runtime=runtime,
                    metadata={
                        "evaluation_policy": ctx["request"].evaluation_policy,
                        "priority": ctx["request"].priority,
                        "labels": ctx["request"].labels,
                        "runner_mode": mode,
                        "fallback_from": run_result.extra.get("fallback_from"),
                        "stubbed": mode == "stub",
                        "async": False,
                        "genie_config_applied": True,
                        "stage_timings_ms": ctx["stage_timings_ms"],
                        "stage_graph": GENIE_STAGE_GRAPH,
                        "notes": (
                            f"Genie rollout executed via {mode} runner. "
                            f"{run_result.frames_generated} frames generated, {run_result.tokens_generated} tokens produced."
                        ),
                    },
                )
            )

        return records

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        sample_id = str(uuid.uuid4())
        return await self.execute_job(request, sample_id)

    def submit_async(self, request: ProduceSampleRequest) -> SampleRecord:
        if self._job_queue is None:
            raise RuntimeError("No job queue attached — cannot submit async")
        temporal = request.temporal
        if temporal is None or not temporal.episode_id:
            raise ValueError("genie-rollout requests require temporal.episode_id")

        sample_id = str(uuid.uuid4())
        task_config = self._effective_task_config(request)
        genie_config = self._effective_genie_config(request, task_config)
        estimate = estimate_rollout_request(task_config)
        self._job_queue.submit(sample_id, request)
        queued_at = time.time()
        queue_lane = GenieQueueLane.HOT_CONTINUATION.value if temporal.state_handle_id else GenieQueueLane.COLD_MATERIALIZE.value

        return SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=self.backend_name,
            model=request.model,
            model_revision=request.model_revision,
            status=SampleStatus.QUEUED,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            temporal=request.temporal,
            token_input=request.token_input,
            task_config=task_config,
            genie_config=genie_config,
            resource_estimate=estimate,
            runtime={
                "runner": f"genie-{self.runner_mode}",
                "runner_mode": self.runner_mode,
                "async": True,
                "genie_config": genie_config.model_dump(mode="json"),
                "stage_graph": GENIE_STAGE_GRAPH,
                "queue_lane": queue_lane,
                "status_history": [{"status": SampleStatus.QUEUED.value, "timestamp": queued_at}],
                "queued_at": queued_at,
            },
            metadata={
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "runner_mode": self.runner_mode,
                "async": True,
                "genie_config_applied": True,
                "stage_graph": GENIE_STAGE_GRAPH,
            },
        )
