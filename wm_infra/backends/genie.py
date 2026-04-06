"""State-aware rollout backend for Genie (STMaskGIT) temporal episodes."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.backends.genie_runner import GenieRunResult, GenieRunner
from wm_infra.controlplane import (
    ArtifactKind,
    ArtifactRecord,
    CheckpointCreate,
    GenieTaskConfig,
    ProduceSampleRequest,
    RolloutTaskConfig,
    RolloutCreate,
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
    """Genie rollout backend with real model execution and persisted artifacts."""

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
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
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
        raw: np.ndarray
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
            raise ValueError(
                "Genie real mode currently only supports genie_config.tokenizer_kind=genie_stmaskgit"
            )

        if genie_config.num_frames > max_frames:
            raise ValueError(
                f"Genie real mode only supports num_frames <= {max_frames}; got {genie_config.num_frames}"
            )

        if genie_config.num_prompt_frames >= genie_config.num_frames:
            raise ValueError(
                "Genie real mode requires num_prompt_frames < num_frames"
            )

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
            raise ValueError(
                f"Genie real mode token input only supports T <= {max_frames}; got {input_tokens.shape[0]}"
            )

        max_seen = int(input_tokens.max())
        min_seen = int(input_tokens.min())
        if min_seen < 0 or max_seen > max_token_id:
            raise ValueError(
                f"Genie real mode token ids must be in [0,{max_token_id}], got [{min_seen},{max_seen}]"
            )

    async def execute_job(self, request: ProduceSampleRequest, sample_id: str) -> SampleRecord:
        if request.task_type not in {TaskType.GENIE_ROLLOUT, TaskType.WORLD_MODEL_ROLLOUT}:
            raise ValueError(f"Backend {self.backend_name} only supports rollout-style temporal tasks")

        stage_timings_ms: dict[str, float] = {}
        total_start = time.perf_counter()

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
        num_frames = genie_config.num_frames

        sample_dir = self._sample_dir(sample_id)
        request_path = sample_dir / "request.json"
        log_path = sample_dir / "runner.log"
        runtime_path = sample_dir / "runtime.json"
        checkpoint_path = sample_dir / "checkpoint.json"
        recovery_path = sample_dir / "recovery.json"

        prep_start = time.perf_counter()
        input_tokens, token_input_runtime, input_artifacts = self._resolve_input_tokens(request, sample_dir)
        if self.runner_mode == "real":
            self._validate_real_mode_request(
                request=request,
                genie_config=genie_config,
                input_tokens=input_tokens,
            )

        request_payload = {
            "sample_id": sample_id,
            "task_type": request.task_type.value,
            "backend": request.backend,
            "model": request.model,
            "model_revision": request.model_revision,
            "sample_spec": request.sample_spec.model_dump(mode="json"),
            "temporal": temporal.model_dump(mode="json") if temporal else None,
            "token_input": request.token_input.model_dump(mode="json") if request.token_input else None,
            "task_config": task_config.model_dump(mode="json") if task_config else None,
            "genie_config": genie_config.model_dump(mode="json"),
            "runner_mode": self._runner.mode,
        }
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True))
        stage_timings_ms["state_token_prep_ms"] = round((time.perf_counter() - prep_start) * 1000.0, 3)

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
                    "runner_mode": self._runner.mode,
                    "prompt": request.sample_spec.prompt,
                    "controls": request.sample_spec.controls,
                    "token_input": token_input_runtime,
                    "genie_config": genie_config.model_dump(mode="json"),
                },
            ),
            status=TemporalStatus.ACTIVE,
        )

        seed = request.sample_spec.seed or 42
        runner_start = time.perf_counter()
        run_result: GenieRunResult = self._runner.run(
            output_dir=sample_dir,
            prompt=request.sample_spec.prompt or "",
            seed=seed,
            num_frames=num_frames,
            input_tokens=input_tokens,
            num_prompt_frames=genie_config.num_prompt_frames,
            maskgit_steps=genie_config.maskgit_steps,
            temperature=genie_config.temperature,
        )
        stage_timings_ms["runner_exec_ms"] = round((time.perf_counter() - runner_start) * 1000.0, 3)

        mode = run_result.mode
        request_payload["runner_mode"] = mode
        if run_result.extra.get("fallback_from"):
            request_payload["fallback_from"] = run_result.extra["fallback_from"]
            request_payload["fallback_error"] = run_result.extra.get("fallback_error")
        persist_start = time.perf_counter()
        request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True))

        status = SampleStatus.SUCCEEDED
        error_info: str | None = None
        if run_result.error:
            status = SampleStatus.FAILED
            error_info = run_result.error

        rollout.metadata["runner_mode"] = mode
        if run_result.extra.get("fallback_from"):
            rollout.metadata["fallback_from"] = run_result.extra["fallback_from"]
            rollout.metadata["fallback_error"] = run_result.extra.get("fallback_error")

        log_lines = [
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
        ]
        if token_input_runtime.get("magvit2_scaffold"):
            log_lines.append("MAGVIT2 scaffold present: raw token path accepted; native MAGVIT2 tokenization is not implemented yet.")
        if run_result.error:
            log_lines.append(f"ERROR: {run_result.error}")
        if run_result.extra.get("fallback_from"):
            log_lines.append(f"FALLBACK: {run_result.extra['fallback_from']} -> {mode}")
        log_path.write_text("\n".join(log_lines) + "\n")

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
                artifact_ids=[f"{sample_id}:tokens", f"{sample_id}:state", f"{sample_id}:checkpoint", f"{sample_id}:recovery"],
                metadata={
                    "runner_mode": mode,
                    "source_backend": self.backend_name,
                    "prompt": request.sample_spec.prompt,
                    "tokens_generated": run_result.tokens_generated,
                    "frames_generated": run_result.frames_generated,
                    "sample_id": sample_id,
                    "token_input": token_input_runtime,
                    "genie_config": genie_config.model_dump(mode="json"),
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
            "timestamps": {"started_at": started_at, "completed_at": completed_at},
        }
        checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2, sort_keys=True))
        recovery_path.write_text(json.dumps({
            **checkpoint_payload,
            "recovery_hint": "Reload tokens_path into token_input or fork from checkpoint/state_handle for retry.",
        }, indent=2, sort_keys=True))

        checkpoint = self.temporal_store.create_checkpoint(
            CheckpointCreate(
                episode_id=episode.episode_id,
                rollout_id=rollout.rollout_id,
                branch_id=temporal.branch_id,
                state_handle_id=output_state.state_handle_id,
                artifact_ids=[f"{sample_id}:checkpoint", f"{sample_id}:recovery", f"{sample_id}:tokens", f"{sample_id}:state"],
                step_index=max(step_count - 1, 0),
                tag="terminal",
                metadata={
                    "runner_mode": mode,
                    "frames_generated": run_result.frames_generated,
                    "checkpoint_path": str(checkpoint_path),
                    "recovery_path": str(recovery_path),
                    "sample_id": sample_id,
                    "genie_config": genie_config.model_dump(mode="json"),
                },
            )
        )

        output_state.checkpoint_id = checkpoint.checkpoint_id
        output_state.artifact_ids = list(dict.fromkeys([
            *output_state.artifact_ids,
            f"{sample_id}:request",
            f"{sample_id}:runtime",
            f"{sample_id}:log",
        ]))
        output_state.metadata["checkpoint_id"] = checkpoint.checkpoint_id
        output_state.metadata["checkpoint_path"] = str(checkpoint_path)
        output_state.metadata["recovery_path"] = str(recovery_path)
        self.temporal_store.state_handles.put(output_state)
        stage_timings_ms["artifact_persist_ms"] = round((time.perf_counter() - persist_start) * 1000.0, 3)

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
        rollout.checkpoint_ids = list(dict.fromkeys([*rollout.checkpoint_ids, checkpoint.checkpoint_id]))
        rollout.artifact_ids = [artifact.artifact_id for artifact in artifacts]
        rollout.metadata["checkpoint_id"] = checkpoint.checkpoint_id
        rollout.metadata["checkpoint_path"] = str(checkpoint_path)
        rollout.metadata["recovery_path"] = str(recovery_path)

        rollout.metrics = {
            "steps": float(step_count),
            "estimated_units": estimate.estimated_units,
            "frames_generated": float(run_result.frames_generated),
            "tokens_generated": float(run_result.tokens_generated),
            "elapsed_s": run_result.elapsed_s,
            "runner_load_ms": stage_timings_ms["runner_load_ms"],
            "state_token_prep_ms": stage_timings_ms["state_token_prep_ms"],
            "runner_exec_ms": stage_timings_ms["runner_exec_ms"],
            "artifact_persist_ms": stage_timings_ms["artifact_persist_ms"],
        }
        temporal_persist_start = time.perf_counter()
        episode.updated_at = time.time()
        self.temporal_store.episodes.put(episode)
        stage_timings_ms["temporal_persist_ms"] = round((time.perf_counter() - temporal_persist_start) * 1000.0, 3)

        total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 3)
        stage_timings_ms["total_elapsed_ms"] = total_elapsed_ms
        rollout.metrics["temporal_persist_ms"] = stage_timings_ms["temporal_persist_ms"]
        rollout.metrics["total_elapsed_ms"] = total_elapsed_ms
        rollout.metadata["stage_timings_ms"] = stage_timings_ms
        self.temporal_store.update_rollout(rollout)

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
            "status_history": [
                {"status": SampleStatus.QUEUED.value, "timestamp": started_at},
                {"status": SampleStatus.RUNNING.value, "timestamp": started_at},
                {"status": status.value, "timestamp": completed_at},
            ],
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_ms": round((completed_at - started_at) * 1000, 2),
            "stage_timings_ms": stage_timings_ms,
        }
        if run_result.extra.get("fallback_from"):
            runtime["fallback_from"] = run_result.extra["fallback_from"]
            runtime["fallback_error"] = run_result.extra.get("fallback_error")
        if error_info:
            runtime["error"] = error_info
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True))

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
                "notes": (
                    f"Genie rollout executed via {mode} runner. "
                    f"{run_result.frames_generated} frames generated, {run_result.tokens_generated} tokens produced."
                ),
            },
        )

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
            },
        )
