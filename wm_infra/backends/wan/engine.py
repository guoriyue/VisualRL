"""In-process Wan engine adapters and multi-stage scheduler."""

from __future__ import annotations

import gc
import hashlib
import importlib
import math
import numpy as np
import os
import random
import sys
import time
from urllib.parse import unquote, urlparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wm_infra.api.metrics import SERVING_COMPILED_PROFILE_EVENTS
from wm_infra.controlplane.schemas import ProduceSampleRequest, WanTaskConfig
from wm_infra.runtime import (
    CallableGenerationStage,
    ComposedGenerationPipeline,
    GenerationPipelineStageSpec,
    GenerationRuntimeConfig,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for pure metadata paths
    torch = None  # type: ignore[assignment]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _clone_tensor_list_to_cpu(tensors: list[Any]) -> list[Any]:
    return [tensor.detach().cpu() for tensor in tensors]


def _move_tensor_list_to_device(tensors: list[Any], device: Any) -> list[Any]:
    return [tensor.to(device) for tensor in tensors]


def _clone_tensor_to_cpu(tensor: Any) -> Any:
    return None if tensor is None else tensor.detach().cpu()


def _move_tensor_to_device(tensor: Any, device: Any, dtype: Any | None = None) -> Any:
    if tensor is None:
        return None
    kwargs = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


def resolve_wan_reference_path(reference: str) -> str:
    if reference.startswith("file://"):
        parsed = urlparse(reference)
        return unquote(parsed.path)
    return reference


@dataclass(slots=True)
class WanExecutionContext:
    """Per-request execution state for Wan pipeline scheduling."""

    sample_id: str
    request: ProduceSampleRequest
    wan_config: WanTaskConfig
    sample_dir: Path
    plan_path: Path
    log_path: Path
    video_path: Path
    runtime_path: Path
    batch_size: int
    batch_index: int
    batch_sample_ids: list[str]
    scheduler_payload: dict[str, Any]
    engine_profile: dict[str, Any]

    @property
    def prompt(self) -> str:
        return (self.request.sample_spec.prompt or "").strip()

    @property
    def negative_prompt(self) -> str:
        return (self.request.sample_spec.negative_prompt or "").strip()

    @property
    def prompt_cache_key(self) -> str:
        return _stable_hash(f"{self.request.model}|{self.prompt}|{self.negative_prompt}")

    @property
    def conditioning_cache_key(self) -> str | None:
        if not self.request.sample_spec.references:
            return None
        return _stable_hash(
            f"{self.request.task_type.value}|"
            f"{'|'.join(self.request.sample_spec.references)}|"
            f"{self.wan_config.width}x{self.wan_config.height}|"
            f"frames={self.wan_config.frame_count}"
        )


@dataclass(slots=True)
class WanStagePlanEntry:
    """One scheduler-visible stage in the Wan execution graph."""

    name: str
    component: str
    device: str
    worker: str
    optional: bool = False


@dataclass(slots=True)
class WanStageUpdate:
    """Result payload returned by a stage implementation."""

    state_updates: dict[str, Any] = field(default_factory=dict)
    runtime_state_updates: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cache_hit: bool | None = None
    status: str = "succeeded"


@dataclass(slots=True)
class WanCompiledGraphDescriptor:
    """Stage-local compiled graph lifecycle metadata."""

    graph_id: str
    graph_key: str
    family_key: str
    stage: str
    device: str
    batch_size: int
    capture_enabled: bool
    capture_state: str
    reuse_count: int
    warmup_runs: int
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WanCompiledGraphEntry:
    """Persistent graph lifecycle entry cached across requests."""

    descriptor: WanCompiledGraphDescriptor
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    capture_count: int = 0
    replay_count: int = 0
    capture_latency_ms: float = 0.0
    capture_backend: str | None = None
    workload: Any | None = None


class WanCompiledStageWorkload:
    """Optional stage-local workload that can back a real graph lifecycle."""

    backend_name = "none"
    last_execute_backend = "none"
    last_replayed = False
    runtime_bound = False

    def execute(self) -> dict[str, Any]:
        self.last_execute_backend = self.backend_name
        self.last_replayed = False
        return {
            "compute_backend": self.backend_name,
            "replayed": False,
            "captured": False,
            "runtime_bound": self.runtime_bound,
        }

    def bind_runtime_value(self, value: Any) -> None:
        self.runtime_bound = value is not None

    def capture(self) -> tuple[bool, float, list[str]]:
        return False, 0.0, ["No compiled workload was registered for this stage family."]

    def replay(self) -> bool:
        return False

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend_name}


class WanTorchCudaGraphWorkload(WanCompiledStageWorkload):
    """Minimal torch CUDA graph workload for stage-local capture/replay."""

    backend_name = "torch.cuda.CUDAGraph"

    def __init__(self, *, logical_shape: tuple[int, ...], device: str, stage_name: str) -> None:
        self.logical_shape = tuple(int(max(dim, 1)) for dim in logical_shape)
        self.device = device
        self.stage_name = stage_name
        self.compute_shape = self._bounded_shape(self.logical_shape)
        self._graph = None
        self._static_input = None
        self._static_output = None
        self._captured = False
        self._eager_runs = 0
        self.last_execute_backend = "none"
        self.last_replayed = False
        self.runtime_bound = False
        self.runtime_shape = list(self.logical_shape)
        self.runtime_dtype = "float32"
        self.runtime_source = "shape_only"

    @staticmethod
    def _bounded_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        bounded: list[int] = []
        total = 1
        for dim in shape:
            if total >= 32768:
                bounded.append(1)
                continue
            next_dim = min(int(dim), max(1, 32768 // total))
            bounded.append(next_dim)
            total *= next_dim
        return tuple(bounded)

    def _supported(self) -> bool:
        return (
            torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "CUDAGraph")
        )

    def _device(self) -> Any:
        assert torch is not None
        return torch.device(self.device)

    def _run_kernel(self) -> None:
        assert self._static_input is not None
        assert self._static_output is not None
        self._static_output.copy_(self._static_input)
        self._static_output.mul_(1.01)
        self._static_output.add_(0.5)
        self._static_output.relu_()

    def _ensure_buffers(self) -> None:
        if self._static_input is not None and self._static_output is not None:
            return
        if torch is not None:
            if self._supported():
                device = self._device()
                self._static_input = torch.arange(
                    math.prod(self.compute_shape),
                    dtype=torch.float32,
                    device=device,
                ).reshape(self.compute_shape)
                self._static_output = torch.empty_like(self._static_input)
            else:
                self._static_input = torch.arange(
                    math.prod(self.compute_shape),
                    dtype=torch.float32,
                ).reshape(self.compute_shape)
                self._static_output = torch.empty_like(self._static_input)
        else:
            raise RuntimeError("WanTorchCudaGraphWorkload requires torch for stage-local compute.")

    def bind_runtime_value(self, value: Any) -> None:
        self.runtime_bound = value is not None
        if value is None:
            return
        source = value
        if isinstance(value, (list, tuple)) and value:
            source = value[0]
        if torch is not None and hasattr(source, "shape"):
            shape = tuple(int(max(dim, 1)) for dim in source.shape)
            dtype = str(getattr(source, "dtype", "float32"))
            device = str(getattr(source, "device", self.device))
        else:
            array = np.asarray(source)
            shape = tuple(int(max(dim, 1)) for dim in array.shape)
            dtype = str(array.dtype)
            device = "cpu"
        if shape and shape != tuple(self.runtime_shape):
            self.runtime_shape = list(shape)
            self.compute_shape = self._bounded_shape(shape)
            self._static_input = None
            self._static_output = None
            self._graph = None
            self._captured = False
        self.runtime_dtype = dtype
        self.runtime_source = device

    def execute(self) -> dict[str, Any]:
        self._ensure_buffers()
        assert self._static_output is not None
        if self._captured and self.replay():
            output = self._static_output.detach().float()
            checksum = round(float(output.sum().item()), 3)
            self.last_execute_backend = "cuda_graph_replay"
            self.last_replayed = True
            return {
                "compute_backend": "cuda_graph_replay",
                "replayed": True,
                "captured": True,
                "checksum": checksum,
                "numel": int(output.numel()),
                "compute_shape": list(self.compute_shape),
                "runtime_shape": list(self.runtime_shape),
                "runtime_dtype": self.runtime_dtype,
                "runtime_source": self.runtime_source,
                "runtime_bound": self.runtime_bound,
            }

        self._run_kernel()
        self._eager_runs += 1
        output = self._static_output.detach().float()
        checksum = round(float(output.sum().item()), 3)
        self.last_execute_backend = "eager_stage_compute"
        self.last_replayed = False
        return {
            "compute_backend": "eager_stage_compute",
            "replayed": False,
            "captured": self._captured,
            "checksum": checksum,
            "numel": int(output.numel()),
            "compute_shape": list(self.compute_shape),
            "runtime_shape": list(self.runtime_shape),
            "runtime_dtype": self.runtime_dtype,
            "runtime_source": self.runtime_source,
            "eager_runs": self._eager_runs,
            "runtime_bound": self.runtime_bound,
        }

    def capture(self) -> tuple[bool, float, list[str]]:
        if not self._supported():
            return False, 0.0, ["CUDA graph capture requested but torch CUDA graphs are unavailable."]
        if self._captured:
            return True, 0.0, ["CUDA graph was already captured for this stage family."]

        assert torch is not None
        device = self._device()
        stream = torch.cuda.Stream(device=device)
        self._ensure_buffers()
        torch.cuda.synchronize(device=device)
        started_at = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._run_kernel()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._run_kernel()
        stream.synchronize()
        self._graph = graph
        self._captured = True
        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        return True, latency_ms, ["Captured a stage-local torch CUDA graph workload."]

    def replay(self) -> bool:
        if not self._captured or self._graph is None:
            return False
        self._graph.replay()
        return True

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "logical_shape": list(self.logical_shape),
                "compute_shape": list(self.compute_shape),
                "captured": self._captured,
                "eager_runs": self._eager_runs,
                "last_execute_backend": self.last_execute_backend,
                "last_replayed": self.last_replayed,
                "runtime_shape": list(self.runtime_shape),
                "runtime_dtype": self.runtime_dtype,
                "runtime_source": self.runtime_source,
                "runtime_bound": self.runtime_bound,
                "device": self.device,
                "stage": self.stage_name,
            }
        )
        return payload


class WanCompiledGraphManager:
    """Best-effort stage-local compiled graph lifecycle scaffold.

    This intentionally starts as a metadata-first lifecycle. It gives the
    service a stable notion of graph families, warmup, capture eligibility, and
    cross-request reuse before real CUDA graph replay is wired in.
    """

    def __init__(self, *, enable_cuda_graphs: bool | None = None, warmup_runs: int = 1) -> None:
        env_flag = os.environ.get("WM_WAN_ENABLE_CUDA_GRAPHS", "")
        self.enable_cuda_graphs = bool(enable_cuda_graphs) if enable_cuda_graphs is not None else env_flag not in {"", "0", "false", "False"}
        self.warmup_runs = max(int(warmup_runs), 1)
        self._entries: dict[str, WanCompiledGraphEntry] = {}

    def begin_stage(self, stage_name: str, stage_device: str, context: WanExecutionContext) -> dict[str, Any]:
        profile = context.engine_profile.get("compiled_profile") or {}
        family = profile.get("execution_family") or context.engine_profile.get("execution_family") or {}
        family_key = family.get("cache_key") or context.engine_profile.get("profile_id", "wan-profile")
        graph_key = _stable_hash(f"{family_key}|{stage_name}|{stage_device}|bs={context.batch_size}")
        graph_id = f"wan-graph:{stage_name}:{graph_key}"
        capture_enabled = self.enable_cuda_graphs and stage_device.startswith("cuda")
        entry = self._entries.get(graph_key)
        if entry is None:
            capture_state = "warmup"
            reuse_count = 0
            notes = ["First execution for this stage/family graph."]
            entry = WanCompiledGraphEntry(
                descriptor=WanCompiledGraphDescriptor(
                    graph_id=graph_id,
                    graph_key=graph_key,
                    family_key=family_key,
                    stage=stage_name,
                    device=stage_device,
                    batch_size=context.batch_size,
                    capture_enabled=capture_enabled,
                    capture_state=capture_state,
                    reuse_count=reuse_count,
                    warmup_runs=self.warmup_runs,
                    notes=list(notes),
                )
            )
            self._entries[graph_key] = entry
        else:
            entry.last_used_at = time.time()
            entry.descriptor.reuse_count += 1
            reuse_count = entry.descriptor.reuse_count
            if capture_enabled and entry.capture_count < self.warmup_runs:
                capture_state = "capture_ready"
                notes = ["Warmup threshold reached; stage is eligible for CUDA graph capture."]
            else:
                capture_state = "reused"
                notes = ["Reused stage-local compiled graph family."]
            entry.descriptor.capture_state = capture_state
            entry.descriptor.notes = list(notes)
            entry.descriptor.batch_size = context.batch_size
            entry.descriptor.device = stage_device
            entry.descriptor.stage = stage_name

        return {
            "graph_id": entry.descriptor.graph_id,
            "graph_key": entry.descriptor.graph_key,
            "family_key": entry.descriptor.family_key,
            "capture_enabled": entry.descriptor.capture_enabled,
            "capture_state": entry.descriptor.capture_state,
            "reuse_count": entry.descriptor.reuse_count,
            "warmup_runs": entry.descriptor.warmup_runs,
            "replay_count": entry.replay_count,
            "capture_count": entry.capture_count,
            "capture_latency_ms": entry.capture_latency_ms,
            "capture_backend": entry.capture_backend,
            "workload": None if entry.workload is None else entry.workload.describe(),
            "notes": list(entry.descriptor.notes),
        }

    def attach_workload(
        self,
        graph_key: str,
        workload: WanCompiledStageWorkload | None,
    ) -> WanCompiledStageWorkload | None:
        entry = self._entries.get(graph_key)
        if entry is None:
            return workload
        if workload is not None and entry.workload is None:
            entry.workload = workload
            entry.capture_backend = workload.backend_name
        return entry.workload or workload

    def maybe_replay(self, graph_key: str) -> dict[str, Any] | None:
        entry = self._entries.get(graph_key)
        if entry is None:
            return None
        entry.last_used_at = time.time()
        if entry.capture_count > 0 and entry.workload is not None and entry.workload.replay():
            entry.replay_count += 1
            entry.descriptor.capture_state = "replayed"
            entry.descriptor.notes = ["Replayed a captured stage-local compiled workload."]
            return {
                "capture_state": entry.descriptor.capture_state,
                "replay_count": entry.replay_count,
                "capture_count": entry.capture_count,
                "capture_latency_ms": entry.capture_latency_ms,
                "capture_backend": entry.capture_backend,
                "workload": entry.workload.describe(),
                "notes": list(entry.descriptor.notes),
            }
        return None

    def finish_stage(self, graph_key: str, *, workload: WanCompiledStageWorkload | None = None) -> dict[str, Any]:
        entry = self._entries.get(graph_key)
        if entry is None:
            return {}
        entry.last_used_at = time.time()
        if workload is not None and entry.workload is None:
            entry.workload = workload
            entry.capture_backend = workload.backend_name
        if (
            entry.descriptor.capture_enabled
            and entry.workload is not None
            and entry.capture_count == 0
            and entry.descriptor.reuse_count >= self.warmup_runs
        ):
            captured, latency_ms, notes = entry.workload.capture()
            if captured:
                entry.capture_count += 1
                entry.capture_latency_ms = latency_ms
                entry.descriptor.capture_state = "captured"
                entry.descriptor.notes = list(notes)
            elif notes:
                entry.descriptor.notes = list(notes)
        if workload is not None and getattr(workload, "last_replayed", False):
            entry.replay_count += 1
            entry.descriptor.capture_state = "replayed"
            entry.descriptor.notes = ["Stage-local compiled workload replayed during execution."]
        SERVING_COMPILED_PROFILE_EVENTS.labels(
            backend="wan-video",
            event=entry.descriptor.capture_state,
        ).inc()
        return {
            "graph_id": entry.descriptor.graph_id,
            "graph_key": entry.descriptor.graph_key,
            "family_key": entry.descriptor.family_key,
            "capture_enabled": entry.descriptor.capture_enabled,
            "capture_state": entry.descriptor.capture_state,
            "reuse_count": entry.descriptor.reuse_count,
            "warmup_runs": entry.descriptor.warmup_runs,
            "replay_count": entry.replay_count,
            "capture_count": entry.capture_count,
            "capture_latency_ms": entry.capture_latency_ms,
            "capture_backend": entry.capture_backend,
            "workload": None if entry.workload is None else entry.workload.describe(),
            "notes": list(entry.descriptor.notes),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "warmup_runs": self.warmup_runs,
            "graph_count": len(self._entries),
            "graphs": [
                {
                    "graph_id": entry.descriptor.graph_id,
                    "graph_key": entry.descriptor.graph_key,
                    "family_key": entry.descriptor.family_key,
                    "stage": entry.descriptor.stage,
                    "device": entry.descriptor.device,
                    "batch_size": entry.descriptor.batch_size,
                    "capture_state": entry.descriptor.capture_state,
                    "capture_enabled": entry.descriptor.capture_enabled,
                    "reuse_count": entry.descriptor.reuse_count,
                    "capture_count": entry.capture_count,
                    "replay_count": entry.replay_count,
                    "capture_latency_ms": entry.capture_latency_ms,
                    "capture_backend": entry.capture_backend,
                }
                for entry in self._entries.values()
            ],
        }


class WanEngineAdapter(ABC):
    """Abstract in-process Wan engine interface."""

    adapter_name = "wan-engine-adapter"
    mode = "real"
    execution_backend = "in_process_stage_scheduler"
    supports_output_video = True
    supports_cross_request_batching = True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.adapter_name,
            "mode": self.mode,
            "execution_backend": self.execution_backend,
            "supports_output_video": self.supports_output_video,
            "supports_cross_request_batching": self.supports_cross_request_batching,
        }

    def stage_device_map(self, context: WanExecutionContext) -> dict[str, tuple[str, str]]:
        """Return ``stage -> (device, worker)`` placement."""
        return {
            "text_encode": ("cpu", "text-encoder"),
            "conditioning_encode": ("cpu", "conditioning-encoder"),
            "diffusion": ("cuda:0", "dit"),
            "vae_decode": ("cuda:0", "vae"),
            "safety": ("cpu", "safety"),
            "postprocess": ("cpu", "postprocess"),
            "persist": ("cpu", "artifact-writer"),
        }

    def build_stage_plan(self, context: WanExecutionContext) -> list[WanStagePlanEntry]:
        placements = self.stage_device_map(context)
        stage_names = ["text_encode"]
        if context.request.sample_spec.references:
            stage_names.append("conditioning_encode")
        stage_names.extend(["diffusion", "vae_decode", "safety", "postprocess", "persist"])
        return [
            WanStagePlanEntry(
                name=stage_name,
                component=stage_name.replace("_", "-"),
                device=placements.get(stage_name, ("cpu", stage_name))[0],
                worker=placements.get(stage_name, ("cpu", stage_name))[1],
                optional=stage_name in {"conditioning_encode", "safety"},
            )
            for stage_name in stage_names
        ]

    async def run_stage(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanStageUpdate:
        handlers = {
            "text_encode": self.run_text_encode,
            "conditioning_encode": self.run_conditioning_encode,
            "diffusion": self.run_diffusion,
            "vae_decode": self.run_vae_decode,
            "safety": self.run_safety,
            "postprocess": self.run_postprocess,
            "persist": self.run_persist,
        }
        handler = handlers.get(stage_name)
        if handler is None:
            raise ValueError(f"Unknown Wan stage: {stage_name}")
        return await handler(context, state)

    @abstractmethod
    async def run_text_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        """Produce prompt encodings."""

    async def run_conditioning_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return WanStageUpdate(notes=["No conditioning inputs were provided."])

    @abstractmethod
    async def run_diffusion(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        """Produce latent video samples."""

    @abstractmethod
    async def run_vae_decode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        """Decode latent samples into frames."""

    async def run_safety(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return WanStageUpdate(
            state_updates={"safety_result": "not_run"},
            runtime_state_updates={"safety_result": "not_run"},
            outputs={"decision": "skipped"},
            notes=["Safety stage is wired but not enforced by this adapter."],
        )

    @abstractmethod
    async def run_postprocess(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        """Assemble decoded frames into output-ready payloads."""

    @abstractmethod
    async def run_persist(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        """Persist stage-local artifacts and output metadata."""

    def build_compiled_stage_workload(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanCompiledStageWorkload | None:
        """Return an optional stage-local compiled workload for graph capture/replay."""
        return None

    @staticmethod
    def _stage_workload(state: dict[str, Any], stage_name: str) -> WanCompiledStageWorkload | None:
        workloads = state.get("_compiled_stage_workloads")
        if not isinstance(workloads, dict):
            return None
        workload = workloads.get(stage_name)
        return workload if isinstance(workload, WanCompiledStageWorkload) else None


class StubWanEngineAdapter(WanEngineAdapter):
    """In-process stub adapter used to exercise the Wan stage scheduler."""

    adapter_name = "stub-wan-engine"
    mode = "stub"
    supports_output_video = False

    def __init__(self) -> None:
        self._prompt_cache: dict[str, str] = {}
        self._conditioning_cache: dict[str, str] = {}

    @staticmethod
    def _stage_workload(state: dict[str, Any], stage_name: str) -> WanCompiledStageWorkload | None:
        workloads = state.get("_compiled_stage_workloads")
        if not isinstance(workloads, dict):
            return None
        workload = workloads.get(stage_name)
        return workload if isinstance(workload, WanCompiledStageWorkload) else None

    def build_compiled_stage_workload(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanCompiledStageWorkload | None:
        if stage_name == "diffusion":
            latent_shape = (
                context.wan_config.frame_count,
                max(1, context.wan_config.height // 8),
                max(1, context.wan_config.width // 8),
                16,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=latent_shape,
                device="cuda:0",
                stage_name=stage_name,
            )
        if stage_name == "vae_decode":
            frame_shape = (
                context.wan_config.frame_count,
                context.wan_config.height,
                context.wan_config.width,
                3,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=frame_shape,
                device="cuda:0",
                stage_name=stage_name,
            )
        return None

    async def run_text_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        cache_key = context.prompt_cache_key
        cache_hit = cache_key in self._prompt_cache
        embedding_id = self._prompt_cache.setdefault(cache_key, f"prompt-embed-{cache_key}")
        negative_embedding_id = None
        if context.negative_prompt:
            negative_key = _stable_hash(f"neg|{context.request.model}|{context.negative_prompt}")
            negative_embedding_id = self._prompt_cache.setdefault(negative_key, f"prompt-embed-{negative_key}")
        return WanStageUpdate(
            state_updates={
                "prompt_embedding_id": embedding_id,
                "negative_prompt_embedding_id": negative_embedding_id,
            },
            runtime_state_updates={
                "prompt_embedding_id": embedding_id,
                "negative_prompt_embedding_id": negative_embedding_id,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(context.prompt.split())),
                "prompt_cache_key": cache_key,
            },
            notes=[
                "Prompt encoder output was materialized inside the process.",
            ],
            cache_hit=cache_hit,
        )

    async def run_conditioning_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        cache_key = context.conditioning_cache_key
        if cache_key is None:
            return WanStageUpdate(notes=["Conditioning stage skipped because there are no references."])
        cache_hit = cache_key in self._conditioning_cache
        conditioning_id = self._conditioning_cache.setdefault(cache_key, f"conditioning-{cache_key}")
        return WanStageUpdate(
            state_updates={"conditioning_id": conditioning_id},
            runtime_state_updates={"conditioning_id": conditioning_id},
            outputs={
                "reference_count": len(context.request.sample_spec.references),
                "conditioning_cache_key": cache_key,
            },
            notes=["Reference inputs were staged for image/video conditioning reuse."],
            cache_hit=cache_hit,
        )

    async def run_diffusion(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        workload = self._stage_workload(state, "diffusion")
        if workload is not None:
            workload.bind_runtime_value(
                (
                    context.wan_config.frame_count,
                    max(1, context.wan_config.height // 8),
                    max(1, context.wan_config.width // 8),
                    16,
                )
            )
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }
        latent_id = f"latent-{_stable_hash(f'{context.sample_id}|{context.wan_config.num_steps}|{context.batch_size}')}"
        latent_shape = [
            context.wan_config.frame_count,
            max(1, context.wan_config.height // 8),
            max(1, context.wan_config.width // 8),
            16,
        ]
        return WanStageUpdate(
            state_updates={
                "latent_id": latent_id,
                "latent_shape": latent_shape,
            },
            runtime_state_updates={
                "latent_id": latent_id,
                "latent_shape": latent_shape,
            },
            outputs={
                "sampler": "stub-dpmpp-2m",
                "num_steps": context.wan_config.num_steps,
                "guidance_scale": context.wan_config.guidance_scale,
                "profile_id": context.engine_profile["profile_id"],
                "compute_profile": compute_profile,
            },
            notes=[
                "Diffusion stage used the in-process scheduler instead of an external generate.py runner.",
                f"Batch occupancy {context.batch_index + 1}/{context.batch_size} under profile {context.engine_profile['profile_id']}.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_vae_decode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        workload = self._stage_workload(state, "vae_decode")
        if workload is not None:
            workload.bind_runtime_value(
                (
                    context.wan_config.frame_count,
                    context.wan_config.height,
                    context.wan_config.width,
                    3,
                )
            )
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }
        return WanStageUpdate(
            state_updates={
                "decoded_frame_count": context.wan_config.frame_count,
            },
            runtime_state_updates={
                "decoded_frame_count": context.wan_config.frame_count,
            },
            outputs={
                "frame_dimensions": [context.wan_config.width, context.wan_config.height],
                "latent_id": state.get("latent_id"),
                "compute_profile": compute_profile,
            },
            notes=[
                "Latent video tiles were decoded into frame tensors.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_safety(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return WanStageUpdate(
            state_updates={"safety_result": "passed"},
            runtime_state_updates={"safety_result": "passed"},
            outputs={"decision": "allow"},
            notes=["Safety path ran as a stage-local no-op policy check."],
        )

    async def run_postprocess(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return WanStageUpdate(
            state_updates={
                "container": "mp4",
                "output_fps": context.request.sample_spec.fps or 16,
            },
            runtime_state_updates={
                "container": "mp4",
                "output_fps": context.request.sample_spec.fps or 16,
            },
            outputs={
                "preview_frame_count": min(context.wan_config.frame_count, 4),
                "watermark_applied": False,
            },
            notes=["Postprocess assembled frame payloads but left video emission disabled in stub mode."],
        )

    async def run_persist(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return WanStageUpdate(
            state_updates={"video_persisted": False},
            runtime_state_updates={"video_persisted": False},
            outputs={
                "log_path": str(context.log_path),
                "output_path": str(context.video_path),
                "runtime_path": str(context.runtime_path),
            },
            notes=["Persist stage wrote scheduler metadata but did not emit a video artifact in stub mode."],
        )


class OfficialWanInProcessAdapter(WanEngineAdapter):
    """Real in-process Wan adapter backed by the official Wan2.2 Python modules."""

    adapter_name = "wan22-official-python"
    mode = "real"
    supports_output_video = True
    supports_cross_request_batching = False

    def __init__(
        self,
        *,
        repo_dir: str | Path,
        default_checkpoint_dir: str | Path | None = None,
        device_id: int = 0,
        default_t5_cpu: bool = True,
        default_convert_model_dtype: bool = True,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.default_checkpoint_dir = None if default_checkpoint_dir is None else Path(default_checkpoint_dir)
        self.device_id = device_id
        self.default_t5_cpu = default_t5_cpu
        self.default_convert_model_dtype = default_convert_model_dtype
        self._modules_loaded = False
        self._torch = None
        self._wan = None
        self._wan_configs = None
        self._size_configs = None
        self._max_area_configs = None
        self._save_video = None
        self._pil_image = None
        self._tv_tf = None
        self._np = None
        self._pipelines: dict[str, Any] = {}
        self._prompt_cache: dict[str, tuple[list[Any], list[Any]]] = {}
        self._conditioning_cache: dict[str, dict[str, Any]] = {}

    def _effective_seed(self, context: WanExecutionContext) -> tuple[int, str]:
        if context.request.sample_spec.seed is not None:
            return context.request.sample_spec.seed, "explicit"
        return random.randint(0, sys.maxsize), "randomized"

    def _guide_scale_pair(self, context: WanExecutionContext) -> tuple[float, float]:
        low_noise = float(context.wan_config.guidance_scale)
        high_noise = (
            low_noise
            if context.wan_config.high_noise_guidance_scale is None
            else float(context.wan_config.high_noise_guidance_scale)
        )
        return low_noise, high_noise

    def _validate_checkpoint_layout(self, task_key: str, checkpoint_dir: Path, config: Any) -> None:
        required_paths: list[tuple[Path, str]] = [
            (checkpoint_dir / config.t5_checkpoint, "T5 checkpoint"),
            (checkpoint_dir / config.t5_tokenizer, "T5 tokenizer"),
            (checkpoint_dir / config.vae_checkpoint, "VAE checkpoint"),
        ]
        if task_key.startswith("i2v-"):
            required_paths.extend(
                [
                    (checkpoint_dir / config.low_noise_checkpoint, "low-noise checkpoint directory"),
                    (checkpoint_dir / config.high_noise_checkpoint, "high-noise checkpoint directory"),
                ]
            )
        for path, label in required_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Incomplete Wan checkpoint layout for {task_key}: missing {label} at {path}"
                )
        for subfolder_attr, label in (
            ("low_noise_checkpoint", "low-noise"),
            ("high_noise_checkpoint", "high-noise"),
        ):
            subfolder = getattr(config, subfolder_attr, None)
            if subfolder is None:
                continue
            shard_dir = checkpoint_dir / subfolder
            has_weight_file = any(shard_dir.glob("*.safetensors")) or any(shard_dir.glob("*.bin"))
            has_index = any(shard_dir.glob("*.index.json"))
            if not has_weight_file and not has_index:
                raise FileNotFoundError(
                    f"Incomplete Wan checkpoint layout for {task_key}: {label} model weights are missing under {shard_dir}"
                )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "repo_dir": str(self.repo_dir),
                "default_checkpoint_dir": None if self.default_checkpoint_dir is None else str(self.default_checkpoint_dir),
                "device": f"cuda:{self.device_id}",
            }
        )
        return payload

    def stage_device_map(self, context: WanExecutionContext) -> dict[str, tuple[str, str]]:
        text_device = "cpu" if context.wan_config.t5_cpu else f"cuda:{self.device_id}"
        return {
            "text_encode": (text_device, "umt5-xxl"),
            "conditioning_encode": (f"cuda:{self.device_id}", "conditioning-vae"),
            "diffusion": (f"cuda:{self.device_id}", "wan-dit"),
            "vae_decode": (f"cuda:{self.device_id}", "wan-vae"),
            "safety": ("cpu", "safety"),
            "postprocess": ("cpu", "video-postprocess"),
            "persist": ("cpu", "artifact-writer"),
        }

    def build_compiled_stage_workload(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanCompiledStageWorkload | None:
        if stage_name == "diffusion":
            latent_shape = (
                context.wan_config.frame_count,
                max(1, context.wan_config.height // 8),
                max(1, context.wan_config.width // 8),
                16,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=latent_shape,
                device=f"cuda:{self.device_id}",
                stage_name=stage_name,
            )
        if stage_name == "vae_decode":
            frame_shape = (
                context.wan_config.frame_count,
                context.wan_config.height,
                context.wan_config.width,
                3,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=frame_shape,
                device=f"cuda:{self.device_id}",
                stage_name=stage_name,
            )
        return None

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Wan repo_dir does not exist: {self.repo_dir}")
        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        import numpy as np
        import torch
        import torchvision.transforms.functional as TF
        import wan
        import wan.modules.attention as wan_attention
        import wan.modules.model as wan_model
        from PIL import Image
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video

        if not wan_attention.FLASH_ATTN_2_AVAILABLE and not wan_attention.FLASH_ATTN_3_AVAILABLE:
            wan_model.flash_attention = wan_attention.attention

        self._np = np
        self._torch = torch
        self._tv_tf = TF
        self._wan = wan
        self._wan_configs = WAN_CONFIGS
        self._size_configs = SIZE_CONFIGS
        self._max_area_configs = MAX_AREA_CONFIGS
        self._save_video = save_video
        self._pil_image = Image
        self._modules_loaded = True

    def _task_key(self, context: WanExecutionContext) -> str:
        model_size = context.wan_config.model_size or "A14B"
        if context.request.task_type.value == "text_to_video":
            return f"t2v-{model_size}"
        if context.request.task_type.value == "image_to_video":
            return f"i2v-{model_size}"
        raise NotImplementedError(f"Official in-process adapter does not support {context.request.task_type.value}")

    def _size_key(self, context: WanExecutionContext) -> str:
        size_key = f"{context.wan_config.width}*{context.wan_config.height}"
        if size_key not in self._size_configs:
            raise ValueError(
                f"Wan official in-process adapter does not support size {size_key}; "
                f"supported sizes are {sorted(self._size_configs.keys())}"
            )
        return size_key

    def _resolve_checkpoint_dir(self, context: WanExecutionContext, task_key: str) -> Path:
        if context.wan_config.ckpt_dir:
            return Path(context.wan_config.ckpt_dir)
        if self.default_checkpoint_dir is not None:
            return self.default_checkpoint_dir
        fallback_map = {
            "t2v-A14B": self.repo_dir / "Wan2.2-T2V-A14B",
            "i2v-A14B": self.repo_dir / "Wan2.2-I2V-A14B",
        }
        ckpt_dir = fallback_map.get(task_key)
        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError(
                f"No Wan checkpoint directory configured for {task_key}. "
                "Set wan_config.ckpt_dir or the backend default checkpoint path."
            )
        return ckpt_dir

    def _pipeline_cache_key(self, task_key: str, checkpoint_dir: Path, context: WanExecutionContext) -> str:
        return "|".join(
            [
                task_key,
                str(checkpoint_dir),
                str(self.device_id),
                str(context.wan_config.t5_cpu),
                str(context.wan_config.convert_model_dtype),
            ]
        )

    def _get_pipeline(self, context: WanExecutionContext) -> tuple[Any, str, Path]:
        self._load_modules()
        task_key = self._task_key(context)
        checkpoint_dir = self._resolve_checkpoint_dir(context, task_key)
        cache_key = self._pipeline_cache_key(task_key, checkpoint_dir, context)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is not None:
            return pipeline, task_key, checkpoint_dir

        config = self._wan_configs[task_key]
        self._validate_checkpoint_layout(task_key, checkpoint_dir, config)
        common_kwargs = {
            "config": config,
            "checkpoint_dir": str(checkpoint_dir),
            "device_id": self.device_id,
            "rank": 0,
            "t5_fsdp": False,
            "dit_fsdp": False,
            "use_sp": False,
            "t5_cpu": context.wan_config.t5_cpu if context.wan_config.t5_cpu is not None else self.default_t5_cpu,
            "convert_model_dtype": (
                context.wan_config.convert_model_dtype
                if context.wan_config.convert_model_dtype is not None
                else self.default_convert_model_dtype
            ),
        }
        if task_key.startswith("t2v-"):
            pipeline = self._wan.WanT2V(**common_kwargs)
        elif task_key.startswith("i2v-"):
            pipeline = self._wan.WanI2V(**common_kwargs)
        else:
            raise NotImplementedError(f"Official in-process adapter does not support task key {task_key}")
        self._pipelines[cache_key] = pipeline
        return pipeline, task_key, checkpoint_dir

    async def run_text_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        pipeline, task_key, checkpoint_dir = self._get_pipeline(context)
        torch = self._torch
        n_prompt = context.negative_prompt or pipeline.sample_neg_prompt
        cache_key = f"{task_key}|{checkpoint_dir}|{context.prompt_cache_key}"
        cache_hit = cache_key in self._prompt_cache

        if cache_hit:
            cached_context, cached_context_null = self._prompt_cache[cache_key]
        else:
            if not pipeline.t5_cpu:
                pipeline.text_encoder.model.to(pipeline.device)
                prompt_context = pipeline.text_encoder([context.prompt], pipeline.device)
                prompt_context_null = pipeline.text_encoder([n_prompt], pipeline.device)
                if context.wan_config.offload_model:
                    pipeline.text_encoder.model.cpu()
            else:
                prompt_context = pipeline.text_encoder([context.prompt], torch.device("cpu"))
                prompt_context_null = pipeline.text_encoder([n_prompt], torch.device("cpu"))
            cached_context = _clone_tensor_list_to_cpu(prompt_context)
            cached_context_null = _clone_tensor_list_to_cpu(prompt_context_null)
            self._prompt_cache[cache_key] = (cached_context, cached_context_null)

        device_context = _move_tensor_list_to_device(cached_context, pipeline.device)
        device_context_null = _move_tensor_list_to_device(cached_context_null, pipeline.device)
        return WanStageUpdate(
            state_updates={
                "pipeline": pipeline,
                "task_key": task_key,
                "checkpoint_dir": str(checkpoint_dir),
                "text_context": device_context,
                "text_context_null": device_context_null,
            },
            runtime_state_updates={
                "task_key": task_key,
                "checkpoint_dir": str(checkpoint_dir),
                "prompt_cache_key": cache_key,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(context.prompt.split())),
                "negative_prompt_used": bool(n_prompt),
            },
            notes=[f"Prompt encoding executed against {task_key} from {checkpoint_dir}."],
            cache_hit=cache_hit,
        )

    async def run_conditioning_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        pipeline = state["pipeline"]
        if context.request.task_type.value != "image_to_video":
            return WanStageUpdate(notes=["Conditioning stage skipped because the request is text-to-video."])
        reference_path = resolve_wan_reference_path(context.request.sample_spec.references[0])
        cache_key = f"{state['task_key']}|{reference_path}|{context.conditioning_cache_key}"
        cache_hit = cache_key in self._conditioning_cache

        if cache_hit:
            cached_conditioning = self._conditioning_cache[cache_key]
            conditioning = {
                "conditioning_tensor": _move_tensor_to_device(cached_conditioning["conditioning_tensor"], pipeline.device),
                "conditioning_shape": list(cached_conditioning["conditioning_shape"]),
                "decoded_size": list(cached_conditioning["decoded_size"]),
                "reference_path": cached_conditioning["reference_path"],
            }
        else:
            image = self._pil_image.open(reference_path).convert("RGB")
            img_tensor = self._tv_tf.to_tensor(image).sub_(0.5).div_(0.5).to(pipeline.device)
            frame_num = context.wan_config.frame_count
            h, w = img_tensor.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                math.sqrt(context.wan_config.width * context.wan_config.height * aspect_ratio)
                // pipeline.vae_stride[1]
                // pipeline.patch_size[1]
                * pipeline.patch_size[1]
            )
            lat_w = round(
                math.sqrt(context.wan_config.width * context.wan_config.height / aspect_ratio)
                // pipeline.vae_stride[2]
                // pipeline.patch_size[2]
                * pipeline.patch_size[2]
            )
            height = lat_h * pipeline.vae_stride[1]
            width = lat_w * pipeline.vae_stride[2]
            mask = self._torch.ones(1, frame_num, lat_h, lat_w, device=pipeline.device)
            mask[:, 1:] = 0
            mask = self._torch.concat(
                [self._torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]],
                dim=1,
            )
            mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
            mask = mask.transpose(1, 2)[0]
            y = pipeline.vae.encode(
                [
                    self._torch.concat(
                        [
                            self._torch.nn.functional.interpolate(
                                img_tensor[None].cpu(),
                                size=(height, width),
                                mode="bicubic",
                            ).transpose(0, 1),
                            self._torch.zeros(3, frame_num - 1, height, width),
                        ],
                        dim=1,
                    ).to(pipeline.device)
                ]
            )[0]
            conditioning = {
                "conditioning_tensor": self._torch.concat([mask, y]),
                "conditioning_shape": [int(lat_h), int(lat_w)],
                "decoded_size": [int(width), int(height)],
                "reference_path": reference_path,
            }
            self._conditioning_cache[cache_key] = {
                "conditioning_tensor": _clone_tensor_to_cpu(conditioning["conditioning_tensor"]),
                "conditioning_shape": list(conditioning["conditioning_shape"]),
                "decoded_size": list(conditioning["decoded_size"]),
                "reference_path": reference_path,
            }

        return WanStageUpdate(
            state_updates={"conditioning": conditioning},
            runtime_state_updates={
                "conditioning_shape": conditioning["conditioning_shape"],
                "conditioning_decoded_size": conditioning["decoded_size"],
                "reference_path": conditioning["reference_path"],
            },
            outputs={"reference_count": len(context.request.sample_spec.references)},
            notes=["Image conditioning tensor and mask were encoded for Wan I2V."],
            cache_hit=cache_hit,
        )

    async def run_diffusion(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        pipeline = state["pipeline"]
        torch = self._torch
        size_key = self._size_key(context)
        seed, seed_policy = self._effective_seed(context)
        seed_g = torch.Generator(device=pipeline.device)
        seed_g.manual_seed(seed)
        low_noise_guidance_scale, high_noise_guidance_scale = self._guide_scale_pair(context)
        sample_solver = context.wan_config.sample_solver

        with (
            torch.amp.autocast("cuda", dtype=pipeline.param_dtype),
            torch.no_grad(),
        ):
            if state["task_key"].startswith("t2v-"):
                width, height = self._size_configs[size_key]
                target_shape = (
                    pipeline.vae.model.z_dim,
                    (context.wan_config.frame_count - 1) // pipeline.vae_stride[0] + 1,
                    height // pipeline.vae_stride[1],
                    width // pipeline.vae_stride[2],
                )
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3])
                    / (pipeline.patch_size[1] * pipeline.patch_size[2])
                    * target_shape[1]
                    / pipeline.sp_size
                ) * pipeline.sp_size
                latents = [
                    torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=pipeline.device,
                        generator=seed_g,
                    )
                ]
                arg_c = {"context": state["text_context"], "seq_len": seq_len}
                arg_null = {"context": state["text_context_null"], "seq_len": seq_len}
            else:
                max_area = self._max_area_configs[size_key]
                frame_num = context.wan_config.frame_count
                lat_h, lat_w = state["conditioning"]["conditioning_shape"]
                max_seq_len = ((frame_num - 1) // pipeline.vae_stride[0] + 1) * lat_h * lat_w // (
                    pipeline.patch_size[1] * pipeline.patch_size[2]
                )
                max_seq_len = int(math.ceil(max_seq_len / pipeline.sp_size)) * pipeline.sp_size
                latents = torch.randn(
                    16,
                    (frame_num - 1) // pipeline.vae_stride[0] + 1,
                    lat_h,
                    lat_w,
                    dtype=torch.float32,
                    generator=seed_g,
                    device=pipeline.device,
                )
                arg_c = {
                    "context": [state["text_context"][0]],
                    "seq_len": max_seq_len,
                    "y": [state["conditioning"]["conditioning_tensor"]],
                }
                arg_null = {
                    "context": state["text_context_null"],
                    "seq_len": max_seq_len,
                    "y": [state["conditioning"]["conditioning_tensor"]],
                }

            boundary = pipeline.boundary * pipeline.num_train_timesteps
            solver_name = sample_solver
            if solver_name == "unipc":
                scheduler = importlib.import_module("wan.utils.fm_solvers_unipc").FlowUniPCMultistepScheduler(
                    num_train_timesteps=pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                scheduler.set_timesteps(
                    context.wan_config.num_steps,
                    device=pipeline.device,
                    shift=context.wan_config.shift,
                )
                timesteps = scheduler.timesteps
            else:
                fm_solvers = importlib.import_module("wan.utils.fm_solvers")
                scheduler = fm_solvers.FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = fm_solvers.get_sampling_sigmas(
                    context.wan_config.num_steps,
                    context.wan_config.shift,
                )
                timesteps, _ = fm_solvers.retrieve_timesteps(
                    scheduler,
                    device=pipeline.device,
                    sigmas=sampling_sigmas,
                )

            for t in timesteps:
                timestep = torch.stack([t]).to(pipeline.device)
                model = pipeline._prepare_model_for_timestep(t, boundary, context.wan_config.offload_model)
                sample_guide_scale = high_noise_guidance_scale if t.item() >= boundary else low_noise_guidance_scale

                if state["task_key"].startswith("t2v-"):
                    noise_pred_cond = model(latents, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model(latents, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents = [temp_x0.squeeze(0)]
                else:
                    latent_model_input = [latents.to(pipeline.device)]
                    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                    if context.wan_config.offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                    if context.wan_config.offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latents = temp_x0.squeeze(0)

            final_latents = latents if isinstance(latents, list) else [latents]
            if context.wan_config.offload_model:
                pipeline.low_noise_model.cpu()
                pipeline.high_noise_model.cpu()
                torch.cuda.empty_cache()

        latent_shape = list(final_latents[0].shape)
        workload = self._stage_workload(state, "diffusion")
        if workload is not None:
            workload.bind_runtime_value(final_latents[0])
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }
        return WanStageUpdate(
            state_updates={"latents": final_latents, "fps": pipeline.config.sample_fps},
            runtime_state_updates={
                "latent_shape": latent_shape,
                "seed": seed,
                "seed_policy": seed_policy,
                "sample_solver": solver_name,
                "guide_scale_pair": [low_noise_guidance_scale, high_noise_guidance_scale],
                "boundary_timestep": boundary,
                "max_area": None if state["task_key"].startswith("t2v-") else max_area,
            },
            outputs={
                "solver": solver_name,
                "num_steps": context.wan_config.num_steps,
                "guidance_scale": [low_noise_guidance_scale, high_noise_guidance_scale],
                "compute_profile": compute_profile,
            },
            notes=[
                f"Diffusion completed on {len(timesteps)} timesteps for {state['task_key']}.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_vae_decode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        pipeline = state["pipeline"]
        video = pipeline.vae.decode(state["latents"])[0]
        if context.wan_config.offload_model:
            gc.collect()
            self._torch.cuda.synchronize()
        workload = self._stage_workload(state, "vae_decode")
        if workload is not None:
            workload.bind_runtime_value(video)
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }
        return WanStageUpdate(
            state_updates={"video_tensor": video},
            runtime_state_updates={
                "decoded_frame_count": int(video.shape[1]),
                "decoded_spatial_size": [int(video.shape[3]), int(video.shape[2])],
            },
            outputs={"fps": state["fps"], "compute_profile": compute_profile},
            notes=[
                "VAE decode completed and produced frame tensors.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_postprocess(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        video = state["video_tensor"]
        return WanStageUpdate(
            state_updates={"output_fps": int(state["fps"])},
            runtime_state_updates={"output_fps": int(state["fps"])},
            outputs={
                "frame_count": int(video.shape[1]),
                "channels": int(video.shape[0]),
            },
            notes=["Postprocess kept the official Wan tensor layout for persistence."],
        )

    async def run_persist(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        self._save_video(
            tensor=state["video_tensor"][None],
            save_file=str(context.video_path),
            fps=state["output_fps"],
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return WanStageUpdate(
            state_updates={"video_persisted": context.video_path.exists()},
            runtime_state_updates={"video_persisted": context.video_path.exists()},
            outputs={
                "output_path": str(context.video_path),
                "fps": state["output_fps"],
            },
            notes=["Persist stage wrote the generated MP4 through the official Wan save_video utility."],
        )


class DiffusersWanI2VAdapter(WanEngineAdapter):
    """Real in-process Wan I2V adapter backed by diffusers."""

    adapter_name = "wan22-diffusers-i2v"
    mode = "real"
    supports_output_video = True
    supports_cross_request_batching = False

    def __init__(
        self,
        *,
        default_model_dir: str | Path,
        device_id: int = 0,
        default_dtype: str = "bfloat16",
    ) -> None:
        self.default_model_dir = Path(default_model_dir)
        self.device_id = device_id
        self.default_dtype = default_dtype
        self._modules_loaded = False
        self._torch = None
        self._np = None
        self._imageio = None
        self._pil_image = None
        self._pipeline_cls = None
        self._pipelines: dict[str, Any] = {}
        self._prompt_cache: dict[str, tuple[Any, Any]] = {}
        self._conditioning_cache: dict[str, Any] = {}

    def _effective_seed(self, context: WanExecutionContext) -> tuple[int, str]:
        if context.request.sample_spec.seed is not None:
            return context.request.sample_spec.seed, "explicit"
        return random.randint(0, sys.maxsize), "randomized"

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "model_dir": str(self.default_model_dir),
                "device": f"cuda:{self.device_id}",
                "task_types": ["image_to_video"],
            }
        )
        return payload

    def stage_device_map(self, context: WanExecutionContext) -> dict[str, tuple[str, str]]:
        return {
            "text_encode": ("cpu", "umt5-xxl"),
            "conditioning_encode": (f"cuda:{self.device_id}", "wan-vae"),
            "diffusion": (f"cuda:{self.device_id}", "wan-transformers"),
            "vae_decode": (f"cuda:{self.device_id}", "wan-vae"),
            "safety": ("cpu", "safety"),
            "postprocess": ("cpu", "video-postprocess"),
            "persist": ("cpu", "artifact-writer"),
        }

    def build_compiled_stage_workload(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanCompiledStageWorkload | None:
        if stage_name == "diffusion":
            latent_shape = (
                context.wan_config.frame_count,
                max(1, context.wan_config.height // 8),
                max(1, context.wan_config.width // 8),
                16,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=latent_shape,
                device=f"cuda:{self.device_id}",
                stage_name=stage_name,
            )
        if stage_name == "vae_decode":
            frame_shape = (
                context.wan_config.frame_count,
                context.wan_config.height,
                context.wan_config.width,
                3,
            )
            return WanTorchCudaGraphWorkload(
                logical_shape=frame_shape,
                device=f"cuda:{self.device_id}",
                stage_name=stage_name,
            )
        return None

    def _load_modules(self) -> None:
        if self._modules_loaded:
            return
        import imageio.v2 as imageio
        import numpy as np
        import torch
        from PIL import Image
        from diffusers import WanImageToVideoPipeline

        self._imageio = imageio
        self._np = np
        self._pil_image = Image
        self._pipeline_cls = WanImageToVideoPipeline
        self._torch = torch
        self._modules_loaded = True

    def _device(self) -> Any:
        return self._torch.device(f"cuda:{self.device_id}")

    def _resolve_model_dir(self, context: WanExecutionContext) -> Path:
        if context.request.task_type.value != "image_to_video":
            raise NotImplementedError("Diffusers Wan adapter only supports image_to_video requests")
        ckpt_dir = context.wan_config.ckpt_dir
        if ckpt_dir:
            ckpt_path = Path(ckpt_dir)
            if (ckpt_path / "model_index.json").exists():
                return ckpt_path
        if not self.default_model_dir.exists():
            raise FileNotFoundError(f"Wan diffusers model_dir does not exist: {self.default_model_dir}")
        return self.default_model_dir

    def _create_pipeline(self, model_dir: Path) -> Any:
        self._load_modules()
        torch = self._torch
        dtype = getattr(torch, self.default_dtype)
        pipeline = self._pipeline_cls.from_pretrained(str(model_dir), torch_dtype=dtype)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_sequential_cpu_offload()
        pipeline.vae.enable_tiling()
        pipeline.vae.enable_slicing()
        return pipeline

    def _get_pipeline(self, model_dir: Path) -> Any:
        cache_key = str(model_dir)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is None:
            pipeline = self._create_pipeline(model_dir)
            self._pipelines[cache_key] = pipeline
        return pipeline

    async def run_text_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        self._load_modules()
        model_dir = self._resolve_model_dir(context)
        pipeline = self._get_pipeline(model_dir)
        cache_key = (
            f"{model_dir}|{context.prompt_cache_key}|cfg={int(context.wan_config.guidance_scale > 1.0)}|"
            f"max_seq={context.scheduler_payload.get('max_sequence_length', 512)}"
        )

        return WanStageUpdate(
            state_updates={
                "pipeline": pipeline,
                "model_dir": str(model_dir),
            },
            runtime_state_updates={
                "task_key": "i2v-A14B-diffusers",
                "checkpoint_dir": str(model_dir),
                "prompt_cache_key": cache_key,
            },
            outputs={
                "prompt_tokens_estimate": max(1, len(context.prompt.split())),
                "negative_prompt_used": bool(context.negative_prompt),
            },
            notes=[
                f"Prompt encode was scheduled against diffusers Wan I2V from {model_dir}.",
                "The actual T5 forward stays inside the verified diffusers pipeline offload path.",
            ],
        )

    async def run_conditioning_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        self._load_modules()
        reference_path = resolve_wan_reference_path(context.request.sample_spec.references[0])
        cache_key = (
            f"{state['model_dir']}|{reference_path}|"
            f"{context.wan_config.width}x{context.wan_config.height}|frames={context.wan_config.frame_count}"
        )
        cache_hit = cache_key in self._conditioning_cache
        if cache_hit:
            reference_image = self._conditioning_cache[cache_key].copy()
        else:
            reference_image = self._pil_image.open(reference_path).convert("RGB")
            self._conditioning_cache[cache_key] = reference_image.copy()

        return WanStageUpdate(
            state_updates={
                "reference_image": reference_image,
            },
            runtime_state_updates={
                "reference_path": reference_path,
                "conditioning_size": [context.wan_config.width, context.wan_config.height],
            },
            outputs={"reference_count": len(context.request.sample_spec.references)},
            notes=["Reference image was staged for diffusers Wan I2V generation reuse."],
            cache_hit=cache_hit,
        )

    async def run_diffusion(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        pipeline = state["pipeline"]
        torch = self._torch
        seed, seed_policy = self._effective_seed(context)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        guidance_scale_2 = (
            context.wan_config.guidance_scale
            if context.wan_config.high_noise_guidance_scale is None
            else context.wan_config.high_noise_guidance_scale
        )
        pipeline_output = pipeline(
            image=state["reference_image"],
            prompt=context.prompt,
            negative_prompt=context.negative_prompt or None,
            height=context.wan_config.height,
            width=context.wan_config.width,
            num_frames=context.wan_config.frame_count,
            num_inference_steps=context.wan_config.num_steps,
            guidance_scale=context.wan_config.guidance_scale,
            guidance_scale_2=guidance_scale_2,
            generator=generator,
            output_type="np",
            return_dict=True,
            max_sequence_length=context.scheduler_payload.get("max_sequence_length", 512),
        )
        frames = pipeline_output.frames[0]
        pipeline.maybe_free_model_hooks()
        gc.collect()
        torch.cuda.empty_cache()
        workload = self._stage_workload(state, "diffusion")
        if workload is not None:
            workload.bind_runtime_value(frames)
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }

        return WanStageUpdate(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "frame_shape": list(frames.shape),
                "seed": seed,
                "seed_policy": seed_policy,
                "guide_scale_pair": [context.wan_config.guidance_scale, guidance_scale_2],
            },
            outputs={
                "solver": pipeline.scheduler.__class__.__name__,
                "num_steps": context.wan_config.num_steps,
                "guidance_scale": [context.wan_config.guidance_scale, guidance_scale_2],
                "compute_profile": compute_profile,
            },
            notes=[
                "Diffusers Wan I2V completed denoise and decode through the pipeline-managed offload path.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_vae_decode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        frames = self._np.asarray(state["video_frames"])
        workload = self._stage_workload(state, "vae_decode")
        if workload is not None:
            workload.bind_runtime_value(frames)
        compute_profile = workload.execute() if workload is not None else {
            "compute_backend": "metadata_only",
            "replayed": False,
            "captured": False,
        }

        return WanStageUpdate(
            state_updates={"video_frames": frames},
            runtime_state_updates={
                "decoded_frame_count": int(frames.shape[0]),
                "decoded_spatial_size": [int(frames.shape[2]), int(frames.shape[1])],
            },
            outputs={"fps": context.request.sample_spec.fps or 16, "compute_profile": compute_profile},
            notes=[
                "VAE decode stage reused diffusers pipeline output from the verified offload path.",
                f"Stage compute backend: {compute_profile['compute_backend']}.",
            ],
        )

    async def run_postprocess(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        frames = self._np.asarray(state["video_frames"])
        if frames.dtype != self._np.uint8:
            frames = self._np.clip(frames * 255.0, 0.0, 255.0).astype(self._np.uint8)
        return WanStageUpdate(
            state_updates={
                "video_frames": frames,
                "output_fps": context.request.sample_spec.fps or 16,
            },
            runtime_state_updates={
                "frame_count": int(len(frames)),
                "output_fps": context.request.sample_spec.fps or 16,
            },
            outputs={"frame_count": int(len(frames))},
            notes=["Postprocess converted decoded tensors into numpy video frames."],
        )

    async def run_persist(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        frames = self._np.asarray(state["video_frames"])
        if frames.dtype != self._np.uint8:
            frames = self._np.clip(frames * 255.0, 0.0, 255.0).astype(self._np.uint8)
        self._imageio.mimsave(str(context.video_path), frames, fps=state["output_fps"])
        return WanStageUpdate(
            state_updates={"video_persisted": context.video_path.exists()},
            runtime_state_updates={"video_persisted": context.video_path.exists()},
            outputs={
                "output_path": str(context.video_path),
                "fps": state["output_fps"],
            },
            notes=["Persist stage wrote the generated MP4 through imageio/ffmpeg."],
        )


class HybridWanInProcessAdapter(WanEngineAdapter):
    """Route Wan requests to the best real in-process adapter per task type."""

    adapter_name = "hybrid-wan-engine"
    mode = "real"
    execution_backend = "in_process_stage_scheduler"
    supports_output_video = True
    supports_cross_request_batching = False

    def __init__(
        self,
        *,
        official_adapter: WanEngineAdapter | None = None,
        image_to_video_adapter: WanEngineAdapter | None = None,
    ) -> None:
        self.official_adapter = official_adapter
        self.image_to_video_adapter = image_to_video_adapter

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.adapter_name,
            "mode": self.mode,
            "execution_backend": self.execution_backend,
            "supports_output_video": self.supports_output_video,
            "supports_cross_request_batching": self.supports_cross_request_batching,
            "delegates": {
                "text_to_video": None if self.official_adapter is None else self.official_adapter.describe(),
                "image_to_video": None
                if self.image_to_video_adapter is None
                else self.image_to_video_adapter.describe(),
            },
        }

    def _adapter_for(self, context: WanExecutionContext) -> WanEngineAdapter:
        if context.request.task_type.value == "image_to_video" and self.image_to_video_adapter is not None:
            return self.image_to_video_adapter
        if self.official_adapter is not None:
            return self.official_adapter
        if self.image_to_video_adapter is not None:
            return self.image_to_video_adapter
        raise RuntimeError("HybridWanInProcessAdapter has no delegate adapters configured")

    def stage_device_map(self, context: WanExecutionContext) -> dict[str, tuple[str, str]]:
        return self._adapter_for(context).stage_device_map(context)

    def build_stage_plan(self, context: WanExecutionContext) -> list[WanStagePlanEntry]:
        return self._adapter_for(context).build_stage_plan(context)

    def build_compiled_stage_workload(
        self,
        stage_name: str,
        context: WanExecutionContext,
        state: dict[str, Any],
    ) -> WanCompiledStageWorkload | None:
        return self._adapter_for(context).build_compiled_stage_workload(stage_name, context, state)

    async def run_text_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_text_encode(context, state)

    async def run_conditioning_encode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_conditioning_encode(context, state)

    async def run_diffusion(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_diffusion(context, state)

    async def run_vae_decode(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_vae_decode(context, state)

    async def run_safety(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_safety(context, state)

    async def run_postprocess(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_postprocess(context, state)

    async def run_persist(self, context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
        return await self._adapter_for(context).run_persist(context, state)


@dataclass(slots=True)
class WanPipelineRun:
    """Normalized result of one in-process Wan scheduler execution."""

    pipeline_metadata: dict[str, Any]
    stage_records: list[dict[str, Any]]
    stage_state: dict[str, Any]
    log_text: str


class WanStageScheduler:
    """SGLang-style composed stage runner for in-process Wan execution.

    The actual Wan stage implementations still live in the adapter. This class
    now owns only the generic stage-composition skeleton and compiled-graph
    lifecycle hooks, mirroring the separation between SGLang Diffusion's
    composed pipeline layer and model-specific stage code.
    """

    def __init__(self, adapter: WanEngineAdapter) -> None:
        self.adapter = adapter
        self._graph_manager = WanCompiledGraphManager()
        self.runtime_config = GenerationRuntimeConfig(local_scheduler=True)

    async def run(self, context: WanExecutionContext) -> WanPipelineRun:
        plan = self.adapter.build_stage_plan(context)
        log_lines = [
            f"Wan in-process scheduler started for sample {context.sample_id}.",
            f"Adapter={self.adapter.adapter_name} mode={self.adapter.mode}.",
        ]

        async def _run_stage(stage_name: str, _context: WanExecutionContext, state: dict[str, Any]) -> WanStageUpdate:
            return await self.adapter.run_stage(stage_name, _context, state)

        def _before_stage(
            spec: GenerationPipelineStageSpec,
            _context: WanExecutionContext,
            state: dict[str, Any],
        ) -> dict[str, Any]:
            compiled_graph = self._graph_manager.begin_stage(spec.name, spec.device, _context)
            workload = self.adapter.build_compiled_stage_workload(spec.name, _context, state)
            workload = self._graph_manager.attach_workload(compiled_graph["graph_key"], workload)
            stage_workloads = state.setdefault("_compiled_stage_workloads", {})
            if workload is not None:
                stage_workloads[spec.name] = workload
            else:
                stage_workloads.pop(spec.name, None)
            state.setdefault("_pending_compiled_graphs", {})[spec.name] = compiled_graph
            return {"compiled_graph": compiled_graph}

        def _after_stage(
            spec: GenerationPipelineStageSpec,
            _context: WanExecutionContext,
            state: dict[str, Any],
            _runtime_state: dict[str, Any],
            _update: WanStageUpdate,
        ) -> dict[str, Any]:
            pending = state.get("_pending_compiled_graphs", {})
            compiled_graph = pending.pop(spec.name)
            workload = state.get("_compiled_stage_workloads", {}).get(spec.name)
            compiled_graph.update(
                self._graph_manager.finish_stage(compiled_graph["graph_key"], workload=workload)
            )
            return {"compiled_graph": compiled_graph}

        stages = [
            CallableGenerationStage(
                GenerationPipelineStageSpec(
                    name=stage.name,
                    component=stage.component,
                    device=stage.device,
                    worker=stage.worker,
                    optional=stage.optional,
                ),
                lambda _context, state, stage_name=stage.name: _run_stage(stage_name, _context, state),
            )
            for stage in plan
        ]
        pipeline = ComposedGenerationPipeline(
            pipeline_name="wan-generation",
            execution_backend=self.adapter.execution_backend,
            stages=stages,
            runtime_config=self.runtime_config,
            before_stage=_before_stage,
            after_stage=_after_stage,
            build_metadata=lambda stage_records, _runtime_state: {
                "adapter": self.adapter.describe(),
                "supports_output_video": self.adapter.supports_output_video,
                "supports_cross_request_batching": self.adapter.supports_cross_request_batching,
                "compiled_graph_lifecycle": self._graph_manager.snapshot(),
            },
            initial_log_lines=log_lines,
        )
        pipeline_run = await pipeline.run(context)

        return WanPipelineRun(
            pipeline_metadata=pipeline_run.pipeline_metadata,
            stage_records=pipeline_run.stage_records,
            stage_state=pipeline_run.stage_state,
            log_text=pipeline_run.log_text,
        )


def load_wan_engine_adapter(
    spec: str | None,
    *,
    repo_dir: str | Path | None = None,
    default_checkpoint_dir: str | Path | None = None,
    i2v_diffusers_dir: str | Path | None = None,
) -> WanEngineAdapter | None:
    """Load an adapter instance from ``module:factory`` or built-in aliases."""

    if spec is None or spec == "":
        return None
    if spec == "disabled":
        return None
    if spec == "stub":
        return StubWanEngineAdapter()
    if spec == "official":
        if repo_dir is None:
            raise ValueError("official wan_engine_adapter requires repo_dir")
        return OfficialWanInProcessAdapter(
            repo_dir=repo_dir,
            default_checkpoint_dir=default_checkpoint_dir,
        )
    if spec == "diffusers-i2v":
        if i2v_diffusers_dir is None:
            raise ValueError("diffusers-i2v wan_engine_adapter requires i2v_diffusers_dir")
        return DiffusersWanI2VAdapter(default_model_dir=i2v_diffusers_dir)
    if spec == "hybrid":
        official_adapter = None
        if repo_dir is not None:
            official_adapter = OfficialWanInProcessAdapter(
                repo_dir=repo_dir,
                default_checkpoint_dir=default_checkpoint_dir,
            )
        image_to_video_adapter = None
        if i2v_diffusers_dir is not None:
            image_to_video_adapter = DiffusersWanI2VAdapter(default_model_dir=i2v_diffusers_dir)
        if official_adapter is None and image_to_video_adapter is None:
            raise ValueError("hybrid wan_engine_adapter requires repo_dir and/or i2v_diffusers_dir")
        return HybridWanInProcessAdapter(
            official_adapter=official_adapter,
            image_to_video_adapter=image_to_video_adapter,
        )
    module_name, separator, attr_name = spec.partition(":")
    if not separator:
        raise ValueError("wan_engine_adapter must use the format module:factory")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    instance = factory() if callable(factory) else factory
    if not isinstance(instance, WanEngineAdapter):
        raise TypeError(f"{spec} did not produce a WanEngineAdapter instance")
    return instance


__all__ = [
    "DiffusersWanI2VAdapter",
    "HybridWanInProcessAdapter",
    "OfficialWanInProcessAdapter",
    "StubWanEngineAdapter",
    "WanEngineAdapter",
    "WanExecutionContext",
    "WanPipelineRun",
    "WanStagePlanEntry",
    "WanStageScheduler",
    "WanStageUpdate",
    "load_wan_engine_adapter",
]
