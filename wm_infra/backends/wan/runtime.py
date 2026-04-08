"""Runtime helpers for Wan queue shaping, warm profiles, and admission hints."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any

from wm_infra.backends.serving_primitives import (
    CompiledProfile,
    ExecutionFamily,
    ResidencyRecord,
    ResidencyTier,
    TransferPlan,
    batch_size_family,
    stable_graph_key,
)
from wm_infra.controlplane.resource_estimator import estimate_wan_request
from wm_infra.controlplane.schemas import ProduceSampleRequest, TaskType, WanTaskConfig


def _round_bucket(value: float, step: float) -> float:
    if step <= 0:
        return round(value, 3)
    return round(round(value / step) * step, 3)


def _align_to(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, int(round(value / multiple) * multiple))


@dataclass(frozen=True, slots=True)
class WanBatchSignature:
    """Compatibility key for Wan diffusion-shaped work."""

    backend: str
    model: str
    runner_mode: str
    task_type: str
    width: int
    height: int
    frame_count: int
    num_steps: int
    guidance_bucket: float
    high_noise_guidance_bucket: float
    shift_bucket: float
    sample_solver: str
    offload_model: bool
    convert_model_dtype: bool
    t5_cpu: bool

    @property
    def aspect_ratio(self) -> float:
        return round(self.width / max(self.height, 1), 4)


@dataclass(slots=True)
class WarmedWanProfile:
    """Tracks one warmed profile family and the batch sizes already seen."""

    profile_id: str
    signature: WanBatchSignature
    prewarmed: bool = False
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    reuse_count: int = 0
    compiled_batch_sizes: set[int] = field(default_factory=set)


def _wan_memory_mode(signature: WanBatchSignature) -> str:
    flags: list[str] = []
    if signature.offload_model:
        flags.append("offload_model")
    if signature.convert_model_dtype:
        flags.append("convert_model_dtype")
    if signature.t5_cpu:
        flags.append("t5_cpu")
    return "+".join(flags) if flags else "default"


def build_wan_execution_family(
    signature: WanBatchSignature,
    *,
    batch_size: int,
    stage: str = "pipeline",
) -> ExecutionFamily:
    return ExecutionFamily(
        backend=signature.backend,
        model=signature.model,
        stage=stage,
        device="cuda" if not signature.t5_cpu else "hybrid",
        dtype="float16" if signature.convert_model_dtype else "float32",
        runner_mode=signature.runner_mode,
        batch_size_family=batch_size_family(batch_size),
        width=signature.width,
        height=signature.height,
        frame_count=signature.frame_count,
        num_steps=signature.num_steps,
        prompt_frames=0,
        memory_mode=_wan_memory_mode(signature),
        layout_key=f"latent:{signature.width}x{signature.height}:{signature.frame_count}f",
        execution_kind=signature.task_type,
    )


def build_wan_transfer_plan(
    *,
    wan_config: WanTaskConfig,
    request: ProduceSampleRequest,
    artifact_io_bytes: int = 0,
) -> TransferPlan:
    input_bytes = 0
    for reference in request.sample_spec.references:
        if isinstance(reference, str) and reference.startswith("file://"):
            input_bytes += 0
    if request.task_type != TaskType.TEXT_TO_VIDEO:
        input_bytes = max(
            input_bytes,
            int(wan_config.width * wan_config.height * max(wan_config.frame_count, 1) * 3),
        )
    plan = TransferPlan(
        overlap_h2d_with_compute=request.task_type != TaskType.TEXT_TO_VIDEO,
        overlap_d2h_with_io=True,
        staging_tier=ResidencyTier.CPU_PINNED_WARM,
        notes=[
            "Wan fast path tries to keep latent state on device until final artifact handoff.",
        ],
    )
    plan.add_h2d(input_bytes)
    plan.add_artifact_io(artifact_io_bytes)
    return plan


def build_wan_residency_records(
    signature: WanBatchSignature,
    *,
    batch_size: int,
    artifact_io_bytes: int = 0,
) -> list[dict[str, Any]]:
    latent_bytes = max(signature.width * signature.height * signature.frame_count * 4, 0)
    records = [
        ResidencyRecord(
            object_id=f"{signature.model}:compiled-profile:{signature.width}x{signature.height}:{signature.num_steps}",
            tier=ResidencyTier.GPU_HOT,
            bytes_size=latent_bytes,
            layout_key=f"latent:{signature.width}x{signature.height}:{signature.frame_count}f",
            pinned=False,
            reusable=True,
            source="wan_pipeline_latent",
        ),
        ResidencyRecord(
            object_id=f"{signature.model}:staging-batch:{batch_size}",
            tier=ResidencyTier.CPU_PINNED_WARM,
            bytes_size=max(signature.width * signature.height * 4, 0),
            layout_key="artifact_staging",
            pinned=True,
            reusable=True,
            source="artifact_handoff",
        ),
    ]
    if artifact_io_bytes > 0:
        records.append(
            ResidencyRecord(
                object_id=f"{signature.model}:artifact-output",
                tier=ResidencyTier.DURABLE_ONLY,
                bytes_size=artifact_io_bytes,
                layout_key="video/mp4",
                pinned=False,
                reusable=False,
                source="artifact_store",
            )
        )
    return [record.as_dict() for record in records]


class WarmedWanEnginePool:
    """Best-effort profile pool for common Wan execution shapes."""

    def __init__(
        self,
        max_profiles: int = 32,
        prewarmed_signatures: list[WanBatchSignature] | None = None,
    ) -> None:
        self.max_profiles = max_profiles
        self._profiles: dict[WanBatchSignature, WarmedWanProfile] = {}
        self._counter = 0
        for signature in prewarmed_signatures or []:
            self._counter += 1
            self._profiles[signature] = WarmedWanProfile(
                profile_id=(
                    f"wan-profile-{self._counter}:"
                    f"{signature.width}x{signature.height}:"
                    f"{signature.frame_count}f:{signature.num_steps}s"
                ),
                signature=signature,
                prewarmed=True,
                compiled_batch_sizes={1},
            )
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def reserve(self, signature: WanBatchSignature, *, batch_size: int) -> dict[str, Any]:
        now = time.time()
        profile = self._profiles.get(signature)
        warm_profile_hit = profile is not None
        if profile is None:
            self._misses += 1
            if len(self._profiles) >= self.max_profiles:
                oldest_key = min(self._profiles, key=lambda key: self._profiles[key].last_used_at)
                self._profiles.pop(oldest_key)
                self._evictions += 1
            self._counter += 1
            profile = WarmedWanProfile(
                profile_id=(
                    f"wan-profile-{self._counter}:"
                    f"{signature.width}x{signature.height}:"
                    f"{signature.frame_count}f:{signature.num_steps}s"
                ),
                signature=signature,
            )
            self._profiles[signature] = profile
        else:
            self._hits += 1

        compiled_batch_size_hit = batch_size in profile.compiled_batch_sizes
        profile.compiled_batch_sizes.add(batch_size)
        profile.reuse_count += 1
        profile.last_used_at = now

        if not warm_profile_hit:
            compile_state = "cold_start"
        elif profile.prewarmed and profile.reuse_count == 1:
            compile_state = "prewarmed"
        elif not compiled_batch_size_hit:
            compile_state = "warm_profile_new_batch_size"
        else:
            compile_state = "warm_profile_batch_hit"

        execution_family = build_wan_execution_family(signature, batch_size=batch_size)
        compiled_profile = CompiledProfile(
            profile_id=profile.profile_id,
            execution_family=execution_family,
            graph_key=stable_graph_key(
                [
                    profile.profile_id,
                    execution_family.cache_key,
                    batch_size_family(batch_size),
                ]
            ),
            compile_state=compile_state,
            warm_profile_hit=warm_profile_hit,
            compiled_batch_size_hit=compiled_batch_size_hit,
            compiled_batch_sizes=sorted(profile.compiled_batch_sizes),
            reuse_count=profile.reuse_count,
            prewarmed=profile.prewarmed,
        )

        return {
            "profile_id": profile.profile_id,
            "compile_state": compile_state,
            "warm_profile_hit": warm_profile_hit,
            "prewarmed": profile.prewarmed,
            "compiled_batch_size_hit": compiled_batch_size_hit,
            "compiled_batch_sizes": sorted(profile.compiled_batch_sizes),
            "reuse_count": profile.reuse_count,
            "last_used_at": profile.last_used_at,
            "graph_family_key": execution_family.cache_key,
            "execution_family": execution_family.as_dict(),
            "compiled_profile": compiled_profile.as_dict(),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "profiles": len(self._profiles),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "prewarmed_profiles": sum(1 for profile in self._profiles.values() if profile.prewarmed),
            "profile_ids": [profile.profile_id for profile in self._profiles.values()],
        }


def build_wan_batch_signature(
    request: ProduceSampleRequest,
    wan_config: WanTaskConfig,
    *,
    runner_mode: str,
) -> WanBatchSignature:
    return WanBatchSignature(
        backend=request.backend,
        model=request.model,
        runner_mode=runner_mode,
        task_type=request.task_type.value,
        width=wan_config.width,
        height=wan_config.height,
        frame_count=wan_config.frame_count,
        num_steps=wan_config.num_steps,
        guidance_bucket=_round_bucket(wan_config.guidance_scale, 0.5),
        high_noise_guidance_bucket=_round_bucket(
            wan_config.guidance_scale
            if wan_config.high_noise_guidance_scale is None
            else wan_config.high_noise_guidance_scale,
            0.5,
        ),
        shift_bucket=_round_bucket(wan_config.shift, 0.5),
        sample_solver=wan_config.sample_solver,
        offload_model=wan_config.offload_model,
        convert_model_dtype=wan_config.convert_model_dtype,
        t5_cpu=wan_config.t5_cpu,
    )


def default_wan_prewarm_signatures(
    backend_name: str,
    runner_mode: str,
) -> list[WanBatchSignature]:
    common_profiles = [
        (832, 480, 9, 4, 4.0, 12.0),
        (832, 480, 9, 8, 4.0, 12.0),
        (1280, 720, 17, 4, 4.0, 12.0),
    ]
    return [
        WanBatchSignature(
            backend=backend_name,
            model="wan2.2-t2v-A14B",
            runner_mode=runner_mode,
            task_type="text_to_video",
            width=width,
            height=height,
            frame_count=frame_count,
            num_steps=num_steps,
            guidance_bucket=_round_bucket(guidance_scale, 0.5),
            high_noise_guidance_bucket=_round_bucket(guidance_scale, 0.5),
            shift_bucket=_round_bucket(shift, 0.5),
            sample_solver="unipc",
            offload_model=True,
            convert_model_dtype=True,
            t5_cpu=True,
        )
        for width, height, frame_count, num_steps, guidance_scale, shift in common_profiles
    ]


def build_wan_batch_key(
    request: ProduceSampleRequest,
    wan_config: WanTaskConfig,
    *,
    runner_mode: str,
) -> tuple[Any, ...]:
    signature = build_wan_batch_signature(request, wan_config, runner_mode=runner_mode)
    return (
        signature.backend,
        signature.model,
        signature.runner_mode,
        signature.task_type,
        signature.width,
        signature.height,
        signature.frame_count,
        signature.num_steps,
        signature.guidance_bucket,
        signature.high_noise_guidance_bucket,
        signature.shift_bucket,
        signature.sample_solver,
        signature.offload_model,
        signature.convert_model_dtype,
        signature.t5_cpu,
    )


def wan_batch_compatibility_score(
    reference_request: ProduceSampleRequest,
    reference_config: WanTaskConfig,
    candidate_request: ProduceSampleRequest,
    candidate_config: WanTaskConfig,
    *,
    runner_mode: str,
) -> float | None:
    reference_signature = build_wan_batch_signature(
        reference_request,
        reference_config,
        runner_mode=runner_mode,
    )
    candidate_signature = build_wan_batch_signature(
        candidate_request,
        candidate_config,
        runner_mode=runner_mode,
    )
    if (
        reference_signature.backend != candidate_signature.backend
        or reference_signature.model != candidate_signature.model
        or reference_signature.runner_mode != candidate_signature.runner_mode
        or reference_signature.task_type != candidate_signature.task_type
        or reference_signature.sample_solver != candidate_signature.sample_solver
    ):
        return None

    width_delta = abs(reference_signature.width - candidate_signature.width) / max(reference_signature.width, 1)
    height_delta = abs(reference_signature.height - candidate_signature.height) / max(reference_signature.height, 1)
    frame_delta = abs(reference_signature.frame_count - candidate_signature.frame_count) / max(reference_signature.frame_count, 1)
    step_delta = abs(reference_signature.num_steps - candidate_signature.num_steps) / max(reference_signature.num_steps, 1)
    guidance_delta = abs(reference_signature.guidance_bucket - candidate_signature.guidance_bucket) / max(reference_signature.guidance_bucket, 1.0)
    high_noise_guidance_delta = abs(
        reference_signature.high_noise_guidance_bucket - candidate_signature.high_noise_guidance_bucket
    ) / max(reference_signature.high_noise_guidance_bucket, 1.0)
    shape_penalty = width_delta + height_delta + frame_delta + step_delta + guidance_delta + high_noise_guidance_delta
    return round(max(0.0, 1.0 - shape_penalty), 4)


def score_wan_batch_compatibility(
    anchor_request: ProduceSampleRequest,
    anchor_config: WanTaskConfig,
    candidate_request: ProduceSampleRequest,
    candidate_config: WanTaskConfig,
    *,
    runner_mode: str,
) -> float | None:
    if anchor_request.task_type != candidate_request.task_type:
        return None
    if anchor_request.task_type not in {TaskType.TEXT_TO_VIDEO, TaskType.IMAGE_TO_VIDEO, TaskType.VIDEO_TO_VIDEO}:
        return None
    if anchor_request.backend != candidate_request.backend:
        return None
    if anchor_request.model != candidate_request.model:
        return None
    if anchor_config.offload_model != candidate_config.offload_model:
        return None
    if anchor_config.convert_model_dtype != candidate_config.convert_model_dtype:
        return None
    if anchor_config.t5_cpu != candidate_config.t5_cpu:
        return None
    if anchor_config.sample_solver != candidate_config.sample_solver:
        return None
    if bool(anchor_request.sample_spec.references) != bool(candidate_request.sample_spec.references):
        return None
    if runner_mode == "official" and anchor_request.task_type == TaskType.VIDEO_TO_VIDEO:
        return None

    area_anchor = max(anchor_config.width * anchor_config.height, 1)
    area_candidate = max(candidate_config.width * candidate_config.height, 1)
    area_ratio = max(area_anchor, area_candidate) / min(area_anchor, area_candidate)
    if area_ratio > 1.35:
        return None

    aspect_delta = abs(
        (anchor_config.width / max(anchor_config.height, 1))
        - (candidate_config.width / max(candidate_config.height, 1))
    )
    if aspect_delta > 0.12:
        return None

    frame_delta = abs(anchor_config.frame_count - candidate_config.frame_count)
    if frame_delta > 8:
        return None

    step_delta = abs(anchor_config.num_steps - candidate_config.num_steps)
    if step_delta > 2:
        return None

    guidance_delta = abs(anchor_config.guidance_scale - candidate_config.guidance_scale)
    if guidance_delta > 1.0:
        return None
    high_noise_guidance_delta = abs(
        (anchor_config.high_noise_guidance_scale or anchor_config.guidance_scale)
        - (candidate_config.high_noise_guidance_scale or candidate_config.guidance_scale)
    )
    if high_noise_guidance_delta > 1.0:
        return None

    score = 100.0
    score -= (area_ratio - 1.0) * 35.0
    score -= aspect_delta * 120.0
    score -= frame_delta * 2.0
    score -= step_delta * 6.0
    score -= guidance_delta * 8.0
    score -= high_noise_guidance_delta * 8.0
    return round(max(score, 0.0), 3)


def expected_occupancy(batch_size: int, max_batch_size: int) -> float:
    if max_batch_size <= 0:
        return 0.0
    return min(max(batch_size / max_batch_size, 0.0), 1.0)


def build_wan_scheduler_payload(
    signature: WanBatchSignature,
    *,
    batch_size: int,
    batch_index: int,
    max_batch_size: int,
    sample_ids: list[str],
) -> dict[str, Any]:
    return {
        "queue_lane": "shape_hot" if batch_size > 1 else "singleton",
        "batched_across_requests": batch_size > 1,
        "execution_mode": "queue_coalesced_serial" if batch_size > 1 else "single_request",
        "batch_id": (
            f"wan:{signature.width}x{signature.height}:"
            f"{signature.frame_count}f:{signature.num_steps}s:{batch_size}"
        ),
        "batch_size": batch_size,
        "batch_index": batch_index,
        "sample_ids": sample_ids,
        "expected_occupancy": expected_occupancy(batch_size, max_batch_size),
        "batch_signature": asdict(signature),
    }


def _fits_limits(
    wan_config: WanTaskConfig,
    *,
    max_units: float | None,
    max_vram_gb: float | None,
) -> bool:
    estimate = estimate_wan_request(wan_config)
    if max_units is not None and estimate.estimated_units > max_units:
        return False
    if max_vram_gb is not None and estimate.estimated_vram_gb is not None and estimate.estimated_vram_gb > max_vram_gb:
        return False
    return True


def _clone_wan_config(wan_config: WanTaskConfig, **updates: Any) -> WanTaskConfig:
    return wan_config.model_copy(update=updates, deep=True)


def build_quality_cost_hints(
    wan_config: WanTaskConfig,
    *,
    max_units: float | None,
    max_vram_gb: float | None,
) -> dict[str, Any]:
    current_estimate = estimate_wan_request(wan_config)
    suggestions: list[dict[str, Any]] = []

    step_candidate = wan_config
    while step_candidate.num_steps > 1 and not _fits_limits(step_candidate, max_units=max_units, max_vram_gb=max_vram_gb):
        next_steps = max(1, int(step_candidate.num_steps * 0.75))
        if next_steps == step_candidate.num_steps:
            next_steps = step_candidate.num_steps - 1
        step_candidate = _clone_wan_config(step_candidate, num_steps=next_steps)
    if step_candidate.num_steps < wan_config.num_steps:
        suggestions.append(
            {
                "policy": "auto_step_reduction",
                "wan_config": step_candidate.model_dump(mode="json"),
                "resource_estimate": estimate_wan_request(step_candidate).model_dump(mode="json"),
            }
        )

    resolution_candidate = wan_config
    while (
        (resolution_candidate.width > 256 or resolution_candidate.height > 256)
        and not _fits_limits(resolution_candidate, max_units=max_units, max_vram_gb=max_vram_gb)
    ):
        scale = 0.85
        resolution_candidate = _clone_wan_config(
            resolution_candidate,
            width=_align_to(max(256, int(resolution_candidate.width * scale)), 32),
            height=_align_to(max(256, int(resolution_candidate.height * scale)), 32),
        )
    if resolution_candidate.width != wan_config.width or resolution_candidate.height != wan_config.height:
        suggestions.append(
            {
                "policy": "resolution_fallback",
                "wan_config": resolution_candidate.model_dump(mode="json"),
                "resource_estimate": estimate_wan_request(resolution_candidate).model_dump(mode="json"),
            }
        )

    preview_width = min(wan_config.width, 832)
    preview_height = min(wan_config.height, 832)
    if preview_width != wan_config.width or preview_height != wan_config.height:
        area_ratio = sqrt((preview_width * preview_height) / max(wan_config.width * wan_config.height, 1))
        preview_width = _align_to(int(wan_config.width * area_ratio), 32)
        preview_height = _align_to(int(wan_config.height * area_ratio), 32)
    preview_candidate = _clone_wan_config(
        wan_config,
        num_steps=min(wan_config.num_steps, 4),
        frame_count=min(wan_config.frame_count, 9),
        width=preview_width,
        height=preview_height,
    )
    if preview_candidate != wan_config:
        suggestions.append(
            {
                "policy": "progressive_preview",
                "wan_config": preview_candidate.model_dump(mode="json"),
                "resource_estimate": estimate_wan_request(preview_candidate).model_dump(mode="json"),
            }
        )

    return {
        "cost_tier": (
            "heavy"
            if current_estimate.estimated_units >= 20
            else "medium"
            if current_estimate.estimated_units >= 8
            else "light"
        ),
        "current_estimate": current_estimate.model_dump(mode="json"),
        "suggested_adjustments": suggestions,
    }
