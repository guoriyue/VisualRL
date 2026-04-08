"""Lightweight resource estimation for video-generation jobs.

The estimator is intentionally simple and scheduler-friendly.
It uses the current first-class task configs instead of guessing from loose
metadata, and it can score both rollout-style and Wan 2.2-style video requests.
"""

from __future__ import annotations

from wm_infra.controlplane.schemas import CosmosTaskConfig, ResourceEstimate, RolloutTaskConfig, VideoMemoryProfile, WanTaskConfig
from wm_infra.rollout_engine import DEFAULT_RESOURCE_UNITS_PER_GB, RolloutRequest

_BASELINE_FRAMES = 9
_BASELINE_WIDTH = 832
_BASELINE_HEIGHT = 480
_BASELINE_STEPS = 4
_BASELINE_VRAM_GB = 28.0


def _memory_multiplier(memory_profile: VideoMemoryProfile | None) -> float:
    if memory_profile == VideoMemoryProfile.LOW_VRAM:
        return 0.65
    if memory_profile == VideoMemoryProfile.HIGH_QUALITY:
        return 1.25
    if memory_profile == VideoMemoryProfile.BALANCED:
        return 0.9
    return 1.0


def _estimate_vram_gb(frame_count: int, width: int, height: int, num_steps: int, memory_profile: VideoMemoryProfile | None) -> float:
    baseline_pixels = _BASELINE_FRAMES * _BASELINE_WIDTH * _BASELINE_HEIGHT
    pixels = max(frame_count, 1) * max(width, 1) * max(height, 1)
    base_gb = _BASELINE_VRAM_GB * (pixels / baseline_pixels)
    step_factor = 1.0 + (max(num_steps, 1) - _BASELINE_STEPS) * 0.02
    return round(max(base_gb * step_factor * _memory_multiplier(memory_profile), 0.1), 2)


def estimate_rollout_request(task_config: RolloutTaskConfig | None) -> ResourceEstimate:
    task_config = task_config or RolloutTaskConfig()
    scheduling_request = RolloutRequest.from_task_config("estimate", task_config)
    estimated_units = scheduling_request.estimate_resource_units()
    estimated_vram_gb = _estimate_vram_gb(
        frame_count=task_config.frame_count or 1,
        width=task_config.width or 256,
        height=task_config.height or 256,
        num_steps=task_config.num_steps,
        memory_profile=task_config.memory_profile,
    )
    return ResourceEstimate(
        estimated_units=estimated_units,
        estimated_vram_gb=estimated_vram_gb,
        bottleneck="frame_pressure",
        notes=[
            "Estimated from frame_count × num_steps × megapixels.",
            "Low-VRAM profiles reduce the relative score; high-quality profiles increase it.",
        ],
    )


def estimate_wan_request(wan_config: WanTaskConfig | None) -> ResourceEstimate:
    wan_config = wan_config or WanTaskConfig()
    rollout_like = RolloutTaskConfig(
        num_steps=wan_config.num_steps,
        frame_count=wan_config.frame_count,
        width=wan_config.width,
        height=wan_config.height,
        offload_model=wan_config.offload_model,
        convert_model_dtype=wan_config.convert_model_dtype,
        t5_cpu=wan_config.t5_cpu,
        memory_profile=wan_config.memory_profile,
    )
    estimate = estimate_rollout_request(rollout_like)
    estimate.notes.append("Wan estimate is calibrated from the Wan 2.2 A14B baseline and first-class wan_config fields.")
    return estimate


def estimate_cosmos_request(
    task_config: RolloutTaskConfig | None,
    cosmos_config: CosmosTaskConfig | None,
) -> ResourceEstimate:
    task_config = task_config or RolloutTaskConfig()
    cosmos_config = cosmos_config or CosmosTaskConfig()
    estimate = estimate_rollout_request(task_config)

    size_factor = {
        "2B": 0.6,
        "7B": 1.0,
        "14B": 1.7,
    }.get(cosmos_config.model_size.upper(), 1.0)
    if cosmos_config.variant.value.endswith("text2world"):
        variant_factor = 0.85
        bottleneck = "text_to_world_generation"
    else:
        variant_factor = 1.1
        bottleneck = "video_conditioned_world_generation"

    estimate.estimated_units = round(estimate.estimated_units * size_factor * variant_factor, 3)
    if estimate.estimated_vram_gb is not None:
        estimate.estimated_vram_gb = round(estimate.estimated_vram_gb * size_factor * variant_factor, 2)
    estimate.bottleneck = bottleneck
    estimate.notes.append(
        "Cosmos estimate scales rollout pressure by Cosmos variant and model_size to reflect world-generation cost."
    )
    return estimate


__all__ = ["estimate_cosmos_request", "estimate_rollout_request", "estimate_wan_request"]
