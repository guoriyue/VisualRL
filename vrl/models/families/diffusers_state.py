"""Shared diffusers denoising state for Wan and Cosmos families.

Stored as ``DenoiseLoopState.model_state``. Both families populate
the common fields; Cosmos-specific fields default to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiffusersDenoiseState:
    """Opaque per-step state for diffusers-backed model families.

    Common fields
    -------------
    latents : [B, C, D, H, W] noise tensor being progressively denoised.
    timesteps : scheduler.timesteps — the full schedule.
    scheduler : diffusers scheduler instance (e.g. FlowMatchEulerDiscreteScheduler).
    prompt_embeds, negative_prompt_embeds : text conditioning tensors.
    guidance_scale : CFG weight.
    do_cfg : whether to run the unconditioned pass.
    pipeline : raw diffusers pipeline reference (for VAE decode, etc.).

    Cosmos-specific (``None`` for Wan)
    ----------------------------------
    init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask :
        Video2World conditioning tensors from ``pipe.prepare_latents()``.
    fps, sigma_data, sigma_conditioning : Cosmos scheduler parameters.
    """

    # Core diffusion
    latents: Any = None
    timesteps: Any = None
    scheduler: Any = None

    # Text conditioning
    prompt_embeds: Any = None
    negative_prompt_embeds: Any | None = None

    # CFG
    guidance_scale: float = 1.0
    do_cfg: bool = False

    # Pipeline ref
    pipeline: Any = None

    # Cosmos-specific (None for Wan)
    init_latents: Any | None = None
    cond_indicator: Any | None = None
    uncond_indicator: Any | None = None
    cond_mask: Any | None = None
    uncond_mask: Any | None = None
    fps: int = 16
    sigma_data: float = 1.0
    sigma_conditioning: float = 0.0001

    # Metadata
    seed: int = 0
    model_family: str = ""
