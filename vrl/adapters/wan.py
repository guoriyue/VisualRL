"""Wan-specific model adapter: dual-expert selection + CFG."""

from __future__ import annotations

from typing import Any

from vrl.experience.types import ExperienceBatch
from vrl.models.families.wan.state import WanDenoiseState


class WanAdapter:
    """Wan-specific model forward: dual-expert selection + CFG.

    Wraps the Wan dual-expert architecture (high/low noise models)
    and classifier-free guidance into the ``ModelAdapter`` protocol.
    """

    def __init__(
        self,
        pipeline: Any,
        boundary: float,
        guidance_scale: float,
        arg_c: dict[str, Any],
        arg_null: dict[str, Any],
        task_key: str,
        cfg: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.boundary = boundary
        self.guidance_scale = guidance_scale
        self.arg_c = arg_c
        self.arg_null = arg_null
        self.task_key = task_key
        self.cfg = cfg

    @classmethod
    def from_denoise_state(cls, denoise_state: WanDenoiseState) -> WanAdapter:
        """Factory: extract forward config from denoise_init() state."""
        # Use the high-noise guidance scale as default; the forward_step
        # will select per-timestep guidance based on boundary.
        return cls(
            pipeline=denoise_state.pipeline,
            boundary=denoise_state.boundary,
            guidance_scale=denoise_state.high_noise_guidance_scale,
            arg_c=denoise_state.arg_c,
            arg_null=denoise_state.arg_null,
            task_key=denoise_state.task_key,
        )

    def _select_model(self, model: Any, timestep_value: float) -> Any:
        """Select high/low noise expert based on timestep boundary."""
        if timestep_value >= self.boundary:
            return model if not hasattr(model, "high_noise_model") else model.high_noise_model
        return model if not hasattr(model, "low_noise_model") else model.low_noise_model

    def _get_guidance_scale(self, timestep_value: float) -> float:
        """Get guidance scale based on timestep boundary (Wan uses dual scale)."""
        return self.guidance_scale

    def forward_step(
        self,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
    ) -> dict[str, Any]:
        """Run Wan model forward for one timestep.

        1. Select expert by boundary (high/low noise)
        2. Forward pass with conditioning and unconditioning
        3. CFG combination

        Returns dict with 'noise_pred', 'noise_pred_cond', 'noise_pred_uncond'.
        """
        import torch

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps
        timestep_value = t[0].item() if t.ndim > 0 else t.item()

        # Select expert model
        active_model = self._select_model(model, timestep_value)

        # Prepare latents for this step
        latents = batch.observations[:, timestep_idx]
        timestep_tensor = torch.stack([t[0]]).to(latents.device) if t.ndim > 0 else torch.stack([t]).to(latents.device)

        # Prepare model input based on task type
        if self.task_key.startswith("t2v-"):
            model_input = [latents[i] for i in range(latents.shape[0])] if latents.ndim > 4 else latents
            if isinstance(model_input, list) and len(model_input) == 1:
                model_input = model_input
            noise_pred_cond = active_model(model_input, t=timestep_tensor, **self.arg_c)[0]
        else:
            model_input = [latents.to(self.pipeline.device)]
            noise_pred_cond = active_model(model_input, t=timestep_tensor, **self.arg_c)[0]

        if self.cfg:
            if self.task_key.startswith("t2v-"):
                noise_pred_uncond = active_model(model_input, t=timestep_tensor, **self.arg_null)[0]
            else:
                noise_pred_uncond = active_model(model_input, t=timestep_tensor, **self.arg_null)[0]

            guidance = self._get_guidance_scale(timestep_value)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred_uncond = None
            noise_pred = noise_pred_cond

        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }
