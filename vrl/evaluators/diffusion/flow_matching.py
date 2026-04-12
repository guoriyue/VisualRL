"""Flow-matching signal extraction for diffusion model training.

Contains the core math (sde_step_with_logprob, compute_kl_divergence)
moved from ``algorithms/flow_matching.py``, plus the ``FlowMatchingEvaluator``
that wraps them behind the ``Evaluator`` protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vrl.adapters.base import ModelAdapter
from vrl.evaluators.types import SignalBatch, SignalRequest
from vrl.experience.types import ExperienceBatch


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------

@dataclass(slots=True)
class SDEStepResult:
    """Result of a single SDE denoising step with log-probability."""

    prev_sample: Any       # x_{t-1} (Tensor)
    log_prob: Any          # per-sample scalar log-prob (Tensor)
    prev_sample_mean: Any  # deterministic mean of x_{t-1} (Tensor)
    std_dev_t: Any         # noise scale sigma_t (Tensor)
    dt: Any | None = None  # step size sqrt(-dt) when requested


# ------------------------------------------------------------------
# Core math (moved from algorithms/flow_matching.py)
# ------------------------------------------------------------------

def sde_step_with_logprob(
    scheduler: Any,
    model_output: Any,
    timestep: Any,
    sample: Any,
    prev_sample: Any | None = None,
    generator: Any | None = None,
    deterministic: bool = False,
    return_dt: bool = False,
) -> SDEStepResult:
    """Compute one SDE step and its log-probability.

    This implements the flow-matching SDE formulation from flow_grpo:

        prev_sample_mean = sample * (1 + sigma^2/(2*sigma) * dt)
                         + model_output * (1 + sigma^2(1-sigma)/(2*sigma)) * dt

        log p(x_{t-1} | x_t) = -||x_{t-1} - mu||^2 / (2 * (sigma*sqrt(-dt))^2)
                               - log(sigma*sqrt(-dt)) - log(sqrt(2*pi))

    All inputs/outputs are tensors.  We import torch lazily so the module
    remains importable without torch for lightweight testing.
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [s + 1 for s in step_index]

    scheduler.sigmas = scheduler.sigmas.to(sample.device)
    ndim = sample.ndim  # 4 for images, 5 for video
    view_shape = (-1,) + (1,) * (ndim - 1)

    sigma = scheduler.sigmas[step_index].view(*view_shape)
    sigma_prev = scheduler.sigmas[prev_step_index].view(*view_shape)
    sigma_max = scheduler.sigmas[1].item()
    sigma_min = scheduler.sigmas[-1].item()
    dt = sigma_prev - sigma

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    if deterministic:
        prev_sample = sample + dt * model_output

    noise_scale = std_dev_t * torch.sqrt(-1 * dt)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * noise_scale**2)
        - torch.log(noise_scale)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # Mean across all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    sqrt_neg_dt = torch.sqrt(-1 * dt) if return_dt else None
    return SDEStepResult(
        prev_sample=prev_sample,
        log_prob=log_prob,
        prev_sample_mean=prev_sample_mean,
        std_dev_t=std_dev_t,
        dt=sqrt_neg_dt,
    )


def compute_kl_divergence(
    prev_sample_mean: Any,
    prev_sample_mean_ref: Any,
    std_dev_t: Any,
    dt: Any | None = None,
) -> Any:
    """KL divergence between current and reference model in latent space.

    From flow_grpo train_wan2_1.py / train_sd3.py:
        kl = ||mu - mu_ref||^2 / (2 * (sigma * dt)^2)

    This is more principled for continuous diffusion than naive log-prob KL.
    """
    denom = 2 * std_dev_t**2
    if dt is not None:
        denom = 2 * (std_dev_t * dt) ** 2
    kl = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(
        dim=tuple(range(1, prev_sample_mean.ndim))
    ) / denom.squeeze()
    return kl


# ------------------------------------------------------------------
# FlowMatchingEvaluator
# ------------------------------------------------------------------

class FlowMatchingEvaluator:
    """Signal extraction for flow-matching diffusion models.

    Uses ``sde_step_with_logprob`` to compute log-probabilities and
    optionally reference model signals for latent-space KL.
    """

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def evaluate(
        self,
        adapter: ModelAdapter,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Run adapter forward -> sde_step_with_logprob -> SignalBatch."""
        import torch

        if signal_request is None:
            signal_request = SignalRequest()

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        # Current latents and next latents for this timestep
        observations = batch.observations[:, timestep_idx]  # x_t
        actions = batch.actions[:, timestep_idx]             # x_{t-1}

        # Forward pass through current model
        fwd = adapter.forward_step(model, batch, timestep_idx)
        noise_pred = fwd["noise_pred"]

        # SDE step with log-prob
        result = sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            t,
            observations,
            prev_sample=actions,
            return_dt=signal_request.need_kl_intermediates,
        )

        ref_log_prob = None
        ref_prev_sample_mean = None
        ref_dt = None

        # Reference model forward for KL
        if signal_request.need_ref and ref_model is not None:
            with torch.no_grad():
                ref_fwd = adapter.forward_step(ref_model, batch, timestep_idx)
                ref_noise_pred = ref_fwd["noise_pred"]

                ref_result = sde_step_with_logprob(
                    self.scheduler,
                    ref_noise_pred,
                    t,
                    observations,
                    prev_sample=actions,
                    return_dt=signal_request.need_kl_intermediates,
                )
                ref_log_prob = ref_result.log_prob
                ref_prev_sample_mean = ref_result.prev_sample_mean
                ref_dt = ref_result.dt

        return SignalBatch(
            log_prob=result.log_prob,
            ref_log_prob=ref_log_prob,
            prev_sample_mean=result.prev_sample_mean,
            ref_prev_sample_mean=ref_prev_sample_mean,
            std_dev_t=result.std_dev_t,
            dt=result.dt if result.dt is not None else ref_dt,
            dist_family="flow_matching",
        )
