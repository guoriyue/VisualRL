"""Flow-matching signal extraction for diffusion model training.

Contains the core math (sde_step_with_logprob, compute_kl_divergence)
moved from ``algorithms/flow_matching.py``, plus the ``FlowMatchingEvaluator``
that wraps them behind the ``Evaluator`` protocol.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

from vrl.rollouts.evaluators.types import SignalBatch, SignalRequest
from vrl.rollouts.types import ExperienceBatch


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
# Core math
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
    noise_level: float = 1.0,
    sde_type: str = "sde",
) -> SDEStepResult:
    """Compute one SDE step and its log-probability.

    Supports two SDE formulations (from flow_grpo):

    **sde** (default, used by WAN):
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
        prev_sample_mean = sample*(1 + std^2/(2*sigma)*dt) + model*(1 + std^2(1-sigma)/(2*sigma))*dt

    **cps** (Consistency Probability Sampler, used by SD3/Flux):
        std_dev_t = sigma_prev * sin(noise_level * pi / 2)
        pred_x0 = sample - sigma * model_output
        pred_x1 = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_x0*(1-sigma_prev) + pred_x1*sqrt(sigma_prev^2 - std^2)

    Args:
        noise_level: Scales diffusion noise (SD3: 0.7-1.5, WAN: 1.0).
        sde_type: "sde" for standard flow-matching, "cps" for consistency probability sampler.
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

    if sde_type == "cps":
        # Consistency Probability Sampler (from sd3_sde_with_logprob.py:70-86)
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output  # predicted x_0
        noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1
        prev_sample_mean = (
            pred_original_sample * (1 - sigma_prev)
            + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
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
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if deterministic:
            prev_sample = sample + dt * model_output

        # CPS: simplified log_prob (no normalization constants, per flow_grpo)
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    else:
        # Standard SDE (from wan_pipeline_with_logprob.py)
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

        # Apply noise_level scaling for SD3 (WAN passes 1.0 = no-op)
        if noise_level != 1.0:
            std_dev_t = torch.sqrt(
                sigma / (1 - torch.where(sigma == 1, torch.tensor(sigma_max, device=sigma.device), sigma))
            ) * noise_level

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

    def __init__(
        self,
        scheduler: Any,
        noise_level: float = 1.0,
        sde_type: str = "sde",
    ) -> None:
        self.scheduler = scheduler
        self.noise_level = noise_level
        self.sde_type = sde_type

    def evaluate(
        self,
        collector: Any,
        model: Any,
        batch: ExperienceBatch,
        timestep_idx: int,
        ref_model: Any | None = None,
        signal_request: SignalRequest | None = None,
    ) -> SignalBatch:
        """Run collector.forward_step() -> sde_step_with_logprob -> SignalBatch.

        Gap 7: When ref_model is the same object as model (LoRA scenario),
        uses disable_adapter() to get base-model predictions — matching
        flow_grpo train_wan2_1.py:940.
        """
        import torch

        if signal_request is None:
            signal_request = SignalRequest()

        timesteps = batch.extras["timesteps"]
        t = timesteps[:, timestep_idx] if timesteps.ndim > 1 else timesteps

        observations = batch.observations[:, timestep_idx]  # x_t
        actions = batch.actions[:, timestep_idx]             # x_{t-1}

        # Forward pass through current model
        fwd = collector.forward_step(model, batch, timestep_idx)
        noise_pred = fwd["noise_pred"]

        # SDE step with log-prob
        result = sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            t,
            observations,
            prev_sample=actions,
            return_dt=signal_request.need_kl_intermediates,
            noise_level=self.noise_level,
            sde_type=self.sde_type,
        )

        ref_log_prob = None
        ref_prev_sample_mean = None
        ref_dt = None

        # Reference model forward for KL
        if signal_request.need_ref and ref_model is not None:
            with torch.no_grad():
                # Gap 7: LoRA disable_adapter() — when ref_model IS model,
                # disable LoRA adapter to get base model output.
                # Port from flow_grpo train_wan2_1.py:940:
                #   with transformer.module.disable_adapter():
                use_adapter_disable = (
                    ref_model is model
                    and hasattr(model, "disable_adapter")
                )
                ctx = model.disable_adapter() if use_adapter_disable else contextlib.nullcontext()

                with ctx:
                    ref_fwd = collector.forward_step(ref_model, batch, timestep_idx)
                    ref_noise_pred = ref_fwd["noise_pred"]

                    ref_result = sde_step_with_logprob(
                        self.scheduler,
                        ref_noise_pred,
                        t,
                        observations,
                        prev_sample=actions,
                        return_dt=signal_request.need_kl_intermediates,
                        noise_level=self.noise_level,
                        sde_type=self.sde_type,
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
