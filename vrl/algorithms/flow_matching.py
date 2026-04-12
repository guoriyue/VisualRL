"""Backward-compatibility shim — canonical location is evaluators.diffusion.flow_matching."""

from vrl.rollouts.evaluators.diffusion.flow_matching import (  # noqa: F401
    SDEStepResult,
    compute_kl_divergence,
    sde_step_with_logprob,
)
