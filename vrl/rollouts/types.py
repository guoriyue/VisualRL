"""Generic training experience collected from rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperienceBatch:
    """Generic training experience collected from rollouts.

    Model-agnostic container: the collector fills in observations/actions
    from the denoising trajectory, and model-specific extras go in ``extras``.
    """

    observations: Any   # x_t — current state [B, T, ...]
    actions: Any         # x_{t-1} — next state (denoised) [B, T, ...]
    rewards: Any         # [B] scalar rewards per sample
    dones: Any           # [B] episode termination flags
    group_ids: Any       # [B] prompt group assignment (for per-prompt normalization)
    extras: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)  # shared metadata (not stacked)
    videos: Any | None = None      # [B, C, T, H, W] decoded frames (for reward scoring)
    prompts: list[str] | None = None


def stack_batches(batches: list[ExperienceBatch]) -> ExperienceBatch:
    """Concatenate multiple ExperienceBatches along batch dim.

    Tensor fields are ``torch.cat``-ed; list fields are concatenated;
    extras are merged from the first batch (non-tensor extras are kept as-is).
    """
    import torch

    if len(batches) == 1:
        return batches[0]

    observations = torch.cat([b.observations for b in batches], dim=0)
    actions = torch.cat([b.actions for b in batches], dim=0)
    rewards = torch.cat([b.rewards for b in batches], dim=0)
    dones = torch.cat([b.dones for b in batches], dim=0)
    group_ids = torch.cat([b.group_ids for b in batches], dim=0)

    # Videos: cat if all present
    if all(b.videos is not None for b in batches):
        videos = torch.cat([b.videos for b in batches], dim=0)
    else:
        videos = None

    # Prompts: concatenate lists
    prompts: list[str] = []
    for b in batches:
        if b.prompts is not None:
            prompts.extend(b.prompts)

    # Extras: cat tensor values from all batches, keep non-tensor from first
    extras: dict[str, Any] = {}
    first = batches[0].extras
    for key in first:
        val = first[key]
        if isinstance(val, torch.Tensor):
            extras[key] = torch.cat([b.extras[key] for b in batches], dim=0)
        else:
            extras[key] = val  # non-tensor: keep from first batch

    # Context: shared metadata — take from first batch (not stacked)
    context: dict[str, Any] = dict(batches[0].context)

    return ExperienceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        group_ids=group_ids,
        extras=extras,
        context=context,
        videos=videos,
        prompts=prompts or None,
    )
