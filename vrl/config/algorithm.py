"""Algorithm-kind to typed-config dispatch shared by schema and builders."""

from __future__ import annotations

import math
from typing import Any


def algorithm_config_class(kind: str) -> type[Any]:
    """Return the runtime config dataclass selected by ``algorithm.kind``."""

    if kind in ("grpo", "dance_grpo"):
        from vrl.algorithms.grpo.continuous import GRPOConfig

        return GRPOConfig
    if kind == "flash_grpo":
        from vrl.algorithms.grpo.continuous import FlashGRPOConfig

        return FlashGRPOConfig
    if kind == "flow_dppo":
        from vrl.algorithms.grpo.continuous import FlowDPPOConfig

        return FlowDPPOConfig
    if kind == "grpo_guard":
        from vrl.algorithms.grpo.continuous import GRPOGuardConfig

        return GRPOGuardConfig
    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPOConfig

        return TokenGRPOConfig
    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPOConfig

        return MultiSegmentTokenGRPOConfig
    if kind == "diffusion_dpo":
        from vrl.algorithms.dpo import DiffusionDPOConfig

        return DiffusionDPOConfig
    if kind == "diffusion_nft":
        from vrl.algorithms.diffusion_nft import DiffusionNFTConfig

        return DiffusionNFTConfig
    if kind == "v_grpo":
        from vrl.algorithms.v_grpo import VGRPOConfig

        return VGRPOConfig
    raise ValueError(f"unsupported algorithm kind: {kind!r}")


def resolve_kl_reward_coef(value: object | None) -> float:
    """Resolve the collector reward-shaping coefficient at its public boundary."""

    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("algorithm.kl_reward_coef must be a finite number >= 0")
    coefficient = float(value)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("algorithm.kl_reward_coef must be a finite number >= 0")
    return coefficient


__all__ = ["algorithm_config_class", "resolve_kl_reward_coef"]
