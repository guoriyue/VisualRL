"""Cross-section rules over a parsed ``RootConfig`` (validation tier 2).

The config package validates in three tiers, one module each:

1. **Section shape** — ``schema.py``. Pydantic models: closed keys, types,
   per-section invariants. Runs inside ``parse_config``.
2. **Cross-section rules** — this module. Pure functions over the typed root
   that relate two or more sections (``algorithm.kind`` against ``rollout``,
   ``model.family`` against ``algorithm``, …). No I/O, no torch, no runtime
   modules: every entrypoint that parses a config pays for them, so they must
   stay import-light. ``RootConfig``'s ``model_validator`` runs
   ``CROSS_SECTION_RULES`` in order, so the rules fire on every construction
   path (``parse_config`` and direct ``RootConfig.model_validate``).
3. **Launch gates** — ``validation.py``. Checks that need the resolved
   precision policy, runtime modules or the filesystem; only training
   launches pay for them (``require_training_config``).

Adding a rule: write ``def rule_<name>(root) -> None`` that raises
``ValueError`` with the offending dotted path, and append it to
``CROSS_SECTION_RULES``. Keep a rule about one relationship; a rule that
needs a runtime module belongs in tier 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from vrl.config.algorithm import resolve_kl_reward_coef
from vrl.models.families.names import normalize_model_family

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

CrossSectionRule = Callable[["RootConfig"], None]


def _family(root: RootConfig) -> str:
    return normalize_model_family((root.model.family or "") if root.model else "")


def rule_kl_reward_shaping_needs_step_kl(root: RootConfig) -> None:
    """``algorithm.kl_reward_coef`` shapes rewards with the collected per-step KL."""

    algo = root.algorithm
    if algo is None:
        return
    if (
        resolve_kl_reward_coef(algo.kl_reward_coef) > 0.0
        and not algo.hyperparameters.config_contract.supports_step_kl_reward
    ):
        raise ValueError(
            "algorithm.kl_reward_coef > 0 requires a diffusion rollout "
            "trajectory with collected per-step KL; "
            f"algorithm.kind={algo.kind!r} does not provide one",
        )


def rule_algorithm_consumes_only_its_surface(root: RootConfig) -> None:
    """Reject sections and explicit fields outside the algorithm's declared surface."""

    algo = root.algorithm
    if algo is None:
        return
    surface = algo.hyperparameters.config_contract.consumed_sections
    if surface is None:
        return
    for section_name, allowed in surface:
        section = getattr(root, section_name)
        if section is None:
            continue
        if allowed is None:
            raise ValueError(f"{algo.kind} does not consume the {section_name} config section")
        unsupported = sorted(section.model_fields_set - allowed)
        if unsupported:
            fields = ", ".join(f"{section_name}.{name}" for name in unsupported)
            raise ValueError(f"{algo.kind} does not consume config field(s): {fields}")


def rule_sft_weight_is_owned_and_backed(root: RootConfig) -> None:
    """Require valid SFT weights and the data source declared by their owner."""

    algo = root.algorithm
    if algo is None:
        return
    raw = getattr(algo.hyperparameters, "sft_weight", None)
    if raw is None:
        return
    try:
        sft_weight = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("algorithm.sft_weight must be a finite number >= 0") from exc
    if not math.isfinite(sft_weight) or sft_weight < 0:
        raise ValueError("algorithm.sft_weight must be a finite number >= 0")
    if sft_weight == 0:
        return
    if algo.hyperparameters.config_contract.sft_source == "latents":
        if root.data is None or not root.data.sft_latents:
            raise ValueError(
                "algorithm.sft_weight > 0 requires data.sft_latents "
                "(the precomputed clean-latents shard; see "
                "vrl/scripts/denoise/encode_targets.py)",
            )
    elif algo.hyperparameters.config_contract.sft_source == "unsupported":
        raise ValueError(
            f"algorithm.sft_weight > 0 is not supported by algorithm.kind={algo.kind!r}",
        )


def rule_sde_objectives_declare_a_sampler(root: RootConfig) -> None:
    """Objectives that collect through the stochastic sampler need ``rollout.sde``.

    Membership of ``sde.type`` itself is the ``SdeConfig`` Literal's job; this
    rule only requires the block to exist.
    """

    algo = root.algorithm
    if algo is None or not algo.hyperparameters.config_contract.needs_sde_rollout:
        return
    if root.rollout is None or root.rollout.sde is None:
        raise ValueError("config missing required field: rollout.sde.type")


def rule_nextstep_token_grpo_needs_noise_level(root: RootConfig) -> None:
    """NextStep-1's continuous-token sampler reads ``rollout.noise_level``."""

    algo = root.algorithm
    if algo is None or algo.kind != "token_grpo" or _family(root) != "nextstep_1":
        return
    if root.rollout is None or root.rollout.noise_level is None:
        raise ValueError("config missing required field: rollout.noise_level")


def rule_janus_r1_pairs_with_multisegment_grpo(root: RootConfig) -> None:
    """``janus_pro_r1`` and ``token_grpo_multisegment`` select each other, and the
    multi-segment protocol's ``rollout.final_image_policy`` is one of two values."""

    algo = root.algorithm
    if algo is None:
        return
    family = _family(root)
    if family == "janus_pro_r1" and algo.kind != "token_grpo_multisegment":
        raise ValueError(
            "model.family=janus_pro_r1 requires algorithm.kind=token_grpo_multisegment",
        )
    if algo.kind != "token_grpo_multisegment":
        return
    if family != "janus_pro_r1":
        raise ValueError("token_grpo_multisegment currently requires model.family=janus_pro_r1")
    # The schema Literal owns the legal values; this pairing requires a value.
    if root.rollout is None or root.rollout.final_image_policy is None:
        raise ValueError("rollout.final_image_policy must be 'always_generate' or 'use_selfcheck'")


CROSS_SECTION_RULES: tuple[CrossSectionRule, ...] = (
    rule_kl_reward_shaping_needs_step_kl,
    rule_algorithm_consumes_only_its_surface,
    rule_sft_weight_is_owned_and_backed,
    rule_sde_objectives_declare_a_sampler,
    rule_nextstep_token_grpo_needs_noise_level,
    rule_janus_r1_pairs_with_multisegment_grpo,
)


def check_cross_section_rules(root: RootConfig) -> None:
    """Run every cross-section rule; the first violation raises."""

    for rule in CROSS_SECTION_RULES:
        rule(root)


__all__ = [
    "CROSS_SECTION_RULES",
    "CrossSectionRule",
    "check_cross_section_rules",
]
