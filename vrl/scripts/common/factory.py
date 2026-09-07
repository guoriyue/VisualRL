"""Factory functions for common online training recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vrl.algorithms.base import Algorithm
from vrl.config.builders import BuiltConfigs
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.registry import ModelFamilyEntry
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rewards import RewardRuntime
from vrl.rewards.base import RewardFunction
from vrl.rollouts.evaluators.base import Evaluator
from vrl.run import ResolvedReward


@dataclass(frozen=True, slots=True)
class AlgorithmEvaluatorPair:
    """Algorithm instance plus its optional evaluator."""

    algorithm: Algorithm
    evaluator: Evaluator | None


def build_reward_function(reward: ResolvedReward) -> RewardFunction:
    """Build the online reward function from the resolved reward inputs.

    Device and parking policy are decided once by ``ResolvedOnlineRun.
    reward_inputs``; this factory validates the components and constructs.
    In-process GPU ownership decides parking: a shared reward must completely
    park its model memory after scoring, while a dedicated reward stays resident.
    HTTP components own their deployment externally and receive no local parking
    policy. YAML selects transport, not lifecycle behavior.
    """

    config = reward.config
    if not any(weight > 0 for weight in config.weights.values()):
        raise ValueError("At least one reward component must have weight > 0.")
    from vrl.rewards.functions.registry import MultiReward

    if reward.memory_parking_required and not config.all_external_inference:
        from vrl.rewards.functions.registry import (
            validate_reward_memory_parking_components,
        )

        validate_reward_memory_parking_components(
            tuple(config.weights),
            device=reward.device,
            reward_kwargs=config.kwargs,
            inference_configs=config.inference_configs,
        )

    return MultiReward.from_dict(
        config.weights,
        device=reward.device,
        reward_kwargs=config.kwargs,
        memory_parking_required=reward.memory_parking_required,
        inference_configs=config.inference_configs,
    )


def build_reward_runtime(reward: ResolvedReward) -> RewardRuntime:
    """Build the collector-facing runtime around the configured reward function."""

    from vrl.rewards.runtime import RewardFunctionRuntime

    return RewardFunctionRuntime(build_reward_function(reward))


def validate_reward_memory_parking(
    *,
    resources: ResolvedDistributedResources,
    built: BuiltConfigs,
    device: str | None = None,
) -> None:
    """Validate shared reward parking without constructing a reward model."""

    if not bool(resources.lifecycle.release_reward_after_score):
        return
    reward = built.reward
    if reward is None or reward.all_external_inference:
        return
    names = tuple(reward.weights)
    if not names:
        return
    from vrl.rewards.functions.registry import (
        validate_reward_memory_parking_components,
    )

    validate_reward_memory_parking_components(
        names,
        device=str(device or resources.reward_torch_device()),
        reward_kwargs=reward.kwargs,
        inference_configs=reward.inference_configs,
    )


def build_algorithm_and_evaluator(
    *,
    family_entry: ModelFamilyEntry,
    built: BuiltConfigs,
    collector_config: Any,
    scheduler: Any | None = None,
) -> AlgorithmEvaluatorPair:
    """Build the algorithm/evaluator pair for a strict online recipe."""

    if not family_entry.supports_policy_replay:
        raise RuntimeError(
            f"{family_entry.family} is generation-only: its runtime exposes no "
            "trainable actions, transition likelihoods, or policy replay evaluator",
        )
    algorithm_config = built.algorithm
    algorithm_section = built.root.algorithm
    if algorithm_section is None:
        raise ValueError("online recipe requires an algorithm section")
    kind = algorithm_section.kind
    if kind == "diffusion_dpo":
        raise ValueError(
            "diffusion_dpo is an offline recipe and is not supported by common online recipe",
        )
    reward = built.reward
    if reward is None:
        raise ValueError("online recipe requires reward configuration")
    diffusion_logprob_kinds = {"grpo", "dance_grpo", "flash_grpo", "flow_dppo", "grpo_guard"}
    precision = built.precision
    if precision.diffusion_math != "fp32" and kind not in diffusion_logprob_kinds:
        raise ValueError(
            "precision.diffusion_math.dtype overrides are supported only by "
            "diffusion log-prob "
            f"objectives; algorithm.kind={kind!r} keeps its protected math in fp32",
        )

    if kind in diffusion_logprob_kinds:
        # All four are flow-matching GRPO-family algorithms on the same SDE
        # evaluator. dance_grpo reuses FlowGRPO unchanged (its delta is the
        # trainer's random timestep selection + multi-reward); flow_dppo /
        # grpo_guard are trust-region variants whose loss reads the rollout
        # proposal mean (sampling.return_prev_sample_mean).
        from vrl.algorithms.grpo.continuous import GRPO, FlashGRPO, FlowDPPO, GRPOGuard

        is_chunk_autoregressive = (
            family_entry.policy_semantics.generation_regime == "chunk_autoregressive"
        )
        if is_chunk_autoregressive and float(getattr(algorithm_config, "sft_weight", 0.0)) > 0:
            raise ValueError(
                f"{family_entry.family} grouped causal-chunk replay does not "
                "implement the full-sequence scheduler target required by "
                "algorithm.sft_weight; set sft_weight=0",
            )
        advantage_estimator = algorithm_config.build_estimator(
            component_weights=reward.weights,
        )
        if kind == "flow_dppo":
            algorithm: object = FlowDPPO(
                algorithm_config,
                advantage_estimator=advantage_estimator,
            )
        elif kind == "grpo_guard":
            algorithm = GRPOGuard(
                algorithm_config,
                advantage_estimator=advantage_estimator,
            )
        elif kind == "flash_grpo":
            algorithm = FlashGRPO(
                algorithm_config,
                advantage_estimator=advantage_estimator,
            )
        else:
            algorithm = GRPO(
                algorithm_config,
                advantage_estimator=advantage_estimator,
            )
        if is_chunk_autoregressive:
            if precision.diffusion_math != "fp32":
                raise ValueError(
                    f"{family_entry.family} uses an exact fp32 Gaussian re-noise "
                    "policy; precision.diffusion_math.dtype overrides are not "
                    "implemented for grouped causal-chunk replay",
                )
            if kind == "dance_grpo":
                raise ValueError(
                    f"{family_entry.family} uses one ordered full-trajectory replay; "
                    "DanceGRPO's random denoise-timestep subset is not defined for "
                    "the [temporal_chunk, denoise_transition] policy axes. Use grpo.",
                )
            if kind in {"flash_grpo", "flow_dppo", "grpo_guard"}:
                raise ValueError(
                    f"{family_entry.family} uses grouped causal-chunk replay; "
                    f"algorithm.kind={kind!r} requires reverse-SDE dt signals that "
                    "the batch re-noise policy does not expose. Use grpo.",
                )
            from vrl.rollouts.evaluators.denoise import (
                ChunkAutoregressiveDenoiseLogProbEvaluator,
            )

            return AlgorithmEvaluatorPair(
                algorithm=algorithm,
                evaluator=ChunkAutoregressiveDenoiseLogProbEvaluator(),
            )

        math_dtype = resolve_torch_dtype(precision.diffusion_math)
        denoise = collector_config.denoise or DenoiseRequestOptions()
        from vrl.rollouts.evaluators.denoise.sde_logprob import (
            DiffusionSDELogProbEvaluator,
        )

        return AlgorithmEvaluatorPair(
            algorithm=algorithm,
            evaluator=DiffusionSDELogProbEvaluator(
                scheduler,
                noise_level=denoise.noise_level,
                sde_type=denoise.sde_type or "flow_grpo",
                math_dtype=math_dtype,
            ),
        )

    if kind == "token_grpo":
        from vrl.algorithms.grpo.token import TokenGRPO

        if family_entry.policy_semantics.action_distribution == "continuous":
            from vrl.rollouts.evaluators.token import ContinuousTokenLogProbEvaluator

            evaluator = ContinuousTokenLogProbEvaluator()
        else:
            from vrl.rollouts.evaluators.token import TokenLogProbEvaluator

            evaluator = TokenLogProbEvaluator()
        return AlgorithmEvaluatorPair(
            algorithm=TokenGRPO(
                algorithm_config,
                advantage_estimator=algorithm_config.build_estimator(
                    component_weights=reward.weights,
                ),
            ),
            evaluator=evaluator,
        )

    if kind == "token_grpo_multisegment":
        from vrl.algorithms.grpo.multisegment import MultiSegmentTokenGRPO
        from vrl.rollouts.evaluators.token import MultiSegmentTokenLogProbEvaluator

        if family_entry.family != "janus_pro_r1":
            raise ValueError(
                "token_grpo_multisegment currently requires model family janus_pro_r1",
            )
        segment_flags = dict(algorithm_config.train_segments or {})
        enabled_segments = tuple(name for name, enabled in segment_flags.items() if bool(enabled))
        return AlgorithmEvaluatorPair(
            algorithm=MultiSegmentTokenGRPO(
                algorithm_config,
                advantage_estimator=algorithm_config.build_estimator(
                    component_weights=reward.weights,
                ),
            ),
            evaluator=MultiSegmentTokenLogProbEvaluator(enabled_segments=enabled_segments),
        )

    if kind == "diffusion_nft":
        from vrl.algorithms.diffusion_nft import DiffusionNFT

        return AlgorithmEvaluatorPair(
            algorithm=DiffusionNFT(algorithm_config),
            evaluator=None,
        )

    if kind == "v_grpo":
        from vrl.algorithms.v_grpo import VGRPO

        return AlgorithmEvaluatorPair(
            algorithm=VGRPO(algorithm_config),
            evaluator=None,
        )

    raise ValueError(f"unsupported online algorithm.kind: {kind!r}")


__all__ = [
    "AlgorithmEvaluatorPair",
    "build_algorithm_and_evaluator",
    "build_reward_function",
    "build_reward_runtime",
    "validate_reward_memory_parking",
]
