"""Multi-reward registry — weighted combination of named reward functions.

Ported from the multi_score() pattern in flow_grpo/rewards.py.

This module is the construction boundary between YAML config and reward code:
``_REWARD_REGISTRY`` maps each config-facing component name to its thin
``RewardFunction`` class under ``vrl.rewards.functions``, and
``MultiReward.from_dict`` (called from ``vrl/scripts/common/factory.py`` with
the resolved ``reward.components`` / ``reward.kwargs``) instantiates them.
The heavyweight scoring networks are never built here — each function only
pins a ``model_factory`` dotted path that the scorer resolves later, on the
resolved device inside its own memory frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.base import (
    DiskArtifactRewardFunction,
    RewardCleanupError,
    RewardFunction,
)
from vrl.rewards.runtime import build_reward_scorer
from vrl.rewards.types import RewardOutput, RewardSample

# Registry of reward function factories.
# Each factory takes (device,) and returns a RewardFunction instance.
_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {}


def get_reward(name: str) -> type[RewardFunction]:
    """Look up a registered reward function class by name.

    The builtins register lazily (importing every reward module is the cost),
    so a first lookup from a config gate populates the registry itself instead
    of depending on ``MultiReward.from_dict`` having run earlier.
    """
    if not _REWARD_REGISTRY:
        _register_builtins()
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function: {name!r}. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]


def _register_builtins() -> None:
    from vrl.rewards.functions.aesthetic import AestheticReward
    from vrl.rewards.functions.animereward_quality import AnimeRewardQualityReward
    from vrl.rewards.functions.codex_image_qa import CodexImageQAReward
    from vrl.rewards.functions.cosmos3_reasoner import Cosmos3ReasonerReward
    from vrl.rewards.functions.countgd import CountGDReward
    from vrl.rewards.functions.geneval import GenEvalReward
    from vrl.rewards.functions.grounded_ocr import GroundedOCRReward
    from vrl.rewards.functions.hpsv3 import HPSv3Reward
    from vrl.rewards.functions.idm_action_following import ActionFollowingReward
    from vrl.rewards.functions.image_sharpness import ImageSharpnessReward
    from vrl.rewards.functions.kling_video_reward import KlingVideoReward
    from vrl.rewards.functions.motion_dynamics import MotionDynamicsReward
    from vrl.rewards.functions.nsfw_safety import NSFWSafetyReward
    from vrl.rewards.functions.ocr import OCRReward
    from vrl.rewards.functions.phymotion import PhyMotionReward
    from vrl.rewards.functions.pickscore import PickScoreReward
    from vrl.rewards.functions.robotics_video_reward import RoboticsVideoReward
    from vrl.rewards.functions.target_dino_similarity import TargetDinoSimilarityReward
    from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
    from vrl.rewards.functions.videocon_physics import VideoConPhysicsReward
    from vrl.rewards.functions.videoscore2 import VideoScore2Reward
    from vrl.rewards.functions.wd_tagger import WDTaggerReward

    _REWARD_REGISTRY.update(
        {
            "aesthetic": AestheticReward,
            "idm_action_following": ActionFollowingReward,
            "animereward_quality": AnimeRewardQualityReward,
            "image_sharpness": ImageSharpnessReward,
            "codex_image_qa": CodexImageQAReward,
            "countgd": CountGDReward,
            "geneval": GenEvalReward,
            "grounded_ocr": GroundedOCRReward,
            "nsfw_safety": NSFWSafetyReward,
            "ocr": OCRReward,
            "pickscore": PickScoreReward,
            "wd_tagger": WDTaggerReward,
            # Future Reward suite (SPRINT_future_reward): DINOv2 perceptual anchor + RAFT
            # motion guard. They replaced the deleted pixel-L1 target_video_similarity (it was
            # reward-hackable, see S1). The IDM action-following signal is designed in S3 but
            # not shipped.
            "target_dino_similarity": TargetDinoSimilarityReward,
            "motion_dynamics": MotionDynamicsReward,
            "robotics_video_reward": RoboticsVideoReward,
            "hpsv3": HPSv3Reward,
            "kling_video_reward": KlingVideoReward,
            "cosmos3_reasoner": Cosmos3ReasonerReward,
            "videocon_physics": VideoConPhysicsReward,
            "videoscore2": VideoScore2Reward,
            "unified_reward_video": UnifiedRewardVideoReward,
            "phymotion": PhyMotionReward,
        }
    )


class MultiReward(RewardFunction):
    """Weighted combination of named reward functions.

    Usage::

        reward_fn = MultiReward.from_dict(
            {"ocr": 1.0, "aesthetic": 0.3},
            device="cuda",
        )
        output = await reward_fn.score_batch([sample])
        # output.scores     -> weighted totals
        # output.components -> {"ocr": (0.87,), "aesthetic": (5.2,)}
    """

    def __init__(
        self,
        rewards: list[tuple[str, float, RewardFunction]],
    ) -> None:
        self.rewards = rewards
        # Composite teardown is retryable: remember children whose shutdown
        # already succeeded so a retry reaches only the ones that actually
        # failed instead of double-shutting siblings.
        self._shutdown_completed_children: set[int] = set()

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether every component yields while its model work executes."""

        return bool(self.rewards) and all(
            reward.scoring_is_nonblocking for _, _, reward in self.rewards
        )

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether every external component proved accelerator isolation."""

        return all(reward.external_accelerator_isolation_verified for _, _, reward in self.rewards)

    async def preflight(self) -> None:
        """Check every component's remote dependency before training starts."""

        for _, _, fn in self.rewards:
            await fn.preflight()

    async def activate(self) -> None:
        """Pre-warm every component at a GPU handoff."""

        for _, _, fn in self.rewards:
            await fn.activate()

    async def shutdown(self) -> None:
        errors: list[BaseException] = []
        for _, _, fn in self.rewards:
            child_id = id(fn)
            if child_id in self._shutdown_completed_children:
                continue
            try:
                await fn.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._shutdown_completed_children.add(child_id)
        if errors:
            raise RewardCleanupError("reward shutdown failures", errors)

    @classmethod
    def from_dict(
        cls,
        score_dict: dict[str, float],
        device: str = "cuda",
        reward_kwargs: dict[str, dict[str, Any]] | None = None,
        memory_parking_required: bool | None = None,
        inference_configs: Mapping[str, RewardInferenceConfig] | None = None,
    ) -> MultiReward:
        """Build from ``{"name": weight}`` dict, like flow_grpo config.reward_fn.

        ``reward_kwargs`` allows passing per-reward init kwargs, keyed by name,
        e.g. ``{"ocr": {"debug_dir": "out/ocr_debug"}}``.

        Config-driven callers pass their already-resolved ``inference_configs``.
        Direct callers may omit it; every component then executes in-process.
        """
        _register_builtins()
        reward_kwargs = reward_kwargs or {}
        configured_weights = {name: float(weight) for name, weight in score_dict.items()}
        reward_classes = {name: get_reward(name) for name in configured_weights}
        resolved_inference_configs: Mapping[str, RewardInferenceConfig] = (
            {name: RewardInferenceConfig() for name in configured_weights}
            if inference_configs is None
            else inference_configs
        )
        if memory_parking_required:
            validate_reward_memory_parking_components(
                tuple(reward_classes),
                device=device,
                reward_kwargs=reward_kwargs,
                inference_configs=resolved_inference_configs,
            )

        triples: list[tuple[str, float, RewardFunction]] = []
        for name, weight in configured_weights.items():
            reward_cls = reward_classes[name]
            # `or {}`: a bare YAML key (kwargs: <name>:) parses as None.
            extra = dict(reward_kwargs.get(name) or {})
            reserved_runtime_keys = sorted(set(extra) & {"scorer", "runtime"})
            if reserved_runtime_keys:
                raise ValueError(
                    f"reward.kwargs.{name} cannot set runtime injection keys "
                    f"{reserved_runtime_keys}; configure reward inference through "
                    "reward.inference.<name>",
                )
            inference = resolved_inference_configs[name]
            if "execution" in extra:
                raise ValueError(
                    f"reward.kwargs.{name}.execution is no longer supported: the "
                    "Ray reward pool was removed and rewards score in-process. "
                    "Drop the key; shared-GPU parking is derived from distributed "
                    "resource topology.",
                )
            if inference.kind == "http":
                if not issubclass(reward_cls, DiskArtifactRewardFunction):
                    raise ValueError(
                        f"reward {name!r} uses in-memory artifacts and cannot use HTTP inference",
                    )
                local_only = sorted(
                    set(extra)
                    & {
                        "device",
                        "memory_parking_residual_bytes_limit",
                        "sleep_offload",
                        "worker_config",
                    },
                )
                if local_only:
                    raise ValueError(
                        f"HTTP reward {name!r} cannot set local execution keys "
                        f"{local_only}; model/device configuration belongs to the "
                        "standalone reward service",
                    )
                component_device = "cpu"
                extra["scorer"] = build_reward_scorer(inference=inference)
            else:
                component_device = reward_cls.resolve_execution_device(
                    device=device,
                    kwargs=extra,
                )
            # The resolved value is passed once through the constructor's common
            # device argument; remove a component override after it has served as
            # the CPU-downgrade input.
            extra.pop("device", None)
            if memory_parking_required is True and component_device.startswith("cuda"):
                # GPU ownership comes from topology. A shared reward cannot rely
                # on every preset remembering an independent parking knob.
                residual_limit = reward_cls.memory_parking_residual_bytes_limit
                if residual_limit is None:
                    raise ValueError(
                        f"reward {name!r} has no complete memory-parking contract",
                    )
                extra["sleep_offload"] = True
                extra["memory_parking_residual_bytes_limit"] = int(residual_limit)
            elif memory_parking_required is not None:
                # A dedicated reward owns its GPU and remains resident even if
                # an inherited reward preset carried the old shared-phase knob.
                # CPU-only components also never receive a GPU parking knob.
                extra.pop("sleep_offload", None)
            triples.append(
                (
                    name,
                    weight,
                    reward_cls(device=component_device, **extra),
                ),
            )
        return cls(triples)

    async def score(self, sample: RewardSample) -> float:
        return (await self.score_batch((sample,))).scores[0]

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Return weighted totals plus per-component observations from one call."""

        totals = [0.0] * len(samples)
        components: dict[str, tuple[float, ...]] = {}
        timing_ms: dict[str, float] = {}
        for name, weight, fn in self.rewards:
            output = await fn.score_batch(samples)
            if len(output.scores) != len(samples):
                raise ValueError(
                    f"reward component {name!r} returned wrong number of scores: "
                    f"scores={len(output.scores)}, samples={len(samples)}",
                )
            components[name] = output.scores
            for key, value in output.timing_ms.items():
                timing_ms[str(key)] = timing_ms.get(str(key), 0.0) + float(value)
            for index, score in enumerate(output.scores):
                totals[index] += weight * score
        return RewardOutput(
            scores=tuple(totals),
            components=components,
            timing_ms=timing_ms,
        )

    async def park_memory(self) -> bool:
        """Actively park every component and report whether any owner parked."""

        parked = False
        errors: list[BaseException] = []
        for name, _, fn in self.rewards:
            try:
                parked = await fn.park_memory() or parked
            except BaseException as error:
                errors.append(RuntimeError(f"reward component {name!r} failed to park"))
                errors[-1].__cause__ = error
        if errors:
            raise RewardCleanupError("reward memory parking failures", errors)
        return parked


def validate_reward_memory_parking_components(
    names: tuple[str, ...],
    *,
    device: str = "cuda",
    reward_kwargs: dict[str, dict[str, Any]] | None = None,
    inference_configs: Mapping[str, RewardInferenceConfig] | None = None,
) -> None:
    """Fail before model construction when a shared reward cannot fully park.

    Config-driven callers pass ``inference_configs`` from ``RewardRuntimeConfig``.
    The fallback preserves this validator as a boundary for direct ``MultiReward``
    construction, where every component executes in-process.
    """

    _register_builtins()
    kwargs_by_name = reward_kwargs or {}
    if inference_configs is None:
        inference_configs = {name: RewardInferenceConfig() for name in names}
    gpu_components = [
        name
        for name in names
        if inference_configs[name].kind == "in_process"
        if get_reward(name)
        .resolve_execution_device(
            device=device,
            kwargs=dict(kwargs_by_name.get(name) or {}),
        )
        .startswith("cuda")
    ]
    if not gpu_components:
        raise ValueError(
            "shared reward GPU topology has no configured GPU reward "
            "component. Declare the reward as CPU execution "
            "(distributed.resources.reward.device=cpu) instead.",
        )
    if len(gpu_components) > 1:
        raise ValueError(
            "shared reward parking supports at most one configured GPU "
            f"component per process, got {gpu_components}. vLLM CuMemAllocator.sleep "
            "is process-wide: tags select which pages are backed up, not which pages "
            "are unmapped. Keep CPU reward siblings or use a dedicated/remote reward.",
        )
    unsupported = [
        name
        for name in gpu_components
        if get_reward(name).memory_parking_residual_bytes_limit is None
    ]
    if unsupported:
        raise ValueError(
            "shared reward GPU requires complete topology-driven memory parking, "
            f"but these reward components do not provide it: {unsupported}. "
            "Use a dedicated reward GPU or a reward with complete parking support.",
        )
