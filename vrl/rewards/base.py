"""RewardFunction base class for async generated-sample scoring.

The plugin layer between ``RewardFunctionRuntime`` above (runtime.py) and the
``RewardScorer`` transports below (inference.py): concrete rewards in
vrl/rewards/functions subclass one of the bases here. The class ladder encodes
capabilities the registry and runtime probe, not taxonomy:
``InferenceRewardFunction`` owns the materialize / score / validate /
release-or-retain seam so every transport (including injected fakes) passes
the same result-identity guard; ``CumemRewardFunction`` declares that all
model CUDA state is built in the tagged pool, enabling verified memory
parking; ``DiskArtifactRewardFunction`` materializes media to disk before
scoring (registry preflight selects it by subclass).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from vrl.rewards.artifacts import (
    ArtifactFormat,
    DiskRewardArtifactStore,
    InMemoryRewardArtifactStore,
    MediaType,
    RewardArtifactStore,
)
from vrl.rewards.inference import (
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.rewards.protocols import (
    ArtifactRetainingError,
    MemoryParkingScorer,
    RemoteReadyScorer,
    RewardScorer,
)
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class RewardCleanupError(RuntimeError):
    """One reward operation accumulated multiple release/teardown failures."""

    def __init__(self, message: str, errors: list[BaseException]) -> None:
        self.errors = tuple(errors)
        details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{message}: {details}")


@dataclass(frozen=True, slots=True)
class ProductionContract:
    """What ``production.<reward>.enabled`` asserts about one reward's config.

    A reward opts into the production gate by declaring one of these in its
    class-declaration block, beside ``model_factory`` and its ``default_*``
    values; a reward that declares none has no production gate, and enabling
    one for it is a config error. Everything the gate compares against is
    declared here, so the check needs the contract and the configured kwargs
    and nothing else.

    Most of what a production config can get wrong (a model it cannot load, a
    row it cannot read, a service that is down) shows up by running the reward
    -- ``python -m vrl.scripts.rewards.preflight`` does that in seconds. What
    stays here is what running the reward cannot answer: the prompt task types
    the reward was validated against, the archive format a production run must
    leave behind, and the loader keys a production config must not carry.
    """

    # Prompt task types (``data.task_type``) this reward is validated for.
    task_types: frozenset[str]
    # The archive a production run must leave behind. Defaults are the disk
    # rewards' own; a reward whose production evidence differs declares its own.
    media_type: MediaType = "video"
    artifact_format: ArtifactFormat = "mp4"

    # ``DiskArtifactRewardFunction.__init__`` prefers ``worker_config["model_factory"]``
    # over the class's own, so this key redirects the reward's model loader. A
    # production config names its model, never its loader. Not per-reward data
    # -- it is a property of how the loader reads its config -- so it lives on
    # this type rather than being restated by every reward.
    LOCKED_WORKER_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({"model_factory"})

    def require(self, name: str, kwargs: Mapping[str, Any], *, task_type: str) -> None:
        """Refuse a production config for component ``name`` that breaks the contract."""

        prefix = f"production.{name} requires"
        if str(kwargs.get("media_type", "")) != str(self.media_type):
            raise ValueError(f"{prefix} reward.kwargs.{name}.media_type={self.media_type}")
        if str(kwargs.get("artifact_format", "")) != str(self.artifact_format):
            raise ValueError(f"{prefix} artifact_format={self.artifact_format}")
        if not str(kwargs.get("reward_name", "")).strip():
            raise ValueError(f"{prefix} reward.kwargs.{name}.reward_name")
        worker_config = kwargs.get("worker_config") or {}
        forbidden = sorted(key for key in self.LOCKED_WORKER_CONFIG_KEYS if key in worker_config)
        if forbidden:
            raise ValueError(
                f"production.{name} worker_config should name the reward model directly; "
                f"remove extra loader fields: {', '.join(forbidden)}",
            )
        if task_type not in self.task_types:
            expected = ", ".join(sorted(self.task_types)) or "<none>"
            raise ValueError(f"{prefix} data.task_type={expected}")


class RewardFunction:
    """Base class for pure scoring functions and reward composition."""

    # The production gate's contract, declared by a reward that has one.
    production: ClassVar[ProductionContract | None] = None

    # None is fail-closed. A specialized base class declares this capability only
    # when all model-owned CUDA state is built in the tagged runtime pool.
    memory_parking_residual_bytes_limit: ClassVar[int | None] = None
    # Most reward constructors expose the selected device as ``device``;
    # exceptional schemas (for example NSFW's classifier_device) override it.
    device_config_key: ClassVar[str] = "device"

    @classmethod
    def resolve_execution_device(
        cls,
        *,
        device: str,
        kwargs: Mapping[str, Any],
    ) -> str:
        """Return the concrete execution device under the resource ceiling.

        The device policy in one place: distributed resources own placement,
        so a component override (``cls.device_config_key`` or
        ``worker_config.device``) may only downgrade from the resolved GPU to
        CPU — never pick a different CUDA ordinal, and never request CUDA on
        a CPU resource plan. A CPU downgrade creates no CuMem owner and can
        coexist with the one GPU reward. Reward classes with a fixed device
        override this hook (OCR forces CPU for its CPU-only engine).
        """

        candidates: list[tuple[str, Any]] = [
            (cls.device_config_key, kwargs.get(cls.device_config_key)),
        ]
        worker_config = kwargs.get("worker_config")
        if isinstance(worker_config, Mapping):
            candidates.append(
                ("worker_config.device", worker_config.get("device")),
            )
        resolved = str(device or "").strip().lower()
        configured = [
            (key, str(value).strip().lower())
            for key, value in candidates
            if str(value or "").strip()
        ]
        distinct = {value for _, value in configured}
        if len(distinct) > 1:
            raise ValueError(
                f"reward component device overrides disagree: {configured}",
            )
        effective = configured[0][1] if configured else resolved
        key = configured[0][0] if configured else "resolved device"
        if resolved.startswith("cuda"):
            if effective.startswith("cuda") and effective != resolved:
                raise ValueError(
                    f"reward {key}={effective!r} conflicts with the "
                    f"distributed-resources CUDA device {device!r}. Remove the "
                    "component override; distributed.resources owns the CUDA "
                    "ordinal.",
                )
            # A component may explicitly downgrade from its GPU ownership
            # ceiling to CPU. It then creates no CuMem owner and can coexist
            # with one GPU reward.
        elif effective.startswith("cuda"):
            raise ValueError(
                f"reward {key}={effective!r} requests CUDA, but distributed "
                f"resources resolved {device!r}. CPU resources cannot launch "
                "a CUDA reward.",
            )
        return effective

    @classmethod
    def worker_config_with_device(
        cls,
        worker_config: Mapping[str, Any] | None,
        *,
        device: str,
    ) -> dict[str, Any]:
        """Copy ``worker_config``, stamping the resolved device as a ceiling.

        The same policy as :meth:`resolve_execution_device`, applied at the
        config-bag boundary: constructors that feed a ``worker_config`` dict
        to a model must stamp the device through the ceiling check, not
        assign it directly.
        """

        cfg = dict(worker_config or {})
        if device:
            cfg["device"] = cls.resolve_execution_device(
                device=str(device),
                kwargs={"worker_config": cfg},
            )
        return cfg

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether this scorer yields while scoring runs elsewhere."""

        return False

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work has been isolated."""

        return True

    async def preflight(self) -> None:
        """Validate dependencies before scoring begins."""

        return None

    async def activate(self) -> None:
        """Pre-warm this reward at a GPU handoff; CPU/remote rewards need none."""

        return None

    async def park_memory(self) -> bool:
        """Release reward-owned accelerator memory and report whether an owner parked."""

        return False

    async def score(self, sample: RewardSample) -> float:
        """Score one generated sample; family-specific scalar extension hook."""

        raise NotImplementedError(f"{type(self).__name__}.score is not implemented")

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Score ordered samples through the batch/runtime hook."""

        return RewardOutput(scores=tuple([await self.score(sample) for sample in samples]))

    async def shutdown(self) -> None:
        """Release reward-owned resources, when applicable."""

        return None


class InferenceRewardFunction(RewardFunction):
    """Reward function backed by one scorer transport and one artifact store."""

    def __init__(
        self,
        *,
        reward_name: str,
        score_key: str,
        scorer: RewardScorer,
        artifact_store: RewardArtifactStore | None = None,
        retain_artifacts: bool = False,
        debug_dir: str = "",
        request_prefix: str = "reward",
        debug_basename: str = "reward",
    ) -> None:
        normalized_reward_name = str(reward_name).strip()
        if not normalized_reward_name:
            raise ValueError("reward_name must be non-empty")
        normalized_score_key = str(score_key).strip()
        if not normalized_score_key:
            raise ValueError("score_key must be non-empty")
        if artifact_store is None:
            # In-memory media is the default transport; the disk base injects
            # the file-backed store.
            artifact_store = InMemoryRewardArtifactStore()
        self.reward_name = normalized_reward_name
        self.score_key = normalized_score_key
        selected_score_keys = tuple(part.strip() for part in normalized_score_key.split("+"))
        if not all(selected_score_keys):
            raise ValueError(
                f"score_key {normalized_score_key!r} contains an empty component; "
                'use "a+b" to sum score keys',
            )
        self._selected_score_keys = selected_score_keys
        self.scorer = scorer
        self.artifact_store = artifact_store
        self._retain_artifacts = bool(retain_artifacts)
        self.debug_dir = str(debug_dir)
        self._request_prefix = request_prefix
        self._debug_basename = debug_basename
        self._inference_started = False

    @property
    def scoring_is_nonblocking(self) -> bool:
        """Whether this scorer yields while inference runs elsewhere."""

        return bool(self.scorer.scoring_is_nonblocking)

    @property
    def external_accelerator_isolation_verified(self) -> bool:
        """Whether out-of-plan reward accelerator work has been isolated."""

        return bool(self.scorer.external_accelerator_isolation_verified)

    async def preflight(self) -> None:
        """Fail before training starts when a remote scoring dependency is broken.

        In-process runtimes have nothing to check here (their model loads
        lazily on the reward device). Remote runtimes expose ``ensure_ready``
        so an unreachable, not-ready, or wrong-model service is reported at
        startup instead of after the first generation batch completes.
        """

        scorer = self.scorer
        if isinstance(scorer, RemoteReadyScorer):
            await scorer.ensure_ready()

    async def activate(self) -> None:
        """Build or wake a parking-capable in-process model at a GPU handoff.

        The inverse of :meth:`park_memory`. Activation marks the model as
        started so a handoff that never scores still releases its GPU lease.
        """

        scorer = self.scorer
        if not isinstance(scorer, MemoryParkingScorer):
            return
        await scorer.activate()
        self._inference_started = True

    async def park_memory(self) -> bool:
        """Park this reward runtime when its model has been activated."""

        scorer = self.scorer
        if not isinstance(scorer, MemoryParkingScorer):
            return False
        if not scorer.requires_memory_parking:
            return False
        if not self._inference_started:
            # This component never activated its model (for example an earlier
            # sibling failed). There is no GPU lease to release.
            return False
        await scorer.park_memory()
        return True

    async def score(self, sample: RewardSample) -> float:
        """Score one sample through the batch inference path."""

        return (await self.score_batch((sample,))).scores[0]

    async def score_batch(self, samples: Sequence[RewardSample]) -> RewardOutput:
        """Materialize, score, validate, and finalize one ordered sample batch."""

        samples = list(samples)
        if not samples:
            return RewardOutput(scores=())

        scorer = self.scorer
        total_started = time.perf_counter()
        materialize_started = time.perf_counter()
        artifacts = self.artifact_store.materialize(samples)
        materialization_ms = (time.perf_counter() - materialize_started) * 1000.0
        operation_error: BaseException | None = None
        output: RewardOutput | None = None
        request_id: str | None = None
        try:
            if len(artifacts) != len(samples):
                raise ValueError(
                    "reward artifact store returned wrong number of artifacts: "
                    f"artifacts={len(artifacts)}, samples={len(samples)}",
                )
            request_id = f"{self._request_prefix}-{uuid.uuid4().hex}"
            request = RewardInferenceRequest(
                request_id=request_id,
                artifacts=tuple(artifacts),
            )
            inference_started = time.perf_counter()
            # Contract enforcement lives at this seam, not inside each runtime,
            # so every runtime (including injected fakes) gets the same result
            # identity guard and request-order re-sort.
            self._inference_started = True
            raw_results = await scorer.score_batch(request)
            results = request.validate_and_order_results(raw_results)
            inference_total_ms = (time.perf_counter() - inference_started) * 1000.0
            total_latency_ms = (time.perf_counter() - total_started) * 1000.0
            self._write_debug(
                request,
                results,
                artifact_materialization_ms=materialization_ms,
                inference_total_ms=inference_total_ms,
                total_reward_latency_ms=total_latency_ms,
            )
            output = RewardOutput(
                scores=tuple(self._select_score(result.scores) for result in results),
                timing_ms=_result_timing_summary(
                    results,
                    materialization_ms=materialization_ms,
                    inference_total_ms=inference_total_ms,
                    total_latency_ms=total_latency_ms,
                ),
            )
        except BaseException as error:
            operation_error = error

        cleanup_error: BaseException | None = None
        retain_for_remote = operation_error is not None and (
            isinstance(operation_error, asyncio.CancelledError)
            or (
                isinstance(operation_error, ArtifactRetainingError)
                and operation_error.retain_reward_artifacts
            )
        )
        if retain_for_remote:
            logger.warning(
                "reward inference did not confirm terminal state; retaining %d "
                "artifact(s) for request_id=%s",
                len(artifacts),
                request_id,
            )
        finalize = (
            self.artifact_store.retain
            if retain_for_remote or self._retain_artifacts
            else self.artifact_store.release
        )
        try:
            finalize(artifacts)
        except BaseException as error:
            cleanup_error = error
        if operation_error is not None and cleanup_error is not None:
            raise RewardCleanupError(
                "reward operation and artifact cleanup both failed",
                [operation_error, cleanup_error],
            )
        if operation_error is not None:
            raise operation_error
        if cleanup_error is not None:
            raise cleanup_error
        assert output is not None
        return output

    def _select_score(self, scores: Mapping[str, Any]) -> float:
        missing = [key for key in self._selected_score_keys if key not in scores]
        if missing:
            raise KeyError(
                "reward inference result missing score keys: "
                f"missing={missing}, requested={self.score_key!r}, "
                f"available={sorted(scores)}",
            )
        value = float(sum(float(scores[key]) for key in self._selected_score_keys))
        if not math.isfinite(value):
            raise ValueError(
                f"reward score_key={self.score_key!r} selected non-finite score: {value}",
            )
        return value

    async def shutdown(self) -> None:
        await self.scorer.shutdown()

    def _write_debug(
        self,
        request: RewardInferenceRequest,
        results: list[RewardInferenceResult],
        *,
        artifact_materialization_ms: float,
        inference_total_ms: float,
        total_reward_latency_ms: float,
    ) -> None:
        if not self.debug_dir:
            return
        debug_path = Path(self.debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        request_row = {
            "request_id": request.request_id,
            "artifact_ids": [artifact.artifact_id for artifact in request.artifacts],
            "reward_name": self.reward_name,
            "score_key": self.score_key,
            "artifact_materialization_ms": artifact_materialization_ms,
            "inference_total_ms": inference_total_ms,
            "total_reward_latency_ms": total_reward_latency_ms,
        }
        requests_file = debug_path / f"{self._debug_basename}_requests.jsonl"
        results_file = debug_path / f"{self._debug_basename}_results.jsonl"
        with requests_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request_row, sort_keys=True) + "\n")
        with results_file.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


class CumemRewardFunction(InferenceRewardFunction):
    """Reward whose model allocations support verified tagged-pool parking."""

    memory_parking_residual_bytes_limit: ClassVar[int] = CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


class DiskArtifactRewardFunction(CumemRewardFunction):
    """Base for rewards whose media is materialized to disk before scoring.

    The heavyweight sibling of the in-memory default: media is written to
    disk via ``DiskRewardArtifactStore`` and scored through the selected
    inference transport instead of riding the request in-memory (registry
    preflight selects HTTP-capable rewards by this subclass).
    ``model_factory`` / ``request_prefix`` / ``debug_basename`` are the only
    per-reward differences — concrete rewards pin their own ``reward_name`` /
    ``score_key`` / ``artifact_format`` defaults before delegating here, so no
    concrete reward copies this wiring. ``sleep_offload`` releases an
    in-process model's physical GPU pages between scores while its contents
    stay in pinned host RAM (the rollout/trainer own the GPU then), mirroring
    the rollout lease's sleep/wake. ``scorer`` injects a ready
    ``RewardScorer`` (HTTP components, tests); it wins over the factory-built
    one. Disk files belong to this reward call and are deleted after terminal
    success or failure; explicit ``retain_artifacts`` or an ambiguous remote
    state transfers them to the debug/output owner instead.
    """

    # Rule-3 collapse: concrete disk rewards differ only in these constants,
    # so each subclass is a declaration block instead of a forwarding
    # __init__. model_factory/request_prefix/debug_basename have no class
    # default on purpose — a subclass that forgets them fails loudly at
    # construction (AttributeError), not silently.
    model_factory: ClassVar[str]
    request_prefix: ClassVar[str]
    debug_basename: ClassVar[str]
    default_reward_name: ClassVar[str]
    default_score_key: ClassVar[str]
    default_artifact_format: ClassVar[str] = "mp4"
    default_media_type: ClassVar[MediaType] = "video"

    def __init__(
        self,
        *,
        reward_name: str | None = None,
        score_key: str | None = None,
        artifact_format: str | None = None,
        media_type: MediaType | None = None,
        artifact_dir: str = "outputs/reward_artifacts",
        debug_dir: str = "",
        device: str | None = None,
        sleep_offload: bool = False,
        memory_parking_residual_bytes_limit: int = 0,
        retain_artifacts: bool = False,
        worker_config: Mapping[str, Any] | None = None,
        scorer: RewardScorer | None = None,
    ) -> None:
        # Deferred: runtime.py imports this module (cycle guard).
        from vrl.rewards.runtime import build_reward_scorer

        model_factory = self.model_factory
        reward_name = self.default_reward_name if reward_name is None else reward_name
        score_key = self.default_score_key if score_key is None else score_key
        artifact_format = (
            self.default_artifact_format if artifact_format is None else artifact_format
        )
        media_type = self.default_media_type if media_type is None else media_type

        artifact_store = DiskRewardArtifactStore(
            artifact_dir,
            media_type=str(media_type),
            artifact_format=str(artifact_format),
        )

        if scorer is None:
            worker_cfg = dict(worker_config or {})
            has_model_factory = bool(
                str(worker_cfg.get("model_factory", "")).strip(),
            )
            # Normalize the model-id key ONCE here so the disk loaders
            # (kling/videocon) read only worker_config["reward_model_name"].
            # Precedence: an explicit worker_config.reward_model_name wins;
            # otherwise fold a top-level reward_name that looks like a HF repo
            # (contains "/") — a bare reward_name stays a logical tag, not a
            # model id.
            reward_name_repo = reward_name if "/" in reward_name else ""
            reward_model_name = str(
                worker_cfg.get("reward_model_name") or reward_name_repo or "",
            ).strip()
            model_path = str(worker_cfg.get("model_path", "")).strip()
            # YAML names the public model; the loader needs the private factory.
            if not has_model_factory:
                # A missing injected scorer is the in-process path: HTTP
                # components inject their ready client in MultiReward before
                # they reach this constructor. Every local disk reward
                # therefore needs its concrete factory even when it is a
                # composite model rather than one Hugging Face repository.
                worker_cfg["model_factory"] = model_factory
            if reward_model_name or model_path:
                if reward_model_name:
                    worker_cfg["reward_model_name"] = reward_model_name
                if not str(worker_cfg.get("reward_model_version", "")).strip():
                    worker_cfg["reward_model_version"] = reward_model_name or model_path
            # Resource resolution is the device source of truth. A nested model
            # override would split lifecycle ownership from real CUDA execution,
            # so apply the shared ceiling even when this constructor runs
            # outside MultiReward.
            if device is not None:
                worker_cfg = self.worker_config_with_device(
                    worker_cfg,
                    device=str(device),
                )
            if sleep_offload:
                worker_cfg["sleep_offload"] = True
                worker_cfg["memory_parking_residual_bytes_limit"] = int(
                    memory_parking_residual_bytes_limit,
                )
            scorer = build_reward_scorer(worker_cfg)

        super().__init__(
            reward_name=str(reward_name),
            score_key=str(score_key),
            scorer=scorer,
            artifact_store=artifact_store,
            retain_artifacts=retain_artifacts,
            debug_dir=debug_dir,
            request_prefix=self.request_prefix,
            debug_basename=self.debug_basename,
        )


def _result_timing_summary(
    results: list[RewardInferenceResult],
    *,
    materialization_ms: float,
    inference_total_ms: float,
    total_latency_ms: float,
) -> dict[str, float]:
    """Aggregate per-result transport timings into one reward-call summary.

    Stage-parallel phases report their slowest member (max over results);
    per-artifact inference cost is additive (sum), falling back to this
    call's measured inference wall time when a transport reports no
    per-result values.
    """

    def stage_max(key: str) -> float:
        return max(
            (float(result.timing_ms[key]) for result in results if key in result.timing_ms),
            default=0.0,
        )

    inference_values = [
        float(result.timing_ms["inference_ms"])
        for result in results
        if "inference_ms" in result.timing_ms
    ]
    return {
        "latency_ms": total_latency_ms,
        "queue_wait_ms": stage_max("queue_wait_ms"),
        "inference_ms": (sum(inference_values) if inference_values else float(inference_total_ms)),
        "artifact_materialization_ms": materialization_ms,
        "artifact_validation_ms": stage_max("service_artifact_validation_ms"),
        "service_inference_wall_ms": stage_max("service_inference_wall_ms"),
        "transport_roundtrip_ms": stage_max("http_roundtrip_ms"),
    }


__all__ = [
    "CumemRewardFunction",
    "DiskArtifactRewardFunction",
    "InferenceRewardFunction",
    "RewardCleanupError",
    "RewardFunction",
]
