"""Tests for rollout collector runtime orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tests.rollouts.collector._helpers import collect_scored
from vrl.config.schema import parse_config
from vrl.generation import (
    GenerationInput,
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.families.registry import get_model_family_entry
from vrl.ray.resources import RayLifecyclePlan
from vrl.rewards import RewardOutput, RewardSample
from vrl.rewards.base import RewardCleanupError
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import CollectorRequest, GenerationRequestBuilder
from vrl.rollouts.orchestration.prompt_collection import collect_prompt_groups
from vrl.rollouts.stats import RolloutStats
from vrl.trajectory import (
    RewardView,
    TrajectoryResolver,
    TrajectoryStoragePolicy,
    build_ar_discrete_trajectory,
    build_chunk_autoregressive_denoise_trajectory,
)


class _RequestBuilder:
    def __init__(self) -> None:
        self._request_index = 0

    def build(
        self,
        inputs: list[Any],
        group_size: int,
        *,
        metadata: dict[str, Any] | None = None,
        request_overrides: dict[str, Any] | None = None,
        runtime_debug: bool = False,
        policy_version: int | None = None,
    ) -> CollectorRequest:
        request_id = f"unit-request-{self._request_index}"
        self._request_index += 1
        request = GenerationRequest(
            request_id=request_id,
            family="unit",
            task="collect",
            inputs=list(inputs),
            samples_per_prompt=group_size,
            sampling=dict(request_overrides or {}),
            runtime_debug=runtime_debug,
            policy_version=policy_version,
        )
        return CollectorRequest(
            request=request,
            metadata={"collector": "metadata", **dict(metadata or {})},
        )


class _Runtime:
    def __init__(
        self,
        *,
        fail_offload: bool = False,
        shutdown_failures: int = 0,
    ) -> None:
        self.requests: list[GenerationRequest] = []
        self.events: list[str] = []
        self.fail_offload = fail_offload
        self.current_policy_version = 0
        self.requires_driver_model_offload = False
        self.shutdown_failures = shutdown_failures
        self.shutdown_calls = 0

    async def preflight(self) -> None:
        return None

    async def activate(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        self.events.append("generate")
        batch_size = len(request.prompts) * request.samples_per_prompt
        sample_rows = _sample_rows(request)
        output = torch.ones(batch_size, 3, 2, 2)
        trajectory = build_ar_discrete_trajectory(
            request=request,
            sample_rows=sample_rows,
            token_ids=torch.arange(batch_size * 2, dtype=torch.long).reshape(batch_size, 2),
            token_log_probs=torch.zeros(batch_size, 2),
            token_mask=torch.ones(batch_size, 2),
            prompt_input_ids=torch.ones(batch_size, 3, dtype=torch.long),
            prompt_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            uncond_input_ids=torch.zeros(batch_size, 3, dtype=torch.long),
            uncond_attention_mask=torch.ones(batch_size, 3, dtype=torch.long),
            context={"collector": "test"},
        )
        return GenerationOutput(
            output=output,
            trajectory=trajectory,
        )

    async def offload(self) -> None:
        self.events.append("offload")
        if self.fail_offload:
            raise RuntimeError("rollout offload failed")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("runtime shutdown failed")


class _RewardRuntime:
    def __init__(
        self,
        runtime: _Runtime | None = None,
        *,
        fail_park: bool = False,
        scoring_is_nonblocking: bool = False,
        external_accelerator_isolation_verified: bool = False,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.generation_runtime = runtime
        self.fail_park = fail_park
        self.scoring_is_nonblocking = scoring_is_nonblocking
        self.external_accelerator_isolation_verified = external_accelerator_isolation_verified
        self.shutdown_failures = 0
        self.shutdown_calls = 0
        self.memory_parked = False

    async def preflight(self) -> None:
        return None

    async def activate(self) -> None:
        self.activate_calls = getattr(self, "activate_calls", 0) + 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_calls <= self.shutdown_failures:
            raise RuntimeError("reward shutdown failed")

    async def score(
        self,
        samples: Sequence[RewardSample],
        *,
        require_memory_release: bool = False,
    ) -> RewardOutput:
        if self.generation_runtime is not None:
            self.generation_runtime.events.append("score")
        self.calls.append(
            {
                "outputs": [sample.output for sample in samples],
                "prompts": [sample.prompt for sample in samples],
                "metadata": [sample.metadata for sample in samples],
                "sample_ids": [sample.sample_id for sample in samples],
            },
        )
        self.memory_parked = False
        if require_memory_release:
            await self.park_memory(required=True)
        return RewardOutput(scores=tuple(float(index) for index in range(len(samples))))

    async def park_memory(
        self,
        *,
        required: bool,
    ) -> None:
        assert required is True
        if self.generation_runtime is not None:
            self.generation_runtime.events.append("reward_park")
        if self.fail_park:
            raise RuntimeError("reward park failed")
        self.memory_parked = True


def _collector(
    *,
    generation_runtime: _Runtime | None = None,
    reward_runtime: _RewardRuntime | None = None,
    lifecycle: RayLifecyclePlan | None = None,
) -> RolloutCollector:
    return RolloutCollector(
        config=RolloutCollectorConfig(),
        request_builder=_RequestBuilder(),
        reward_runtime=reward_runtime or _RewardRuntime(),
        generation_runtime=generation_runtime,
        lifecycle=lifecycle,
    )


def test_collector_requires_runtime_before_collect() -> None:
    import asyncio

    collector = _collector()

    with pytest.raises(RuntimeError, match="runtime is not initialized"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))


@pytest.mark.asyncio
async def test_collector_shutdown_retries_in_safe_runtime_then_reward_order() -> None:
    runtime = _Runtime(shutdown_failures=1)
    reward_runtime = _RewardRuntime()
    reward_runtime.shutdown_failures = 1
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
    )

    with pytest.raises(RuntimeError, match="runtime shutdown failed"):
        await collector.shutdown()
    assert runtime.shutdown_calls == 1
    assert reward_runtime.shutdown_calls == 0
    assert collector._generation_runtime is runtime
    assert collector._reward_shutdown_complete is False

    with pytest.raises(RuntimeError, match="reward shutdown failed"):
        await collector.shutdown()
    assert runtime.shutdown_calls == 2
    assert reward_runtime.shutdown_calls == 1
    assert collector._generation_runtime is None
    assert collector._reward_shutdown_complete is False

    await collector.shutdown()
    await collector.shutdown()

    assert runtime.shutdown_calls == 2
    assert reward_runtime.shutdown_calls == 2
    assert collector._generation_runtime is None
    assert collector._reward_shutdown_complete is True


def test_collector_routes_request_through_runtime_reward_and_trajectory_batch() -> None:
    """One collect call builds one request (prompts, group size as samples_per_prompt, overrides
    as sampling, policy version), scores every sample through the reward runtime with the
    collector metadata attached, and returns a batch whose rewards, group ids and sample rows
    line up prompt-major.
    """
    import asyncio

    runtime = _Runtime()
    reward_runtime = _RewardRuntime()
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
    )

    batch = asyncio.run(
        collect_scored(
            collector,
            ["p0", "p1"],
            group_size=2,
            request_overrides={"seed": 5},
            policy_version=7,
        ),
    )

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.prompts == ["p0", "p1"]
    assert request.samples_per_prompt == 2
    assert request.sampling == {"seed": 5}
    assert request.policy_version == 7
    assert all(
        metadata["collector"] == "metadata" for metadata in reward_runtime.calls[0]["metadata"]
    )
    assert reward_runtime.calls[0]["prompts"] == ["p0", "p0", "p1", "p1"]
    assert torch.stack(reward_runtime.calls[0]["outputs"]).shape == (4, 3, 2, 2)
    assert runtime.events == ["generate"]
    assert batch.rewards.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert batch.context == {"collector": "test"}
    assert batch.trajectory is not None
    assert batch.group_ids.tolist() == [0, 0, 1, 1]
    assert [row.prompt_index for row in batch.trajectory.sample_rows] == [0, 0, 1, 1]


def test_nextstep_noise_level_reaches_generation_request_from_rollout_owner() -> None:
    cfg = OmegaConf.create({"rollout": {"noise_level": 0.37}})
    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("nextstep_1"),
        config=RolloutCollectorConfig.from_root(parse_config(cfg)),
    )

    request = builder.build(["draw text"], group_size=1).request

    assert request.denoise is not None
    assert request.denoise.noise_level == pytest.approx(0.37)
    assert "noise_level" not in request.sampling


@pytest.mark.asyncio
async def test_profiled_collector_builds_cpu_batch_without_trainer_cuda_sync(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VRL_PROFILE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_trainer_sync(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("collector touched the trainer CUDA device")

    monkeypatch.setattr(torch.cuda, "synchronize", reject_trainer_sync)
    collector = _collector(
        generation_runtime=_Runtime(),
        reward_runtime=_RewardRuntime(),
    )

    batch = await collect_scored(collector, ["p0"], group_size=1)

    assert batch.rewards.device.type == "cpu"
    assert batch.group_ids.device.type == "cpu"


def test_collector_offloads_runtime_memory_before_reward_scoring() -> None:
    """Under a lifecycle plan that shares the reward GPU, scoring is bracketed by a rollout
    offload before the reward model activates and a reward park afterwards, and the phase-final
    offload repeats both.
    """
    import asyncio

    runtime = _Runtime()
    reward_runtime = _RewardRuntime(runtime)
    # Shared reward GPU: the lifecycle plan (not the runtime) tells the collector
    # to park rollout GPU memory before the in-process reward model scores.
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=True,
        rollout_and_reward_share_gpu=True,
        trainer_and_reward_share_gpu=True,
    )
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
        lifecycle=lifecycle,
    )

    asyncio.run(collect_scored(collector, ["p0"], group_size=1))
    asyncio.run(collector.offload_generation_runtime_memory())

    assert runtime.events == [
        "generate",
        "offload",
        "score",
        "reward_park",
        "offload",
        "reward_park",
    ]
    assert collector.requires_driver_model_offload_for_reward is True


def test_collector_does_not_offload_runtime_before_independent_reward() -> None:
    """Checks collector keeps rollout active for an independent reward."""
    import asyncio

    runtime = _Runtime()
    reward_runtime = _RewardRuntime(runtime)
    # Dedicated reward GPU: the plan keeps both roles resident, so the collector
    # never releases before reward.
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=False,
        rollout_and_reward_share_gpu=False,
        trainer_and_reward_share_gpu=False,
    )
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
        lifecycle=lifecycle,
    )

    asyncio.run(collect_scored(collector, ["p0"], group_size=1))

    assert runtime.events == ["generate", "score"]
    assert collector.requires_driver_model_offload_for_reward is False


@pytest.mark.parametrize(
    (
        "rollout_handoff",
        "trainer_handoff",
        "scorer_supports_overlap",
        "expected",
    ),
    [
        (False, False, True, True),
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_collector_derives_reward_generation_overlap_from_topology_and_scorer(
    rollout_handoff: bool,
    trainer_handoff: bool,
    scorer_supports_overlap: bool,
    expected: bool,
) -> None:
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=False,
        rollout_and_reward_share_gpu=rollout_handoff,
        trainer_and_reward_share_gpu=trainer_handoff,
    )
    collector = _collector(
        reward_runtime=_RewardRuntime(
            scoring_is_nonblocking=scorer_supports_overlap,
            external_accelerator_isolation_verified=scorer_supports_overlap,
        ),
        lifecycle=lifecycle,
    )

    assert collector.supports_reward_generation_overlap is expected
    assert collector.supports_continuous_reward_execution is expected


def test_collector_keeps_continuous_admission_for_no_reward() -> None:
    collector = _collector(
        reward_runtime=_RewardRuntime(
            scoring_is_nonblocking=False,
            external_accelerator_isolation_verified=True,
        ),
    )

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is True


def test_collector_separates_nonblocking_scoring_from_accelerator_isolation() -> None:
    runtime = _RewardRuntime()
    runtime.scoring_is_nonblocking = True
    runtime.external_accelerator_isolation_verified = False
    collector = _collector(reward_runtime=runtime)

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is False

    runtime.external_accelerator_isolation_verified = True

    assert collector.supports_reward_generation_overlap is True
    assert collector.supports_continuous_reward_execution is True


def test_dedicated_local_reward_allows_concurrent_collects_without_streaming() -> None:
    runtime = _RewardRuntime(external_accelerator_isolation_verified=True)
    collector = _collector(reward_runtime=runtime)

    assert collector.supports_reward_generation_overlap is False
    assert collector.supports_continuous_reward_execution is True


def test_collector_blocks_trainer_handoff_when_reward_parking_fails() -> None:
    """A failed reward park remains terminal even after rollout itself parks."""
    import asyncio

    class _FailingRewardRuntime(_RewardRuntime):
        async def score(
            self,
            samples: Sequence[RewardSample],
            *,
            require_memory_release: bool = False,
        ) -> RewardOutput:
            await super().score(
                samples,
                require_memory_release=False,
            )
            assert require_memory_release is True
            raise RuntimeError("reward park failed")

    runtime = _Runtime()
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=True,
        rollout_and_reward_share_gpu=True,
        trainer_and_reward_share_gpu=True,
    )
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=_FailingRewardRuntime(runtime, fail_park=True),
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))
    with pytest.raises(RuntimeError, match="reward park failed"):
        asyncio.run(collector.offload_generation_runtime_memory())

    assert runtime.events == [
        "generate",
        "offload",
        "score",
        "offload",
        "reward_park",
    ]


def test_collector_phase_final_gate_retries_reward_parking() -> None:
    """The final rollout offload retries a transient reward park failure."""
    import asyncio

    class _FlakyRewardRuntime(_RewardRuntime):
        def __init__(self, runtime: _Runtime) -> None:
            super().__init__(runtime)
            self.park_attempts = 0

        async def park_memory(
            self,
            *,
            required: bool,
        ) -> None:
            self.park_attempts += 1
            if self.generation_runtime is not None:
                self.generation_runtime.events.append("reward_park")
            if self.park_attempts == 1:
                raise RuntimeError("transient reward park failure")
            return await super().park_memory(required=required)

    runtime = _Runtime()
    reward_runtime = _FlakyRewardRuntime(runtime)
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=True,
        rollout_and_reward_share_gpu=True,
        trainer_and_reward_share_gpu=True,
    )
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="transient reward park failure"):
        asyncio.run(collect_scored(collector, ["p0"], group_size=1))

    asyncio.run(collector.offload_generation_runtime_memory())

    assert reward_runtime.park_attempts == 2
    assert reward_runtime.memory_parked is True


@pytest.mark.parametrize("reward_park_fails", [False, True])
def test_collector_attempts_reward_park_after_rollout_offload_failure(
    reward_park_fails: bool,
) -> None:
    """Rollout failure cannot skip reward cleanup; dual failures aggregate."""
    import asyncio

    runtime = _Runtime(fail_offload=True)
    reward_runtime = _RewardRuntime(runtime, fail_park=reward_park_fails)
    lifecycle = RayLifecyclePlan(
        trainer_and_rollout_share_gpu=True,
        rollout_and_reward_share_gpu=True,
        trainer_and_reward_share_gpu=True,
    )
    collector = _collector(
        generation_runtime=runtime,
        reward_runtime=reward_runtime,
        lifecycle=lifecycle,
    )
    collector._reward_phase_started = True

    if reward_park_fails:
        with pytest.raises(RewardCleanupError) as error:
            asyncio.run(collector.offload_generation_runtime_memory())
        assert len(error.value.errors) == 2
    else:
        with pytest.raises(RuntimeError, match="rollout offload failed"):
            asyncio.run(collector.offload_generation_runtime_memory())

    assert runtime.events == ["offload", "reward_park"]


def _reward_sample_rows(
    request_id: str,
    prompts: list[str],
) -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=index,
            sample_index=0,
            prompt=prompt,
            sample_id=f"{request_id}:sample:{index}",
        )
        for index, prompt in enumerate(prompts)
    ]


def _reward_sample_builder(
    request_id: str,
    prompts: list[str],
    *,
    outputs: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> TrajectoryRolloutBatchBuilder:
    import asyncio

    request = GenerationRequest(
        request_id=request_id,
        family="unit",
        task="collect",
        inputs=prompts,
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    output.trajectory.sample_rows = _reward_sample_rows(request_id, prompts)
    if outputs is not None:
        output.output = outputs
    return TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata=dict(metadata or {}), device="cpu"),
    )


def test_reward_samples_reject_sample_row_output_mismatch() -> None:
    builder = _reward_sample_builder(
        "request-0",
        ["p0"],
        outputs=torch.ones(2, 3),
    )

    with pytest.raises(ValueError, match="sample-row/output batch mismatch"):
        builder.reward_samples()


def test_reward_samples_preserve_prompt_identity_and_metadata() -> None:
    samples = _reward_sample_builder(
        "request-0",
        ["p0", "p1"],
        outputs=torch.ones(2, 3),
        metadata={"target_text": "caption"},
    ).reward_samples()

    assert [sample.prompt for sample in samples] == ["p0", "p1"]
    assert [sample.sample_id for sample in samples] == [
        "request-0:sample:0",
        "request-0:sample:1",
    ]
    assert all(sample.metadata["target_text"] == "caption" for sample in samples)
    assert [sample.metadata["reward_group_id"] for sample in samples] == [
        "request-0:prompt:0",
        "request-0:prompt:1",
    ]


def test_collector_uses_one_reward_call_and_splits_scores_per_group() -> None:
    """One reward call preserves every group's metadata and sample identity."""
    import asyncio

    reward_runtime = _RewardRuntime()
    collector = _collector(
        generation_runtime=_Runtime(),
        reward_runtime=reward_runtime,
    )

    async def _collect_groups():
        first = await collector.collect_unscored(
            ["same prompt"],
            group_size=2,
            metadata={"target_text": "group-0"},
            policy_version=4,
        )
        second = await collector.collect_unscored(
            ["same prompt"],
            group_size=3,
            metadata={"target_text": "group-1"},
            policy_version=5,
        )
        return await collector.score_rollouts([first, second])

    first, second = asyncio.run(_collect_groups())

    assert len(reward_runtime.calls) == 1
    call = reward_runtime.calls[0]
    assert call["prompts"] == ["same prompt"] * 5
    assert [metadata["target_text"] for metadata in call["metadata"]] == [
        "group-0",
        "group-0",
        "group-1",
        "group-1",
        "group-1",
    ]
    assert len(set(call["sample_ids"])) == 5
    reward_group_ids = [metadata["reward_group_id"] for metadata in call["metadata"]]
    assert len(set(reward_group_ids[:2])) == 1
    assert len(set(reward_group_ids[2:])) == 1
    assert reward_group_ids[0] != reward_group_ids[2]
    assert all(group_id.endswith(":prompt:0") for group_id in reward_group_ids)
    assert first.rewards.tolist() == [0.0, 1.0]
    assert second.rewards.tolist() == [2.0, 3.0, 4.0]


def test_collector_attaches_components_to_their_exact_rollout_groups() -> None:
    """Prefetched reward observations travel with the batch they describe."""
    import asyncio

    class _ComponentRewardRuntime(_RewardRuntime):
        async def score(
            self,
            samples: Sequence[RewardSample],
            *,
            require_memory_release: bool = False,
        ) -> RewardOutput:
            output = await super().score(
                samples,
                require_memory_release=require_memory_release,
            )
            return RewardOutput(
                scores=output.scores,
                components={
                    "observer": tuple(value + 10.0 for value in output.scores),
                },
            )

    collector = _collector(
        generation_runtime=_Runtime(),
        reward_runtime=_ComponentRewardRuntime(),
    )

    async def _collect_two_groups():
        pending = [
            await collector.collect_unscored(["p0"], group_size=2),
            await collector.collect_unscored(["p1"], group_size=3),
        ]
        return await collector.score_rollouts(pending)

    first, second = asyncio.run(_collect_two_groups())

    assert first.rewards.tolist() == [0.0, 1.0]
    assert first.extras["reward_components"]["observer"].tolist() == [10.0, 11.0]
    assert second.rewards.tolist() == [2.0, 3.0, 4.0]
    assert second.extras["reward_components"]["observer"].tolist() == [12.0, 13.0, 14.0]


def test_collect_prompt_groups_folds_reward_timing_into_stats() -> None:
    """Checks reward runtime timing reaches RolloutStats."""
    import asyncio

    class _TimedRewardRuntime(_RewardRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        async def score(
            self,
            samples: Sequence[RewardSample],
            *,
            require_memory_release: bool = False,
        ) -> RewardOutput:
            del require_memory_release
            self.batch_sizes.append(len(samples))
            return RewardOutput(
                scores=tuple(float(index + 1) for index in range(len(samples))),
                timing_ms={
                    "latency_ms": 12.0,
                    "queue_wait_ms": 3.0,
                    "inference_ms": 9.0,
                    "artifact_validation_ms": 2.0,
                },
            )

    reward_runtime = _TimedRewardRuntime()
    collector = RolloutCollector(
        config=RolloutCollectorConfig(),
        request_builder=_RequestBuilder(),
        reward_runtime=reward_runtime,
        generation_runtime=_Runtime(),
    )
    stats = RolloutStats()

    batches = asyncio.run(
        collect_prompt_groups(
            collector=collector,
            prompts=["p0"],
            group_size=2,
            runtime_debug=False,
            policy_version=None,
            stats=stats,
        ),
    )

    assert reward_runtime.batch_sizes == [2]
    assert len(batches) == 1
    assert batches[0].rewards.tolist() == [1.0, 2.0]
    assert stats.as_phase_dict()["reward.latency_s"] == 0.012
    assert stats.reward_queue_wait_ms == 3.0
    assert stats.reward_inference_ms == 9.0
    assert stats.reward_extra_ms == {"artifact_validation_ms": 2.0}
    assert stats.as_phase_dict()["reward.queue_wait_s"] == 0.003
    assert stats.as_phase_dict()["reward.artifact_validation_s"] == 0.002


def test_reward_view_selection_fails_fast_when_ambiguous() -> None:
    """Two reward views that both point at ``GenerationOutput.output`` cannot be disambiguated;
    the builder refuses instead of picking one.
    """
    import asyncio

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    assert output.trajectory is not None
    output.trajectory.reward_views["alternate"] = RewardView(
        name="alternate",
        metadata={"output_ref": "GenerationOutput.output"},
    )

    with pytest.raises(RuntimeError, match="multiple reward views"):
        TrajectoryRolloutBatchBuilder(
            output,
            RolloutBatchBuildContext(metadata={}),
        ).reward_outputs()


def test_reward_output_is_independent_of_trajectory_storage_dtype() -> None:
    """Trainer replay compression must not change the artifact being scored."""
    import asyncio

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    canonical = torch.tensor(
        [[[[0.1234567, -0.2345678], [0.3456789, -0.4567891]]]],
        dtype=torch.float32,
    )
    output.output = canonical

    builder = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(
            metadata={},
            trajectory_storage_policy=TrajectoryStoragePolicy(dtype="float16"),
        ),
    )
    replay_log_probs = builder.trajectory.segments["image_tokens"].tensors["old_log_prob"].value

    assert replay_log_probs.dtype == torch.float16
    assert output.output is canonical
    assert torch.equal(
        builder.reward_outputs(),
        ((canonical + 1.0) * 0.5).clamp(0.0, 1.0),
    )


@pytest.mark.parametrize(
    "metadata",
    [{}, {"output_ref": "TrajectoryBatch.output"}],
)
def test_reward_output_rejects_missing_or_unsupported_output_ref(
    metadata: dict[str, str],
) -> None:
    import asyncio

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    assert output.trajectory is not None
    output.trajectory.reward_views["image"] = RewardView(
        name="image",
        value_range="tanh",
        metadata=metadata,
    )

    with pytest.raises(RuntimeError, match="no supported output_ref"):
        TrajectoryRolloutBatchBuilder(
            output,
            RolloutBatchBuildContext(metadata={}),
        ).reward_outputs()


def test_chunk_denoise_kl_reward_sums_chunk_and_transition_axes() -> None:
    """Latent Gaussian trajectories use denoise packing and per-sample KL."""

    request = GenerationRequest(
        request_id="batch-request",
        family="causvid",
        task="text_to_video",
        inputs=["p0"],
        samples_per_prompt=2,
    )
    sample_rows = _sample_rows(request)
    batch_size, chunk_count, transition_count = 2, 2, 3
    policy_shape = (batch_size, chunk_count, transition_count)
    trajectory = build_chunk_autoregressive_denoise_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.zeros(*policy_shape, 1),
        actions=torch.ones(*policy_shape, 1),
        old_log_prob=torch.zeros(policy_shape),
        mask=torch.ones(policy_shape),
        timesteps=torch.zeros(policy_shape),
        finalized_chunk_latents=torch.zeros(batch_size, chunk_count, 1),
        replay_tensors={},
        context={},
        kl=torch.stack(
            (
                torch.ones(chunk_count, transition_count),
                torch.full((chunk_count, transition_count), 2.0),
            )
        ),
    )
    output = GenerationOutput(
        output=torch.zeros(batch_size, 3, 2, 2),
        trajectory=trajectory,
    )

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}, kl_reward_coef=0.25),
    ).build(torch.tensor([10.0, 20.0]))

    resolver = TrajectoryResolver.from_batch(packed)
    assert resolver.role_value("denoise", "observation").shape[:3] == policy_shape
    assert resolver.role_value("denoise", "action").shape[:3] == policy_shape
    assert packed.rewards.tolist() == pytest.approx([8.5, 17.0])
    assert packed.trajectory is not None
    assert packed.trajectory.primary_segment == "denoise"


def test_nonlatent_gaussian_keeps_autoregressive_packing() -> None:
    """Gaussian token policies do not get mistaken for latent denoise policies."""

    import asyncio

    request = GenerationRequest(
        request_id="token-gaussian-request",
        family="unit",
        task="text_to_image",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    assert output.trajectory is not None
    output.trajectory.segments["image_tokens"].distribution = "gaussian"

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    ).build(torch.tensor([1.0]))

    resolver = TrajectoryResolver.from_batch(packed)
    assert resolver.tensor_value("image_tokens", "prompt_input_ids").shape == (1, 3)
    assert resolver.role_value("image_tokens", "action").shape == (1, 2)
    assert packed.trajectory is not None
    assert packed.trajectory.primary_segment == "image_tokens"


def test_collector_forwards_reference_metadata_to_request() -> None:
    """A ``GenerationInput.reference_image`` reaches both the request input (for the executor) and
    the collector metadata (for the reward side).
    """
    from vrl.models.families.registry import get_model_family_entry
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    collector_request = builder.build(
        [GenerationInput(prompt="prompt", reference_image="/tmp/reference.png")],
        1,
    )

    assert collector_request.request.inputs[0].reference_image == "/tmp/reference.png"
    assert collector_request.metadata["reference_image"] == "/tmp/reference.png"


def test_collector_forwards_target_metadata_to_request() -> None:
    """Checks collector forwards target artifact metadata to rewards."""
    from vrl.models.families.registry import get_model_family_entry
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder

    builder = GenerationRequestBuilder(
        entry=get_model_family_entry("cosmos-predict2"),
        config=RolloutCollectorConfig(request_sampling={"num_steps": 1}),
    )

    targets = {"target_image": "/tmp/target.png", "target_video": "/tmp/target.mp4"}
    collector_request = builder.build(
        [
            GenerationInput(
                prompt="prompt",
                reference_image="/tmp/reference.png",
            ),
        ],
        1,
        metadata=dict(targets),
    )

    assert collector_request.metadata["target_image"] == "/tmp/target.png"
    assert collector_request.metadata["target_video"] == "/tmp/target.mp4"


def _sample_rows(request: GenerationRequest) -> list[GenerationSampleRow]:
    rows: list[GenerationSampleRow] = []
    for prompt_index, prompt in enumerate(request.prompts):
        for sample_index in range(request.samples_per_prompt):
            sample_id = f"{request.request_id}:prompt:{prompt_index}:sample:{sample_index}"
            rows.append(
                GenerationSampleRow(
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    sample_id=sample_id,
                ),
            )
    return rows


def test_reward_outputs_reconstructs_uint8_wire_video_exactly() -> None:
    """Checks uint8 wire-packed video reconstructs to k/255 floats.

    The worker packs decoded video as uint8 before the wire (wire diet T1);
    reward_outputs must hand consumers [0, 1] floats that round-trip
    bit-exactly through every downstream to_uint8 quantization, keeping
    reward scores identical to the fp32-over-wire path.
    """
    import asyncio

    from vrl.utils.media import to_uint8

    request = GenerationRequest(
        request_id="unit-request",
        family="unit",
        task="collect",
        inputs=["p0"],
        samples_per_prompt=1,
    )
    output = asyncio.run(_Runtime().generate(request))
    packed = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    output.output = packed

    reconstructed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}),
    ).reward_outputs()

    assert reconstructed.dtype == torch.float32
    assert float(reconstructed.min()) >= 0.0
    assert float(reconstructed.max()) <= 1.0
    # The G3 guarantee: re-quantizing recovers every byte value exactly.
    assert torch.equal(to_uint8(reconstructed), packed)


def test_uint8_quantization_roundtrip_is_exact_for_all_byte_values() -> None:
    """Checks k/255 floats survive both downstream quantization formulas.

    Pins the mechanism behind reward-score equality: to_uint8 (reward
    models) and the *255-round mp4 path must both map k/255 back to k.
    """
    k = torch.arange(256, dtype=torch.float32)
    grid = k / 255.0

    from vrl.utils.media import to_uint8

    assert torch.equal(to_uint8(grid), k.to(torch.uint8))
    mp4_path = (grid * 255.0).round().clamp(0, 255).to(torch.uint8)
    assert torch.equal(mp4_path, k.to(torch.uint8))


def test_collect_phase_timings_are_per_call_not_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase timings live on each call's rollouts; no shared collector state."""
    import asyncio

    monkeypatch.setenv("VRL_PROFILE", "1")
    collector = _collector(
        generation_runtime=_Runtime(),
        reward_runtime=_RewardRuntime(),
    )

    async def _run() -> tuple[Any, Any]:
        first = await collector.collect_unscored(["p0"], group_size=1)
        second = await collector.collect_unscored(["p1"], group_size=1)
        await collector.score_rollouts([first, second])
        return first, second

    first, second = asyncio.run(_run())

    # Generation time is owned per collect_unscored call.
    assert "collect.engine_generate" in first.phases
    assert "collect.engine_generate" in second.phases
    # Call-level score/build timings land on the first group only, so summing
    # phases over groups never multiplies the same wall time.
    assert "collect.reward_score" in first.phases
    assert "collect.batch_build" in first.phases
    assert "collect.reward_score" not in second.phases
    assert "collect.batch_build" not in second.phases
