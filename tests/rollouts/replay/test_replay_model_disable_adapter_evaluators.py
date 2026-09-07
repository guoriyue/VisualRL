"""Evaluator tests for the trainer-facing ReplayModel reference context."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import ReplayResult, ReplaySegmentResult
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.token.continuous_token_logprob import (
    ContinuousTokenLogProbEvaluator,
)
from vrl.rollouts.evaluators.token.token_logprob import TokenLogProbEvaluator
from vrl.rollouts.evaluators.types import SignalRequest
from vrl.trajectory import (
    TrajectoryResolver,
    build_ar_continuous_trajectory,
    build_ar_discrete_trajectory,
)

_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="req",
        family="janus_pro",
        task="ar_t2i",
        inputs=["draw text"],
        samples_per_prompt=2,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    request = _request()
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt=request.prompts[0],
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]


def _discrete_batch(context: dict | None = None) -> RolloutBatch:
    token_ids = torch.tensor([[1, 2], [2, 3]])
    trajectory = build_ar_discrete_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        token_ids=token_ids,
        token_log_probs=torch.zeros_like(token_ids, dtype=torch.float32),
        token_mask=torch.ones_like(token_ids, dtype=torch.float32),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "janus_pro", **(context or {})},
    )
    return RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )


def _continuous_batch() -> RolloutBatch:
    tokens = torch.ones(2, 2, 3)
    trajectory = build_ar_continuous_trajectory(
        request=_request(),
        sample_rows=_sample_rows(),
        tokens=tokens,
        saved_noise=torch.zeros_like(tokens),
        token_log_probs=torch.zeros(2, 2),
        token_mask=torch.ones(2, 2),
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        uncond_input_ids=torch.zeros(2, 3, dtype=torch.long),
        uncond_attention_mask=torch.ones(2, 3, dtype=torch.long),
        context={"model_family": "nextstep_1"},
    )
    return RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )


class _DiscreteReplayModel:
    precision = _PRECISION

    def __init__(self) -> None:
        self.disable_calls = 0
        self._disabled = False

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        self.disable_calls += 1
        self._disabled = True
        try:
            yield
        finally:
            self._disabled = False

    def replay_forward(self, batch: RolloutBatch, timestep_idx: int = 0, **_) -> ReplayResult:
        del timestep_idx
        actions = TrajectoryResolver.from_batch(batch).role_value("image_tokens", "action")
        logits = torch.zeros(actions.shape[0], actions.shape[1], 8)
        boost = 1.0 if self._disabled else 4.0
        logits.scatter_(-1, actions.unsqueeze(-1), boost)
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"logits": logits},
                ),
            },
        )

    def load_trainable_state(self, state_dict):
        del state_dict


class _ContinuousReplayModel:
    precision = _PRECISION

    def __init__(self) -> None:
        self.disable_calls = 0
        self._disabled = False

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        self.disable_calls += 1
        self._disabled = True
        try:
            yield
        finally:
            self._disabled = False

    def replay_forward(self, batch: RolloutBatch, timestep_idx: int = 0, **_) -> ReplayResult:
        del timestep_idx
        value = 0.5 if self._disabled else 2.0
        actions = TrajectoryResolver.from_batch(batch).role_value("image_tokens", "action")
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"log_probs": torch.full(actions.shape[:2], value)},
                ),
            },
        )

    def load_trainable_state(self, state_dict):
        del state_dict


def test_token_logprob_evaluator_applies_rollout_temperature() -> None:
    """Checks replay renormalizes with the recorded sampling temperature."""
    batch = _discrete_batch(context={"temperature": 0.5})
    model = _DiscreteReplayModel()

    signals = TokenLogProbEvaluator().evaluate(
        model,
        batch,
    )

    actions = TrajectoryResolver.from_batch(batch).role_value("image_tokens", "action")
    logits = torch.zeros(2, 2, 8)
    logits.scatter_(-1, actions.unsqueeze(-1), 4.0)
    expected = (
        torch.nn.functional.log_softmax(logits / 0.5, dim=-1)
        .gather(
            -1,
            actions.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    assert torch.allclose(signals.primary.log_prob, expected, atol=1e-6)


def test_token_logprob_evaluator_uses_replay_model_disable_adapter() -> None:
    """The token evaluator computes ``ref_log_prob`` inside the replay model's ``disable_adapter``
    (exactly one call), and that adapter-off reference differs from the policy log-prob.
    """
    batch = _discrete_batch()
    model = _DiscreteReplayModel()

    signals = TokenLogProbEvaluator().evaluate(
        model,
        batch,
        signal_request=SignalRequest(need_ref=True),
    )

    assert model.disable_calls == 1
    assert signals.primary.ref_log_prob is not None
    assert not torch.equal(signals.primary.log_prob, signals.primary.ref_log_prob)


def test_continuous_logprob_evaluator_uses_replay_model_disable_adapter() -> None:
    """Same contract for the continuous-token evaluator: one ``disable_adapter`` call yields a
    reference log-prob distinct from the policy's.
    """
    batch = _continuous_batch()
    model = _ContinuousReplayModel()

    signals = ContinuousTokenLogProbEvaluator().evaluate(
        model,
        batch,
        signal_request=SignalRequest(need_ref=True),
    )

    assert model.disable_calls == 1
    assert signals.primary.ref_log_prob is not None
    assert not torch.equal(signals.primary.log_prob, signals.primary.ref_log_prob)
