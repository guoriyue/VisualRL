"""Tests for R1 multi-segment token log-prob replay."""

from __future__ import annotations

import contextlib

import torch

from vrl.config.precision import RolePrecision
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
)
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.token.multi_segment_token_logprob import (
    MultiSegmentTokenLogProbEvaluator,
)
from vrl.trajectory import build_ar_multisegment_trajectory

_PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


def _sample_rows() -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="draw a chart",
            sample_id="s0",
        ),
        GenerationSampleRow(
            prompt_index=0,
            sample_index=1,
            prompt="draw a chart",
            sample_id="s1",
        ),
    ]


def _trajectory_segment(
    name: str,
    token_ids: torch.Tensor,
    *,
    modality: str,
) -> dict[str, torch.Tensor | str | bool]:
    batch_size = token_ids.shape[0]
    return {
        "name": name,
        "visual": modality == "image",
        "token_ids": token_ids,
        "token_log_probs": torch.zeros_like(token_ids, dtype=torch.float32),
        "token_mask": torch.ones_like(token_ids, dtype=torch.float32),
        "prompt_embeds": torch.ones(batch_size, 3, 4),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
    }


def _trajectory_batch(context: dict | None = None) -> RolloutBatch:
    initial_ids = torch.tensor([[1, 2], [2, 3]])
    final_ids = torch.tensor([[3, 4, 5], [4, 5, 6]])
    selfcheck_ids = torch.tensor([[7, 8], [8, 9]])
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw a chart"],
        samples_per_prompt=2,
        train_segments={
            "initial_image": True,
            "selfcheck_text": True,
            "final_image": True,
        },
    )
    trajectory = build_ar_multisegment_trajectory(
        request=request,
        sample_rows=_sample_rows(),
        segments={
            "initial_image": _trajectory_segment(
                "initial_image",
                initial_ids,
                modality="image",
            ),
            "selfcheck_text": _trajectory_segment(
                "selfcheck_text",
                selfcheck_ids,
                modality="text",
            ),
            "final_image": _trajectory_segment(
                "final_image",
                final_ids,
                modality="image",
            ),
        },
        primary_segment="final_image",
        context=context or {},
    )
    return RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        extras={},
        trajectory=trajectory,
    )


class _SegmentReplayModel:
    precision = _PRECISION

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def replay_forward(
        self,
        batch,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx
        names = request.segment_names if request is not None and request.segment_names else ()
        segments = {}
        for name in names:
            trajectory_segment = self._segment_payload(name)
            values = self.replay_r1_segment(segment_name=name, segment=trajectory_segment)
            segments[name] = ReplaySegmentResult(segment=name, values=values)
        return ReplayResult(segments=segments)

    def disable_adapter(self):
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict):
        del state_dict

    def replay_r1_segment(self, *, segment_name, segment):
        modality = str(segment["modality"])
        self.calls.append((segment_name, modality))
        token_ids = segment["token_ids"]
        logits = torch.zeros(token_ids.shape[0], token_ids.shape[1], 20)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 3.0)
        return {"logits": logits, "token_ids": token_ids}

    @staticmethod
    def _segment_payload(name: str):
        token_ids = {
            "selfcheck_text": torch.tensor([[7, 8], [8, 9]]),
            "final_image": torch.tensor([[3, 4, 5], [4, 5, 6]]),
            "initial_image": torch.tensor([[1, 2], [2, 3]]),
        }[name]
        modality = "text" if name == "selfcheck_text" else "image"
        return {"token_ids": token_ids, "modality": modality}


def test_evaluator_can_replay_text_segment_without_using_image_path() -> None:
    """Enabling only the text segment replays only it (the image path is never touched) and makes
    it the primary segment.
    """
    batch = _trajectory_batch()
    assert batch.trajectory is not None
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text",),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == [("selfcheck_text", "text")]
    assert signals.primary_segment == "selfcheck_text"
    assert signals.segments["selfcheck_text"].log_prob.shape == (2, 2)


def test_evaluator_applies_rollout_temperature_to_all_segments() -> None:
    """Checks replay renormalizes every segment with the recorded temperature."""
    batch = _trajectory_batch(context={"temperature": 0.5})
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text", "final_image"),
    )

    signals = evaluator.evaluate(model, batch)

    token_ids = torch.tensor([[7, 8], [8, 9]])
    logits = torch.zeros(2, 2, 20)
    logits.scatter_(-1, token_ids.unsqueeze(-1), 3.0)
    expected = (
        torch.nn.functional.log_softmax(logits / 0.5, dim=-1)
        .gather(
            -1,
            token_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    assert torch.allclose(signals.segments["selfcheck_text"].log_prob, expected, atol=1e-6)
    assert signals.context == {"temperature": 0.5}


def test_evaluator_reads_r1_segments_from_canonical_trajectory_fields() -> None:
    """With both R1 segments enabled, replay reads them from the canonical trajectory segments in
    enabled order, the last enabled segment is primary, and no segment bookkeeping leaks into
    the signal context.
    """
    batch = _trajectory_batch()
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text", "final_image"),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == [("selfcheck_text", "text"), ("final_image", "image")]
    assert signals.primary_segment == "final_image"
    assert signals.primary.log_prob.shape == (2, 3)
    assert signals.segments["selfcheck_text"].log_prob.shape == (2, 2)
    assert signals.segments["final_image"].old_log_prob.shape == (2, 3)
    assert "primary_segment" not in signals.context
    assert "segment_order" not in signals.context
