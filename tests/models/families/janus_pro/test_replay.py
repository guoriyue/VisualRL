"""Core Janus-Pro replay ownership tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tests.models.steps.token.fixtures import build_stub_janus_model
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProModel,
)
from vrl.models.interfaces import ReplayResult
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import TrajectoryResolver, build_ar_discrete_trajectory

HIDDEN = 32
TEXT_VOCAB = 64


# ---------------------------------------------------------------------------
# Stubs — mirror tests/models/test_janus_wrapper.py + tests/rollouts/...
# ---------------------------------------------------------------------------


class _StubLM(nn.Module):
    """Identity trunk: last_hidden_state == inputs_embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)

    @property
    def model(self) -> _StubLM:
        # Property — not attribute — so ``train()`` does not infinite-recurse.
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


def _build_stub_model(*, unfreeze_gen_head: bool = False) -> JanusProModel:
    return build_stub_janus_model(
        language_model=_StubLM(),
        hidden_size=HIDDEN,
        image_vocab_size=JANUS_IMAGE_VOCAB_SIZE,
        unfreeze_gen_head=unfreeze_gen_head,
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


def _discrete_batch() -> RolloutBatch:
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
        context={"model_family": "janus_pro"},
    )
    return RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )


def test_janus_model_replay_forward_returns_typed_replay_result() -> None:
    """Replay returns the fused vocab-head payload (hidden, weight, bias, token ids) instead of
    materialized logits, and its ``logprobs(actions)`` agrees with the eager
    ``forward_image_logits`` gather.
    """
    model = _build_stub_model()
    batch = _discrete_batch()

    result = model.replay_forward(batch)

    assert isinstance(result, ReplayResult)
    segment = result.segments["image_tokens"]
    assert segment.segment == "image_tokens"
    # Fused vocab-head payload: logits are never materialized during replay.
    assert set(segment.values) == {
        "head_hidden",
        "head_weight",
        "head_bias",
        "image_token_ids",
    }
    assert segment.values["head_weight"].shape[0] == JANUS_IMAGE_VOCAB_SIZE
    actions = TrajectoryResolver.from_batch(batch).role_value("image_tokens", "action")
    assert torch.equal(segment.values["image_token_ids"], actions)

    # The contract path must agree with the eager forward_image_logits gather.
    from vrl.math.token.logprob import gather_categorical_log_probs

    replay, _ = model._resolve_image_token_replay(batch, 0, None)
    logits = model.forward_image_logits(
        model.language_model.get_input_embeddings()(replay["prompt_input_ids"]),
        replay["prompt_attention_mask"],
        actions,
    )
    torch.testing.assert_close(
        segment.logprobs(actions),
        gather_categorical_log_probs(logits, actions),
        rtol=1e-5,
        atol=1e-5,
    )
