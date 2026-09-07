"""Janus runner tests for the paged-attention AR hook."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.models.steps.token.fixtures import RecordingHead, build_stub_janus_model
from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.generation.types import GenerationRequest
from vrl.models.families.janus_pro.model import (
    JANUS_IMAGE_VOCAB_SIZE,
    JanusProModel,
)
from vrl.models.families.janus_pro.runner import JanusProARModelRunner
from vrl.models.families.janus_pro.runtime import JanusProBatchExecutor
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
)
from vrl.nn.modules.ar_attention_backends import build_vllm_attention_backend

# Default cache dtype resolved from the source signature, so a default change
# (e.g. "auto" -> a concrete dtype) auto-flows into this assertion.
_DEFAULT_CACHE_DTYPE = (
    inspect.signature(build_vllm_attention_backend).parameters["cache_dtype"].default
)

HIDDEN = 8
TEXT_VOCAB = 32

# Module level: every test here drives `_RecordingPagedBackend`, and a helper
# class cannot carry a function decorator.
pytestmark = pytest.mark.real_cover(
    "tests/generation/bindings/token_autoregressive/test_janus_vllm_paged_attention_backend.py"
    "::test_janus_vllm_paged_attention_matches_hf_llama_one_step",
    why=(
        "the recording backend returns fixed hidden states so the runner's prefill/step "
        "sequencing is assertable on CPU; a real backend needs CUDA plus vLLM's worker "
        "internals, and the gpu-lane test checks its output against HF Llama instead"
    ),
)


@dataclass(frozen=True, slots=True)
class _PagedState:
    tokens: int


class _RecordingPagedBackend(ARAttentionBackend):
    def __init__(self) -> None:
        super().__init__(ARAttentionConfig(family="janus_pro"))
        self.prefill_requests: list[ARAttentionPrefillInput] = []
        self.step_requests: list[ARAttentionStepInput] = []

    def prefill(
        self,
        request: ARAttentionPrefillInput,
    ) -> ARAttentionPrefillOutput:
        self.prefill_requests.append(request)
        batch = request.inputs_embeds.shape[0]
        hidden = torch.zeros(batch, HIDDEN, device=request.inputs_embeds.device)
        hidden[:, 0] = 10.0 if request.branch == "cond" else 11.0
        states = tuple(
            _PagedState(
                tokens=request.attention_mask.shape[1],
            )
            for _ in range(batch)
        )
        return ARAttentionPrefillOutput(
            last_hidden=hidden,
            sequence_states=states,
        )

    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        self.step_requests.append(request)
        batch = request.input_embeds.shape[0]
        hidden = torch.zeros(batch, HIDDEN, device=request.input_embeds.device)
        hidden[:, 0] = 12.0
        states = tuple(
            _PagedState(
                tokens=state.tokens + 1,
            )
            for state in request.sequence_states
        )
        return ARAttentionStepOutput(
            last_hidden=hidden,
            sequence_states=states,
        )


class _RecordingLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> _RecordingLM:
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            last_hidden_state=torch.zeros(1, 1, HIDDEN),
            past_key_values=None,
        )


def test_janus_runner_can_drive_one_paged_attention_image_step() -> None:
    """With a paged backend installed the HF language model is never called: prefill goes through
    the backend once per branch (cond, uncond) and one [4,1,H] step follows, yielding the same
    first- and second-token hidden rows as the KV-cache path.
    """
    torch.manual_seed(0)
    model = _model()
    backend = _RecordingPagedBackend()

    TokenAutoregressiveLoop(
        runner=JanusProARModelRunner(model, attention_backend=backend),
        scheduler_batch_size=2,
        init_args=_prompt_tensors(),
        init_kwargs={"image_token_num": 2},
    ).run()

    assert model.mmgpt.language_model.calls == []
    assert [request.branch for request in backend.prefill_requests] == ["cond", "uncond"]
    assert len(backend.step_requests) == 1
    step = backend.step_requests[0]
    assert step.input_embeds.shape == (4, 1, HIDDEN)

    first_logits_hidden = model.mmgpt.gen_head.inputs[0]
    assert torch.equal(
        first_logits_hidden[:, 0, 0],
        torch.tensor([10.0, 10.0, 11.0, 11.0]),
    )
    second_logits_hidden = model.mmgpt.gen_head.inputs[1]
    assert torch.equal(
        second_logits_hidden[:, 0, 0],
        torch.tensor([12.0, 12.0, 12.0, 12.0]),
    )


def test_janus_none_image_token_count_uses_model_default() -> None:
    model = _model()
    backend = _RecordingPagedBackend()

    init = JanusProARModelRunner(model, attention_backend=backend).init_token(
        *_prompt_tensors(),
        image_token_num=None,
    )

    assert init.step_count == model.config.image_token_num
    assert init.state.total_token_num == model.config.image_token_num
    assert [request.max_new_tokens for request in backend.prefill_requests] == [
        model.config.image_token_num,
        model.config.image_token_num,
    ]


def test_janus_zero_image_token_count_fails_before_prefill() -> None:
    backend = _RecordingPagedBackend()
    runner = JanusProARModelRunner(_model(), attention_backend=backend)

    with pytest.raises(ValueError, match="image_token_num must be >= 1"):
        runner.init_token(
            *_prompt_tensors(),
            image_token_num=0,
        )

    assert backend.prefill_requests == []


def test_janus_runtime_uses_vllm_paged_attention_by_default(monkeypatch) -> None:
    """Without an explicit backend, ``ar_paged_block_size`` in sampling selects the vLLM paged
    backend built for the model's family with that block size and the default cache dtype.
    """
    model = _model()
    backend = _RecordingPagedBackend()

    def build_backend(
        passed_model: JanusProModel,
        *,
        family: str,
        block_size: int,
        cache_dtype: str,
    ) -> _RecordingPagedBackend:
        assert passed_model is model
        assert family == "janus_pro"
        assert block_size == 32
        assert cache_dtype == _DEFAULT_CACHE_DTYPE
        return backend

    monkeypatch.setattr(
        "vrl.nn.modules.ar_attention_backends.build_vllm_attention_backend",
        build_backend,
    )
    request = _request()
    request.sampling.update(
        {
            "ar_paged_block_size": 32,
        }
    )

    runner = JanusProBatchExecutor(model)._ar_runner(request)

    assert isinstance(runner, JanusProARModelRunner)
    assert runner.attention_backend is backend


def _model() -> JanusProModel:
    return build_stub_janus_model(
        language_model=_RecordingLM(),
        hidden_size=HIDDEN,
        image_vocab_size=JANUS_IMAGE_VOCAB_SIZE,
        gen_head=RecordingHead(image_vocab_size=JANUS_IMAGE_VOCAB_SIZE),
        image_token_num=2,
    )


def _prompt_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cond = torch.zeros(2, 3, HIDDEN)
    uncond = torch.zeros(2, 3, HIDDEN)
    cond_mask = torch.ones(2, 3, dtype=torch.long)
    uncond_mask = torch.ones(2, 3, dtype=torch.long)
    return cond, uncond, cond_mask, uncond_mask


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="test-janus-paged",
        family="janus_pro",
        task="ar_t2i",
        inputs=["test prompt"],
        samples_per_prompt=2,
    )
