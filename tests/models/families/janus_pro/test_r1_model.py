"""Janus-Pro-R1 model/executor contract tests with fake weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.models.steps.token.fixtures import StubVQ, build_stub_janus_model
from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.generation.execution.planner import EnginePlan
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.models.families.janus_pro import JANUS_R1_SEGMENTS
from vrl.models.families.janus_pro.model import (
    JanusProModel,
)
from vrl.models.families.janus_pro.runtime import (
    JanusProR1BatchExecutor,
    JanusProR1GenerationBatchGatherer,
)
from vrl.models.interfaces import ReplayRequest, ReplayResult
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory import TrajectoryResolver, build_ar_multisegment_trajectory

HIDDEN = 16
TEXT_VOCAB = 128
IMAGE_VOCAB = 64


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        if text == "Yes":
            return [self.bos_token_id, 3]
        if text == "No":
            return [self.bos_token_id, 4]
        return [self.bos_token_id, *[(ord(ch) % (TEXT_VOCAB - 5)) + 5 for ch in text]]

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, padding, truncation
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for text in texts:
            ids = self.encode(text)[:max_length]
            pad_len = max_length - len(ids)
            rows.append([*ids, *([self.pad_token_id] * pad_len)])
            masks.append([*(1 for _ in ids), *(0 for _ in range(pad_len))])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class _Processor:
    tokenizer = _Tokenizer()


class _LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(TEXT_VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, TEXT_VOCAB)

    @property
    def model(self) -> _LM:
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: object = None,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, output_hidden_states
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            logits=self.lm_head(inputs_embeds),
            past_key_values=past_key_values,
        )


def _model() -> JanusProModel:
    return build_stub_janus_model(
        language_model=_LM(),
        hidden_size=HIDDEN,
        image_vocab_size=IMAGE_VOCAB,
        gen_vision_model=StubVQ(vocab_size=IMAGE_VOCAB, latent_channels=4),
        embed_ids="clamp",
        processor=_Processor(),
        device="cpu",
        image_token_num=4,
    )


def _sample_rows() -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=index,
            prompt="draw text",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]


def _segment_payload(
    name: str,
    token_ids: torch.Tensor,
    *,
    visual: bool,
) -> dict[str, torch.Tensor | str | bool]:
    batch = token_ids.shape[0]
    prompt_len = 3
    return {
        "name": name,
        "visual": visual,
        "token_ids": token_ids,
        "token_log_probs": torch.zeros_like(token_ids, dtype=torch.float32),
        "token_mask": torch.ones_like(token_ids, dtype=torch.float32),
        "prompt_embeds": torch.zeros(batch, prompt_len, HIDDEN),
        "attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch, prompt_len, dtype=torch.long),
    }


def _r1_rollout_batch() -> RolloutBatch:
    final_ids = torch.tensor([[3, 4, 5], [4, 5, 6]])
    selfcheck_ids = torch.tensor([[7, 8], [8, 9]])
    initial_ids = torch.tensor([[1, 2], [2, 3]])
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw text"],
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
            "initial_image": _segment_payload(
                "initial_image",
                initial_ids,
                visual=True,
            ),
            "selfcheck_text": _segment_payload(
                "selfcheck_text",
                selfcheck_ids,
                visual=False,
            ),
            "final_image": _segment_payload(
                "final_image",
                final_ids,
                visual=True,
            ),
        },
        primary_segment="final_image",
        context={},
    )
    return RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.tensor([0, 0]),
        trajectory=trajectory,
    )


def test_generate_with_refine_returns_three_segments_and_selects_final_image() -> None:
    """One selfcheck refine round samples twice (initial 10s, regenerated 20s) and returns all
    three JANUS_R1_SEGMENTS; the final image follows the per-row verdict, Yes keeping the
    initial candidate and No taking the regenerated one.
    """
    model = _model()
    sample_calls: list[int] = []

    def image_sampler(
        cond_inputs_embeds: torch.Tensor,
        uncond_inputs_embeds: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        *,
        guidance_scale: float | None = None,
        temperature: float | None = None,
        image_token_num: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del (
            cond_inputs_embeds,
            uncond_inputs_embeds,
            cond_attention_mask,
            uncond_attention_mask,
            guidance_scale,
            temperature,
        )
        value = 10 if not sample_calls else 20
        sample_calls.append(value)
        tokens = torch.full((2, int(image_token_num or 4)), value, dtype=torch.long)
        log_probs = torch.full(tokens.shape, -0.1 * len(sample_calls))
        return tokens, log_probs

    def decode_image_tokens(
        image_token_ids: torch.Tensor,
        *,
        image_size: int = 384,
    ) -> torch.Tensor:
        del image_size
        return image_token_ids[:, :1].float().view(-1, 1, 1, 1).expand(-1, 3, 2, 2)

    def sample_selfcheck_text(
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        yes_token_id: int,
        no_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del prompt_embeds, prompt_attention_mask, temperature, eos_token_id
        text = torch.tensor(
            [
                [yes_token_id, pad_token_id, pad_token_id],
                [no_token_id, pad_token_id, pad_token_id],
            ],
            dtype=torch.long,
        )[:, :max_new_tokens]
        log_probs = torch.full(text.shape, -0.25)
        mask = torch.zeros_like(log_probs)
        mask[:, 0] = 1.0
        return text, log_probs, mask, torch.tensor([True, False])

    model.decode_image_tokens = decode_image_tokens  # type: ignore[method-assign]
    model._sample_selfcheck_text = sample_selfcheck_text  # type: ignore[method-assign]

    prompt_ids = torch.tensor([[5, 6, 0], [7, 8, 0]], dtype=torch.long)
    prompt_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long)
    out = model.generate_with_refine(
        prompt_ids,
        prompt_mask,
        guidance_scale=5.0,
        temperature=0.9,
        image_token_num=4,
        max_reflect_len=3,
        refine_mode="selfcheck",
        image_sampler=image_sampler,
    )

    assert sample_calls == [10, 20]
    assert out["context"] == {
        "temperature": 0.9,
        "guidance_scale": 5.0,
        "refine_mode": "selfcheck",
    }
    assert set(out["segments"]) == set(JANUS_R1_SEGMENTS)
    assert out["segments"]["initial_image"]["token_ids"].shape == (2, 4)
    assert out["segments"]["selfcheck_text"]["token_ids"].shape == (2, 3)
    assert out["segments"]["final_image"]["token_ids"].shape == (2, 4)
    assert torch.equal(
        out["segments"]["final_image"]["token_ids"][0],
        torch.full((4,), 10),
    )
    assert torch.equal(
        out["segments"]["final_image"]["token_ids"][1],
        torch.full((4,), 20),
    )
    assert torch.equal(out["final_image"][0], torch.full((3, 2, 2), 10.0))
    assert torch.equal(out["final_image"][1], torch.full((3, 2, 2), 20.0))


def test_r1_model_replay_forward_returns_requested_replay_segments() -> None:
    """``ReplayRequest.segment_names`` selects exactly the requested segments: the text segment
    carries materialized logits, the image segment the fused vocab-head payload, and its token
    ids equal the trajectory's recorded actions.
    """
    model = _model()
    batch = _r1_rollout_batch()

    result = model.replay_forward(
        batch,
        request=ReplayRequest(segment_names=("selfcheck_text", "final_image")),
    )

    assert isinstance(result, ReplayResult)
    assert set(result.segments) == set(JANUS_R1_SEGMENTS[1:])
    assert result.segments["selfcheck_text"].values["logits"].shape == (2, 2, TEXT_VOCAB)
    # Image segments use the fused vocab-head payload (no materialized logits).
    assert result.segments["final_image"].values["head_weight"].shape[0] == IMAGE_VOCAB
    assert result.segments["final_image"].values["head_hidden"].shape[:2] == (2, 3)
    actions = TrajectoryResolver.from_batch(batch).role_value("final_image", "action")
    assert torch.equal(
        result.segments["final_image"].values["token_ids"],
        actions,
    )


class _ExecutorModel:
    processor = _Processor()
    device = torch.device("cpu")
    language_model = _LM()

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            guidance_scale=6.25,
            temperature=0.45,
        )
        self.sampling_calls: list[tuple[float, float]] = []

    def generate_with_refine(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        guidance_scale: float,
        temperature: float,
        image_token_num: int,
        max_reflect_len: int,
        uncond_input_ids: torch.Tensor,
        uncond_attention_mask: torch.Tensor,
        image_size: int,
        refine_mode: str,
    ) -> dict[str, object]:
        self.sampling_calls.append((guidance_scale, temperature))
        del (
            guidance_scale,
            temperature,
            max_reflect_len,
            uncond_input_ids,
            uncond_attention_mask,
            image_size,
            refine_mode,
        )
        batch = prompt_input_ids.shape[0]
        token_mask = torch.ones(batch, image_token_num)
        prompt_embeds = torch.zeros(batch, prompt_input_ids.shape[1], HIDDEN)
        image = torch.ones(batch, 3, 2, 2)
        segments = {
            name: {
                "name": name,
                "token_ids": torch.full(
                    (batch, image_token_num),
                    idx,
                    dtype=torch.long,
                ),
                "token_log_probs": torch.zeros(batch, image_token_num),
                "token_mask": token_mask,
                "prompt_embeds": prompt_embeds,
                "attention_mask": prompt_attention_mask,
                "prompt_attention_mask": prompt_attention_mask,
                "visual": name != "selfcheck_text",
                "cfg": name != "selfcheck_text",
            }
            for idx, name in enumerate(JANUS_R1_SEGMENTS, start=1)
        }
        return {
            "initial_image": image * 1,
            "final_image": image * 2,
            "selfcheck": torch.zeros(batch, dtype=torch.bool),
            "segments": segments,
            "context": {"source": "fake"},
        }


@pytest.mark.parametrize(
    ("sampling_overrides", "expected"),
    [
        ({}, (6.25, 0.45)),
        (
            {
                "guidance_scale": 4.0,
                "temperature": 0.8,
            },
            (4.0, 0.8),
        ),
    ],
)
def test_r1_executor_uses_request_overrides_then_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
    sampling_overrides: dict[str, float],
    expected: tuple[float, float],
) -> None:
    model = _ExecutorModel()
    executor = JanusProR1BatchExecutor(model)
    monkeypatch.setattr(
        executor,
        "_r1_image_sampler",
        lambda *, request, scheduler_batch_size: object(),
    )
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw text"],
        samples_per_prompt=1,
        sampling={
            "image_token_num": 4,
            "image_size": 32,
            "max_text_length": 8,
            "max_reflect_len": 3,
            "final_image_policy": "always_generate",
            **sampling_overrides,
        },
    )

    executor.forward_batch(
        request,
        GenerationSampleBatch(
            prompt_index=0,
            sample_start=0,
            sample_count=1,
        ),
    )

    assert model.sampling_calls == [expected]


def test_r1_executor_forward_emits_canonical_family_and_segment_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The R1 executor emits the canonical trajectory: no decoded segment, an image reward view
    that references ``GenerationOutput.output`` rather than tensor refs, every R1 segment
    present, and the scheduler batch size threaded to the image sampler.
    """
    executor = JanusProR1BatchExecutor(
        _ExecutorModel(),
        gatherer=JanusProR1GenerationBatchGatherer(),
    )
    scheduler_batch_sizes: list[int | None] = []
    monkeypatch.setattr(
        executor,
        "_r1_image_sampler",
        lambda *, request, scheduler_batch_size: (
            scheduler_batch_sizes.append(scheduler_batch_size) or object()
        ),
    )
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw text"],
        samples_per_prompt=2,
        sampling={
            "image_token_num": 4,
            "image_size": 32,
            "max_text_length": 8,
            "max_reflect_len": 3,
            "final_image_policy": "always_generate",
            "ar_scheduler_batch_size": 1,
        },
    )
    specs = request.sample_rows()

    out = executor.forward_plan(request, specs, EnginePlan.from_request(request))

    assert out.output.shape == (2, 3, 2, 2)
    assert scheduler_batch_sizes == [1]
    assert out.runtime_debug is None
    assert out.trajectory is not None
    assert "decoded" not in out.trajectory.segments
    assert out.trajectory.reward_views["image"].tensor_refs == ()
    assert out.trajectory.reward_views["image"].metadata == {
        "output_ref": "GenerationOutput.output"
    }
    assert set(out.trajectory.segments) >= set(JANUS_R1_SEGMENTS)
    assert out.trajectory.segments["final_image"].tensors["token_ids"].value.shape == (2, 4)
    assert out.trajectory.segments["selfcheck_text"].tensors["token_ids"].value.shape == (2, 4)
