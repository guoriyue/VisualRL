"""Tests for diffusion request layout helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionRequestLayout,
    GenericDiffusionBatchExecutor,
)
from vrl.generation.steps.denoise.config import DenoiseRequestOptions
from vrl.generation.types import GenerationRequest


def test_denoise_options_reject_oversized_sde_window() -> None:
    """A window wider than its range is refused when the options are built."""
    with pytest.raises(ValueError, match="window_size"):
        DenoiseRequestOptions(sde_window_size=10, sde_window_range=(0, 5))


def test_diffusion_layout_rejects_sde_window_past_the_schedule() -> None:
    """The range upper bound is checked against the request's num_steps."""
    request = _request(denoise=DenoiseRequestOptions(sde_window_range=(0, 30)))

    with pytest.raises(ValueError, match="num_steps"):
        _layout().parse_sampling_params(request)


@pytest.mark.parametrize("denoise_mode", ["native", "sde"])
def test_diffusion_layout_always_builds_sde_math_params(denoise_mode: str) -> None:
    """Checks both denoise modes carry the non-optional loop math contract."""
    request = _request(denoise=DenoiseRequestOptions(denoise_mode=denoise_mode))

    params = _layout().parse_sampling_params(request)

    assert params.denoise_mode == denoise_mode
    assert params.sde.sde_type == "flow_grpo"
    assert params.sde_window_size == 0
    assert params.sde_window_range == (0, 20)


def test_diffusion_layout_selects_request_owned_sde_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks request window policy resolves before entering the denoise loop."""
    layout = _layout()
    params = layout.parse_sampling_params(
        _request(denoise=DenoiseRequestOptions(sde_window_size=2, sde_window_range=(3, 8))),
    )
    monkeypatch.setattr(
        "vrl.generation.bindings.full_sequence_denoise.layout.random.randint",
        lambda lo, hi: hi,
    )

    assert layout.select_sde_window(params) == (6, 8)

    no_window = layout.parse_sampling_params(_request())
    assert layout.select_sde_window(no_window) is None


@pytest.mark.parametrize("window_range", [(2, 2), "bad"])
def test_denoise_options_reject_invalid_sde_window_range(window_range: object) -> None:
    """A malformed window range fails when the typed options are built."""
    with pytest.raises(ValueError, match="window_range"):
        DenoiseRequestOptions(sde_window_range=window_range)  # type: ignore[arg-type]


def test_denoise_options_reject_unknown_denoise_mode() -> None:
    with pytest.raises(ValueError, match="denoise_mode"):
        DenoiseRequestOptions(denoise_mode="custom")  # type: ignore[arg-type]


def test_diffusion_layout_repeat_batch_rejects_unexpected_batch_size() -> None:
    """``repeat_batch`` broadcasts a batch-1 tensor to the sample count, returns an already-sized
    tensor as the same object, and refuses any other batch size.
    """
    layout = _layout()

    repeated = layout.repeat_batch(torch.ones(1, 2), 3)
    assert repeated.shape == (3, 2)

    already_sized = torch.ones(3, 2)
    assert layout.repeat_batch(already_sized, 3) is already_sized

    with pytest.raises(ValueError, match="cannot repeat tensor batch=2"):
        layout.repeat_batch(torch.ones(2, 2), 3)


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_extra"),
    [
        (None, {}),
        (123, {"max_sequence_length": 123}),
    ],
)
def test_diffusion_executor_only_projects_real_text_length(
    max_sequence_length: int | None,
    expected_extra: dict[str, int],
) -> None:
    """An absent family/request value stays absent at both model boundaries."""

    model = SimpleNamespace()
    executor = GenericDiffusionBatchExecutor(
        model,
        family="test",
        task="t2i",
        max_sequence_length=max_sequence_length,
    )
    params = executor.parse_sampling_params(_request())
    assert params.max_sequence_length == max_sequence_length
    assert params.text_encode_kwargs() == {
        "guidance_scale": 4.5,
        **expected_extra,
    }


def test_sde_window_is_resolved_once_at_parse_time() -> None:
    """The RESOLVED window lives on the params, so every sample batch of a
    request — and therefore every sample of a prompt group — shares one window
    (Flash-GRPO's iso-temporal grouping)."""

    params = _layout().parse_sampling_params(
        _request(
            {"seed": 7},
            denoise=DenoiseRequestOptions(sde_window_size=1, sde_window_range=(0, 10)),
        ),
    )
    assert params.sde_window is not None
    lo, hi = params.sde_window
    assert hi - lo == 1
    assert 0 <= lo < 10

    no_window = _layout().parse_sampling_params(_request())
    assert no_window.sde_window is None


def test_seeded_sde_window_is_deterministic_per_request() -> None:
    """Same request seed -> same window on every parse (multi-rank engines and
    re-parses agree without relying on the worker RNG sync); the seed does
    actually steer the draw."""

    denoise = DenoiseRequestOptions(sde_window_size=1, sde_window_range=(0, 10))
    first = _layout().parse_sampling_params(_request({"seed": 1234}, denoise=denoise))
    second = _layout().parse_sampling_params(_request({"seed": 1234}, denoise=denoise))
    assert first.sde_window == second.sde_window

    windows = {
        _layout().parse_sampling_params(_request({"seed": seed}, denoise=denoise)).sde_window
        for seed in range(30)
    }
    assert len(windows) > 1, "30 distinct seeds all drew the same window"


def _layout() -> DiffusionRequestLayout:
    """A layout with explicit fallbacks (the executor is the real source)."""
    return DiffusionRequestLayout(
        default_num_frames=1,
        default_fps=None,
        default_max_sequence_length=512,
        sde_type="flow_grpo",
    )


def _request(
    extra_sampling: dict[str, object] | None = None,
    *,
    denoise: DenoiseRequestOptions | None = None,
) -> GenerationRequest:
    sampling = {
        "num_steps": 20,
        "guidance_scale": 4.5,
        "height": 64,
        "width": 64,
        **(extra_sampling or {}),
    }
    return GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        inputs=["p0"],
        samples_per_prompt=1,
        sampling=sampling,
        denoise=denoise,
    )
