"""Tests for trainer replay model contracts."""

from __future__ import annotations

import pytest

from tests.models.interfaces import registered_replay_model_classes
from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayRequestContract,
    ReplayResult,
    ReplaySegmentResult,
)

# ReplayModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))


def test_replay_result_requires_non_empty_segments() -> None:
    with pytest.raises(ValueError, match="segments must be non-empty"):
        ReplayResult(segments={})


def test_replay_result_requires_matching_segment_key() -> None:
    with pytest.raises(ValueError, match="must match"):
        ReplayResult(
            segments={
                "wrong": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )


def test_replay_request_requires_non_empty_segment_names() -> None:
    with pytest.raises(ValueError, match="segment_names"):
        ReplayRequest(segment_names=("",))


class _DenoiseOnlyContract(ReplayRequestContract):
    replay_segments = ("denoise",)
    replay_indexes_timesteps = False


def test_replay_segment_guard_rejects_unsupported_selection() -> None:
    with pytest.raises(ValueError, match="supports segments"):
        _DenoiseOnlyContract().reject_unsupported_replay_segments(
            ReplayRequest(segment_names=("unsupported",)),
        )


def test_replay_timestep_guard_rejects_nonzero_index() -> None:
    contract = _DenoiseOnlyContract()
    contract.reject_replay_timestep_selection(0)
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        contract.reject_replay_timestep_selection(1)


@pytest.mark.parametrize(
    "family",
    sorted(registered_replay_model_classes()),
)
def test_registered_family_replay_model_satisfies_contract(family: str) -> None:
    """Every registered family's replay-model class satisfies ReplayModel.

    Runs over the family registry (not a hand-written list) so a newly
    registered family cannot silently skip the contract. The check is
    class-level — ``callable(getattr(cls, m))`` like ``_missing_callables`` —
    because instantiating a real family model needs weights/GPU.
    """
    replay_cls = registered_replay_model_classes()[family]
    missing = [m for m in _REPLAY_MODEL_METHODS if not callable(getattr(replay_cls, m, None))]
    assert not missing, f"{family}: {replay_cls.__name__} missing ReplayModel methods {missing}"
