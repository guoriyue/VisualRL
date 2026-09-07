"""Tests for the continuous rollout staleness policy."""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy


def test_staleness_difference_and_none() -> None:
    """``staleness`` is ``current - produced`` and ``None`` whenever either version is unknown."""
    policy = StalenessPolicy(max_stale_policy_versions=1)
    assert policy.staleness(3, 5) == 2
    assert policy.staleness(None, 5) is None
    assert policy.staleness(3, None) is None


def test_too_stale_and_future() -> None:
    # Mechanism-only boundary: production continuous config requires >= 1.
    policy = StalenessPolicy(max_stale_policy_versions=0)
    assert policy.too_stale(4, 5) is True
    assert policy.too_stale(5, 5) is False
    assert policy.is_future(6, 5) is True
    assert policy.is_future(5, 5) is False


def test_negative_bound_rejected() -> None:
    with pytest.raises(ValueError, match="max_stale_policy_versions"):
        StalenessPolicy(max_stale_policy_versions=-1)
