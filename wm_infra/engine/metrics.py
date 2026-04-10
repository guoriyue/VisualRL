"""Engine-level Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter

API_AUTH_FAILURES = Counter(
    "wm_api_auth_failures_total",
    "API authentication failures",
    ["endpoint"],
)
