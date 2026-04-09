"""Prometheus metrics for world model serving.

Exposes counters, histograms, and gauges for monitoring request
throughput, latency, batching efficiency, and resource usage.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ─── Request-level metrics ───

REQUEST_TOTAL = Counter(
    "wm_request_total",
    "Total rollout requests",
    ["status"],
)

REQUEST_DURATION = Histogram(
    "wm_request_duration_seconds",
    "End-to-end rollout request duration",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

API_AUTH_FAILURES = Counter(
    "wm_api_auth_failures_total",
    "API authentication failures",
    ["endpoint"],
)

SAMPLE_TOTAL = Counter(
    "wm_sample_total",
    "Total sample-production requests",
    ["backend", "status"],
)

SAMPLE_DURATION = Histogram(
    "wm_sample_duration_seconds",
    "End-to-end sample-production request duration",
    ["backend", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ─── Engine-level metrics (re-exported from wm_infra.engine.metrics) ───

from wm_infra.engine.metrics import (  # noqa: E402
    ACTIVE_ROLLOUTS,
    BATCH_FILL_RATIO,
    BATCH_SIZE,
    EXECUTION_CHUNK_SIZE,
    EXECUTION_CHUNK_TOTAL,
    QUEUE_DEPTH,
    SERVING_COMPILED_PROFILE_EVENTS,
    SERVING_TRANSFER_BYTES,
    STEP_DURATION,
    VRAM_USED_BYTES,
)

# ─── Low-level serving metrics (api-only) ───

SERVING_GRAPH_COMPILE_SECONDS = Histogram(
    "wm_serving_graph_compile_seconds",
    "Latency of graph/profile compilation or capture",
    ["backend", "stage"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

SERVING_STAGING_BYTES = Gauge(
    "wm_serving_staging_bytes",
    "Bytes reserved in staging buffers for low-level serving",
    ["backend"],
)
