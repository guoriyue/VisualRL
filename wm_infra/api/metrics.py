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
    ["status"],  # "success", "error"
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

# ─── Engine-level metrics ───

BATCH_SIZE = Histogram(
    "wm_batch_size",
    "Number of rollouts in each engine step batch",
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

STEP_DURATION = Histogram(
    "wm_step_duration_seconds",
    "Duration of a single engine step (batch prediction)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

QUEUE_DEPTH = Gauge(
    "wm_queue_depth",
    "Number of jobs waiting in submission queue",
)

ACTIVE_ROLLOUTS = Gauge(
    "wm_active_rollouts",
    "Number of currently active rollouts in the engine",
)

VRAM_USED_BYTES = Gauge(
    "wm_vram_used_bytes",
    "GPU memory used by the state cache (bytes)",
)
