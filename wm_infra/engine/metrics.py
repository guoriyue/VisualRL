"""Engine-level Prometheus metrics.

These metric definitions live in the engine package so that the engine
layer has no reverse dependency on the api layer.  The api layer
re-exports them from ``wm_infra.api.metrics`` for backward compatibility.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ─── Batching / step metrics ───

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

EXECUTION_CHUNK_SIZE = Histogram(
    "wm_execution_chunk_size",
    "Number of homogeneous rollout entities in each execution chunk",
    ["stage", "mode"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)

EXECUTION_CHUNK_TOTAL = Counter(
    "wm_execution_chunk_total",
    "Total homogeneous execution chunks processed by the runtime",
    ["stage", "mode"],
)

BATCH_FILL_RATIO = Histogram(
    "wm_batch_fill_ratio",
    "Chunk size divided by logical batch size for each executed chunk",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0],
)

# ─── Engine resource gauges ───

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

# ─── Low-level serving metrics ───

SERVING_COMPILED_PROFILE_EVENTS = Counter(
    "wm_serving_compiled_profile_events_total",
    "Compiled profile lifecycle events for low-level serving runtimes",
    ["backend", "event"],
)

SERVING_TRANSFER_BYTES = Histogram(
    "wm_serving_transfer_bytes",
    "Bytes moved across low-level serving transfer boundaries",
    ["backend", "kind"],
    buckets=[0, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456],
)
