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

# ─── Genie stage runtime metrics ───

GENIE_STAGE_DURATION = Histogram(
    "wm_genie_stage_duration_seconds",
    "Duration of a Genie runtime stage",
    ["stage", "lane", "runner_mode"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

GENIE_CHUNK_SIZE = Histogram(
    "wm_genie_chunk_size",
    "Number of Genie execution entities in a chunk",
    ["stage", "lane"],
    buckets=[1, 2, 4, 8, 16, 32],
)

GENIE_CHUNK_FILL_RATIO = Histogram(
    "wm_genie_chunk_fill_ratio",
    "Chunk size divided by scheduler chunk capacity for Genie runtime",
    ["stage", "lane"],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0],
)

GENIE_STATE_MATERIALIZE_BYTES = Histogram(
    "wm_genie_state_materialize_bytes",
    "Bytes materialized into Genie runtime state",
    buckets=[0, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216],
)

GENIE_STATE_MATERIALIZE_SECONDS = Histogram(
    "wm_genie_state_materialize_seconds",
    "Latency of Genie state materialization",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

GENIE_TRANSITION_FRAMES_TOTAL = Counter(
    "wm_genie_transition_frames_total",
    "Total number of Genie frames advanced through transition windows",
    ["runner_mode"],
)

GENIE_TRANSITION_TOKENS_TOTAL = Counter(
    "wm_genie_transition_tokens_total",
    "Total number of Genie tokens advanced through transition windows",
    ["runner_mode"],
)

GENIE_CHECKPOINT_DELTA_BYTES = Histogram(
    "wm_genie_checkpoint_delta_bytes",
    "Size in bytes of Genie checkpoint deltas",
    buckets=[0, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
)

GENIE_CHECKPOINT_BUILD_SECONDS = Histogram(
    "wm_genie_checkpoint_build_seconds",
    "Latency of Genie checkpoint delta construction",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

GENIE_PERSIST_BACKLOG = Gauge(
    "wm_genie_persist_backlog",
    "Outstanding Genie persistence tasks waiting to be written",
)

GENIE_PROMPT_REUSE_EVENTS = Counter(
    "wm_genie_prompt_reuse_events_total",
    "Prompt reuse events in Genie runtime",
    ["outcome"],
)

GENIE_RESIDENCY_EVENTS = Counter(
    "wm_genie_residency_events_total",
    "Residency tier transitions in Genie runtime",
    ["tier"],
)

GENIE_GPU_OCCUPANCY_ESTIMATE = Gauge(
    "wm_genie_gpu_occupancy_estimate",
    "Best-effort Genie runtime occupancy estimate for the selected chunk",
)

# Backward-compatible aliases for older Genie metric call sites.
GENIE_PROMPT_REUSE_HITS_TOTAL = GENIE_PROMPT_REUSE_EVENTS.labels(outcome="hit")
GENIE_PROMPT_REUSE_MISSES_TOTAL = GENIE_PROMPT_REUSE_EVENTS.labels(outcome="miss")
GENIE_RESIDENCY_EVENTS_TOTAL = GENIE_RESIDENCY_EVENTS
