# Genie Stage Runtime

This document defines a stage-oriented execution runtime for `genie-rollout`.

Implementation status as of the current repo state:

- landed:
  - explicit stage graph in `wm_infra/backends/genie.py`
  - execution-plane objects in `wm_infra/backends/genie_runtime.py`
  - stage-local scheduler in `wm_infra/backends/genie_scheduler.py`
  - checkpoint delta helpers in `wm_infra/backends/genie_checkpoint.py`
  - cross-request transition batching in `wm_infra/backends/genie_batcher.py`
- not landed:
  - a global multi-stage runtime that batches every stage across requests
  - backend-independent many-world scheduler shared by Genie and other runtimes

The goal is not to turn `wm-infra` into a general-purpose game engine.
The goal is to make `genie-rollout` behave like a many-world temporal runtime:

- keep `episode / branch / rollout / checkpoint / state_handle` as durable control-plane objects
- split Genie execution into explicit stage-local work units
- batch by homogeneous frame-window work instead of request-shaped calls
- make state locality, checkpoint cadence, and artifact persistence visible to the scheduler

This document is a backend-specific refinement of the broader execution direction in:

- `docs/ECS_EXECUTION_RUNTIME.md`
- `docs/GENIE_ECS_BASELINE.md`
- `docs/GENIE_ECS_BATCHING_POLICY.md`

## Why this redesign exists

`genie-rollout` now has an explicit execution-plane skeleton, but it is still an intermediate system rather than the final target runtime.

The backend already has:

- explicit temporal identity (`episode`, `branch`, `rollout`, `checkpoint`, `state_handle`)
- first-class Genie knobs (`num_frames`, `num_prompt_frames`, `maskgit_steps`, `temperature`)
- persisted artifacts (`tokens`, `state`, `request`, `runtime`, `checkpoint`, `recovery`)

The old blocking shape looked like this:

```python
run_result = self._runner.run(
    output_dir=sample_dir,
    prompt=request.sample_spec.prompt or "",
    seed=seed,
    num_frames=num_frames,
    input_tokens=input_tokens,
    num_prompt_frames=genie_config.num_prompt_frames,
    maskgit_steps=genie_config.maskgit_steps,
    temperature=genie_config.temperature,
)
```

That shape makes several important things invisible:

- which work is runnable right now
- which requests are batch-compatible
- whether prompt state is already hot
- whether a request is continuation-heavy or cold-start-heavy
- which checkpoint and persistence costs belong on the hot path

That is no longer the hot path in the current implementation.
The backend now uses:

- `prepare_inputs(...)`
- `build_transition_entities(...)`
- `GenieScheduler.schedule(...)`
- `GenieTransitionBatcher.run_transition(...)`
- `persist_outputs(...)`

What is still missing is broader many-request batching outside the transition stage.

## Design principles

1. Keep the control plane durable and explicit.
2. Move Genie execution from whole-request calls to stage-local window execution.
3. Batch by shared execution signature, not by raw queue arrival.
4. Prefer continuation locality when prompt/state materialization is expensive.
5. Keep checkpoint policy explicit and scheduler-visible.
6. Separate hot GPU transition work from cold materialization and artifact persistence.
7. Preserve backend-specific behavior instead of hiding Genie behind vague generic abstractions.

## Non-goals

- replacing the current control plane with runtime-only entities
- forcing `wan-video` and `genie-rollout` into a fake shared execution contract
- redesigning the public northbound sample API
- inventing distributed execution before single-node profiling is strong
- rewriting the Genie model internals before the runtime boundary is fixed

## Current backend shape

The current `genie-rollout` path roughly does this:

1. validate temporal references
2. load the runner
3. resolve input tokens
4. validate Genie config against model limits
5. create rollout metadata
6. prepare prompt state with `GenieRunner.prepare_inputs(...)`
7. split runnable work into transition entities
8. schedule per-window chunks
9. batch compatible transition windows across concurrent requests
10. persist logs, tokens, checkpoint files, and state handles

This gives correct artifacts, lineage, and stage-local scheduler visibility.
The main remaining gap is that only the transition stage is cross-request batched today.

## Target layering

The target layering for Genie should be:

```text
northbound request
  -> admission
  -> state materialize
  -> prompt prepare
  -> transition window execution
  -> checkpoint delta build
  -> artifact persist
  -> control-plane commit
```

Each of these is a stage.
Each stage produces or updates transient runtime objects.
Only selected stage boundaries commit durable control-plane state.

## Runtime object model

### Durable control-plane objects

These remain the source of truth:

- `episode`
- `branch`
- `rollout`
- `checkpoint`
- `state_handle`
- `sample manifest`

They answer:

- what temporal state exists
- what lineage produced it
- what artifacts depend on it
- what the durable checkpoint and recovery surface is

### Transient execution-plane objects

These are internal runtime objects and may be created or destroyed frequently:

- `GenieExecutionEntity`
- `GenieBatchSignature`
- `GenieRuntimeState`
- `GenieExecutionChunk`
- `GenieCheckpointDelta`
- `GeniePersistTask`

They answer:

- what work is runnable right now
- what work can batch together
- where prompt and generated token state currently lives
- which checkpoint and persistence work can be deferred

## Core execution objects

### `GenieExecutionEntity`

This is the smallest schedulable Genie work item.
It should usually represent a bounded frame window, not a whole request.

Suggested shape:

```python
@dataclass(slots=True)
class GenieExecutionEntity:
    entity_id: str
    rollout_id: str
    episode_id: str
    branch_id: str | None
    sample_id: str
    input_state_handle_id: str | None
    current_stage: str
    next_stage: str | None
    window_start_frame: int
    window_num_frames: int
    total_frames: int
    num_prompt_frames: int
    checkpoint_every_n_frames: int
    priority: float
    deadline_s: float | None
    batch_signature: "GenieBatchSignature"
```

Recommended entity kinds:

- `materialize_window`
- `prompt_prepare_window`
- `transition_window`
- `checkpoint_window`
- `persist_window`

### `GenieBatchSignature`

This is the execution-plane equivalent of a Genie archetype key.
Entities with different signatures should not share a chunk.

Suggested shape:

```python
@dataclass(frozen=True, slots=True)
class GenieBatchSignature:
    backend: str
    model_name: str
    stage: str
    device: str
    dtype: str
    tokenizer_kind: str
    spatial_h: int
    spatial_w: int
    window_num_frames: int
    num_prompt_frames: int
    maskgit_steps: int
    temperature_bucket: str
    checkpoint_every_n_frames: int
    runner_mode: str
    needs_persist: bool
```

This is the key idea:

Genie work should batch because execution shape matches, not because several requests happen to be queued.

### `GenieRuntimeState`

This stores hot execution state without changing durable control-plane semantics.

Suggested shape:

```python
@dataclass(slots=True)
class GenieRuntimeState:
    rollout_id: str
    prompt_tokens_ref: object | None
    generated_tokens_ref: object | None
    last_completed_frame: int
    resident_tier: str
    ancestor_state_ref: str | None
    checkpoint_delta_ref: str | None
    materialized_bytes: int
    dirty_since_checkpoint: bool
```

Important note:

`state_handle` remains durable identity.
`GenieRuntimeState` is only the hot execution view of that identity.

### `GenieExecutionChunk`

This is the unit sent to a Genie GPU worker.

Suggested shape:

```python
@dataclass(slots=True)
class GenieExecutionChunk:
    chunk_id: str
    signature: GenieBatchSignature
    entity_ids: list[str]
    runnable_stage: str
    frame_range: tuple[int, int]
    estimated_vram_bytes: int
    estimated_transfer_bytes: int
    estimated_flops: float
```

## Stage graph

The default stage graph for Genie should be:

```text
admission
  -> state_materialize
  -> prompt_prepare
  -> transition
  -> checkpoint
  -> artifact_persist
  -> controlplane_commit
```

Not every entity must pass every stage on every step.
For example, terminal-only checkpoint policies may skip intermediate checkpoint work.

## Stage definitions

### 1. `AdmissionStage`

Responsibilities:

- normalize the request into a Genie execution entity
- resolve the effective Genie config
- validate northbound temporal references
- classify cold-start vs continuation work
- assign initial queue lane and priority

Inputs:

- `ProduceSampleRequest`
- `TemporalRefs`
- `GenieTaskConfig`
- sample metadata

Outputs:

- `GenieExecutionEntity`
- initial `GenieBatchSignature`
- admission estimate for scheduler use

What this stage must not do:

- model execution
- file persistence
- whole-request materialization

### 2. `StateMaterializeStage`

Responsibilities:

- resolve prompt tokens from `token_input`, `state_handle`, or checkpoint delta chain
- hydrate a model-ready token window
- record source tier and movement cost
- surface whether the state is hot, warm, or cold

Inputs:

- `token_input`
- `input_state_handle_id`
- prior checkpoint or state-handle lineage

Outputs:

- `GenieRuntimeState.prompt_tokens_ref`
- materialization metadata
- scheduler-visible transfer cost

This stage is the main bridge between durable lineage and transient runtime state.

### 3. `PromptPrepareStage`

Responsibilities:

- validate token shape and tokenizer compatibility
- clip or align prompt windows to model limits
- build model-ready prompt tensors and masks
- stamp the transition signature for the next stage

Inputs:

- materialized prompt tokens
- `num_prompt_frames`
- `tokenizer_kind`
- model geometry and vocab constraints

Outputs:

- prompt-prepared runtime buffers
- transition-ready `GenieBatchSignature`

### 4. `TransitionStage`

This is the hot GPU stage.

Responsibilities:

- execute one bounded frame window
- batch homogeneous entities into one GPU chunk
- update generated token buffers
- emit stage-local timing and occupancy signals

Inputs:

- prompt-prepared runtime buffers
- `window_num_frames`
- `maskgit_steps`
- `temperature`
- current frame range

Outputs:

- generated token window
- updated `GenieRuntimeState`
- checkpoint trigger signal

This stage should not persist terminal artifacts directly.
It should focus on forward progress and batch throughput.

### 5. `CheckpointStage`

Responsibilities:

- build checkpoint deltas at the configured cadence
- update state lineage without forcing deep copies
- produce replayable checkpoint metadata
- surface checkpoint cost to metrics and scheduling

Inputs:

- generated token state
- `checkpoint_every_n_frames`
- current frame index

Outputs:

- `GenieCheckpointDelta`
- updated checkpoint metadata
- optional `state_handle` update request

Recommended policy:

- `checkpoint_every_n_frames = 0` means terminal-only checkpointing
- otherwise emit checkpoint work after each configured frame group

### 6. `ArtifactPersistStage`

Responsibilities:

- persist `tokens`, `state`, `runtime`, `request`, `log`, `checkpoint`, and `recovery`
- attach persisted artifact references to the sample and temporal entities
- keep persistence work off the hot transition lane when possible

Inputs:

- generated token outputs
- checkpoint payloads
- runtime timing data

Outputs:

- persisted files
- artifact metadata
- ready-to-commit control-plane payloads

### 7. `ControlPlaneCommitStage`

Responsibilities:

- create or update `rollout`
- create or update `state_handle`
- create or update `checkpoint`
- finalize `SampleRecord`

This stage is durable state commit.
It should be explicit and auditable.

## Queue lanes

Genie should use explicit queue lanes instead of a single undifferentiated queue.

Recommended lanes:

- `hot_continuation`
- `cold_materialize`
- `checkpoint_heavy`
- `persist_only`

Purpose:

- keep hot continuation windows close to their state
- stop cold state hydration from poisoning low-latency continuation work
- isolate heavy checkpoint and persistence work from the GPU transition lane

## Scheduler inputs

The Genie scheduler should stop scoring only request count or coarse resource units.
It should score runnable chunks using:

- `batch_signature cardinality`
- `expected occupancy`
- `estimated transfer bytes`
- `prompt state hotness`
- `continuation locality`
- `checkpoint_due`
- `artifact persist backlog`
- `queue lane`
- `priority`
- `deadline`

Recommended ordering rule:

1. prefer hot continuation chunks
2. prefer chunks with larger safe batch fill
3. avoid mixing checkpoint-heavy work into throughput-oriented chunks
4. prevent persist backlog from growing without bound

## State and reuse model

Genie should make prompt and branch reuse explicit.

### Prompt reuse

If multiple requests share the same prompt token prefix or same input state handle, the runtime should reuse the materialized prompt state rather than reconstructing it repeatedly.

### Branch reuse

If a branch forks from an existing checkpoint, the runtime should prefer copy-on-write checkpoint deltas over deep copying full token windows.

### Residency tiers

Suggested residency tiers:

- `hot_gpu`
- `warm_pinned_cpu`
- `cold_file`

The scheduler should know which tier currently holds the relevant state.

## Suggested code boundaries

This document does not require an exact file layout, but the runtime should roughly separate:

```text
wm_infra/
  backends/
    genie.py
    genie_runner.py
    genie_runtime.py
    genie_scheduler.py
    genie_checkpoint.py
    genie_persist.py
```

Suggested responsibilities:

- `genie.py`: backend API and control-plane orchestration
- `genie_runner.py`: model-facing execution primitives
- `genie_runtime.py`: runtime entities, stage graph, state objects
- `genie_scheduler.py`: chunk scheduling and lane policy
- `genie_checkpoint.py`: delta and checkpoint logic
- `genie_persist.py`: artifact persistence and durable commit helpers

## Stage API direction

The current monolithic runner API should move toward explicit stage-level entry points.

Suggested shape:

```python
class GenieRunner:
    def prepare_inputs(...)
    def run_window(...)
    def build_checkpoint_delta(...)
    def persist_outputs(...)
```

This does not require rewriting the model internals on day one.
The first milestone is only to expose stage boundaries cleanly enough for runtime scheduling.

## Metrics

Genie should preserve end-to-end metrics, but add stage-local metrics.

Recommended metrics:

- `wm_genie_stage_duration_seconds{stage=...}`
- `wm_genie_chunk_size{stage=...}`
- `wm_genie_chunk_fill_ratio`
- `wm_genie_state_materialize_bytes`
- `wm_genie_state_materialize_seconds`
- `wm_genie_transition_frames_total`
- `wm_genie_transition_tokens_total`
- `wm_genie_checkpoint_delta_bytes`
- `wm_genie_checkpoint_build_seconds`
- `wm_genie_persist_backlog`
- `wm_genie_prompt_reuse_hits_total`
- `wm_genie_prompt_reuse_misses_total`
- `wm_genie_residency_events_total{tier=...}`
- `wm_genie_gpu_occupancy_estimate`

At minimum, the runtime should allow future profiling of:

- queue wait
- materialization latency
- prompt prepare latency
- transition latency
- checkpoint latency
- artifact persistence latency
- transfer bytes
- reuse hit rate

## Validation requirements

The redesign is only real if it improves measurable behavior.

Each runtime iteration should answer:

1. what exact workload was used
2. what stage was changed
3. what metric should improve
4. what metric actually improved
5. what tradeoff regressed

Important Genie-specific validation axes:

- `num_frames`
- `num_prompt_frames`
- `maskgit_steps`
- checkpoint cadence
- cold-start vs warm-start
- branch fan-out

## Migration plan

### Phase 1: expose stage boundaries without changing external behavior

- keep the current backend contract
- split `GenieRunner.run(...)` into internal stage helpers
- emit stage-local timings
- keep persistence semantics unchanged

### Phase 2: introduce runtime entities and chunk signatures

- add `GenieExecutionEntity`
- add `GenieBatchSignature`
- add runtime state objects
- route execution through a stage graph

### Phase 3: add chunk scheduler and lane policy

- schedule chunks instead of whole requests
- add hot continuation and cold materialization lanes
- add chunk-level metrics

### Phase 4: make checkpointing incremental

- consume `checkpoint_every_n_frames` as a real runtime policy
- add checkpoint delta build path
- replace deep persistence coupling with stage-local checkpoint work

### Phase 5: add reuse-aware residency management

- cache hot prompt state
- add copy-on-write branch reuse
- expose reuse and transfer metrics

## Definition of success

The redesign is successful when:

- Genie execution is no longer one opaque whole-request call
- the scheduler can form real homogeneous Genie chunks
- checkpoint cadence is execution-visible
- hot continuation work can stay hot
- artifact persistence stops dominating the GPU transition lane
- profiling can attribute latency to a specific stage rather than a single blob

## References

- `docs/ECS_EXECUTION_RUNTIME.md`
- `wm_infra/backends/genie.py`
- `wm_infra/backends/genie_runner.py`
- `wm_infra/controlplane/schemas.py`
- `wm_infra/controlplane/temporal.py`
