# ECS-Like Execution Runtime for Temporal World Models

This document defines the execution-plane redesign for `wm-infra`.

The goal is not to turn the repository into a game engine.
The goal is to make the runtime see large amounts of homogeneous temporal state updates so it can build GPU-friendly batches and stage-local task graphs.

That means:

- keep the existing temporal control plane as the durable product surface
- redesign the execution plane as a data-oriented runtime
- treat temporal state, stage locality, and chunked rollout work as first-class scheduling inputs

## Why this redesign exists

The current runtime has the right high-level concepts but the wrong execution grain.

Today:

- the scheduler selects a batch
- the engine still loops over jobs one by one
- state is managed per rollout object rather than per homogeneous execution chunk
- stage boundaries are implicit
- cache and residency are not explicit scheduler inputs

This limits:

- true GPU batching
- state reuse
- stage-local observability
- branch-aware placement
- disaggregated execution across encode, transition, decode, and persistence

The redesign should fix those limits without collapsing `episode`, `branch`, `rollout`, `checkpoint`, or `state_handle` into runtime-only objects.

## Layering

The repository should keep a strict split between durable temporal identity and transient execution identity.

### Durable control-plane objects

These remain the northbound and persisted source of truth:

- `episode`
- `branch`
- `rollout`
- `checkpoint`
- `state_handle`
- sample/artifact manifests

These objects answer:

- what world state exists
- where it came from
- which branch it belongs to
- which artifacts and evaluations depend on it

### Transient execution-plane objects

These are internal runtime objects and may be created, merged, split, and destroyed frequently:

- `ExecutionEntity`
- `ExecutionChunk`
- `StageTask`
- `ResidencyRecord`
- `CheckpointDeltaRecord`

These objects answer:

- what work is runnable right now
- which work is homogeneous enough to batch
- where the hot state lives
- which stage should run next

## Design principles

1. Batch by homogeneous state updates, not by requests.
2. Make stage boundaries explicit.
3. Separate persistent state identity from execution residency.
4. Prefer continuation locality over global fairness when state movement is expensive.
5. Encode branch reuse and window overlap as runtime-managed references, not ad hoc Python conditionals.
6. Make latent-only execution a first-class path.
7. Keep Wan and Genie backend behavior explicit; do not force fake genericity.

## Non-goals

- replacing the control plane with an ECS
- forcing Wan diffusion execution into a rollout-shaped engine
- inventing a distributed runtime before local single-node behavior is well profiled
- pretending world-model execution is the same as text-only prefill/decode serving

## Current execution-plane conflicts

The current runtime conflicts with an ECS-like design in three direct ways:

1. `WorldModelEngine._execute_batch()` still performs per-job execution inside a scheduled batch.
2. `LatentStateManager` stores state as per-rollout Python objects with deep-copy fork semantics, not chunked or shared state records.
3. `RolloutScheduler` schedules requests, not homogeneous stage-local work units.

See:

- `wm_infra/core/engine.py`
- `wm_infra/core/state.py`
- `wm_infra/core/scheduler.py`
- `wm_infra/backends/rollout.py`

## Runtime object model

### ExecutionEntity

An `ExecutionEntity` is the smallest schedulable temporal work item.
It should not represent a whole episode.
It should usually represent a bounded shard of rollout work.

Recommended entity kinds:

- `rollout_window`
- `transition_step_group`
- `decode_window`
- `checkpoint_task`
- `artifact_task`

Suggested shape:

```python
@dataclass(slots=True)
class ExecutionEntity:
    entity_id: str
    entity_kind: str
    rollout_id: str
    episode_id: str | None
    branch_id: str | None
    state_handle_id: str | None
    current_stage: str
    next_stage: str | None
    batch_signature: "BatchSignature"
    residency: "ResidencyRef"
    priority: float
    deadline_s: float | None
```

### BatchSignature

`BatchSignature` is the execution-plane equivalent of an ECS archetype key.
Entities with different signatures should not be batched together.

Recommended fields:

```python
@dataclass(frozen=True, slots=True)
class BatchSignature:
    backend_family: str
    model: str
    stage: str
    dtype: str
    device_kind: str
    tokenizer_kind: str | None
    width: int | None
    height: int | None
    num_tokens: int | None
    window_frames: int | None
    prompt_frames: int | None
    maskgit_steps: int | None
    needs_decode: bool
    checkpoint_every_n_frames: int | None
```

This is the key idea:

The runtime should batch work because the batch signature matches, not because many requests happen to be queued.

### ExecutionChunk

An `ExecutionChunk` groups homogeneous entities into a single stage-local batch.

```python
@dataclass(slots=True)
class ExecutionChunk:
    chunk_id: str
    signature: BatchSignature
    entity_ids: list[str]
    device_id: str
    estimated_bytes: int
    estimated_flops: float
    runnable_stage: str
```

`ExecutionChunk` is the unit sent to the GPU worker.

## Components

The runtime should model execution-relevant state as components attached to entities.

### Temporal identity components

- `EpisodeRef`
- `BranchRef`
- `RolloutRef`
- `StateHandleRef`
- `CheckpointRef`

### Shape and workload components

- `WindowShape`
- `TokenLayout`
- `ActionLayout`
- `RenderPolicy`
- `CheckpointPolicy`
- `BatchSignature`

### Residency and reuse components

- `StateResidency`
- `HotOnGPU`
- `PinnedOnCPU`
- `NeedsMaterialization`
- `AncestorStateRef`
- `CheckpointDeltaRef`
- `WindowOverlapRef`
- `SharedPromptEmbeddingRef`

### Scheduling components

- `Priority`
- `Deadline`
- `QueueLane`
- `ExpectedTransferBytes`
- `EstimatedRuntime`
- `AdmissionBudget`

### Artifact components

- `ArtifactRequest`
- `DecodeRequest`
- `PersistRequest`
- `EvaluationHookRequest`

## Systems

The execution plane should be driven by explicit systems.

### 1. AdmissionSystem

Converts a northbound request into one or more execution entities.

Responsibilities:

- normalize the request into workload shape
- decide cold-start vs warm-start lane
- create initial `ExecutionEntity` records
- attach the first `BatchSignature`

### 2. StateMaterializeSystem

Materializes runtime state from a `state_handle`, checkpoint delta chain, or encoded prompt input.

Responsibilities:

- resolve `state_handle`
- load or reconstruct latent state
- reuse ancestor or overlap state when possible
- update residency metadata

This system is the correct home for:

- state hydration latency
- checkpoint delta replay cost
- prompt token decode cost

### 3. BatchBuildSystem

Builds `ExecutionChunk` objects from runnable entities.

Responsibilities:

- group by `BatchSignature`
- enforce per-device memory and occupancy budgets
- prefer hot-state locality
- produce stage-local chunks

### 4. TransitionSystem

Runs world-model latent transition on a chunk.

Responsibilities:

- gather homogeneous latent tensors into one batched tensor
- gather actions into one batched tensor
- call the model once per chunk, not once per job
- write outputs back to transient state buffers

This is the highest-priority performance system for `genie-rollout` and `rollout-engine`.

### 5. DecodeSystem

Runs latent-to-frame decode only when required.

Responsibilities:

- decode batched latent windows
- support latent-only continuation when decode is unnecessary
- keep render work out of the transition hot path

### 6. CheckpointCompactSystem

Builds checkpoint deltas and copy-on-write branch records.

Responsibilities:

- emit delta rather than full materialization when possible
- record parent-child lineage for branch reuse
- expose checkpoint build cost to metrics

### 7. ResidencySystem

Maintains hot/cold state placement.

Responsibilities:

- move state between GPU and CPU tiers
- pin active state on the correct worker
- evict based on recompute plus transfer cost, not raw LRU alone

### 8. ArtifactPersistSystem

Persists output artifacts and runtime metadata after compute stages complete.

Responsibilities:

- keep artifact I/O off the transition critical path
- emit `state_handle` and checkpoint updates
- synchronize with the control plane

## Stage DAG

World-model execution should not be modeled as a single monolithic loop.
It should be modeled as a stage DAG.

Recommended default DAG:

1. `ingest`
2. `visual_encode`
3. `state_materialize`
4. `transition`
5. `decode`
6. `checkpoint`
7. `artifact_persist`
8. `evaluation_hooks`

Not every workload uses every stage.

Examples:

- `rollout-engine` may use `state_materialize -> transition -> decode`
- `genie-rollout` may use `state_materialize -> transition -> checkpoint -> artifact_persist`
- latent-only evaluation may stop after `transition`

## Scheduler design

The scheduler should move from request scheduling to chunk scheduling.

### Current scheduler problems

The current scheduler:

- sorts `RolloutRequest`
- selects request IDs
- leaves the engine to execute per request

That misses the real optimization target.

### Target scheduler inputs

The target scheduler should score runnable chunks using:

- `batch fullness`
- `expected gpu occupancy`
- `state locality hit`
- `estimated transfer bytes`
- `chunk age`
- `priority`
- `deadline`
- `online vs offline lane`

### Queue lanes

Use separate lanes for:

- `interactive_continuation`
- `interactive_cold_start`
- `offline_generation`
- `decode_backfill`
- `artifact_backfill`

This avoids letting cold-start and I/O-heavy work poison interactive continuation latency.

### Placement policy

Default placement rules:

1. keep continuation work on the worker holding the hot state
2. co-locate branch children with the parent worker when possible
3. split only when the state transfer cost is lower than expected queue delay
4. isolate decode-heavy work from transition-heavy work when GPUs differ in bottleneck behavior

## Cache and residency design

The runtime should stop thinking in terms of a single cache.
It needs at least four reusable state classes:

1. prompt/video token cache
2. visual encoder output cache
3. latent state residency cache
4. checkpoint delta store

### Residency record

```python
@dataclass(slots=True)
class ResidencyRef:
    residency_id: str
    device_tier: str  # gpu, cpu_pinned, disk
    worker_id: str | None
    bytes_size: int
    shared_ref_count: int
    materialized_from: str | None
    last_used_at_s: float
```

### Eviction policy

Eviction should rank candidates by:

- estimated recompute cost
- estimated transfer cost
- branch fan-out reuse probability
- near-term continuation probability
- age

That is materially better than plain LRU for temporal workloads.

### Branch reuse

Branch reuse should use copy-on-write references instead of deep copies.

Today `LatentStateManager.fork()` deep-copies the full source history.
That is not scalable for long-horizon branching.

Target behavior:

- parent state is shared
- child writes create delta pages
- checkpoints can be reconstructed from parent plus delta chain

## GPU worker model

The runtime should expose stage-specialized workers even on a single host.

Recommended local worker roles:

- `transition_worker`
- `decode_worker`
- `artifact_worker`

Optional later roles:

- `visual_encode_worker`
- `checkpoint_compact_worker`

This does not require a distributed cluster first.
It only requires the runtime to stop assuming one synchronous loop owns all stages.

## Northbound to execution-plane mapping

The current API surface can remain intact.
The redesign happens behind it.

### `POST /v1/rollout`

Maps to:

- create rollout execution entity
- optionally create decode request
- return streamed stage progress or final result

### `POST /v1/samples`

Maps to:

- create durable sample intent
- enqueue backend-specific execution entities
- emit durable artifacts and temporal records after completion

### Temporal control-plane mapping

Execution completion should write back to:

- `rollout.metrics`
- `state_handle`
- `checkpoint`
- artifact manifests

The control plane remains the owner of durable lineage.

## Module plan

Recommended new execution-plane modules:

```text
wm_infra/core/execution/
  entities.py
  components.py
  chunking.py
  task_graph.py
  systems/
    admission.py
    state_materialize.py
    transition.py
    decode.py
    checkpoint.py
    residency.py
    persist.py
  workers/
    transition_worker.py
    decode_worker.py
  scheduler.py
```

Recommended existing-module changes:

- `wm_infra/core/engine.py`
  - stop owning the whole loop directly
  - become a coordinator over systems and workers
- `wm_infra/core/scheduler.py`
  - schedule chunks, not request IDs
- `wm_infra/core/state.py`
  - move from rollout object store to residency plus delta-aware state store
- `wm_infra/backends/rollout.py`
  - build execution entities rather than directly constructing a thin `RolloutJob`

## Concrete interface sketch

```python
class StageSystem(Protocol):
    stage_name: str

    def can_run(self, entity: ExecutionEntity) -> bool: ...
    def build_chunks(self, entities: list[ExecutionEntity]) -> list[ExecutionChunk]: ...
    def run_chunk(self, chunk: ExecutionChunk, runtime: "ExecutionRuntime") -> "StageResult": ...


class ExecutionRuntime:
    def submit(self, intent: "ExecutionIntent") -> str: ...
    def tick(self) -> None: ...
    def next_runnable_chunks(self) -> list[ExecutionChunk]: ...
    def apply_stage_result(self, result: "StageResult") -> None: ...
```

## Metrics that must exist

The redesign is incomplete without metrics.

Add execution-plane metrics for:

- `wm_chunk_size`
- `wm_chunk_build_seconds`
- `wm_transition_chunk_seconds`
- `wm_decode_chunk_seconds`
- `wm_state_materialize_seconds`
- `wm_state_transfer_bytes`
- `wm_state_transfer_seconds`
- `wm_state_locality_hit_rate`
- `wm_chunk_gpu_occupancy_estimate`
- `wm_branch_copy_on_write_bytes`
- `wm_checkpoint_delta_bytes`
- `wm_warm_start_ratio`
- `wm_transition_only_ratio`
- `wm_decode_backlog`

## Workload profiles to optimize first

The runtime should optimize real workloads first, not abstract elegance.

### 1. Genie online continuation

Goal:

- continue from a hot temporal state with low tail latency

Primary metrics:

- warm-start ratio
- state locality hit rate
- p95 transition latency

### 2. Offline synthetic data generation

Goal:

- maximize GPU occupancy and throughput on long batch jobs

Primary metrics:

- chunk fullness
- transition throughput
- checkpoint delta size

### 3. Branch-heavy evaluation

Goal:

- fork many short rollouts from shared ancestors cheaply

Primary metrics:

- branch materialization cost
- copy-on-write memory overhead
- ancestor reuse hit rate

## Migration plan

### Phase 1: Make batching real

- add `BatchSignature`
- add `ExecutionChunk`
- tensorize `TransitionSystem`
- stop per-job `predict_next(...)` inside `_execute_batch()`

### Phase 2: Make stages explicit

- split transition, decode, and persistence into separate systems
- expose stage timing metrics everywhere
- support latent-only completion without implicit decode

### Phase 3: Make state residency explicit

- introduce residency records
- add continuation stickiness
- add warm vs cold queue lanes

### Phase 4: Make branching cheap

- replace deep-copy fork with copy-on-write checkpoint deltas
- add ancestor-aware reuse

### Phase 5: Add worker specialization

- transition worker
- decode worker
- artifact worker

## Risks and tradeoffs

1. A pure ECS rewrite would be overkill.
   Keep ECS-like ideas inside the execution plane only.

2. Too much genericity will erase backend-specific constraints.
   Wan and Genie should share runtime primitives, not fake identical request surfaces.

3. Chunk scheduling can improve throughput while hurting latency.
   Keep explicit queue lanes and latency budgets.

4. Copy-on-write branching can complicate checkpoint reconstruction.
   Do not adopt it without strong validation and observability.

## Definition of success

The redesign is successful when:

- the runtime forms real homogeneous GPU batches
- continuation work stays near hot state
- branch-heavy workloads reuse ancestor state cheaply
- stage-local metrics explain where time goes
- control-plane lineage remains intact and explicit

At that point `wm-infra` becomes:

- a temporal workflow and control plane on top
- a many-world, ECS-like execution runtime underneath

That is the correct shape for world-model infrastructure.
