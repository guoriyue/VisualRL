# Rollout Runtime Refactor Plan

## Purpose

This plan replaces the overly broad idea that the whole repo should converge on
a single "Neural ECS Engine" architecture.

The correct target is narrower:

- keep `wm-infra` framed as temporal sample-production infra
- keep `POST /v1/samples` and the control plane as first-class repo surfaces
- treat `matrix-game` as the current concrete temporal-rollout backend path
- treat `rollout-engine` as lower-level runtime substrate and bring-up surface
- allow a future learned-simulator execution engine to exist underneath those
  surfaces without redefining the repo around one model family

## Root Problem To Fix

The repo currently has two different rollout-oriented execution stories:

- `wm_infra/core/` still carries the older rollout-engine path
- `wm_infra/runtime/env/` carries the newer learned-env / explicit-state path

That split is real and worth reducing.
What should not happen is collapsing the repo narrative into "Genie runtime"
or replacing durable temporal entities with an in-memory vector-env API.

## Design Rules

1. Keep workload/category framing above backend naming.
   Top-level language should say "temporal rollout" or "learned-simulator
   runtime", not "Genie architecture".
2. Keep backend names concrete.
   `matrix-game` and `rollout-engine` are acceptable concrete backend/runtime
   identifiers and should remain explicit where contracts depend on them.
3. Keep model identity separate from backend identity.
   `backend` selects runtime behavior.
   `model` selects the logical model identifier.
4. Keep temporal state first-class.
   `episode_id`, `branch_id`, `rollout_id`, `checkpoint_id`, and
   `state_handle_id` remain durable control-plane entities.
5. Keep trainer-facing APIs southbound.
   A future vec-env or send/recv API can be added as a specialized consumer
   interface, but it must not replace the durable northbound/control-plane
   surface.
6. Measure before committing to engine rewrites.
   Paged state, continuous batching, preemption, and prefix sharing should only
   move from design to implementation after profiling the real `matrix-game`
   path.

## What Stays

- `POST /v1/samples` remains the primary product-facing API
- `matrix-game` remains the current concrete temporal-rollout backend path
- `rollout-engine` remains available for low-level bring-up and experiments
- explicit temporal control-plane entities remain the source of truth
- backend-specific knobs stay first-class instead of moving into opaque metadata

## What Changes

### Phase 0: Fix Framing

Goal:
- stop describing rollout runtime work as the target architecture for the whole
  repo

Changes:
- narrow docs to "learned-simulator runtime" or "temporal-rollout runtime"
- stop using one closed model family as a top-level category
- describe `matrix-game` as one concrete backend path, not the repo's
  defining identity

Exit criteria:
- top-level docs no longer imply that `wm-infra` is primarily a Genie runtime

### Phase 1: Define The Rollout Substrate Boundary

Goal:
- make it explicit which pieces are generic rollout substrate versus
  Matrix-specific backend behavior

Changes:
- keep `wm_infra/runtime/env/` as the rollout execution substrate
- keep dynamics backend behavior explicit instead of pushing it into generic
  runtime substrate
- document `wm_infra/core/` as legacy rollout-engine substrate rather than the
  preferred extension point for learned-simulator work

Exit criteria:
- there is one clearly documented place for new learned-simulator runtime work
- backend-specific behavior is not hidden in generic runtime modules

### Phase 2: Reduce Duplicate Rollout Engines

Goal:
- stop growing two independent rollout execution stacks

Changes:
- audit `wm_infra/core/` and `wm_infra/runtime/env/` for duplicated concepts:
  scheduling, state residency, async dispatch, and persistence coupling
- pick `wm_infra/runtime/env/` as the preferred substrate for future
  learned-simulator execution work
- keep `rollout-engine` compatible, but stop adding major new execution
  features there unless required for bring-up

Exit criteria:
- new rollout/runtime work lands in one substrate by default
- duplication is shrinking instead of growing

### Phase 3: Move Generic Hot-Path Improvements Into The Substrate

Goal:
- improve rollout execution without tying the design to one model family

Changes:
- move persistence planning and commit boundaries fully behind
  `runtime/env/persistence.py`
- add rollout metrics that match real bottlenecks:
  queue wait, schedule time, transition time, reward time, persist time,
  transfer bytes, state bytes, and locality metrics
- make async dispatch semantics honest about whether they are true engine-loop
  batching or just threadpool submission

Exit criteria:
- execution hot paths are measurable
- persistence is structurally separated from transition execution

### Phase 4: Profile The Real Matrix Path

Goal:
- prove the dominant bottleneck before rewriting the engine

Changes:
- benchmark the concrete `matrix-game` path, not toy abstractions
- capture baseline throughput, latency breakdown, state bytes, queue wait, and
  control-plane commit cost
- record whether the first limiting factor is GPU forward cost, host-device
  transfer, persistence, or scheduler overhead

Exit criteria:
- at least one profiling note or benchmark artifact identifies the current
  dominant bottleneck on the real Matrix path

### Phase 5: Introduce Measured Runtime Mechanisms One By One

Goal:
- add lower-level execution mechanisms only where profiling justifies them

Candidate changes, in this order:
1. state residency / memory-pool improvements
2. locality-aware batching for continuation-heavy workloads
3. persistence off the critical path
4. continuous admission/ejection for variable-length rollout workloads
5. optional prefix/state reuse for explicit branch-heavy search workloads

Rules:
- one bottleneck per iteration
- one attributable systems change per patch
- no megarefactor that mixes API reframing with runtime optimization

Exit criteria:
- each new mechanism is tied to a measured before/after result

### Phase 6: Optional Southbound Trainer API

Goal:
- support RL or search workloads without replacing durable repo surfaces

Changes:
- if needed, add an experimental vec-env or send/recv adapter on top of the
  rollout substrate
- keep it clearly southbound and workload-specific
- keep northbound sample/control-plane APIs unchanged

Exit criteria:
- trainer-facing APIs exist as an adapter layer, not as the repo's new top
  identity

## Non-Goals

- do not collapse Wan, Matrix, and Cosmos into a generic config blob
- do not replace durable temporal entities with server-owned in-memory sessions
- do not turn `wm-infra` into a generic inference-engine rewrite
- do not commit to paged state / ECS / multi-process engine work before profiling

## Immediate Next Step

The immediate next refactor target should be:

1. keep `runtime/env` as the preferred learned-simulator substrate
2. continue thinning duplicated rollout concepts versus `core/`
3. finish isolating persistence and execution metrics on the `matrix-game`
   path
4. only then decide whether paged state residency or continuous batching is the
   next justified systems step
