# Repo Roadmap

This document translates product strategy into concrete repository work.

## Current state

The repo already has a credible temporal runtime skeleton:
- latent rollout API
- engine
- scheduler
- state manager
- Wan backend path
- Genie backend path
- kernels / ops / layers layout
- benchmark harness

That is useful, but the repo still needs cleaner product framing and sharper separation between runtime substrate and backend-facing control-plane APIs.

## What the repo must become

A layered codebase with a clean separation between:
- runtime execution
- control-plane metadata
- evaluation hooks
- training/export interfaces

## Layered target structure

```text
wm_infra/
  api/
  core/
  controlplane/
    schemas.py
    manifests.py
    experiments.py
    storage.py
  evaluation/
    interfaces.py
    taxonomy.py
    scorers/
  training/
    exports.py
    datasets.py
  backends/
    base.py
    world_model.py
    diffusion_video.py
```

## Near-term coding tasks

### 0. Redesign the execution runtime around homogeneous temporal work

The current runtime has the right temporal nouns but not yet the right execution grain.
The next runtime milestone should redesign the execution plane so it batches and schedules homogeneous temporal state updates instead of request-shaped objects.

Target outcome:

- keep `episode / branch / rollout / checkpoint / state_handle` as durable control-plane objects
- introduce transient execution objects such as `ExecutionEntity`, `ExecutionChunk`, and stage-local tasks
- batch by shared execution signature instead of raw request count
- make state residency, branch reuse, and stage locality explicit scheduler inputs
- turn the current rollout-engine batch from logical batching into real batched execution

Reference:

- `docs/ECS_EXECUTION_RUNTIME.md`

### 1. Introduce control-plane schemas
Done first because every downstream feature depends on it.

Add first-class models for:
- sample specs
- artifacts
- experiment refs
- evaluation records
- training export specs

### 2. Reframe API surface
Current low-level API is rollout-centric.
That is fine for the runtime substrate, but the production-facing API should center temporal sample production across concrete backends such as Wan and Genie.

Add a higher-level request shape around:
- `task_type`
- `backend`
- `experiment_id`
- `sample_spec`
- `artifact_policy`
- `evaluation_policy`

### 3. Add persistence boundary
Need an abstraction for storing:
- sample manifests
- artifact pointers
- status transitions
- acceptance decisions

### 4. Add evaluation interfaces
Not the full evaluator stack yet.
Just the interfaces and contracts.

### 5. Add training export interfaces
Make it possible to export accepted samples into:
- scorer training format
- pairwise ranking format
- finetune manifests

## What stays out of scope for now

- full UI
- giant orchestration platform
- heavyweight distributed runtime redesign
- training a base video model

## Coding standard for the repo

- runtime modules should stay small and benchmarkable
- control-plane code should be schema-first
- APIs should be explicit, typed, and stable
- any production entity should have an ID and lineage hooks
- avoid hiding data transformations inside handlers
- execution batching should be built from homogeneous temporal work, not ad hoc per-request loops

## Definition of progress

A new feature is only real if it improves one of:
- reproducibility
- debuggability
- usable sample yield
- exportability
- runtime efficiency
