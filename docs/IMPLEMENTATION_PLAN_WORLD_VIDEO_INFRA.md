# World/Video Infra Implementation Plan

## Purpose

This plan is for `wm-infra` as infrastructure for the three primary workload paths in this repo:

- `wan-video` as the current video-model serving path
- `matrix-game` as the current dynamics / learned-simulator path
- `cosmos-predict` as the current Cosmos-family world/video generation path

This is not a generic inference-platform rewrite plan.
The goal is to make the repository safer and more effective for future systems work on video-model and world-model serving.

The plan below is intentionally sized for roughly five hours of focused implementation work.
Each step includes anti-rot checks so the repo does not accumulate new legacy layers, duplicate namespaces, or unnecessary backward-compatibility shims.

## What This Plan Optimizes For

Short term:

- cleaner substrate for `matrix-game`
- cleaner boundaries between runtime, workload/domain code, and control-plane code
- fewer namespace leaks and fewer compatibility shims
- better observability for the world-model path
- safer foundation for later performance work

Medium term:

- paged state / latent residency
- better batching and state-locality execution
- persistence off the critical path
- cleaner performance experiments for `wan-video`, `matrix-game`, and `cosmos-predict`

This plan does **not** try to fully implement:

- multi-process `EngineCore` architecture
- paged latent pool
- continuous batching
- preemption / swap
- automatic prefix/state cache

Those are next-phase systems tasks and should follow cleaner structure plus measurement.

## Current Practical Constraints

At the time of writing, the repo already has:

- `wm_infra/runtime/execution/` with chunking/scheduling substrate
- `wm_infra/runtime/env/` with learned-env runtime substrate
- `wm_infra/core/` with an older rollout-engine execution path
- `wm_infra/backends/wan*` for the current video-model product path
- `wm_infra/backends/matrix_game/` for the current world-model product path
- `wm_infra/backends/cosmos*` for the current Cosmos-family world/video generation product path
- `wm_infra/workloads/reinforcement_learning/` for RL-style learned-env domain code

The biggest practical structure risks are:

- `runtime` re-exporting workload/domain types
- duplicate engine concepts between `core/` and `runtime/env/`
- persistence logic still too close to execution hot paths
- compatibility wrappers surviving after refactors
- world-model changes accidentally drifting into generic-engine abstraction

## Five-Hour Implementation Plan

### Step 0: Freeze Baseline And Surface Map

Time budget:

- 20 minutes

Goal:

- capture current test baseline
- capture import surface
- capture the current workload/runtime namespace map before edits

Tasks:

1. Run a focused baseline:
   - `pytest -q tests/api/test_server.py tests/core/test_engine.py tests/backends/test_matrix_game.py tests/backends/test_wan.py tests/backends/test_cosmos.py tests/workloads/reinforcement_learning`
2. Record current runtime/workload import usage:
   - `rg -n "from wm_infra\\.(runtime\\.env|workloads\\.reinforcement_learning|core)" wm_infra tests`
3. Record current collected test count:
   - `pytest tests/ --co -q`

Anti-rot checks:

- do not change code in this step
- save the import and test-collection baseline before edits
- do not begin deleting wrappers until current users are known

Exit condition:

- baseline tests green
- current import graph known

### Step 1: Make `runtime` Pure Substrate

Time budget:

- 45 minutes

Goal:

- `wm_infra/runtime/env` should stop re-exporting domain/workload types
- workload types should only come from `wm_infra/workloads/reinforcement_learning`

Tasks:

1. Edit `wm_infra/runtime/env/__init__.py` to export only runtime substrate types.
2. Remove exports of:
   - closed-model adapter exports
   - `GoalReward`
   - `ToyContinuousWorldModel`
   - `ToyLineWorldModel`
   - `ToyLineWorldSpec`
   - `ToyWorldSpec`
3. Update all imports in:
   - `wm_infra/api/`
   - `wm_infra/workloads/reinforcement_learning/`
   - tests
   so workload/domain imports come from `wm_infra.workloads.reinforcement_learning.*`.

Anti-rot checks:

Run:

- `rg -n "from wm_infra\\.runtime\\.env import .*Toy|.*GoalReward" wm_infra tests`
- `rg -n "from wm_infra\\.runtime\\.env\\.(toy|rewards)" wm_infra tests`

Required result:

- zero matches

Exit condition:

- `runtime/env` exports only substrate
- all domain types come from workload modules

### Step 2: Remove Runtime-to-Workload Hard Dependencies

Time budget:

- 50 minutes

Goal:

- `runtime/env` must depend on protocols/registry, not on concrete workload implementations

Tasks:

1. Review `wm_infra/runtime/env/catalog.py`.
2. Ensure runtime does not directly construct concrete workload env classes.
3. Push built-in registration wiring to:
   - `wm_infra/workloads/reinforcement_learning/defaults.py`
   - or app wiring in `wm_infra/api/server.py`
4. Keep `EnvRegistry` / `LearnedEnvCatalog` as runtime-facing substrate interfaces only.

Anti-rot checks:

Run:

- `rg -n "from wm_infra\\.workloads\\.reinforcement_learning|import wm_infra\\.workloads\\.reinforcement_learning" wm_infra/runtime`

Required result:

- zero matches

Exit condition:

- `runtime` no longer imports concrete workload implementations
- workload registration is injected, not hard-coded in runtime substrate

### Step 3: Finish Thinning The Env Manager

Time budget:

- 60 minutes

Goal:

- `TemporalEnvManager` should be a real facade, not a compatibility dump

Tasks:

1. Review `wm_infra/runtime/env/manager.py`.
2. Remove compatibility wrappers that only forward to:
   - `SessionStore`
   - `TransitionExecutor`
3. Move any test usage that still calls manager internals onto:
   - `wm_infra/runtime/env/session_store.py`
   - `wm_infra/runtime/env/transition_executor.py`
4. Keep manager northbound methods only:
   - session lifecycle API
   - step/predict API
   - query helpers actually needed by the northbound surfaces

Anti-rot checks:

Run:

- `rg -n "def _load_stateless_context|def _persist_state_handle|def _ensure_stateless_trajectory|def _execute_transition_batch" wm_infra/runtime/env`
- `rg -n "\\._load_stateless_context|\\._persist_state_handle|\\._ensure_stateless_trajectory|\\._execute_transition_batch" wm_infra tests`

Required result:

- either zero matches, or only unavoidable executor/session-store internals

Exit condition:

- `manager.py` is clearly thinner
- private compatibility wrappers are removed or minimized

### Step 4: Push Persistence Out Of The Execution Loop

Time budget:

- 70 minutes

Goal:

- `matrix-game` path should stop doing scattered store writes inside the core execution loop

Tasks:

1. Review `wm_infra/runtime/env/transition_executor.py`.
2. Refactor execution so a batch/chunk produces:
   - execution result
   - persist intent / persist plan
3. Route persistence writes through `wm_infra/runtime/env/persistence.py`.
4. Batch commit state-handle / transition / checkpoint / trajectory updates through the persistence layer.
5. Do not add background threads yet; keep semantics stable first.

Why this matters:

- this is directly useful for world-model stepping throughput
- it reduces the amount of control-plane bookkeeping mixed into hot execution logic

Anti-rot checks:

Run:

- `rg -n "create_transition|update_trajectory|create_checkpoint|create_state_handle" wm_infra/runtime/env/transition_executor.py`

Required result:

- direct scattered writes should be gone or reduced to a narrow persistence-layer call surface

Exit condition:

- executor prepares persistence, persistence layer commits it

### Step 5: Add World-Model Runtime Metrics That Matter

Time budget:

- 55 minutes

Goal:

- `matrix-game` path should expose stage and execution metrics that make future optimization grounded

Tasks:

1. Extend runtime outputs and benchmark surfaces to preserve:
   - `schedule_ms`
   - `transition_ms`
   - `reward_ms`
   - `persist_ms`
   - `chunk_count`
   - `chunk_sizes`
   - `avg_chunk_size`
   - `state_bytes`
   - `state_count`
   - `queue_wait_ms`
2. Ensure at least one benchmark test or runtime test validates these fields exist.
3. Keep metrics product-relevant for world-model stepping; do not add generic metrics no one uses.

Why this matters:

- later systems work should target measured cost centers, not vibes
- `matrix-game` needs the same level of serious instrumentation that `wan-video` already deserves

Anti-rot checks:

- no dead metrics
- no duplicate names for the same concept
- no metrics that are never asserted or surfaced

Exit condition:

- benchmark/runtime outputs visibly include per-stage and per-batch metrics

### Step 6: Remove Namespace And Test Duplication

Time budget:

- 45 minutes

Goal:

- a single canonical path for each concept

Tasks:

1. Review `tests/` for duplicate old/new locations.
2. Review `benchmarks/` for mismatched source/test file movement.
3. Remove stale convenience re-exports and duplicated package roots.
4. Ensure workload tests live under workload-oriented directories.

Anti-rot checks:

Run:

- `pytest tests/ --co -q`
- `find tests -maxdepth 1 -name 'test_*.py'`
- `rg -n "backward|compat|shim|legacy" wm_infra tests`

Required result:

- no duplicate test locations for the same module
- no top-level stale tests left behind unless intentional
- compatibility wording only where truly necessary

Exit condition:

- one canonical path per test and per module family

### Step 7: Final Integrity Pass

Time budget:

- 35 minutes

Goal:

- confirm the repo did not rot during the refactor

Tasks:

1. Compile all Python:
   - `python -m py_compile $(find wm_infra -name '*.py' | tr '\n' ' ')`
2. Run full tests:
   - `pytest tests/ --tb=short -q`
3. Re-run anti-rot checks:
   - `rg -n "from wm_infra\\.runtime\\.env import .*Toy|.*GoalReward" wm_infra tests`
   - `rg -n "from wm_infra\\.workloads\\.reinforcement_learning" wm_infra/runtime`
4. Inspect final diff:
   - `git diff --stat`

Exit condition:

- full tests green
- runtime remains workload-agnostic
- no obvious compatibility leftovers

## What This Plan Deliberately Leaves For The Next Iteration

After the five-hour plan above is done, the next systems iteration should focus on one concrete bottleneck at a time.

For `matrix-game`, the next serious systems tasks are:

1. paged state / latent residency
2. state-locality-aware batching
3. persistence off the critical path with real background execution
4. continuous step batching

For `wan-video`, the next serious systems tasks are:

1. stage-local metrics preservation for encode / diffusion / decode
2. compiled profile / graph lifecycle observability
3. reducing unnecessary host/device movement between video stages
4. shape-family execution and warm profile reuse

## Non-Goals During This Plan

Do not do these during this five-hour implementation pass:

- do not introduce a generic multi-process serving framework
- do not add `ZMQ` or remote worker architecture
- do not redesign `wan-video` around a generic LLM engine model
- do not kill existing execution vocabulary without a replacement already landed
- do not add broad backward-compatibility shims unless they are proven necessary

## Codex Execution Checklist

When executing this plan, the working loop should be:

1. pick one step only
2. make the smallest coherent change for that step
3. run that step's anti-rot checks
4. run focused tests
5. only then move to the next step

Never stack several speculative architecture changes into one patch.
If a step reveals a different bottleneck or a hidden dependency, update the next step before proceeding.
