# wm-infra

Temporal model serving and control-plane infrastructure for video sample production.

`wm-infra` is not a generic inference stack.
It is a repo for **temporal model infra**: running, tracking, and extending systems that produce time-based outputs such as video generations and world-model rollouts.

Today, the real implemented paths are:
- **Wan 2.2 video generation** via the `wan-video` backend
- **Dynamics-style temporal rollouts** via the `matrix-game` backend
- **World/video generation** via the `cosmos-predict` backend
- the legacy low-level **rollout engine** API for internal/runtime bring-up

That distinction matters. The repo should read like a temporal infra system with concrete backend paths, not like a broad "serve any model" project.

## Product thesis

Most model infra stops at request/response execution.
That is not enough for temporal workloads.

Real video and world-model teams need to:
- launch long-running jobs
- expose backend-specific execution knobs cleanly
- persist artifacts and per-sample metadata
- trace outputs back to prompts, configs, and source state
- inspect failures and quality issues
- export accepted samples into later eval/training loops

So the product direction is:

> **a serving-first temporal sample-production stack**

The serving layer is the entry point.
The durable value is in the control plane around samples, artifacts, lineage, evaluation hooks, and future export paths.

## What this repo is

- A runtime + control plane for **temporal sample production**
- A schema-first API for producing and tracking video/world-model outputs
- A backend-oriented codebase that can grow around real temporal model families

## What this repo is not

- Not a generic omni-model inference framework
- Not a chatbot-serving stack
- Not a claim that `wm-infra` already beats vLLM/sglang on broad runtime maturity
- Not a random collection of kernels with a vague product story

## Implemented backend paths

### 1. Wan 2.2 video (`wan-video`)
Implemented and wired through `POST /v1/samples`.

Current capabilities:
- task types: `text_to_video`, `image_to_video`, `video_to_video`
- async job submission + queue-backed execution
- persisted sample manifests and artifacts
- first-class `wan_config` with video execution knobs
- explicit model identifiers such as `wan2.2-t2v-A14B` and `wan2.2-i2v-A14B`
- runner modes:
  - `stub`
  - `shell`
  - `official` (local Wan 2.2 repo + conda env)

Important reality:
- this is the clearest current path for **video generation serving** in the repo
- the control plane is already shaped around Wan 2.2's real operational constraints: frame count, resolution, steps, and low-VRAM/offload mode
- the currently wired official runner path is aligned with `wan2.2-t2v-A14B` and `wan2.2-i2v-A14B`

### 2. Dynamics rollout backend (`matrix-game`)
Implemented and wired through `POST /v1/samples`.

Current capabilities:
- temporal rollout-style sample production
- explicit action-conditioned latent stepping
- `dynamics` world-model classification in the control plane
- matrix-style action traces through `sample_spec.controls.actions`
- reusable rollout-engine substrate for bring-up

Important reality:
- this is the clearest current path for **world-model / temporal rollout serving** in the repo
- it is video-model-based dynamics infra, not a generic env platform
- the current concrete implementation stabilizes the northbound `dynamics` contract before a dedicated Matrix runtime lands

### 3. Cosmos generation backend (`cosmos-predict`)
Implemented and wired through `POST /v1/samples`.

Current capabilities:
- Cosmos-family text/video to world generation
- `generation` world-model classification in the control plane
- queue-backed async execution
- explicit `cosmos_config` for variant and runner behavior

### 4. Legacy rollout engine (`rollout-engine`)
Still present and useful, but lower-level.

Purpose:
- runtime bring-up
- scheduler/state-manager testing
- benchmark harness support
- low-level rollout API experimentation

This path should be treated as the runtime substrate, not the main product narrative.

## Architecture

`wm-infra` is easiest to extend when you keep a strict separation between runtime execution and production metadata.

### Runtime layer
Responsible for actually executing temporal model workloads.

Examples in this repo:
- `wan-video`
- `matrix-game`
- `cosmos-predict`
- low-level rollout engine

### Control plane
Responsible for describing and tracking what was requested and produced.

Examples in this repo:
- `ProduceSampleRequest`
- `SampleRecord`
- artifact manifests
- experiment references
- temporal lineage entities

### Evaluation/export hooks
Still early, but the schema space is already reserved.

This repo should grow toward:
- QC/failure taxonomy
- review signals
- acceptance decisions
- exportable manifests for downstream training/eval

## API surfaces

### Higher-level temporal sample API
This is the main product-facing surface.

- `POST /v1/samples`
- `GET /v1/samples`
- `GET /v1/samples/{sample_id}`
- artifact metadata/content endpoints
- `GET /v1/backends`
- queue status endpoints

This is where backend-specific temporal workloads should land.

### Low-level rollout API
This is still available:
- `POST /v1/rollout`
- `GET /v1/rollout/{job_id}`

Useful for engine work, but not the primary framing for the repo.

### Stateless transition API
For trainer-style world-model stepping, prefer explicit state references over
server-owned sessions.

- `POST /v1/transitions/initialize`
- `POST /v1/transitions/predict`
- `POST /v1/transitions/predict_many`

These routes return durable resource IDs such as `state_handle_id`,
`trajectory_id`, `branch_id`, and `episode_id`. Callers continue work by
submitting those IDs explicitly rather than relying on an in-memory session.

## Repo layout

```text
wm_infra/
  api/            HTTP surface for temporal sample production and rollout bring-up
  backends/       concrete temporal backend adapters (Wan 2.2, Matrix, Cosmos, rollout-engine)
  controlplane/   sample schemas, manifests, temporal lineage, resource estimates
  rollout_engine/ legacy low-level rollout runtime substrate
  execution/      shared execution substrate
  env_runtime/    learned-env runtime substrate
  models/         model interfaces and registry
  tokenizer/      token/video tokenizer code
  ops/            backend ops / kernels
  kernels/        Triton kernels
  layers/         neural network building blocks
benchmarks/       microbenchmarks and runtime tests
docs/             strategy, profiling, and architecture notes
tests/            unit and integration tests
```

## Configuration model

The config should mirror the actual product shape:
- core runtime config
- server config
- control-plane storage config
- backend-specific config for Wan 2.2 and Cosmos queue/runner behavior

Current environment variables include:
- `WM_MANIFEST_STORE_ROOT`
- `WM_WAN_OUTPUT_ROOT`
- `WM_WAN_SHELL_RUNNER`
- `WM_WAN_REPO_DIR`
- `WM_WAN_CONDA_ENV`
- `WM_WAN_MAX_QUEUE_SIZE`
- `WM_WAN_MAX_CONCURRENT_JOBS`
- `WM_COSMOS_OUTPUT_ROOT`
- `WM_COSMOS_BASE_URL`
- `WM_COSMOS_MODEL_NAME`
- `WM_COSMOS_SHELL_RUNNER`
- `WM_COSMOS_MAX_QUEUE_SIZE`
- `WM_COSMOS_MAX_CONCURRENT_JOBS`

## Why the memory-aware framing exists

The verified Wan2.2 baseline on a 32GB RTX 5090 showed something important:
- frame count, resolution, and step count materially affect whether a run fits and how long it takes
- low-VRAM/offload behavior is not incidental metadata; it is part of request semantics and scheduling

That is why `wm-infra` makes these fields first-class in request/config models instead of burying them in opaque metadata blobs.

## Near-term repo priorities

1. Keep Wan 2.2, Matrix, and Cosmos paths explicit and honest
2. Keep the control plane schema-first
3. Avoid vague generic-inference messaging
4. Make backend onboarding easier by separating runtime code from sample/control-plane code
5. Extend evaluation/export interfaces without bloating core runtime modules

## Development

```bash
pip install -e .[dev]
pytest
wm-serve
```

Defaults:
- sample manifests: `${TMPDIR:-/tmp}/wm_infra`
- Wan 2.2 outputs: `${TMPDIR:-/tmp}/wm_infra_wan`
- Cosmos outputs: `${TMPDIR:-/tmp}/wm_infra_cosmos`

For `POST /v1/samples`:
- rollout-style execution parameters belong in `task_config`
- Wan 2.2-specific parameters belong in `wan_config`
- Cosmos-specific parameters belong in `cosmos_config`
- some legacy metadata backfilling still exists for compatibility, but new callers should use first-class config objects

## Benchmarking and comparison tooling

Practical measurement tooling now exists for temporal/video workloads:
- `benchmarks/bench_rollout.py` for rollout microbenchmarks
- `benchmarks/bench_samples_api.py` for `POST /v1/samples` submit latency, queue latency, and terminal latency
- `benchmarks/compare_runs.py` for strict comparison between structured result files
- `wm_infra/benchmarking.py` for shared latency summaries and workload comparability checks

Example:

```bash
python benchmarks/bench_samples_api.py --in-process --workload wan --iterations 5
python benchmarks/compare_runs.py run_a.json run_b.json
```

Important constraint:
- the repo can now produce honest benchmark artifacts and compare them
- it does **not** pretend that vLLM or sglang support the same temporal/video workloads unless they actually do in a like-for-like setup

## Supporting docs

- `docs/STARTUP_STRATEGY.md`
- `docs/REPO_ROADMAP.md`
- `docs/WAN22_BASELINE.md`
- `docs/WAN22_PROFILING_PLAN.md`
- `docs/WAN_SERVING_SCAFFOLD.md`

## One-line pitch

> `wm-infra` turns temporal model execution into a reproducible sample-production system.
