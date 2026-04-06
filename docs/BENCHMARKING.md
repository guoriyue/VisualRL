# Benchmarking and profiling

This repo now has a minimal but real benchmark/profiling layer for temporal and video serving workflows.

## What is measurable today

### 1. Rollout microbenchmarks
Use the existing rollout microbenchmark:

```bash
python benchmarks/bench_rollout.py --device cpu --steps 16 --batch 4
python benchmarks/bench_rollout.py --device cuda --steps 16 --batch 4
```

This measures:
- world-model rollout elapsed time
- steps/sec
- per-step latency
- peak GPU memory when run on CUDA
- best-effort GPU utilization / memory sampling via `nvidia-smi`

### 2. Sample API / queue latency
Use the new sample benchmark harness:

```bash
python benchmarks/bench_samples_api.py --in-process --workload wan --iterations 5 --device cuda
```

Or against a running server:

```bash
python benchmarks/bench_samples_api.py \
  --base-url http://127.0.0.1:8000 \
  --workload wan \
  --iterations 20 \
  --concurrency 2 \
  --runner-name official-wan
```

This measures:
- `POST /v1/samples` submit latency
- queue-to-terminal latency for async sample jobs
- polling counts until completion
- per-iteration status and terminal payloads
- aggregated p50 / p95 / p99 latency summaries
- reproducibility metadata such as Python, platform, and git context
- best-effort GPU utilization / memory sampling via `nvidia-smi`
- execution device recorded in the workload artifact so CPU and CUDA runs do not get compared accidentally

To embed a baseline comparison directly into the artifact, add `--baseline-file`:

```bash
python benchmarks/bench_samples_api.py \
  --in-process \
  --workload wan \
  --iterations 5 \
  --baseline-file benchmarks/results/wan-baseline.json
```

When the baseline and current run share the same workload axes, the artifact includes metric deltas for submit latency, terminal latency, and success rate.

The output is written to a structured JSON artifact under `benchmarks/results/` by default.

## Result format

Benchmark outputs follow a simple structured schema:
- `system`: who produced the run (`wm-infra`, `vllm`, `sglang`, etc.)
- `run_context`: runtime, platform, and git metadata captured at benchmark time
- `workload`: canonical workload definition used for comparability checks
- `baseline`: optional embedded comparison against a prior run file
- `summary`: aggregated counts and latency stats
- `gpu_profile`: sampled GPU utilization and memory summary when a GPU is visible
- `samples`: raw per-iteration observations

The comparison rule is strict on purpose: if workload axes differ, the run is not comparable.

## Honest comparison against vLLM / sglang

There are meaningful comparisons we can prepare for, and ones we should not fake.

### Comparable today

If the external system can actually execute the same workload, the new comparison tooling can compare:
- submit latency
- terminal latency
- success rate
- throughput under the same prompt count / concurrency / frame count / resolution / step count

### Not honestly comparable yet

Do **not** claim direct vLLM or sglang comparisons for:
- native Wan2.2 text-to-video execution, unless that system truly runs the same Wan workload
- Genie temporal rollout execution, unless that system truly supports the same token/stateful temporal workload
- sample lineage, artifact persistence, or checkpoint semantics as if they were generic inference-server features

That means today the repo supports **comparison scaffolding**, not a fake benchmark victory lap.

## Comparing two runs

Once you have two result files with the same workload definition:

```bash
python benchmarks/compare_runs.py left.json right.json
```

If the workload differs, the script exits non-zero and prints the mismatch.

## Suggested experiments

### wm-infra only
- rollout-engine latency vs `num_steps`
- Wan queue latency vs concurrency
- Wan request latency vs `frame_count`
- Wan request latency vs resolution
- Genie rollout latency vs frame count and checkpoint cadence

### Cross-system only when supported
- same prompt count
- same frame count
- same resolution
- same step count
- same model family / same serving mode
- same hardware

Without those controls, comparisons are marketing, not measurement.
