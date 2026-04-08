# Architectural Critique: wm-infra ECS Runtime vs Industrial-Grade Inference Engines

## Executive Summary

The current architecture is a **staged pipeline with homogeneous batching, marketed as ECS**. It works for toy/research workloads but has 10 fundamental structural deficiencies that would prevent it from scaling to production serving loads. Below we dissect each one against vLLM/SGLang/Madrona/EnvPool patterns.

---

## Part 1: 10 Critical Deficiencies

### Deficiency 1: Static Batching, Not Continuous Batching

**Current** (`core/scheduler.py:116-152`):
```
schedule_batch() → collect ALL active → sort → select top N → execute ALL → wait → repeat
```

This is textbook **static batching**. Every request in the batch must complete its step before the next batch starts. If one rollout is at step 1 and another at step 99, they still wait for each other per iteration.

**vLLM/SGLang**: Token-level continuous batching. Finished sequences are ejected and new sequences admitted **within the same iteration**. GPU never idles waiting for the longest sequence.

**Impact**: At 64 concurrent rollouts with variable lengths (1-100 steps), static batching wastes 30-60% of GPU cycles on padding/waiting.

---

### Deficiency 2: No Memory Pooling — Fragmented Per-Tensor Allocation

**Current** (`core/state.py:52-330`):
```python
# Every step creates a new independent tensor
state.latent_states.append(predicted_state)  # line 247
self._register_tensor(predicted_state)        # line 251
```

Each latent state is a separate `torch.Tensor` with its own CUDA allocation. The "memory management" is just tracking `data_ptr` counts and doing LRU eviction of entire rollouts.

**vLLM**: Pre-allocates a contiguous **block table** on GPU memory at startup. KV cache lives in fixed-size blocks (e.g., 16 tokens per block). New tokens map into free blocks via a page table. Zero fragmentation, zero allocation overhead during inference.

**SGLang**: Radix tree over contiguous token buffers. Shared prefixes point to the same physical memory.

**Impact**: CUDA malloc/free per step is ~100-500us. At 1000 steps/sec across 64 rollouts, that's 6-32ms/iteration wasted on allocation alone. Plus memory fragmentation reduces effective capacity by 20-40%.

---

### Deficiency 3: "ECS" is a Misnomer — It's Just GroupBy

**Current** (`runtime/execution/scheduler.py:27-80`):
```python
class HomogeneousChunkScheduler:
    def schedule(self, *, work_items, ...):
        grouped_items = OrderedDict()
        for item in work_items:
            grouped_items.setdefault(item.entity.batch_signature, []).append(item)
        # split into chunks of max_chunk_size
```

This is `itertools.groupby` with a class name. A real ECS (Madrona, Bevy, flecs) has:
- **Persistent entities** with component archetypes stored in contiguous SoA arrays
- **System queries** that iterate archetypes with zero overhead
- **Structural change detection** for reactive scheduling

`ExecutionEntity` is created, lives for one function call, and is discarded. There is no entity lifecycle, no component storage, no archetype table. Calling this "ECS" confuses the architecture.

---

### Deficiency 4: CPU Scheduling on GPU Critical Path

**Current** (`core/engine.py:137-176, step()`):
```
admit() → schedule_batch() → _execute_batch_chunked() → _finalize_job()
  CPU        CPU                GPU + CPU                   CPU
```

All CPU work (scheduling, chunk building, persistence) is **synchronous** with GPU execution. The GPU must wait for CPU to finish scheduling before it can start, and CPU must wait for GPU to finish before it can persist.

**SGLang v0.4**: Zero-overhead scheduler runs **one batch ahead**. While GPU executes batch N, CPU prepares batch N+1. GPU never waits for CPU.

**vLLM V1**: Separate EngineCore process runs a busy-loop scheduler via ZMQ, decoupled from GPU worker processes.

**Impact**: CPU scheduling overhead is typically 2-15ms per iteration. At 100 iterations/sec, that's 2-15% throughput loss from pure CPU blocking.

---

### Deficiency 5: Two Divergent Execution Engines

The codebase has **two completely independent execution paths**:

| Feature | `WorldModelEngine` (core/) | `TransitionExecutor` (runtime/env/) |
|---------|---------------------------|--------------------------------------|
| Scheduling | `RolloutScheduler` | None (inline) |
| State mgmt | `LatentStateManager` | `TemporalStore` state handles |
| Batching | `HomogeneousChunkScheduler` | `TransitionStagePipeline` |
| Async | `AsyncWorldModelEngine` loop | `AsyncTransitionDispatcher` threadpool |
| Persistence | In-memory only | Full temporal store |

These should be **one engine** with different execution modes. The current split means every optimization must be implemented twice, and the two paths will inevitably diverge in capability.

---

### Deficiency 6: Persistence on GPU Critical Path

**Current** (`runtime/env/transition_executor.py:198-296`):
```python
for executed in pipeline.execute_chunk(...):  # GPU work
    # IMMEDIATELY AFTER GPU:
    self._persist_state_handle(...)      # CPU + memory
    self.temporal_store.create_transition(...)  # CPU + memory
    self.temporal_store.update_trajectory(...)  # CPU + memory
```

Every single transition is persisted synchronously before the next chunk executes. Persistence is ~0.1-1ms per transition. At batch size 32, that's 3-32ms of CPU work blocking the GPU pipeline.

**vLLM**: Outputs are streamed to a detokenizer running in a separate process. The engine never waits for output processing.

---

### Deficiency 7: No Pipeline Parallelism Between Stages

**Current** (`runtime/env/pipeline.py:311-327, run_chunk()`):
```python
def run_chunk(self, prepared, ...):
    execution = self.transition(prepared)   # GPU: dynamics model
    reward = self.reward(execution)         # GPU/CPU: reward computation
    persist = self.build_persist_plan(...)  # CPU: build intents
    return ...
```

For N chunks, the execution is strictly: `T1 -> R1 -> P1 -> T2 -> R2 -> P2 -> ...`

With pipeline parallelism: `T1 -> T2 -> T3 ...` while `R1 -> R2 ...` and `P1 -> P2 ...` run on separate CUDA streams or CPU threads.

**Impact**: 3-stage pipeline could theoretically achieve 3x throughput if stages are balanced.

---

### Deficiency 8: AsyncTransitionDispatcher is a ThreadPool Pretending to Be a Batch Queue

**Current** (`runtime/env/async_runtime.py:67-312`):

The class has 312 lines of complexity for what amounts to:
1. Queue items by `batch_key`
2. When queue reaches `max_batch_size`, submit to `ThreadPoolExecutor`
3. `collect()` blocks on `Future.result()`

Problems:
- `ThreadPoolExecutor` is for CPU-bound work. GPU work needs CUDA stream management
- The default `_default_batch_runner` executes payloads **sequentially** in the thread (line 302-311) — there is no actual batching
- `max_workers=2` means at most 2 batches in flight, but GPU can only execute one at a time anyway

---

### Deficiency 9: No Preemption or Priority-Based Eviction

**Current** (`core/scheduler.py`): The scheduler has `MEMORY_AWARE` policy that sorts by resource units, but once a rollout is admitted and executing, it **cannot be preempted**. If a high-priority request arrives, it must wait for all active rollouts to complete their current step.

**vLLM**: Supports preemption via **swapping** (GPU->CPU) or **recomputation** (evict KV cache, recompute from prompt when rescheduled). This is critical for meeting SLA deadlines.

---

### Deficiency 10: No State Sharing Across Rollouts (No Prefix Cache Analog)

World model rollouts that start from the **same initial observation** share the same initial latent state. Rollouts that share the first N actions share the first N transition states. This is the exact analog of prefix caching in LLMs.

**Current**: Each rollout independently stores its full state history. The `fork()` method in `LatentStateManager` does data_ptr sharing, but this only works for explicit fork operations. There is no **automatic** detection of shared state prefixes.

**vLLM**: Hash-based block matching automatically detects and shares identical KV cache blocks.

**SGLang**: Radix tree automatically discovers shared prefixes across requests.

---

## Part 2: Current Module Disposition

### What to Delete

| Current Module | Verdict |
|---------------|---------|
| `ExecutionEntity` / `BatchSignature` / `ExecutionChunk` | **Kill**. Replace with paged pool + page table. The "entity" abstraction adds naming overhead with zero benefit. |
| `HomogeneousChunkScheduler` | **Kill**. A `groupby` doesn't need a class. Continuous batching scheduler subsumes this. |
| `AsyncTransitionDispatcher` | **Kill**. ThreadPoolExecutor-based dispatch is wrong for GPU work. Replace with engine-loop + recv buffer. |
| `TransitionExecutor` | **Merge** into unified engine. |
| `TemporalEnvManager._backward_compat_*` (20+ methods) | **Kill immediately**. 20 private wrapper methods that call through to composed objects — this is dead weight. |
| `ExecutionBatchPolicy` | **Simplify**. `mode`, `min_ready_size`, `return_when_ready_count` are unused complexity. Continuous batching has one policy: "execute everything that's ready." |

### What to Keep

| Current Module | Verdict |
|---------------|---------|
| `TransitionStagePipeline` | **Refactor** into `TaskGraph` with CUDA stream overlap. The stage decomposition (materialize/transition/reward/persist) is correct; the sequential execution is wrong. |
| `SessionStore` | **Keep**. Session management layered on top of ECS. |
| `LearnedEnvCatalog` / `EnvRegistry` | **Keep**. Registration pattern is sound. |
| `TransitionPersistenceLayer` | **Keep but move off critical path**. Async background thread. |

---

## Part 3: Current vs Target Comparison

| Dimension | Current | Target (Industrial) |
|-----------|---------|---------------------|
| Batching | Static (schedule -> execute -> wait) | Continuous (admit/eject within iteration) |
| Memory | Per-tensor malloc + LRU eviction | Pre-allocated block pool + page table |
| Scheduling | Sort-and-select, no preemption | Block-budget-aware, preemption + swap |
| CPU/GPU overlap | Zero (sequential) | Full (scheduler runs 1 batch ahead) |
| Persistence | Synchronous on critical path | Async fire-and-forget |
| Stage pipeline | Sequential per chunk | Multi-CUDA-stream overlap |
| Prefix sharing | Explicit fork only | Automatic radix/hash cache |
| Process model | Single process, single thread | Multi-process (API / EngineCore / Workers) |
| Execution engines | Two divergent paths | One unified engine |

---

## Reference Systems Studied

### vLLM
- PagedAttention and block-level KV cache management
- Continuous batching scheduler (token-level)
- Engine/Worker multi-process separation (V1 via ZMQ)
- Hash-based automatic prefix caching
- Preemption via swap (GPU<->CPU) and recomputation
- Tensor parallelism via Ray or native multiprocessing
- Sources: [vLLM Architecture](https://docs.vllm.ai/en/latest/design/arch_overview/), [Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)

### SGLang
- RadixAttention tree-based KV cache (token-level granularity)
- Zero-overhead batch scheduler (CPU scheduling overlapped with GPU computation)
- Cache-aware scheduling to maximize radix tree hit rates
- Rust-based model gateway router with hybrid PD routing
- Frontend language + backend runtime co-design
- Sources: [RadixAttention Blog](https://www.lmsys.org/blog/2024-01-17-sglang/), [v0.4 Zero-Overhead Scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)

### Madrona
- GPU-native ECS with SoA component storage across ALL environments
- Single-table-per-archetype design (not per-environment)
- Megakernel compilation (all systems compiled into one CUDA kernel)
- Persistent-threads execution (warp-level work fetching)
- TaskGraph DAG for system scheduling
- Zero-copy tensor export via DLPack/nanobind
- 1.9M steps/sec (Hide and Seek), 40M steps/sec (Overcooked)
- Sources: [Madrona SIGGRAPH 2023](https://madrona-engine.github.io/shacklett_siggraph23.pdf), [GitHub](https://github.com/shacklettbp/madrona)

### EnvPool
- C++ threadpool with lock-free circular buffers (ActionBufferQueue, StateBufferQueue)
- Async send/recv API (batch_size < num_envs hides latency)
- Zero-copy memory transfer via pybind11
- CPU affinity pinning, 2-3x envs per thread for load balancing
- Auto-reset embedded in C++ engine (no Python callback)
- 1M FPS (Atari), 3M FPS (MuJoCo) on DGX-A100
- Sources: [EnvPool Paper](https://arxiv.org/abs/2206.10558), [GitHub](https://github.com/sail-sg/envpool)
