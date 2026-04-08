# Neural ECS Engine: Target Architecture for Learned-Simulator Runtime in wm-infra

## Scope

This document describes a possible future execution-plane architecture for
the learned-simulator / temporal-rollout runtime in `wm-infra`, with the
current `matrix-game` path as the main motivating workload.

It is not the top-level target architecture for all of `wm-infra`.
It does not replace the repo's control-plane and sample-production framing,
and it does not redefine the `wan-video` path as a special case of this
runtime.

## The Core Problem

A learned-simulator environment server is simultaneously an **inference
engine** AND an **environment server**. No existing system handles both:

| System | What It Nails | Why It's Wrong Alone |
|--------|--------------|---------------------|
| **vLLM/SGLang** | Paged memory, continuous batching, GPU utilization | Request-response model. No persistent environments, episodes, resets, or policy coupling. |
| **Madrona** | GPU-native ECS, SoA tables, megakernel, 1M+ envs/sec | Assumes env logic is hand-written C++/CUDA. A PyTorch world model cannot be compiled into a megakernel. |
| **EnvPool** | Lock-free send/recv, zero-copy, auto-reset, async batching | CPU threadpool. Threads all block on GPU forward pass. Useless when env step = neural net. |

### The Fundamental Constraint

> **The environment step IS a neural network forward pass.**

- Madrona: env step = CUDA kernel (~1us) -> megakernel works
- EnvPool: env step = CPU game logic (~0.3us) -> threadpool works
- **wm-infra**: env step = PyTorch dynamics model (~1-50ms) -> **neither works**

We need vLLM's batching to amortize GPU cost, Madrona's ECS to manage thousands of env instances, and EnvPool's API to feed RL trainers. The combination is a new, unsolved infrastructure problem.

---

## Architecture Overview

```
+======================================================================+
|  Layer 4: EnvPool-style Python API                                    |
|                                                                       |
|  vec_env = NeuralVecEnv("world-model-v1", num_envs=4096,             |
|                          batch_size=256, device="cuda:1")             |
|                                                                       |
|  obs, info = vec_env.reset()              # -> GPU tensors            |
|  vec_env.send(actions, env_ids)           # non-blocking              |
|  obs, rew, done, trunc, info = vec_env.recv()  # first 256 ready     |
|                                                                       |
|  Policy GPU (cuda:1) <---- zero-copy ----> Engine GPU (cuda:0)       |
+=============================|========================================+
                              | ZMQ / shared CUDA tensor
+=============================v========================================+
|  Layer 3: Engine Core (dedicated process on cuda:0)                   |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  |  ECS World (Madrona-inspired)                                  |   |
|  |                                                                |   |
|  |  +- Entity Table (SoA, one row per env instance) -----------+ |   |
|  |  | env_id | phase    | step | latent_blocks | goal_blk | ...| |   |
|  |  |   u32  |  enum    | u32  |  BlockRef[]   | BlockRef |    | |   |
|  |  |--------+----------+------+---------------+----------+----| |   |
|  |  |   0    | STEPPING |  7   | [b2, b14]     | [b0]     |    | |   |
|  |  |   1    | ENCODING |  0   | []            | [b1]     |    | |   |
|  |  |   2    | WAITING  |  --  | --            | --       |    | |   |
|  |  |   3    | RESETTING|  50  | [b3, b6, b11] | [b4]     |    | |   |
|  |  |  ...   |          |      |               |          |    | |   |
|  |  |  4095  | STEPPING |  3   | [b9]          | [b5]     |    | |   |
|  |  +--------+----------+------+---------------+----------+----+ |   |
|  |                                                                |   |
|  |  Phases: WAITING -> ENCODING -> STEPPING <-> DONE -> RESETTING |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  |  Paged Latent Pool (vLLM-inspired)                             |   |
|  |                                                                |   |
|  |  Pre-allocated: pool[NUM_BLOCKS, BLOCK_SIZE, N, D] on GPU     |   |
|  |  Free list: [b7, b8, b10, b12, b13, b15, ...]                 |   |
|  |  Page table: env_id -> [block_ids]                              |   |
|  |  Prefix tree: hash(obs, a0..ak) -> shared block_id             |   |
|  |                                                                |   |
|  |  +-----+-----+-----+-----+-----+-----+-----+-----+-----+    |   |
|  |  | b0  | b1  | b2  | b3  | b4  | b5  | b6  | ... | bM  |    |   |
|  |  +-----+-----+-----+-----+-----+-----+-----+-----+-----+    |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  |  Continuous Batching Scheduler                                 |   |
|  |                                                                |   |
|  |  Every iteration:                                              |   |
|  |  1. ECS query: phase == ENCODING -> encode batch               |   |
|  |  2. ECS query: phase == STEPPING -> step batch (block budget)  |   |
|  |  3. ECS query: phase == DONE -> free blocks, auto-reset        |   |
|  |  4. ECS query: phase == WAITING -> admit if blocks available   |   |
|  |  5. If OOM: preempt lowest-priority (swap GPU->CPU)            |   |
|  |                                                                |   |
|  |  Output: SchedulerOutput {encode_batch, step_batch, resets}    |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  +---------------------------------------------------------------+   |
|  |  TaskGraph Executor (Madrona DAG + CUDA streams)               |   |
|  |                                                                |   |
|  |  [GatherStates] ---> [DynamicsForward] ---> [Scatter+Write]   |   |
|  |       |                                          |             |   |
|  |       +-----------> [RewardCompute] -------------+             |   |
|  |                          |                                     |   |
|  |                     [UpdateECS]                                |   |
|  |                          |                                     |   |
|  |                [AutoReset + Admit]                             |   |
|  |                                                                |   |
|  |  Each node = CUDA kernel / PyTorch op on a stream              |   |
|  |  Edges = CUDA event synchronization                            |   |
|  +---------------------------------------------------------------+   |
+======================================================================+
```

---

## Layer-by-Layer Design

### Layer 1: ECS World (from Madrona)

**What we borrow**: SoA entity table with implicit env_id, one table across ALL environments, phase-based system queries.

**What we don't borrow**: Megakernel compilation (PyTorch models can't be compiled into megakernels), C++ requirement, hand-written CUDA.

Madrona's insight: one table across ALL environments, not one table per env. This is critical because:
- The scheduler can query "all entities in phase X" as a vectorized mask operation
- State for 4096 envs is one contiguous tensor per component column
- No per-env allocation/deallocation -- entity creation is just flipping a row from WAITING to ENCODING

```python
class ECSWorld:
    """Madrona-style SoA entity table. GPU-resident."""

    def __init__(self, max_envs: int, device: str):
        # Component columns -- one contiguous array per component
        self.phase     = torch.zeros(max_envs, dtype=torch.int32, device=device)
        self.step_idx  = torch.zeros(max_envs, dtype=torch.int32, device=device)
        self.reward    = torch.zeros(max_envs, dtype=torch.float32, device=device)
        self.done      = torch.zeros(max_envs, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(max_envs, dtype=torch.bool, device=device)
        self.episode_id= torch.zeros(max_envs, dtype=torch.int64, device=device)
        # Block references (page table) -- variable-length per env
        self.block_refs: list[list[int]] = [[] for _ in range(max_envs)]
        self.goal_refs:  list[int]       = [0] * max_envs

    def query_by_phase(self, phase: int) -> torch.Tensor:
        """Madrona system query: return env_ids matching phase. O(1) GPU op."""
        return torch.nonzero(self.phase == phase, as_tuple=False).squeeze(-1)
```

**Madrona TaskGraph -> CUDA stream DAG**:

Madrona compiles all systems into a single megakernel. We can't do that because our "system" is a PyTorch model. Instead, we use Madrona's DAG concept but execute each node on a CUDA stream:

```python
class TaskGraph:
    """Madrona-style execution DAG using CUDA streams instead of megakernel."""

    def __init__(self):
        self.nodes: list[TaskNode] = []
        self.edges: list[tuple[int, int]] = []  # (from, to) dependency

    def execute(self):
        """Execute DAG respecting dependencies via CUDA events."""
        events: dict[int, torch.cuda.Event] = {}
        for node in self._topological_order():
            # Wait for all predecessors
            for pred in self._predecessors(node):
                node.stream.wait_event(events[pred.id])
            # Execute on assigned stream
            with torch.cuda.stream(node.stream):
                node.fn()
                events[node.id] = torch.cuda.Event()
                events[node.id].record()
```

---

### Layer 2: Paged Latent Pool (from vLLM/SGLang)

**What we borrow**: Pre-allocated block pool, page table mapping, hash/radix prefix caching, COW fork via ref counting, preemption via swap (GPU<->CPU).

**What we don't borrow**: Multi-process engine (premature for v1), ZMQ IPC, detokenizer process.

```python
class PagedLatentPool:
    """vLLM-style block pool for world model latent states.

    Pre-allocates ALL GPU memory at startup. Rollouts map to
    sequences of block IDs via a page table. Zero allocation
    during inference.
    """

    def __init__(self, num_blocks: int, block_size: int,
                 latent_shape: tuple[int, int], device: str):
        # One allocation. Ever.
        self.pool = torch.zeros(
            num_blocks, block_size, *latent_shape,
            device=device, dtype=torch.float16,
        )
        self.free_blocks = deque(range(num_blocks))
        self.ref_count = torch.zeros(num_blocks, dtype=torch.int32)

    def gather_batch(self, env_ids: list[int], world: ECSWorld) -> torch.Tensor:
        """Gather latest latent states for a batch. Zero-copy via indexing."""
        block_ids = [world.block_refs[eid][-1] for eid in env_ids]
        slots = [world.step_idx[eid].item() % self.block_size for eid in env_ids]
        # Single GPU indexing op -- no malloc, no copy
        return self.pool[block_ids, slots]  # [B, N, D]

    def scatter_results(self, env_ids: list[int], states: torch.Tensor,
                        world: ECSWorld):
        """Write next states back to pool. Zero-copy via indexing."""
        block_ids = [world.block_refs[eid][-1] for eid in env_ids]
        next_slots = [(world.step_idx[eid].item() + 1) % self.block_size
                      for eid in env_ids]
        self.pool[block_ids, next_slots] = states

    def fork(self, source_id: int, new_id: int, world: ECSWorld):
        """COW fork: new page table entry, same block refs. O(num_blocks)."""
        world.block_refs[new_id] = list(world.block_refs[source_id])
        for block_id in world.block_refs[new_id]:
            self.ref_count[block_id] += 1

    def swap_out(self, env_id: int, world: ECSWorld) -> "SwapHandle":
        """Preemption: GPU blocks -> CPU pinned memory."""
        blocks = world.block_refs[env_id]
        cpu_buffer = self.pool[blocks].to("cpu", non_blocking=True)
        handle = SwapHandle(env_id=env_id, blocks=blocks, cpu_data=cpu_buffer)
        self.free_blocks.extend(blocks)
        world.block_refs[env_id] = []
        world.phase[env_id] = Phase.SWAPPED
        return handle

    def swap_in(self, handle: "SwapHandle", world: ECSWorld):
        """Resume: CPU pinned memory -> GPU blocks."""
        new_blocks = [self.free_blocks.popleft() for _ in handle.blocks]
        self.pool[new_blocks] = handle.cpu_data.to(self.pool.device, non_blocking=True)
        world.block_refs[handle.env_id] = new_blocks
        world.phase[handle.env_id] = Phase.STEPPING
```

**Radix State Cache (SGLang-style prefix sharing for world models)**:

Rollouts starting from the same observation and taking the same initial actions share identical latent state history. This is the world-model analog of KV cache prefix sharing. Critical for MCTS / tree search.

```python
class RadixStateCache:
    """SGLang-style prefix cache for world model trajectories.

    Node key = hash(initial_obs, action_0, action_1, ..., action_k)
    Node value = block_id containing latent state at step k
    """

    def match_prefix(self, obs_hash: int, action_sequence: list[int]
            ) -> tuple[int, list[int]]:
        """Find longest matching prefix.
        Returns (matched_steps, shared_block_ids).

        If a rollout with same initial obs and first 5 actions exists,
        we skip recomputing those 5 steps and reuse the blocks.
        """
        node = self.root
        matched = 0
        blocks = []
        for action in action_sequence:
            child = node.children.get(action)
            if child is None:
                break
            blocks.append(child.block_id)
            node = child
            matched += 1
        return matched, blocks
```

---

### Layer 3: Continuous Batching Scheduler (from vLLM + Madrona queries)

**What we borrow**: Capacity-aware admission based on free blocks (vLLM), phase-based ECS queries for batch construction (Madrona), preemption/swap (vLLM).

```python
class ContinuousBatchingScheduler:
    """vLLM/SGLang-style continuous batching + Madrona ECS queries.

    Key difference from LLM continuous batching:
    - LLM: each iteration generates 1 token per sequence
    - WM: each iteration generates 1 latent state per rollout
    - LLM prefill/decode split -> WM encode/step split
    """

    def __init__(self, pool: PagedLatentPool, world: ECSWorld,
                 config: SchedulerConfig):
        self.pool = pool
        self.world = world
        self.waiting: deque[EnvRequest] = deque()
        self.swap_space: dict[int, SwapHandle] = {}

    def schedule(self) -> SchedulerOutput:
        """Called once per engine iteration.

        Algorithm:
        1. Try swap-in: bring back preempted high-priority rollouts
        2. Try admit: move WAITING -> ENCODING if blocks available
        3. Build batch from ECS phase queries
        4. If OOM: preempt lowest-priority running rollout (swap-out)
        5. Eject completed rollouts, admit replacements immediately
        """
        # Phase 1: Swap-in preempted rollouts that now have capacity
        self._try_swap_in()

        # Phase 2: Admit new rollouts
        while self.waiting and self._can_admit(self.waiting[0]):
            req = self.waiting.popleft()
            self._admit(req)

        # Phase 3: Build batch from ALL running rollouts (Madrona-style query)
        encode_ids = self.world.query_by_phase(Phase.ENCODING)
        step_ids   = self.world.query_by_phase(Phase.STEPPING)

        return SchedulerOutput(
            encode_requests=encode_ids,
            step_requests=step_ids,
            num_free_blocks=len(self.pool.free_blocks),
        )

    def on_step_complete(self, env_id: int):
        """Continuous admission: eject done env, admit next, same iteration."""
        if self.world.done[env_id] or self.world.truncated[env_id]:
            self._eject(env_id)  # free blocks immediately
            # Admit next waiting request in the SAME iteration
            if self.waiting and self._can_admit(self.waiting[0]):
                self._admit(self.waiting.popleft())

    def _can_admit(self, request: EnvRequest) -> bool:
        blocks_needed = math.ceil(request.max_episode_steps / self.pool.block_size)
        return len(self.pool.free_blocks) >= blocks_needed
```

---

### Layer 4: EnvPool-style Python API

**What we borrow**: `send(actions, env_ids)` / `recv()` async interface, `batch_size < num_envs` for latency hiding, auto-reset embedded in engine, zero-copy GPU tensor exchange.

**What we don't borrow**: C++ threadpool (wrong for GPU work), CPU affinity, lock-free circular buffer (bottleneck is GPU not CPU).

```python
class NeuralVecEnv:
    """EnvPool-style vectorized environment backed by neural world model.

    Key differences from EnvPool:
    - Env step = PyTorch forward pass (not CPU game logic)
    - State lives on GPU (not CPU memory)
    - Batching is done by the inference engine (not a threadpool)
    """

    def __init__(self, env_name: str, num_envs: int, batch_size: int,
                 device: str = "cuda:0", policy_device: str | None = None):
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.engine = NeuralECSEngine(
            env_name=env_name, max_envs=num_envs, device=device,
        )
        self.policy_device = policy_device or device

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments. Returns initial observations on GPU."""
        self.engine.reset_all()
        return self.recv()

    def send(self, actions: torch.Tensor, env_ids: torch.Tensor):
        """Non-blocking. Enqueue actions for specified envs.
        Actions stay on GPU -- no CPU roundtrip."""
        self.engine.enqueue_actions(env_ids, actions)

    def recv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor, dict]:
        """Block until batch_size envs complete their step.
        Returns GPU tensors directly -- zero copy to policy."""
        obs, rew, done, trunc, info, env_ids = self.engine.dequeue_results(
            self.batch_size
        )
        if self.policy_device != self.engine.device:
            # Different GPUs: async peer copy
            obs = obs.to(self.policy_device, non_blocking=True)
            rew = rew.to(self.policy_device, non_blocking=True)
        return obs, rew, done, trunc, {**info, "env_id": env_ids}

    def step(self, actions: torch.Tensor,
             env_ids: torch.Tensor | None = None):
        """Sync convenience: send + recv."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=actions.device)
        self.send(actions, env_ids)
        return self.recv()
```

**Async latency hiding (batch_size < num_envs)**:

When `num_envs=4096, batch_size=256`, the engine doesn't wait for ALL 4096 to finish. It returns the first 256 that are ready. The policy processes these while the other 3840 continue stepping:

```
Time ->
Policy:  [process 256] [process 256] [process 256] ...
Engine:  [step 4096..........................................]
                ^ first 256 ready      ^ next 256 ready
```

---

## The Unified Engine Loop

Where everything comes together. One loop iteration:

```python
class NeuralECSEngine:
    """Unified engine combining Madrona ECS + vLLM batching + EnvPool API."""

    def __init__(self, env_name: str, max_envs: int, device: str):
        self.world = ECSWorld(max_envs, device)
        self.pool = PagedLatentPool(
            num_blocks=..., block_size=16,
            latent_shape=(N, D), device=device,
        )
        self.scheduler = ContinuousBatchingScheduler(self.pool, self.world, ...)
        self.prefix_cache = RadixStateCache(self.pool)  # optional
        self.task_graph = TaskGraph()
        self.persist_queue = queue.Queue()  # async persistence

    def engine_loop(self):
        """Main loop. One iteration = one GPU batch."""
        while not self._shutdown:
            # == Phase 1: Drain input queue (CPU, overlapped with prev GPU) ==
            # EnvPool-style: collect actions that were send()'d
            new_actions = self._drain_action_queue()
            for env_id, action in new_actions:
                self.action_buffer[env_id] = action
                self.world.phase[env_id] = Phase.STEPPING

            # == Phase 2: Schedule (CPU, ECS queries) ==
            # Madrona-style: query entity table by phase
            schedule = self.scheduler.schedule()

            # == Phase 3: Execute TaskGraph (GPU, pipelined) ==
            self.task_graph.execute({
                # Node 1: Encode new observations -> latent states
                "encode": (schedule.encode_requests, self.model.encode),
                # Node 2: Gather states + step dynamics model
                "step": (schedule.step_requests, self._batched_step),
                # Node 3: Compute rewards (can overlap with encode)
                "reward": (schedule.step_requests, self.reward_fn.evaluate),
                # Node 4: Scatter results to pool + update ECS
                "scatter": (schedule.step_requests, self._scatter_and_update),
            })

            # == Phase 4: Auto-reset + Admit (CPU, while GPU finishes) ==
            # EnvPool-style: auto-reset done envs, admit waiting envs
            done_ids = self.world.query_by_phase(Phase.DONE)
            self._auto_reset(done_ids)
            self._admit_waiting()

            # == Phase 5: Notify recv() (EnvPool-style) ==
            completed = self.world.query_by_phase(Phase.RESULT_READY)
            if len(completed) >= self.batch_size:
                self._push_to_recv_buffer(completed[:self.batch_size])

            # == Phase 6: Async persistence (off critical path) ==
            self.persist_queue.put(PersistBatch(...))

    def _batched_step(self, step_ids: torch.Tensor):
        """Core step: gather from pool -> dynamics forward -> scatter to pool."""
        # Gather (zero-copy from paged pool)
        states = self.pool.gather_batch(step_ids, self.world)
        actions = self.action_buffer[step_ids]
        # Forward (the expensive part -- batched for GPU efficiency)
        next_states = self.model.predict_next(states, actions)
        # Scatter (zero-copy write back to pool)
        self.pool.scatter_results(step_ids, next_states, self.world)
        # Update ECS columns
        self.world.step_idx[step_ids] += 1

    def _auto_reset(self, done_ids: torch.Tensor):
        """EnvPool-style auto-reset: free blocks, sample new state, re-admit."""
        for eid in done_ids.tolist():
            self.pool.free_blocks.extend(self.world.block_refs[eid])
            self.world.block_refs[eid] = []
        self.world.phase[done_ids] = Phase.RESETTING
        self.world.episode_id[done_ids] += 1
        self.world.step_idx[done_ids] = 0
        # Sample new initial states (batched)
        new_obs = self.initial_state_sampler.sample(len(done_ids))
        self._enqueue_encode(done_ids, new_obs)
```

---

## Why This Synthesis Is Novel

The key insight nobody has built yet:

> **In a neural environment, the "system" that advances entities IS the neural network forward pass.**

This means:
1. **ECS entity table IS the batch buffer** -- the SoA columns for "latent state block refs" directly feed into the batched gather for the dynamics forward pass
2. **Continuous batching IS auto-reset** -- when an env finishes (phase -> DONE), its blocks are freed and the next WAITING env is admitted, all in the same iteration
3. **The paged pool IS the state manager** -- no separate KV cache vs state store, they're the same thing
4. **The TaskGraph IS the stage pipeline** -- but with CUDA stream overlap instead of sequential execution

---

## Comparison Matrix

| Property | Madrona | vLLM/SGLang | EnvPool | **Neural ECS (Target)** |
|----------|---------|-------------|---------|------------------------|
| Env step execution | CUDA megakernel | PyTorch forward | CPU thread | **PyTorch forward, batched** |
| State storage | SoA columns (raw) | Paged blocks (hash) | C++ structs | **SoA columns + paged blocks** |
| Scheduling | TaskGraph DAG | Continuous batching | Lock-free queue | **Continuous batching + DAG** |
| Entity lifecycle | Persistent, typed | Transient request | Implicit slot | **Persistent, phase-based** |
| Memory mgmt | Virtual memory + SoA | Block table + eviction | Pre-allocated arrays | **Block table + SoA metadata** |
| RL API | Gymnasium via nanobind | OpenAI-compatible HTTP | send/recv + Gymnasium | **send/recv (GPU tensors)** |
| Auto-reset | Per-env phase change | N/A | C++ engine-internal | **ECS phase transition** |
| Prefix sharing | N/A | Hash/radix cache | N/A | **Radix cache (tree search)** |
| Multi-env on 1 GPU | Yes (1M+ steps/sec) | N/A | No (CPU-based) | **Yes (target: 100K+ steps/sec)** |
| Preemption | N/A | Swap GPU<->CPU | N/A | **Swap GPU<->CPU** |
| Policy coupling | Zero-copy DLPack | HTTP/gRPC | numpy arrays | **GPU tensor, zero-copy** |

---

## Migration from Current Architecture

| Current Module | Action | Target Module |
|---------------|--------|---------------|
| `ExecutionEntity` / `BatchSignature` | **Kill** | ECS entity (row in SoA table) |
| `HomogeneousChunkScheduler` | **Kill** | `ECSWorld.query_by_phase()` + block budget |
| `LatentStateManager` | **Refactor** | `PagedLatentPool` |
| `RolloutScheduler` | **Refactor** | `ContinuousBatchingScheduler` |
| `TransitionStagePipeline` | **Refactor** | `TaskGraph` with CUDA streams |
| `AsyncTransitionDispatcher` | **Kill** | Engine loop + recv buffer |
| `TransitionExecutor` | **Merge** | `NeuralECSEngine` |
| `WorldModelEngine` | **Merge** | `NeuralECSEngine` |
| `TemporalEnvManager` | **Refactor** | `NeuralVecEnv` |
| `SessionStore` | **Keep** | Session mgmt layered on ECS |
| `LearnedEnvCatalog` / `EnvRegistry` | **Keep** | Registration still needed |
| `TransitionPersistenceLayer` | **Keep, move off critical path** | Async persist thread |

---

## Performance Targets

| Metric | Current (estimated) | Target | How |
|--------|-------------------|--------|-----|
| Steps/sec (toy env, CPU) | ~10K | ~50K | Zero-copy + batch |
| Steps/sec (neural env, GPU) | ~500 | ~100K+ | Continuous batching + paged pool + pipeline |
| Memory efficiency | ~40% (fragmented tensors) | ~95% (paged pool) | Pre-allocation, block table |
| GPU utilization | ~30-50% (CPU blocking) | ~90%+ | Stream pipelining + async persist |
| Env reset overhead | ~1ms (Python callback) | ~0us (phase flip) | Engine-internal auto-reset |
| Max concurrent envs | ~64 (memory-limited) | ~4096+ | Paged pool + eviction + swap |
| MCTS tree search | Not supported efficiently | First-class | Radix state cache + COW fork |

---

## Implementation Roadmap

| Phase | Change | Impact | Risk |
|-------|--------|--------|------|
| **P0** | `PagedLatentPool` (replace `LatentStateManager`) | Eliminate CUDA malloc, 2-3x memory efficiency | Medium |
| **P1** | `ECSWorld` SoA entity table + phase queries | Foundation for continuous batching | Medium |
| **P2** | `ContinuousBatchingScheduler` | 30-60% throughput gain on variable-length workloads | Medium |
| **P3** | Unified `NeuralECSEngine` (merge two engines) | Single optimization target | High |
| **P4** | `NeuralVecEnv` EnvPool-style API | RL trainer compatibility | Low |
| **P5** | Async persistence (off critical path) | 10-20% latency reduction per step | Low |
| **P6** | `TaskGraph` with CUDA stream overlap | 2-3x throughput if stages balanced | Medium |
| **P7** | Preemption + swap (GPU<->CPU) | Meet SLA deadlines under load | Medium |
| **P8** | `RadixStateCache` prefix sharing | 10-50% compute savings for tree-search RL | Low |
| **P9** | Multi-process engine (API / EngineCore / Workers) | Horizontal scaling, tensor parallelism | High |
