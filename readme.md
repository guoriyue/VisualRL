  Your Layout vs. SGLang: Side-by-Side Map                                                                
  ┌─────────────────┬───────────────────────────┬───────────────────────────────────┬───────────────────┐ 
  │     Concern     │      SGLang location      │         wm-infra location         │      Verdict      │ 
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ HTTP serving    │ sglang.srt.server         │ gateway/                          │ You have this.    │
  │                 │                           │                                   │ Fine.             │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │                 │                           │                                   │ You're ahead.     │
  │                 │                           │                                   │ SGLang's          │
  │                 │                           │                                   │ diffusion         │
  │ Request         │                           │ engine/managers/scheduler.py +    │ scheduler is a    │
  │ batching /      │ managers/scheduler.py     │ engine_loop.py (continuous        │ trivial dequeue.  │
  │ scheduling      │ (simple ZMQ event loop)   │ batching, paged pool, preemption) │ Yours has         │
  │                 │                           │                                   │ block-budget      │
  │                 │                           │                                   │ admission, radix  │
  │                 │                           │                                   │ prefix cache,     │
  │                 │                           │                                   │ swap.             │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ GPU worker /    │                           │ engine/model_executor/worker.py + │ Comparable. Yours │
  │ execution       │ managers/gpu_worker.py    │  task_graph.py                    │  has DAG + CUDA   │
  │                 │                           │                                   │ stream support.   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │                 │                           │                                   │ You're ahead.     │
  │ Memory          │ (none for diffusion)      │ engine/mem_cache/paged_pool.py +  │ SGLang diffusion  │
  │ management      │                           │ radix_cache.py                    │ has no paged      │
  │                 │                           │                                   │ memory.           │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Model forward   │ ComposedPipelineBase +    │ VideoGenerationModel (5 abstract  │ This is the gap.  │
  │ contract        │ PipelineStage             │ methods)                          │                   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Stage           │ stages/encoding.py,       │ Inlined in each model file        │ This is the gap.  │
  │ implementations │ denoising.py, decoding.py │                                   │                   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Pipeline config │ configs/pipeline_configs/ │ (none)                            │ This is the gap.  │
  │  (callbacks)    │                           │                                   │                   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Component       │ loader/component_loaders/ │ Ad-hoc per model                  │ Minor gap         │
  │ loaders         │                           │                                   │                   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │                 │ registry.py               │ models/registry.py                │                   │
  │ Model registry  │ (auto-discover            │ (decorator-based)                 │ Comparable        │
  │                 │ EntryClass)               │                                   │                   │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │ Control-plane / │                           │                                   │ You're ahead.     │
  │  sample         │ (none)                    │ controlplane/                     │ SGLang doesn't    │
  │ tracking        │                           │                                   │ have this.        │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────┼───────────────────┤
  │                 │ Req (200+ field           │ state: dict[str, Any] +           │ Different         │
  │ State bag       │ dataclass)                │ VideoGenerationRequest            │ approach, both    │
  │                 │                           │                                   │ work              │
  └─────────────────┴───────────────────────────┴───────────────────────────────────┴───────────────────┘
  The Honest Assessment