# wm-infra Startup Strategy

## One-line positioning

**wm-infra is a serving-first platform for temporal sample production.**

In the current codebase, that means concrete support for Wan 2.2 video generation and Genie-style temporal rollouts.

It starts as infra because that is the wedge.
It becomes durable when it owns the full sample lifecycle:

`generate -> track -> evaluate -> review -> export -> train`

## The market mistake to avoid

Do not position this as "another general inference stack".
That drags the product into direct competition with:
- vLLM
- Triton / TensorRT-LLM / NIM
- cloud model serving platforms
- broad multimodal infra projects

That is the wrong fight.

## The right wedge

Start with the hardest practical problem for video teams:

> Running large-scale video generation jobs reliably and understanding which outputs are actually usable.

That means the initial product should feel excellent at:
- long-running workloads
- throughput under heterogeneous video jobs
- cost observability
- sample traceability
- quality filters
- replay/debugging failed samples

## Why this wins

Users do not really want "an inference engine".
They want:
- usable video samples
- lower cost per accepted sample
- fewer review hours
- reproducibility
- confidence that a dataset was built correctly

That shifts the value from low-level model execution to production control.

## Product surface

### Core serving product
The initial sellable product.

Capabilities:
- multi-backend execution
- queueing and scheduling
- artifact storage integration
- request / sample metadata tracking
- experiment-level reporting
- benchmarkable runtime performance

### Data loop product
The force multiplier.

Capabilities:
- auto-QC
- failure taxonomy
- pairwise review
- acceptance policies
- export to training and eval formats

### Training loop product
The long-term lock-in.

Capabilities:
- scorer training datasets
- reranker datasets
- accepted-sample curation
- hard-case mining
- LoRA / adapter export pathways

## ICP: ideal customer profile

Best early users:
- teams generating internal video training/eval data
- model labs iterating on video quality and controllability
- robotics / simulation teams producing rollout data
- applied AI teams needing repeatable synthetic video generation

Less attractive early users:
- hobby image generation users
- generic chatbot serving customers
- teams who only need an OpenAI-style API wrapper

## North-star metrics

### Runtime metrics
- accepted samples / GPU hour
- cost per accepted sample
- queue latency for long jobs
- backend utilization
- artifact IO overhead

### Data quality metrics
- auto-QC precision / recall
- human-review agreement
- accepted-sample yield
- failure-tag coverage
- reproducibility rate

### Product metrics
- time to onboard a new backend
- time to debug a bad sample
- time to produce a benchmark/eval set
- time to export a trainable manifest

## Moat

The moat is not raw serving speed by itself.
The moat is:
- video-native job model
- sample-level lineage
- accumulated failure taxonomy
- human + auto review loop
- export compatibility with downstream training systems

## Build order

### Stage 1 — Runtime credibility
Ship a clean runtime with measurable performance and an honest API.

### Stage 2 — Sample registry
Every output becomes a tracked sample with lineage and artifacts.

### Stage 3 — Evaluation loop
Add scoring, failure tags, review queues, and acceptance decisions.

### Stage 4 — Training loop
Make accepted data exportable for scorer/ranker/LoRA training.

## Things not to do early

- Do not build a giant UI before the data model is stable.
- Do not promise support for every model family.
- Do not overfit to chatbot-style online serving metrics.
- Do not try to train a giant base video model on day one.

## The message to customers

Not:
- "We host your video model."

Better:
- "We help you produce, evaluate, and ship video datasets reliably."

Best:
- "We turn video generation from an artisanal workflow into a reproducible data factory."
