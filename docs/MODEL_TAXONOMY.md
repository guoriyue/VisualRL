# Model Taxonomy

VRL classifies the **trainable policy exposed by a registry entry**, not an
entire checkpoint and not its package directory. Hybrid checkpoints can contain
several generation stages with different mathematics; only the stage producing
RL actions determines `PolicySemantics`.

## Typed policy semantics

`vrl.models.families.semantics.PolicySemantics` records four typed facts:

| Field | Values | Meaning |
|---|---|---|
| `generation_regime` | `full_sequence`, `token_autoregressive`, `chunk_autoregressive` | Whether policy generation updates all output positions together, advances one token from a prefix, or advances one temporal chunk from earlier chunks |
| `step_kind` | `denoise`, `token` | The unit advanced by the policy loop |
| `action_distribution` | `continuous`, `categorical` | The policy action space; continuous includes flow/Gaussian transitions |
| `trajectory_layout` | `denoise`, `token`, `multisegment_token` | The replay/reward record shape; multisegment is not a generation regime |

These three labels normalize paper-familiar vocabulary; they are not a formal
three-value standard defined by one paper. [ACDiT](https://arxiv.org/abs/2412.07720)
uses “full-sequence diffusion”, “token-wise autoregression”, and “blockwise
autoregressive”; [MAGI-1](https://arxiv.org/abs/2505.13211) uses chunk-wise
autoregressive denoising. `causal` remains available for an actual dependency or
attention property such as `causal_attention` or
`block_causal_attention`; it is not a generation-regime value. `full_sequence`
also does not mean one-shot generation: a denoise policy still takes multiple
solver steps, with each step updating the full output field.

## Current executable profiles

| Profile | Current families |
|---|---|
| `full_sequence + denoise + continuous + denoise` | SD3.5, Flux, Qwen-Image, SANA, Lumina2, Hunyuan image/video, Mochi, PixArt-Sigma, CogVideoX, Wan, Cosmos variants, Anima, Echo |
| `token_autoregressive + token + categorical + token` | Janus-Pro, Emu3, GLM-Image, LlamaGen |
| `token_autoregressive + token + continuous + token` | NextStep-1 |
| `token_autoregressive + token + categorical + multisegment_token` | Janus-Pro R1 |
| `chunk_autoregressive + denoise + continuous + denoise` | CausVid (trainable Gaussian re-noise); MAGI-1 (generation-only trajectory) |

## Chunk-autoregressive support

The first two integrations deliberately share the typed trajectory/executor
boundary while retaining family-owned temporal schedules:

| Family | Profile | Integration status |
|---|---|---|
| [CausVid](https://github.com/tianweiy/CausVid) ([paper](https://arxiv.org/abs/2412.07772), [weights](https://huggingface.co/tianweiy/CausVid/tree/main/autoregressive_checkpoint)) | `chunk_autoregressive + denoise + continuous + denoise` | Executable rollout and differentiable full-prefix GRPO replay. The released schedule performs three x0 predictions per latent chunk and records its two stochastic Gaussian re-noise transitions. Source and weight revisions are immutable; runtime requires an explicit acknowledgement of their CC BY-NC-SA 4.0 / non-commercial terms. DanceGRPO/Flow-DPPO/GRPO-Guard are rejected until their timestep/trust-region semantics are defined for the two policy axes. Real-weight RL promotion is still pending. |
| [MAGI-1](https://github.com/SandAI-org/MAGI-1) ([paper](https://arxiv.org/abs/2505.13211), [weights](https://huggingface.co/sand-ai/MAGI-1)) | `chunk_autoregressive + denoise + continuous + denoise` | Executable generation through the pinned official 4.5B CLI in an isolated dependency environment. Its 24-video-frame chunks and diagonal multi-chunk denoise schedule are preserved upstream. The release exposes final-video inference under no-grad/inference mode, but no transition likelihood or autograd replay surface, so VRL records no fake actions/log-probs and rejects collector-to-trainer batch construction and replay. |

Both are `FAMILY_REGISTRY` entries. `CausVid` is the RL-capable implementation;
`MAGI-1` is generation-only until upstream exposes an autograd model
and replayable transition distribution. Registration therefore does not imply
identical trainability.

Self-Forcing remains a related candidate only when named by exact executable
variant. Its released chunk-wise DMD policy fits this profile, while the
Self-Forcing family name alone does not: the method also includes a frame-wise
variant with different generation regime.

Two hybrid cases show why this scope matters:

- GLM-Image exposes a trainable token-autoregressive categorical-token prior
  followed by a frozen full-sequence denoise renderer. Its policy semantics are
  `token_autoregressive + token`, not “mixed model”.
- Cosmos3 contains a causal reasoner and a full-sequence vision generator, but
  VRL trains the vision policy stream. Its current entry is
  `full_sequence + denoise`.
- MiniMax-H3 denoises video and audio latents jointly in one packed
  sequence. VRL's action is the video latent (`full_sequence + denoise`); the
  audio rows are a deterministic side stream the family steps itself and
  records for replay, so they never enter the policy ratio.

If one checkpoint supports multiple executable policies, register distinct
entries or variants. Do not mutate a checkpoint-level label based on the
selected algorithm; `janus_pro` and `janus_pro_r1` demonstrate this rule.

## Semantics versus runtime bindings

Semantics must not select unrelated implementation behavior. The registry binds
the executor and gatherer explicitly and separately publishes concrete runtime
capabilities such as torch-compile support, request chunk-size arguments, CuMem
parking, and frozen-component parking.

The former flattened `collector_kind` has been removed. Production code reads
`policy_semantics` for policy classification and the explicit executor,
gatherer, or runtime capability for implementation behavior. A future
chunk-autoregressive denoise entry must therefore register that profile directly; it
cannot pretend to be conventional full-sequence denoise merely to reuse a branch.

## Physical layout

The repository now uses a family-first physical layout. Directories answer
ownership questions; `PolicySemantics` answers classification questions:

```text
vrl/models/
  families/
    registry.py               canonical family wiring and lazy build descriptors
    names.py                  canonical names and external aliases
    semantics.py              task and trainable-policy taxonomy
    <family>/                 checkpoint-, backbone-, and replay-specific code
  steps/{denoise,token}/      shared model contracts, builders, and step helpers

vrl/generation/
  steps/denoise/              denoise config, hot loop, and TeaCache
  steps/token/                token-step protocol
  composition/token_autoregressive/   reusable ordered-prefix state machine
  bindings/full_sequence_denoise/     full-sequence × denoise binding
  bindings/token_autoregressive/      token-autoregressive binding
  bindings/chunk_autoregressive_denoise/ temporal-chunk × denoise contract
  execution/                  step-neutral chunk planning, pipelining, and workers
  ray/                        distributed lifecycle and transport
```

Model and experiment presets are family-first too:
`vrl/config/presets/model/<family>/` and
`vrl/config/presets/experiment/<family>/`. Do not restore `ar/` or `diffusion/`
as routing directories. A family path is stable even if a later executable
variant uses different policy semantics.

There is intentionally no empty `composition/full_sequence` or generalized
`composition/chunk_autoregressive` state machine. The two causal-chunk families
now prove a shared typed result/gather/replay-axis boundary, which lives in
`bindings/chunk_autoregressive_denoise`; their cache lifecycles and denoise
schedules remain family-owned because those algorithms differ materially.
`SampleChunk` is execution batching over requests/samples, not an
autoregressive temporal chunk.

The older `Diffusion*` and `AR*` class names remain where renaming them would add
symbol churn without clarifying ownership. They are implementation API names,
not taxonomy. New imports should use the family-first and axis-based package
paths above.

## Architecture hygiene and non-goals

Thin family `runtime.py` and `runner.py` files stay only when they own a real
model protocol, lazy import, tensor adapter, or state machine. Binding
`__init__.py` facades and gatherers stay because registry import paths and the
driver/worker handoff are protocol boundaries. Cross-family consistency here is
more valuable than reducing a few lines.

The three package-root family modules remain import-light even though they are
co-located with the implementations. Config and generation may import the
registry; they must not import a concrete `<family>` package at module scope.

`FAMILY_REGISTRY` remains a deliberately isolated taxonomy/config table, and
`GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR` remains an import-path protocol value
used across the neutral registry/worker boundary. They are legitimate
module-level constants; do not duplicate them into parallel `SUPPORTED_*`
vocabularies or mix provider capability tables into workflow code.

This reorganization does not introduce a UNet/DiT taxonomy, rename mathematical
algorithm names such as DiffusionNFT, flatten family facades, or create symmetric
modules without a real owner and consumer.
