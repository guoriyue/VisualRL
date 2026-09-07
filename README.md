# visual-rl

RL post-training for visual generative models.

`visual-rl` trains visual generative policies with one layered-config trainer and
one Collector -> Evaluator -> Algorithm loop. Policies are classified by generation
regime (`full_sequence`, `token_autoregressive`, or `chunk_autoregressive`) and
policy step (`denoise` or `token`), rather than forcing every model into an
AR-or-diffusion bucket.
Recipes are marked as validated only after real training runs show optimizer
updates, non-flat rewards, generated artifacts, and changed weights.

## Why It Exists

- **One training loop.** The same online RL loop drives full-sequence denoise and
  token-autoregressive policies instead of keeping a separate training script per model.
- **Layered configs.** Model, sampling, reward, dataset, algorithm, rollout, and
  distributed choices are composed from bundled YAML layers under
  `vrl/config/presets/`.
- **Decoupled rewards.** OCR, aesthetic, CLIP, PickScore, Kling VideoReward, physics,
  and safety-style rewards share the same scoring contract.
- **Validation-first recipes.** Runnable wiring is not treated as a working recipe
  until a real run clears the promotion bar.

## Why not an LLM-RL framework?

RL frameworks built for text LLMs (slime, verl, OpenRLHF, TRL) assume one causal
categorical-token shape. Visual policies span several shapes, so forcing all of
them through that abstraction fights the core contracts:

- **Generation regime varies** — policies update a full latent sequence, emit
  one token from a prefix, or advance one causal temporal chunk while denoising
  inside that chunk. These are distinct time organizations, not AR/diffusion
  synonyms.
- **The policy step varies** — denoise policies record continuous flow/Gaussian
  transitions, while token policies record categorical or continuous token
  actions.
- **The reward is on decoded pixels/video** (VAE decode → image/video reward
  model), not on text.
- **Conditioning is world-model-shaped** (reference image/video, I2V, V2W), not a
  text prefix.

`visual-rl` keeps these policy semantics explicit behind one rollout / replay /
algorithm contract. See [`docs/MODEL_TAXONOMY.md`](docs/MODEL_TAXONOMY.md) for
the typed semantics for the full
positioning and roadmap.

## Status Policy

| Status | Meaning |
| --- | --- |
| ✅ **Validated** | A real run proved optimizer updates, non-flat reward, artifacts, and changed weights. |
| 🧪 **Runnable** | Config, entrypoint, runtime path, and structural tests exist; training quality is not yet proven. |
| 🔌 **Integrated** | Model/runtime wiring and rollout parity exist, but a complete experiment recipe or environment contract is still missing. |
| 🚧 **Planned** | Targeted, not wired end-to-end yet. |

## Supported Models

| Family | Modality | Generation regime / step | Algorithms | Status |
| --- | --- | --- | --- | --- |
| **SD3.5** | text -> image | full_sequence / denoise | GRPO | ✅ OCR GRPO |
| **FLUX** | text -> image | full_sequence / denoise | GRPO-Guard, DanceGRPO, DiffusionNFT, Flow-DPPO | 🧪 Runnable |
| **Qwen-Image** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **SANA** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Lumina-Image-2** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **HunyuanImage-2.1** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **PixArt-Sigma** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **CogVideoX** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **HunyuanVideo** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Mochi-1** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Wan2.1** | text/image -> video | full_sequence / denoise | GRPO, DPO | 🧪 Runnable |
| **Wan2.2** | image -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Cosmos-Predict2** | video -> world | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Cosmos-Predict2.5** | text -> world | full_sequence / denoise | GRPO, DiffusionNFT | 🧪 Runnable |
| **Cosmos-Anima** | text -> image | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Echo** | text -> video | full_sequence / denoise | GRPO | 🧪 Runnable |
| **Janus-Pro** | text -> image | token_autoregressive / token (R1: multisegment) | GRPO, R1-GRPO | 🧪 Runnable |
| **NextStep-1** | text -> image | token_autoregressive / token (continuous action) | GRPO | 🧪 Runnable |
| **Emu3** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **GLM-Image** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **LlamaGen** | text -> image | token_autoregressive / token | Token-GRPO | 🔌 Integrated |
| **Cosmos3** | text -> video | full_sequence / denoise | — | 🔌 Integrated |
| **MiniMax-H3** (Hailuo 3.0) | text -> video (+ audio side stream) | full_sequence / denoise | GRPO (recipe only) | 🔌 Integrated (33B + 32B conditioner, ~144 GB: needs multi-GPU FSDP; CPU tiny-real parity only; diffusers>=0.40) |
| **CausVid** | text -> video | chunk_autoregressive / denoise | GRPO | 🔌 Integrated (real-weight proof pending; non-commercial weights) |
| **MAGI-1 4.5B** | text/image/video -> video | chunk_autoregressive / denoise | generation only | 🔌 Integrated (isolated upstream runtime; no replay API) |

`FAMILY_REGISTRY` is the canonical runtime roster. This table reports user-facing
recipe readiness as well, so a registered family remains Integrated until a complete
experiment config and its dependency contract are committed.

## Supported Algorithms

| Algorithm | Config base |
| --- | --- |
| GRPO | `vrl/config/presets/base/algorithm/grpo.yaml` |
| GRPO-Guard | `vrl/config/presets/base/algorithm/grpo_guard.yaml` |
| DanceGRPO | `vrl/config/presets/base/algorithm/dance_grpo.yaml` |
| DiffusionNFT | `vrl/config/presets/base/algorithm/diffusion_nft.yaml` |
| Flow-DPPO | `vrl/config/presets/base/algorithm/flow_dppo.yaml` |
| Token-GRPO | `vrl/config/presets/base/algorithm/token_grpo{,_multisegment}.yaml` |
| DPO | `vrl/config/presets/base/algorithm/dpo.yaml` |

## Architecture

The online trainer runs:

```text
collect -> evaluate -> advantage -> loss -> backward -> step
```

Core contracts:

- **Collector** produces decoded images/video and records the policy trajectory.
- **Reward** scores the rollout through a common reward interface.
- **Evaluator** replays the trajectory through the current model.
- **Algorithm** consumes trajectory signals and computes the loss.
- **Trainer** applies the update and syncs weights for the next rollout.

## Repository Layout

```text
vrl/
  models/
    families/  family-owned checkpoints, backbones, and replay projections
    steps/     shared denoise/token model contracts and builders
  generation/
    steps/        denoise loop and token-step protocol
    composition/  reusable generation-regime state machines
    bindings/     full-sequence, token-AR, and temporal-chunk denoise assemblies
    execution/ ray/  step-neutral execution and distributed lifecycle
  families/    policy semantics and canonical runtime registry
  rollouts/    collector, orchestration, and replay evaluation
  rewards/     reward objectives, reward models, scoring transport
  algorithms/  GRPO, flow-matching, DPO, DiffusionNFT
  trainers/    online and offline trainers, weight sync, checkpointing
  trajectory/  trajectory build, resolve, and storage
  config/      OmegaConf loading, typed schema, and bundled YAML presets
  nn/ math/ utils/    shared kernels and helpers
  scripts/     training and data preparation entrypoints
datasets/   committed prompt datasets and dataset build scripts
docs/       architecture notes, sprint notes, training examples
third_party/  vendored submodules + editable-install wrappers
```

## Setup

A bare `git clone` does **not** fetch submodules, so run once after cloning:

```bash
make setup
```

That fetches the vendored submodules and editable-installs the **base** package
plus the vendored submodule wrappers. It is the only setup step; re-run it after a
submodule bump. Base install ≠ feature extras — see **Dependencies** below for the
one or two extras your use case needs (the quickstart needs `.[cosmos,ocr]`).

The supported install unit is currently a source checkout, not a standalone
wheel: runtime configs, datasets, reward assets, and vendored backends live beside
the `vrl/` package. CI therefore verifies an editable source install and config
resolution instead of publishing an incomplete wheel artifact.

### Why a setup step (vendored submodules)

Some model/reward backends are upstream code that ships no Python packaging
(JoyAI-Echo's `ltx_*`, videophy's `mplug_owl_video`). They live as git
submodules under `third_party/`. The single thin editable-install wrapper at
`third_party/pyproject.toml` exposes their packages, so `vrl/` contains **no**
`sys.path` injection. `make setup` fetches the submodules, then installs that
wrapper. Adding a vendored dependency only requires extending the wrapper's
explicit source roots and package allowlist under `third_party/`. See
[`third_party/README.md`](third_party/README.md) for the convention.

## Dependencies

`make setup` installs the **base** package only. Each use case adds one or two
optional-dependency groups. Most groups compose in a single `pip install`; the
table and isolation notes below call out environments that must remain separate:

| Use case | Install | Brings (why) |
|---|---|---|
| Full-sequence denoise families (SD3.5 / Flux / Cosmos / Wan / Qwen …) | `.[cosmos]` | diffusers + transformers + peft + torchvision |
| CausVid causal-chunk rollout/replay | `.[cosmos]` + CUDA `flash-attn` | pinned upstream CausVid/Wan runtime; cross-attention requires FlashAttention |
| Token-autoregressive families (Janus-Pro / NextStep) | `.[cosmos]` | transformers/peft model runtime (vLLM accel is separate — see note) |
| OCR reward (the validated quickstart) | `.[ocr]` | paddleocr |
| Video / VLM reward (Kling, VideoScore2, UnifiedReward) | `.[reward]` | transformers≥5.13, qwen-vl-utils, opencv |
| CPU tag inference | `.[detection]` | pinned CPU onnxruntime |
| Dataset prep (video-world, pickapic) | `.[data]` | datasets, pyarrow, av |
| Fixed video-eval suite (VBench) | dedicated `.[videoeval]` environment | vbench 0.1.5 |
| Full-param 8-bit Adam (Cosmos trustworthy-curve recipe) | `.[optim8bit]` | bitsandbytes (int8 Adam state, RL-safe) |
| Tests / lint | `uv sync --group test --group lint` | pytest, ruff |

MAGI-1 is intentionally not a project extra: its official code pins an
incompatible Torch/diffusers/transformers/FlashAttention stack. Create
`third_party/MAGI-1/.venv` from that submodule's requirements and point
`model.python_executable` at it. VRL invokes the audited official CLI in that
environment and records a generation-only trajectory; trainer/replay startup
fails before loading weights. Ray never uploads virtual environments: the
driver resolves a path-like executable to an absolute path before launch, and
every rollout node must provide that path through a shared mount or container
image (or override the preset with its node-local absolute executable). The
official 4.5B process also fixes the DiT to BF16 and T5 to FP32, so its launch
config must use rollout `dtype: bf16`, `outer_autocast: false`, prompt-encoder
`dtype: fp32`, and `float32_precision: ieee`; unsupported precision knobs fail
before source probing or weight download.

CausVid stays in the main model environment, but its pinned Wan cross-attention
calls FlashAttention directly. After installing `.[cosmos]`, install a CUDA-
compatible `flash-attn` build (typically
`uv pip install --no-build-isolation flash-attn`) before resolving the
multi-gigabyte weights. Run `make setup` after
fetching the new submodule so the editable `causvid` package is refreshed. Ray
archives omit nested Git databases; both adapters retain strict
commit/clean-tree checks in a developer checkout and verify an audited SHA256
of their executable source subset in the packaged worker tree.

Example — the SD3.5-OCR quickstart below needs `pip install -e ".[cosmos,ocr]"`.
(The `cosmos` group is the core model-runtime extra and is misnamed for history —
it serves both full-sequence denoise and token-autoregressive families, not just Cosmos.)

> **`ar-vllm` is optional; a separate environment is recommended.** Token-autoregressive families
> run in the main env without it via `sampling.attention_backend=torch_native`;
> `.[ar-vllm]` only adds vLLM's internal paged-attention / blockwise-fp8 kernels.
> vLLM pins its Torch/TorchVision/TorchAudio ABI. The current lock resolves it with
> `.[cosmos]`, but a dedicated venv keeps this large, tightly pinned accelerator
> stack isolated — the repo already ships one at `.venvs/vllm-omni`.

> **Shared-GPU topologies need the `vllm` package importable in the run env.**
> Separate from the `ar-vllm` kernel extra: whenever rollout and reward or trainer
> share a card (`sleep_offload`), physical memory parking is CuMem-only and fails
> loud at policy/model build without vLLM's `CuMemAllocator`. There is no CPU-move
> fallback — the one that used to exist measured 6.2x slower per park cycle and
> hid misconfiguration. `pyproject.toml` still declares `ar-vllm` conflicting with
> `cosmos`/`reward` for ABI reasons, so install it into the run env directly
> (`pip install "vllm>=0.21.0,<0.22" --no-deps`) rather than via the extra.
> `--no-deps` also drops vLLM's own declared dependencies. The internal entry
> points VRL imports (`CuMemAllocator`, the paged-attention kernels, the fp8
> utils) load seven of them that the core/`cosmos`/`reward` closure does not
> already carry, so install those alongside:
> `pip install cbor2 gguf "mistral-common[image]" openai openai-harmony cloudpickle py-cpuinfo`.
> (Measured on vllm 0.21.0 by importing those entry points and intersecting the
> loaded modules with vLLM's `Requires-Dist`; without them the import chain
> stops at `cbor2`, then `gguf`, and `tests/nn/kernels/test_vllm_paged_attention_real_ops.py`
> fails loud instead of skipping.)
> The core environment declares `pyzmq`, which vLLM imports while initializing
> this allocator even though VRL does not use vLLM's distributed serving path.

> **`videoeval` also requires its own environment.** VBench 0.1.5 pins
> `transformers==4.33.2`, while `cosmos` and `reward` require Transformers 5.13+
> APIs. The conflict is declared in `pyproject.toml`, so uv can lock both valid
> environments but rejects an invalid combined sync. Create the evaluation
> environment directly from the repository lock:
>
> ```bash
> UV_PROJECT_ENVIRONMENT=.venvs/videoeval \
>   uv sync --frozen --extra videoeval
> ```

For the reproducible contributor/CI environment, install directly from the
committed lock instead of resolving floating versions. The exact sync prunes
packages outside the main project metadata, so reinstall the source-only vendored
wrapper immediately afterward:

```bash
uv sync --frozen --extra dev --extra cosmos
uv pip install --python .venv/bin/python --no-deps --no-build-isolation --editable third_party
```

## Quickstart

After `make setup`, install the two extras the validated recipe needs and launch
it — SD3.5 text-to-image GRPO with an OCR reward:

```bash
pip install -e ".[cosmos,ocr]"
vrl-train --config experiment/sd3_5/online_grpo_ocr
```

`--config` accepts a bundled config name (no extension) or an absolute YAML path.
Trailing arguments compose independent presets with `+group=option` and set
individual values with OmegaConf dotlist overrides (`vrl-train --help`):

```bash
# shorter smoke run
vrl-train --config experiment/sd3_5/online_grpo_ocr \
    trainer.total_epochs=2 trainer.seed=0
```

For a new combination, use a reward/data-neutral execution recipe rather than
adding another model-by-reward experiment YAML:

Anima has only two execution templates: `online_grpo` (LoRA) and
`online_grpo_fullparam`. Neither selects rewards or data; compose those at launch.

```bash
python -m vrl.scripts.train \
    --config experiment/anima_preview3/online_grpo \
    +reward=ocr +dataset=ocr \
    actor.optim.lr=1e-5 trainer.total_epochs=2 \
    trainer.output_dir=outputs/anima_ocr_composed
```

This demonstrates composition, not a recommended Anima training recipe. The
standard OCR dataset must be available at the paths declared by `dataset/ocr`.
The same arguments work with `python -m vrl.scripts.supervise`. Preset overlays
merge in order; ordinary dotlist values apply last. `+reward=` is additive, not
replacement: multiple different reward presets produce a multi-reward config.
See [runtime configuration composition](docs/CONFIGURATION.md) for independent
judge/rubric selection, evaluation, and historical recipe migration.

Within the first few epochs you should see optimizer steps and a **non-flat**
`reward_mean`. A flat reward is a bug, not a result (see Status Policy). Every
recipe lives under `vrl/config/presets/experiment/` — browse it to see what runs.

## Current Focus

- Promote video recipes only after real training validation.
- Add training recipes for the runtime-verified Emu3, GLM-Image, and LlamaGen families.
- Validate DiffusionNFT and DanceGRPO on more model families.
- Expand multi-card and cross-node online training coverage.

## Docs

- [`docs/MODEL_TAXONOMY.md`](docs/MODEL_TAXONOMY.md) — policy axes, current family profiles, and physical layout.
- [`docs/ADDING_A_MODEL_FAMILY.md`](docs/ADDING_A_MODEL_FAMILY.md) — add a model module, registry descriptor, presets, and contract tests without forking the trainer.
- [`docs/PRECISION.md`](docs/PRECISION.md) — base dtypes, selective FP8 quantization, protected diffusion math, and frozen rollout components.
