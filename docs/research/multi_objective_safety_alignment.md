# Multi-objective safety alignment: method and recipe audit

Research date: 2026-08-25

## Conclusion

Adding another reward combiner is not the next useful experiment for the Anima
safety-quality problem. DanceGRPO, Safe-RLHF, Flow-Multi, and PAMA all reduce
multiple objectives to one update of the same unconditioned policy parameters.
They can change the selected trade-off and optimization quality, but none
provides a mechanism that prevents a safety gradient from changing benign image
quality.

Shared parameters alone are not an impossibility theorem: a policy explicitly
conditioned on a preference vector or safety mode can represent multiple
behaviors in one network. VRL's current LoRA is not preference-conditioned,
however, and a combiner supplies one aggregate training signal rather than a
runtime control. Adding conditioning would still require a trusted router and
would not replace an output safety decision.

[Rewarded Soups](https://arxiv.org/abs/2306.04488) is the only reviewed method
that separates objectives during training. It trains one expert per reward, but
its static parameter interpolation still produces a global trade-off model. A
hard isolation design must keep adapters independent and route requests; an
output safety gate remains the final decision boundary.

The `fix/wan-hpsv3-recipe-correctness` branch fixes a genuine Wan HPSv3 policy-
gradient error, but that error is not the direct cause of the Anima multi-reward
results. The Anima logs do expose a smaller, matched rollout/replay drift that
must be gated before another long run.

## Evidence boundary

The previously described raw artifacts are not present in the current checkout
or elsewhere under `/home/mingfeiguo`:

- `tasks/wbhy6vo3d.output`
- `scratchpad/paper.txt`

The only surviving local summary is
[`SPRINT_multi_objective_advantage.md`](../sprints/done/SPRINT_multi_objective_advantage.md).
This audit therefore does not treat the interrupted transcript as evidence.
PAMA and every other method below were checked again against primary papers or
author-maintained code.

## What the Wan branch fixes

The old Wan HPSv3 recipe combined `native` denoising, CPS SDE scoring, a fixed
strided 51% timestep subset, ordinary GRPO, and an inherited `1e-4` PPO clip.
On the native path, the rollout first produces a deterministic scheduler action
and then evaluates that action under an SDE proposal. The resulting value is not
the log-density of an action sampled from that proposal, so the policy-gradient
identity does not apply.

The corrected [Wan HPSv3 preset](../../vrl/config/presets/experiment/wan_2_1/online_grpo_hpsv3_fsdp_4x_l40s.yaml)
resolves to:

```yaml
actor:
  optim:
    lr: 1.0e-4
  timestep_selection: sde_window
  timestep_fraction: 1.0

algorithm:
  kind: flash_grpo
  clip_ratio: 1.0e-3
  kl_coef: 0.0

rollout:
  denoise_mode: sde
  noise_level: 1.0
  sde:
    type: flow_grpo
    window_size: 1
    window_range: [0, 10]
```

It also scopes compilation to rollout and adds the reference EMA settings. The
branch does not modify reward combination, advantage estimation, or the trainer
path used to combine reward components.

After merging `main` into the fix branch, its branch-only surface is limited to
the preset, fixed-prompt HPSv3 evaluation and tests, a shared score-summary
module, and the robotics evaluator's use of that module. The shared thin module
is justified because it removes two copies of the same statistical protocol;
its schema/key constants are protocol and reproducibility boundaries rather
than business vocabulary.

## Why the Wan failure does not explain Anima

The weighted, normalized, and Lagrangian Anima runs all resolve to the same
rollout/training shape:

```yaml
algorithm:
  kind: grpo
  clip_ratio: 0.003

actor:
  timestep_fraction: 1.0

rollout:
  denoise_mode: sde
  noise_level: 0.3
  sde:
    type: flow_grpo
```

They therefore do not contain Wan's deterministic-action/SDE-density mismatch,
CPS variance behavior, fixed half-step selection, or `1e-4` clip. Ordinary GRPO
is an intentional Anima choice and should not be changed to FlashGRPO merely to
copy the Wan preset.

There is still a shared confound in the recorded Anima metrics:

| Arm | Pre-update clip | Active clip | Mean max log-prob difference |
|---|---:|---:|---:|
| weighted | 15.40% | 7.05% | 0.0509 |
| normalized | 15.37% | 7.19% | 0.0550 |
| Lagrangian | 15.38% | 7.25% | 0.0492 |

The epoch-zero maximum differences are `0.143`, `0.235`, and `0.131`, all above
the current first-step guard's default `0.01` threshold. About 7% of the active
surrogate was already on a clipped branch before the first optimizer update.

This cannot explain Lagrangian's relative loss to weighted scalarization because
the drift is almost identical across arms. It does prevent describing these
runs as a perfectly clean proof of one universal frontier. Any new long run
must first pass:

```yaml
trainer:
  debug:
    first_step: true
```

A failure should be traced to rollout/replay parity. Merely widening the PPO
clip would hide the symptom rather than establish a correct likelihood ratio.

## Method audit

| Method | Actual mechanism | One shared policy update? | Evidence for antagonist NSFW/quality | Decision for VRL |
|---|---|---:|---|---|
| Fixed scalarization | Weighted reward or advantage sum | Yes | Directly tested locally; coefficient-sensitive | Keep as baseline, not as a safety guarantee |
| DanceGRPO | Normalize each reward's group advantage, then sum | Yes | Paper tests HPS + CLIP, not NSFW | Already represented by `normalized_sum`; no new run |
| Safe-RLHF | Expected-cost constraint with a learned dual variable | Yes | Helpfulness/harmlessness, not image safety | Local Lagrangian arm was dominated; do not retry unchanged |
| Flow-Multi | Pareto-mask dominated samples, then fixed-sum survivors | Yes | Four mostly compatible image objectives | Does not remove the shared-LoRA conflict |
| PAMA | Closed-form combination of scalar advantage surrogates | Yes | Text PPO only; no image/NSFW test | Do not add for the next long run |
| Rewarded Soups | Train one expert per reward, then interpolate weights | No during training; yes after merge | Includes an antagonist aesthetic/NSFW case | Keep experts separate and route; do not rely on a static soup for hard safety |
| Guardrail | Input decision/rewrite plus output classifier | No generator update | Public production architecture evidence | Preferred product boundary; measure risk versus coverage |

### Fixed scalarization was not universally disproved

The old statement that fixed weights were documented to fail across coefficient
sweeps is too broad. [Safe-RLHF](https://arxiv.org/abs/2310.12773) tests seven
static coefficients—`0.01`, `0.5`, `1`, `2`, `5`, `10`, and `100`—in one
helpfulness/harmlessness setting and reports that none matches its constrained
method. That supports coefficient sensitivity in that experiment, not a general
impossibility result.

Rewarded Soups reports that independently trained scalarized MORL policies often
trace a similar Pareto front. More importantly, its text-to-image appendix uses
aesthetic and NSFW-related rewards, observes their antagonism, and reports
scalarized MORL outperforming soup interpolation in that case. This is the
closest external evidence to the Anima problem and directly contradicts a
universal fixed-weight-failure claim.

The precise conclusion is: static scalarization selects one preference, is
coefficient-sensitive, and can miss unsupported points on a non-convex return
set. It does not structurally expand or isolate the policy, but it can still be
the best optimizer among tested methods at a particular operating point.

### DanceGRPO is the existing `normalized_sum`

[DanceGRPO](https://arxiv.org/abs/2505.07818) computes a group-relative
advantage separately for each reward and then sums them:

```text
A_i = sum_k (r_i^k - mean(r^k)) / std(r^k)
```

This is the essential behavior of VRL's `normalized_sum`. It removes raw reward
scale dominance, but the final result remains one gradient applied to one LoRA.
Normalization also changes the effective raw-scale weight each batch to roughly
`weight / group_std`; it is an adaptive scalarization, not a fixed raw utility.

The paper's HPS/CLIP experiment supports scale balancing for compatible image
objectives. It supplies no NSFW, hard-constraint, or strongly antagonist test.

### The local Lagrangian result is narrower than a rejection of Safe-RLHF

Safe-RLHF optimizes expected reward subject to an expected cost constraint and
alternates actor and dual-variable updates. It still feeds a combined reward/
cost advantage to the same actor, and an expectation constraint does not ensure
that every generated sample is safe.

The local run shows that its tested threshold, dual cap, scales, and optimizer
dynamics produced a point dominated by weighted scalarization. A dual variable
pinned at its cap can mean the constraint is infeasible for the current policy,
the cap is too low, scales are mismatched, the actor has not solved the inner
problem, or primal/dual time scales are unsuitable. Therefore the run rejects
this configuration for the product goal; it does not disprove constrained RL as
a mathematical family.

### Flow-Multi is Pareto masking followed by fixed weights

[Flow-Multi](https://doi.org/10.3390/s26041120) masks prompt-group samples that
are Pareto dominated, applies a fixed weighted sum to the survivors, normalizes
survivor scores, assigns zero advantage to dominated samples, and updates one
LoRA. The [author code](https://github.com/2JAE22/Flow-Multi) confirms this
sequence. It is not a weight-free Pareto ranking method.

Its objectives are aesthetic, PickScore, CLIP, and GenEval. With four samples
and four objectives, many samples can be mutually non-dominated; the paper does
not provide the decisive keep-ratio and no-Pareto equal-weight ablation needed
to isolate the filter's contribution. VRL removed its own `pareto_filter`
without a real Anima run, so it should be called untested locally—not failed.
The published mechanism still gives no parameter isolation.

### PAMA is not a convexification of full RLHF

[PAMA](https://arxiv.org/abs/2508.07768) introduces Noon PPO, which removes
negative per-objective advantages, then upper-bounds a parameter-gradient MGDA
problem with a scalar advantage-space surrogate. The simplex subproblem has a
closed-form solution. The full neural policy optimization remains non-convex.

The guarantee is convergence to a Pareto-stationary point under assumptions,
not Pareto optimality or expansion of the attainable frontier. Its 7B evaluation
uses one shared LoRA, has no image or NSFW objective, and reports one seed. In
its own Table 4, PAMA improves harmlessness but produces a lower length reward
than the two reported baselines; that is another trade-off rather than two-axis
dominance.

Discarding negative advantages is also a poor default for an experiment whose
safety signal must suppress bad samples. PAMA could fit technically inside
`GroupAdvantageEstimator`, but the evidence gives no reason to spend another
long run on it.

### Rewarded Soups separates training, not the final merged behavior

Rewarded Soups trains one policy per reward from a common initialization and
interpolates the resulting parameters. Independent training prevents one
objective's gradient from modifying another expert. A static merge loses that
hard boundary and depends on linear mode connectivity; the theory covers only
restricted simplified settings.

For this problem, the useful version is not a soup utility. It is independent
adapters plus deterministic request routing:

```text
prompt -> input policy/classifier -> reject, rewrite, or select adapter
                                      |
                                      v
                              generated image
                                      |
                                      v
                              output NSFW gate
```

Benign requests can use the untouched quality path. Unsafe requests can be
rejected, rewritten, or sent to an independently trained safe behavior path.
The output gate handles classification errors and residual unsafe generations.

### Guardrails shift cost to coverage

The [Sora system card](https://openai.com/index/sora-system-card/) publicly
documents input classifiers, prompt transformations, blocklists, and output
classifiers before display. It supports a layered guardrail design. It does not
establish that production labs never safety-tune their generators or use RL
only for aesthetics; those categorical claims should be removed.

Because an external gate sends no gradient into the generator, it avoids this
specific training-time quality leakage. The total product cost is not zero: it
appears as rejection/coverage, false positives, latency, and possible selection
effects in accepted outputs. Evaluation should therefore report risk versus
coverage, following the selective-prediction framing represented by
[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), alongside
quality on a fixed benign set.

## Recommended sequence

1. Require the first-step parity gate to pass before another expensive run.
2. Keep `weighted_sum_raw` as the simple reference and `normalized_sum` as the
   scale-robust reference. Do not add PAMA or another shared-LoRA combiner now.
3. For a production safety objective, implement input decision/rewrite and an
   output NSFW gate. Report false-positive rate, unsafe pass-through rate,
   coverage, latency, and fixed-set benign quality.
4. For a learned research path, train independent adapters from the same base
   and retain them as routed experts. Do not interpret a static soup as a hard
   safety mechanism.
5. Describe the existing Anima experiment as a relative comparison under one
   matched setup, not proof that every combiner must land on one universal
   frontier.

## Primary sources

- Dai et al., [Safe RLHF](https://arxiv.org/abs/2310.12773).
- Ramé et al., [Rewarded Soups](https://arxiv.org/abs/2306.04488).
- Xue et al., [DanceGRPO](https://arxiv.org/abs/2505.07818).
- Lee and Choi, [Flow-Multi](https://doi.org/10.3390/s26041120) and the
  [author repository](https://github.com/2JAE22/Flow-Multi).
- He and Maghsudi, [PAMA](https://arxiv.org/abs/2508.07768) and the
  [official ECML-PKDD preprint](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_141.pdf).
- Yang et al., [multi-objective policy adaptation and convex coverage sets](https://proceedings.neurips.cc/paper/2019/file/4a46fbfca3f1465a27b210f4bdfe6ab3-Paper.pdf).
- Paternain et al., [Constrained RL Has Zero Duality Gap](https://proceedings.neurips.cc/paper/2019/file/c1aeb6517a1c7f33514f7ff69047e74e-Paper.pdf).
- OpenAI, [Sora System Card](https://openai.com/index/sora-system-card/).
- Geifman and El-Yaniv, [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html).
