# SPRINT: Multi-objective advantage combination

状态：**done（seam 2026-08-23 落地，实验与 combiner 裁决 2026-08-25 收口；2026-09-06 归档）**。
下文 "Experiment status" 的第 3 项（per-prompt objective masking）与第 4 项（非 GRPO 算法接线）
是**带触发条件的搁置项**，不是待办：前者的唯一消费者是 anima 质量-vs-安全线（等外部评审），
后者按原文只在某个 NFT recipe 真的需要多目标时才做。再次扫描 planned 时不应把它们当增量。

## What landed (2026-08-23)

A pluggable **advantage-combination seam** so multiple reward components are no
longer forced through a single summed scalar. Motivated by the Anima
quality-vs-safety runs: summing a low-variance quality reward with a
high-variance NSFW penalty let the penalty dominate the group-relative
advantage and collapse quality (see [[nsfw-trigger-structurally-immovable]],
[[codex-judge-same-frontier]]).

- `vrl/algorithms/advantages.py` — `GroupAdvantageEstimator` binds the selected
  strategy, normalization settings, and resolved reward weights to one algorithm
  instance. The public `group_relative_advantages(...)` function remains the
  shared stateless math boundary. Two strategies:
  - `weighted_sum_raw` (default, legacy): `normalize(Σ wᵢ·rᵢ)` — one scalar,
    high-variance component dominates. It consumes the authoritative weighted
    total already produced by `MultiReward` rather than rebuilding it.
  - `normalized_sum` (DanceGRPO-style, arxiv 2505.07818): normalize EACH
  component to a per-group advantage first, then `Σ wᵢ·advᵢ`, then clamp — no
    reward dominates by scale/variance. Component advantages are combined before
    the one final clamp.
- `vrl/algorithms/grpo/continuous.py` — `GroupAdvantageConfig` exposes only
  `advantage_combine`; reward weights are runtime data, not public algorithm
  config. Its `build_estimator(...)` method is the single mapping from config
  fields to the runtime estimator. `GRPO` owns the estimator and exposes the
  optional component-advantage capability inherited by FlashGRPO / FlowDPPO /
  GRPOGuard / TokenGRPO / MultiSegmentTokenGRPO.
- `vrl/scripts/common/factory.py` — supplies resolved `reward.components`
  weights to the config-owned estimator construction, then injects the result
  into the algorithm. The reward section remains the only source of component
  weights.
- `vrl/algorithms/base.py` and `vrl/trainers/online/trainer.py` — an explicit
  runtime-checkable capability replaces strategy-string inspection and
  `hasattr`. Capable algorithms receive both the authoritative weighted total
  and raw component observations; other algorithms keep the scalar path.
- Schema: allowed `algorithm.*` keys are auto-derived from the dataclass fields,
  so `algorithm.advantage_combine` needs no schema edit.
- Test: `tests/algorithms/test_advantage_combine.py` (proves normalized_sum
  de-dominates a 10x-variance component and clamps only after aggregation).

Enable per recipe: `algorithm.advantage_combine=normalized_sum`.

## Experiment status and next steps

1. **Real Anima validation — completed.** `normalized_sum` was run against the
   codex/pickscore + nsfw recipe and compared with `weighted_sum_raw` on the
   fixed anatomy/safety set. It selected a different operating point but did
   not establish dominance. Keep `weighted_sum_raw` as the default. Any repeat
   must first pass the first-step rollout/replay parity gate described below.

2. **More aggregation strategies — evaluated and removed (2026-08-25).**
   `lagrangian` was evaluated on the Anima codexqa+nsfw run and was strictly
   dominated by the `weighted_sum_raw` baseline. `pareto_filter` was implemented
   and removed without a real Anima run; it must be described as untested, not
   as an empirical failure. The external Flow-Multi method is also not
   weight-free: it Pareto-masks samples and then applies a fixed weighted sum to
   the survivors.

   - Lagrangian ck14 external eval: sharpness **8.86** (WORST of all — below
     base 13.4, normalized_sum 11.18, weighted_sum 9.48) AND trigger **12/24**
     (worse than weighted_sum's 10/24). The dual variable λ pinned at its cap
     (2.0) for the entire run and never decayed, so the intended "release
     pressure once safe" behavior never occurred. This explains the effective
     fixed-penalty behavior, but a capped λ alone cannot distinguish an
     infeasible constraint from a low cap, reward scaling, insufficient primal
     optimization, or mismatched primal/dual time scales.
   - In these matched runs, `weighted_sum_raw`, `normalized_sum`, and
     `lagrangian` produced points consistent with the same observed
     quality/safety trade-off. This is evidence about the tested setup, not a
     theorem that every combiner must reach the same frontier. A combiner can
     change which point is found and whether optimization reaches a
     non-dominated point, but none of these methods structurally isolates the
     two objectives inside one unconditioned LoRA.
   - All three logs also show nearly identical pre-update active clipping
     (7.05%, 7.19%, and 7.25%). That matched drift does not explain the relative
     ranking, but it weakens any absolute frontier claim. A future long run must
     pass `trainer.debug.first_step=true` before its likelihood-ratio results are
     accepted.
   - Removed cleanly: no `lagrangian`/`pareto_filter` in `_STRATEGIES`; the λ
     runtime-state/checkpoint/logging coupling into the trainer is gone; only
     `weighted_sum_raw` (default) and `normalized_sum` remain on the seam.
     `pareto_filter` was removed untested-on-the-real-run. Re-add it from git
     history only for a workload whose sample-selection behavior is the actual
     research question.

   The real lever is not another combiner but **structural decoupling** or, more
   to the point, how production models actually handle this — see §Production.

   - `max` / `min` / `softmin` combiners for worst-objective emphasis (not built).

3. **Per-prompt objective masking** (cheap, orthogonal): on anatomy prompts zero
   the nsfw component, on explicit prompts zero the quality component, so each
   group optimizes a single objective. Implement as a combiner variant that
   reads a per-sample component mask (needs the prompt-type tag threaded into
   reward_components extras).

4. **Wire the non-GRPO algorithms.** The component capability lives on the GRPO
   family; add it to `diffusion_nft` (and any future group-advantage algorithm)
   only if those recipes need multi-objective combination.

5. **Independent adapters / Rewarded Soups** (separate workflow, not this seam):
   train one LoRA per reward from a shared init. Static interpolation supplies a
   global trade-off dial but no hard safety boundary; the closest published
   aesthetic+NSFW experiment reports scalarized MORL outperforming the soup.
   Keep adapters separate and route requests when parameter isolation matters.

## Production safety boundary (2026-08-25)

Why the tested RL combiners hit the same wall: they update one unconditioned
LoRA, so the safety and quality objectives act on the same parameters. The
observed interference is real, but public evidence does not justify claiming
that every shared model must degrade identically or that specific labs never use
safety tuning. Public system cards instead support a layered design:

1. **Training-data curation and model-side safety work** reduce unsafe priors.
2. **Inference guardrails** classify or transform prompts before generation and
   scan outputs before display.
3. **Reward-driven tuning** can improve aesthetics and prompt alignment, but the
   public sources reviewed here do not establish that it is reserved only for
   those goals.

Implication for this repo's Anima-safety goal: the production path is an
**inference guardrail** (input+output classifiers — the Falconsai NSFW model is
already wired as a reward and can be reused as an output gate), optionally plus
a **data-filtered SFT** on safe-only high-quality anime. A separate gate does not
change generator weights, but its product cost is not zero: it moves to
rejection/coverage, false positives, latency, and accepted-output selection.
Measure risk versus coverage as well as quality on a fixed benign set.

## Deep-research backing

The interrupted transcript could not be recovered, so the methods were checked
again against primary sources. The corrected result is:

- Safe-RLHF uses an expected-cost Lagrangian and reports one seven-coefficient
  static sweep where its constrained method performs better.
- DanceGRPO is the existing normalize-each-reward-then-sum strategy; it handles
  scale imbalance, not structural gradient conflict.
- Flow-Multi Pareto-filters samples and then still uses fixed scalarization on a
  shared LoRA.
- PAMA solves a closed-form scalar advantage surrogate, not the full non-convex
  policy problem, and still updates one shared policy.
- Rewarded Soups is the only reviewed method that separates parameters during
  training, but a static merge remains a trade-off rather than a hard boundary.

Fixed scalarization is coefficient-sensitive and cannot isolate shared
parameters, but the reviewed literature does **not** document its universal
failure. Rewarded Soups reports scalarized MORL matching it broadly and beating
it in the closest antagonist aesthetic+NSFW experiment. The full evidence and
the Wan/Anima recipe audit are in
[`docs/research/multi_objective_safety_alignment.md`](../../research/multi_objective_safety_alignment.md).
