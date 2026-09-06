# SPRINT: 给 Anima RL 找一个真正可训的目标

Color/light dataset consolidation (2026-09-04): `datasets/anima/color_light`
is now the only active color/light dataset, selected with
`+dataset=anima_color_light_ddrl`. It keeps the former v2 training manifest
(256 prompts) and formal evaluation manifest (96 prompts) byte-identical.
The former v1 evaluation manifest is retained as `development_prompts.jsonl`
(32 prompts), not mixed into formal evaluation. The 64 v1 training prompts
already existed in v2 and were deduplicated by retaining that existing manifest.
All 384 distinct prompts survive in one dataset with three disjoint splits.
Old 64-image reference/latent assets remain historical; this merge did not
generate the complete 256-image reference set. Original files are in
`/home/mingfeiguo/Desktop/vrl-color-light-merge-zMa45M/before.tar.gz`.
Versioned paths in historical sections below are evidence, not current inputs.

Configuration migration (2026-09-04): historical model/reward/dataset experiment
names and commands below record past runs. The combination-only Anima presets
have been retired in favor of [runtime composition](../CONFIGURATION.md).
Use each retained run's `resolved_config.yaml` for exact reproduction; current
neutral defaults are not substitutes for the original experiment settings.

Final Anima composition migration (2026-09-04): the remaining eight
model/reward experiment combinations were retired, including aesthetic and
safety variants previously retained for fixture/inheritance compatibility.
Only the reward/data-neutral LoRA and full-parameter execution templates remain.
Callers now compose independent presets at launch. Three anime rubric names
changed from `codex_image_qa_anima_*` to `codex_image_qa_anime_*`; scoring text
and policy values did not change. Historical commands below use their original
names and remain evidence, not current launch instructions. Original YAMLs and
call sites are in
`/home/mingfeiguo/Desktop/vrl-anima-composition-archive-VIWBEO/before.tar.gz`;
see that archive directory's `README.md` for the exact inventory.

Online-critic retirement (2026-09-04): the Anima-bound canary launcher and both
online critic reward registrations/presets were removed. The canary hard-coded
one experiment/dataset, and the production deployment path unconditionally
rejected launch because its qualification producers were absent. The offline
network, annotation, training, calibration, and benchmark work were initially retained;
historical mentions of a completed online canary boundary below are superseded
by this retirement. Shared CountGD, grounded OCR, and tag-adherence rewards were
retained because their scoring inputs do not depend on Anima generator identity.

Offline-person-research retirement (2026-09-04): the dedicated critic network,
annotation/data tools, training/calibration/qualification code, tests, protocol
assets, person-count/integrity datasets, and rejected Luna person-count rubric
have now been archived outside the active repository. They are not required by
ordinary Anima training or shared reward components. The findings and historical
commands below remain unchanged as experiment evidence, not current entrypoints.
Retired code/config/text-data paths below refer to members in
`/home/mingfeiguo/Desktop/vrl-person-research-archive-jB20H6/before.tar.gz`.
Moved datasets, media, and probes retain their repository-relative layout under
that archive directory's `files/`; its `README.md` lists the exact inventory.
These retired paths no longer resolve in this checkout. CountGD, grounded OCR, tag adherence,
the shared Codex exact-count scoring mechanism, and reusable exact-count evaluation
remain available without these research datasets.

Evaluation-experiment retirement (2026-09-04): the Anima OCR qualification
reporter, its dedicated dataset generators, five OCR dataset presets and their
data, the rejected requested-token grounding rubric, the anatomy probe/report
chain, and the old objective-C tag-adherence evaluation entrypoint have been
archived outside the active repository. The shared tag reward and its separate
NSFW datasets, configs, and run evidence are unchanged. Shared OCR and grounded-OCR
rewards, text-matching logic, and the standard `dataset/ocr` preset remain available;
reusable checkpoint, fixed-panel, and exact-count evaluation are also retained.
Historical findings, commands, and links below are preserved as evidence, not
current entrypoints. Retired source/config paths identify members in
`/home/mingfeiguo/Desktop/vrl-eval-cleanup-archive-3AI2zQ/before.tar.gz`;
moved assets retain their repository-relative layout under that archive's `files/`.
See the [evaluation cleanup inventory](/home/mingfeiguo/Desktop/vrl-eval-cleanup-archive-3AI2zQ/README.md)
for the exact scope and recovery instructions.

状态：**exploration done（2026-08-28 ~ 08-29）**。本文是实验日志，不是计划。
Sections 0--9 preserve that exploration as it happened. Section 10 records the
subsequent objective qualification on 2026-08-30 and supersedes Section 8's
choice of the first trainable target. Section 10.13 records the completed formal
v2 endpoint on 2026-08-31 and supersedes the pending gate in Section 10.12.
Section 11 records the later exact-person-count attempt and supersedes the old
count-headroom claim below. The weakness is real; the Luna reward failed an
exact same-pixel retest. Section 11.8 records the later CountGD reward that
passed the frozen base-distribution gate, while Section 11.10 records its
post-training policy-shift failure and the completed checkpoint-12 endpoint.
Section 11.11 freezes that failure as a 140-image qualification benchmark and
records the rejection of fixed anime-detector/pose voting.
Section 11.14 records the independent 384-image critic-pilot screen and rejects
critic training because the fragment class and review agreement are inadequate.
Section 11.15 records the early stop of detector-disagreement mining and the
qualification of close multi-person interaction as the replacement source axis.
Section 11.17 records the append-only source wave, the still-unmet independent
human-review gate, and the now-complete bounded-canary execution boundary.
姊妹文档：[SPRINT_anima_rl_5090.md](SPRINT_anima_rl_5090.md)（同一条实验线的前史）、
[done/SPRINT_multi_objective_advantage.md](done/SPRINT_multi_objective_advantage.md)（combiner 那一段）。

触发：quality/safety 的多目标路线全部失败后，问题变成「那到底该训什么」。
本轮把「先量空间、再投 GPU」当成硬纪律，全部结论都有本机实测数字。

---

## 0. 结论先行

1. **仓库里所有 2026-08-28 之前的 LoRA GRPO run，重要性比率都是坏的。** rollout 与
   replay 的 log-prob 最大差 **0.203**（限 0.01）。根因是 `torch_compile` 与
   generation/replay 批形状**两个因素叠加**，必须同时修。症状是首步
   `pre_update_active_clip_fraction` 恒为 **7.0–7.3%**。修复后为 **0.00%**。
2. **codex/luna 判官不能当在线奖励。** 在质量、光影、手部三种任务、九种配置下测了
   重测一致性：噪声与信号同量级。判官输出最多作为离线标注建议或
   auxiliary pretraining signal；未经 Anima-domain blind annotation and adjudication，
   不能当作训练真值。
3. **Anima base 在本轮已测的大部分语义轴上已经很强**：标签遵循 98%、自然语言遵循
   93%、否定 100%。早期自动测得的计数 88–100% 不能代表严格人数正确率；Section 11
   的新人工审计只有 **36/64（56.25%）** 严格命中 4/5 人。结构解剖与精确人数都是
   真实弱点。CountGD 曾通过冻结分布资格门，但训练后的策略分布审计证明它不能单独作
   在线奖励（Sections 11.8--11.10）；结构解剖仍缺少合格奖励，文字渲染见 Section 10。
4. **修好平价后的干净对照 run 得到了诚实的零结果**：held-out 无显著提升
   （z=0.77 / 0.52），但**不再崩塌**——灾难性失败模式已解决。
5. **动漫域手部定位器可用**（imgutils 100% 检出），photo 域的 MediaPipe 不可用（0%）。
   这只解决裁切与粗粒度计数；它不判断手部结构是否正确，不能单独构成解剖奖励
   （见 Section 10）。
6. **The corrected 80-update v2 OCR run is a qualified null result.** The final
   checkpoint scored **-0.007080** versus base on the frozen OCR grid, with a
   prompt-cluster 95% CI of **[-0.026753, +0.005268]**. Luna and blinded visual
   review also found no general-quality gain. The checkpoint must not replace
   base; Section 10.13 records the complete decision.
7. **The exact-person-count canary is invalid as an improvement result.** On
   identical pixels, a clean reward rerun changed `base 15/64 -> checkpoint
   19/64` into `base 17/64 -> checkpoint 15/64`; binary reward agreement was
   only 90.6%/87.5%. The staged continuation was stopped at the last durable
   checkpoint 4. That conclusion applies to the Luna reward, not to the target.
8. **Pinned CountGD passed the frozen base-distribution gate but failed the
   post-training policy-shift gate.** Its frozen 64-image audit had `TP=27,
   FP=0, FN=9, TN=28`, 75% recall, 8/8 active groups, and exact repeatability.
   Checkpoint 3 was neutral. Extending the same run to checkpoint 12 moved the
   automatic metric `44/128 -> 50/128`, but its prompt-cluster 95% CI still
   crossed zero at **[-0.031250, +0.117188]**. More importantly, blinded human
   review of every reward flip found unchanged human exact-count status in
   19/20 cells; the remaining apparent gain erased distant people into malformed
   fragments. This is detector instability/reward hacking, not an Anima
   improvement. Checkpoint 12 is rejected and the run must not be extended.
9. **Fixed detector voting does not repair the reward.** Intersecting CountGD
   with anime head/person and RTMW signals still left four false policy reward
   flips while reducing frozen recall from 0.750 to 0.556. The 140 human-labeled
   images are now a standalone qualification-only benchmark; the next candidate
   must learn count and person integrity from separate training data.
10. **The first critic corpus failed its data gate before training.** A blind
    image-level screen of 384 paired base/checkpoint-12 images found only 6
    fragments from 4 prompt groups, below the predeclared minimum of 32. On 44
    crowded images with a separate blind Codex re-review, both reviews called
    only one same image fragmented (`kappa=0.097`). No critic or RL canary is
    authorized from this pool; the next asset must add targeted hard negatives
    and instance boxes.
11. **Detector disagreement was the wrong source axis; close interaction is the
    first measured replacement.** The hard-negative v2 review stopped after 308
    unique images with only two provisional fragments, both from the uniform
    stratum and none from 117 detector-mined/control images. A separate blinded
    64-image interaction probe produced three confirmed non-confounded malformed
    people among 60 eligible images (`5.0%` versus `0.65%`, a `7.7x` enrichment).
    This authorizes scale-up and box annotation, not critic training. A frozen
    96-source/32-eval prompt corpus now defines that next data collection.

---

## 1. 平价 bug：所有 LoRA GRPO run 的梯度都受污染

`trainer.debug.first_step=true` 从来没有在 LoRA recipe 里开过。开了之后立刻失败：

```
RuntimeError: full first-step log-prob parity failed before optimizer step:
finite=True, max_abs_diff=0.202691, limit=0.01
```

GRPO 在 `ppo_epochs=1` 下整个成立的前提是「比率 ≈ 1」。差 0.203 意味着比率
e^0.203 ≈ **1.22，偏离 22%**，而 `clip_ratio` 只有 0.003。这个数值漂移被当作策略改进
喂进了梯度。

### 隔离矩阵（5 个小 run，各 1 次优化器更新）

| generation batch | replay batch | torch_compile | max_abs_diff | |
|---|---|---|---|---|
| 16 | 1（默认） | on | **0.2027** | ✗ 原配置 |
| 4 | 1 | on | 0.0857 | ✗ |
| 4 | 1 | off | 0.0665 | ✗ 光关编译不够 |
| 1 | 1 | on | 0.0910 | ✗ 光对齐形状不够 |
| **1** | **1** | **off** | — | **✓ 唯一通过** |

- `actor.samples_per_replay_batch` 默认就是 1，所以原配置是 **generation 16 / replay 1
  的形状错配**。
- 试图把 replay 提到 4 或 16 以匹配 generation：**两者都 CUDA OOM**（replay 要对每个
  去噪步反传）。因此 replay 被显存钉死在 1，generation 必须跟着降到 1。
- 代价：生成从 1.33 s/样本 变成 **2.36 s/样本**，**每次更新的样本数不受影响**。
- 错误信息建议的另一条路 `precision_correction.recompute_old_logprob='on'`
  在本仓**未实现**，构造即抛 `NotImplementedError`。

### 独立佐证：首步裁剪率

| run | ep0 | 全程均值 |
|---|---|---|
| **修复后 luna_anchored** | **0.00%** | **0.00%** |
| luna fullparam（本就 compile off + batch 1） | 0.00% | 0.00% |
| codexqa_nsfw (weighted) | 6.62% | 7.05% |
| codexqa_nsfw_normsum | 6.82% | 7.19% |
| codexqa_nsfw_lagrangian | 6.95% | 7.25% |

**受污染的三个 run 在第一步就有约 7% 的样本被裁剪**——那不是学到了什么，是数值漂移。
它们关于「质量/安全前沿」的结论仍有肉眼与 held-out 证据支撑，但任何依赖
likelihood-ratio 的结论都要打折。

**落地**：`experiment/anima_preview3/online_grpo_codex_quality_luna_anchored` 把三项修复
写进注释与配置。**今后任何 GRPO run 都必须带 `trainer.debug.first_step=true`，并在相信
结果之前确认 `pre_update_active_clip_fraction == 0.00%`。**

---

## 2. 判官重测一致性：九种配置，全部不合格

方法：对**同一批已存 rollout** 重新打分，与训练时记录的分数（或第二遍打分）比较。
GRPO 的 advantage 符号由**组内排序**决定，所以组内 Spearman 是关键量。

### 2.1 质量 rubric

| 数据 | 每次调用图数 | 组内 rank ρ | 噪声 σ | 组内展布 σ | 真实方差占比 |
|---|---|---|---|---|---|
| luna fullparam（8 样本组） | 4 | +0.558 | 0.077 | 0.107 | 49% |
| luna fullparam（8 样本组） | **8** | **+0.750** | 0.055 | 0.109 | 74% |
| anchored（16 样本组） | 4 | +0.531 ~ +0.637 | 0.030–0.041 | 0.096–0.101 | 82–91% |
| anchored（16 样本组） | **8** | **+0.714** | 0.040 | 0.118 | 88% |
| anchored（16 样本组） | 16 | +0.598 | 0.043 | 0.091 | 78% |

跨 112 张图的总体：Pearson r=0.824，但**组内** rank ρ 仅 **+0.506**，
**14 组里有 2 组 ρ ≤ 0**（排序完全洗牌）——约 14% 的 prompt 组梯度方向近乎随机。

### 2.2 光影 rubric

| 每次调用图数 | 组内 rank ρ | 噪声 σ | 真实占比 |
|---|---|---|---|
| 4 | **+0.782** | 0.043 | 81% |
| 8 | +0.499 | 0.061 | 35% |

**分块最优值跟 rubric 走，不能通用**：质量 rubric 是 8 格最好，光影 rubric 是 4 格最好，
16 格在两者上都更差（4×4×512 = 2048px 大概超过判官的有效图像分辨率）。

### 2.3 组内展布不是区别所在

曾假设「光影任务组内分数差更大所以更好训」。用已存训练分数大样本核对，**证伪**：

| rubric | 组数 | 组内 std | 组内极差 | ≥0.8 |
|---|---|---|---|---|
| 光影 | 4 | 0.0868 | 0.243 | 69% |
| 质量（anchored n=16） | 160 | 0.0826 | 0.295 | 81% |
| 质量（fullparam n=8） | 40 | 0.1065 | 0.335 | 55% |

---

## 3. 修好平价后的对照 run：诚实的零结果

`online_grpo_codex_quality_luna_anchored`，10 epoch，三项修复齐备：
compile off + generation/replay 批均为 1（平价）、`kl_coef=0.004`（LoRA 靠
`disable_adapter`，零显存开销）、**256 样本/更新**（失败的 luna run 是 32）、
supervisor 的 grad_norm 尖峰早停。

训练侧全程健康：`pre_update_active_clip` **恒为 0.00%**，grad_norm 最高 0.033
（早停线 0.5 从未触发），KL 缓升到 0.0004。奖励 0.783 → 0.858(ep1) → 0.821(ep9)，
**在 0.81–0.86 稳住，没有崩塌**。

Held-out（44 prompt × 2 seed，盲测）：

| 臂 | luna | vs base | 配对胜/平/负 |
|---|---|---|---|
| base | 0.8539 | — | — |
| checkpoint-5 | 0.8628 | +0.0090（z=+0.77） | 13 / 26 / 5 |
| checkpoint-10 | 0.8600 | +0.0061（z=+0.52） | 11 / 26 / 7 |

评测自判 `heldout_luna_gain_supported: false`。

**校准数字坐实这是噪声**：同一个 base、**同一批 44 条 prompt**（manifest sha256 一致），
上一次评测得 **0.8801**，这次 **0.8539**——**仅换 seed 就漂 0.026，是所谓提升的 3 倍**。

对比失败的 fullparam run（ck10 held-out **0.391**，43/44 输，图像撕裂）：
**灾难性崩塌被解决了，但奖励本身带不来提升。** 44 个 prompt 里 **26 个完全平局**，
判官在 60% 的 prompt 上无法区分。

---

## 4. Headroom 普查：Anima 在语义层面已经很强

全部用**客观检测器**测（WD14 tagger + imgutils 人物检测），零判官参与。

| 轴 | base 表现 | 有空间？ |
|---|---|---|
| 整体质量（luna held-out，176 张） | 均分 0.867，<0.7 仅 **2%** | ❌ |
| 标签 prompt 遵循（32 张） | 召回 **0.980**，81% 完美 | ❌ |
| 自然语言遵循（20 张） | 召回 **0.943**，80% 完美 | ❌ |
| 计数（含 3 人） | **88–100%** | ❌ |
| 否定/缺失（12 prompt，22 个禁止标签） | **100% 遵守** | ❌ |
| **结构解剖（手）** | **0% 完美**，6–38% 崩坏 | ✅ **当时这轮普查唯一** |

按 bucket 拆分（held-out base，每类 16 张）：

| bucket | 均分 | 最低 | <0.7 |
|---|---|---|---|
| **hand_focus** | 0.808 | **0.28** | **12%** |
| standing_side | 0.849 | 0.31 | 6% |
| 其余 9 类 | 0.844–0.905 | ≥0.71 | **0%** |

### 两个测量陷阱（都实际踩过）

1. **tagger 词表**：首次算出召回 0.731，是因为 `feet_visible` / `both_hands_visible` /
   `arms_visible` **不在 WD14 的 7786 词表里**（数据集自造标签），被计为 100% 失败。
   用真实词表过滤后是 0.980。**任何 tag-based 奖励必须先校验词表。**
2. **小样本**：否定一开始测出 67%（n=3），扩到 n=12 后是 **100%**。

### 数据集缺陷

`datasets/danbooru/anatomy` 的 `hand_focus` bucket（2118 条）内容其实是**全身镜头**
（`full body, standing, both hands visible...`），手只有约 30px。**模型没有足够像素去学，
判官也没有足够像素去看。** 真要训手必须重建成上半身/近景取景。

---

## 5. FRELAN 参考图集：判官方向相反，路线否决

用户提供 `FRELAN_rednote_images.zip`（652 张 NAI 动漫图，137 帖，1080×1566）。
思路是当作 `reference_listwise` 的目标锚。

- 流程验证通过：中心裁切到 512（上偏 1/4 避免切脸）→ luna 生成 caption（8/8 可用，
  40–70 词）→ base 按 caption 生成 → 判官同框对比。
- **结果与直觉相反**：

| rubric | 参考图均分 | base 均分 | 参考胜场 |
|---|---|---|---|
| 现有 color_light | 0.842 | **0.900** | **1/8** |
| 新写的「插画完成度」 | 0.832 | 0.820 | 2/8 |

**判官认为 base 比参考图更好。** 根因是 color_light rubric 明文写着「更亮/更暗/更饱和/
更戏剧化**永远不构成更好**」——这条防 reward-hacking 的规则做得太成功，
**奖励干净平淡、惩罚有氛围的艺术表现**，恰好与该参考集的价值相反。
换 rubric 只修好了最糟的一例（平涂糊手图 0.92 → 0.46，gap +0.35），整体仍是 base ≈ 参考。

**方法论偏差**：参考图 1080×1566 压到 512 损失厚涂笔触，base 原生 512 无损，
对比本身对参考不利。

**结论**：该路线是**风格迁移**（干净赛璐璐 → 厚涂插画），不是「改善光影」，
且现有判官无法为它提供正确梯度。已否决。
（使用条款注记：README 写明「可自用、作参考；请勿擅自用于盈利」。）

---

## 6. 手部专项：目标成立，判官不成立，检测器成立

### 6.1 空间（三种取景，每张打两遍取平均）

| 取景 | 手的像素 | 均分 | broken(≤0.4) | **perfect(≥0.95)** |
|---|---|---|---|---|
| 全身 + 4 图拼图 | ~30px | 0.672 | 6% | **0%** |
| 全身 + 单图 | ~30px | 0.577 | 22% | **0%** |
| 近景 + 单图 | ~300px | 0.542 | **38%** | **0%** |
| 裁切放大 + 单图 | 512px | 0.625 | 17% | 4% |

**base 的手几乎从不完全正确。** 这是普查中唯一有大量空间的轴。

### 6.2 判官：五种配置全部不合格，且与分辨率无关

| 配置 | 手的像素 | test-retest r | 噪声 σ |
|---|---|---|---|
| 全身 + 拼图 | ~30px | 0.111 | 0.138 |
| 全身 + 单图 | ~30px | 0.331 | 0.132 |
| 近景 + 单图 | ~300px | 0.404 | 0.146 |
| **裁切到 512px 全画幅** | 512px | **0.424** | 0.137 |
| 近景 + 两两比较 | ~300px | 8/12 一致（**67%**，随机 50%），4/12 换位翻转 | — |

**噪声底线恒定在 ~0.135，与手的像素数无关**——把手裁到占满 512px 画面仍只有 r=0.42。
**分辨率假说被证伪：判官就是判不了动漫手部解剖。**

### 6.3 检测器：动漫域可用，photo 域不可用

| 图集 | **imgutils.detect_hands** | MediaPipe HandLandmarker |
|---|---|---|
| 全身 | **100%**（1.81 只/张，框占边长 9.0%） | 62%（疑似误检） |
| **上半身** | **100%**（1.58 只/张，框占 **49.1%**） | **0%** |
| 近景 | 92%（框占 61.9%） | **0%** |
| FRELAN 参考 | 88% | 25% |

MediaPipe 在我们真正关心的取景上完全失效（动漫域外）；**imgutils 全取景可用**，
且上半身的手部框占画面边长 49%，**足以裁切放大到 512px**。

**检测器自带一个免费的客观信号**：24 张上半身图检出手数为 `1,2,3,5,6`
（计数 13/7/2/1/1），而 prompt 只可能要 1 或 2 只手。**检出 5–6 只 = 重复/融合畸形**
（已肉眼确认：手指严重畸形导致检测器在指簇上重复触发）。

---

## 7. 本轮落地到代码的东西

| 文件 | 改动 |
|---|---|
| `vrl/scripts/supervise.py` | 新增 `max_grad_norm` 健康门（与 `min_grad_norm` 对称，默认 `inf` 禁用）。顺带修复 `health_verdict.json` 中 `json.dumps(math.inf)` 产生非法 JSON 的问题（禁用时序列化为 `null`）。+5 测试，`tests/scripts` 550 全绿。 |
| `.../online_grpo_codex_quality_luna_anchored.yaml` | 新建。平价三件套 + KL 锚定 + 256 样本/更新 + `images_per_call: 8`，注释里带实测数字。 |
| `vrl/algorithms/advantages.py` 等 | **移除** `lagrangian` 与 `pareto_filter`（见 multi_objective sprint）。seam 只保留 `weighted_sum_raw` + `normalized_sum`。The shared standardizer now obtains mean and variance from one `torch.var_mean` reduction so constant decimal rewards cannot create fake advantages. |

一次性验证产物（`*_probe`，位于 session scratchpad，结论已记录于本文，可删）：
`luna_reliability_probe.py`、`luna_grouping_probe.py`、`parity_probe{,2,3}.sh`、
`ref_caption_probe.py`、`ref_gap_probe.py`、`hand_probe*.py`、`hand_pairwise_probe.py`、
`mp_hand_probe.py`、`anime_hand_probe.py`、`crop_hands.py`、`det_signal.py`、
`tag_recall*.py`、`build_nl_probe.py`、`nl_score*.py`、隔离环境 `handenv/`。

The hand, natural-language, and FRELAN probe manifests were one-shot inputs,
not recipe dependencies. On 2026-09-04 they were retired together with their
generated images and hand crops. Their results remain in Sections 4--6; the
original FRELAN archive is unchanged. See the worktree-cleanup record below
for the recoverable archive location.

---

## 8. 当时的下一步（未开工，已被 Section 10 取代）

This was the anatomy plan at the end of the 2026-08-29 sweep. It remains useful
qualification work, but it is not the first RL target because no available
anatomy scorer passed reward validation.

1. **把结构解剖做成客观奖励**。两段式：
   (a) 免费的检测器信号——检出手数 vs prompt 预期数量、检测置信度；
   (b) 把 luna 一次性蒸馏成手部裁切图的分类器（RLHF 的做法是蒸馏，不是每步重复调用；
   实测要靠重复打分把信噪比拉到质量 run 的 62% 需要 **k≈8** 次/张，成本不可接受）。
2. **先确认「解剖」是宽目标还是窄目标**：用同一套方法测脚与复杂姿势下的四肢，
   看是否同样是 0% 完美的模式。是则值得投入，否则手部只是局部修补。
3. **防 reward hacking**：手数奖励可被「把手藏起来」规避。需在奖励中要求检出预期数量的
   手，并保留 luna 作为事后护栏 + 定期肉眼检查。
4. **重建手部训练集**：现有 `hand_focus` 是全身取景，须换成上半身/近景，
   模板已在当时的上半身 probe 验证可用（该一次性输入现已归档）
   （注意：`hands large in frame` + `sharp line art` 会把模型推向单色漫画极端特写，
   正确写法是 `waist-up portrait, medium shot, full color, cel shading`）。

## 9. 可复用的纪律（本轮反复验证）

- **投 GPU 之前先量 headroom。** 本轮靠这条否决了三个方向（质量、光影/FRELAN、语义遵循），
  每次只花几十次 codex 调用与几分钟 GPU。
- **任何奖励先测 test-retest。** 判官在四个任务上都比预期噪声大，且噪声与直觉不符
  （光影 rubric 比质量 rubric 稳，手部最差）。
- **训练奖励会骗人。** 本轮每一个 run 的训练曲线都与 held-out 反向或无关。
  结论必须过 held-out 盲测。
- **held-out 也要校准。** 同模型同 prompt 换 seed 漂 0.026；小于这个量级的差异不算数。
- **小样本会骗人。** 否定 67%(n=3) → 100%(n=12)；hand_focus 12%(n=16) 是全部普查中
  唯一在放大后依然成立的弱点。

---

## 10. Qualified first target: prompt-conditioned text rendering

**Status (2026-08-31): complete and rejected for promotion.** The first
20-update OCR run had no clear held-out gain. The later v1 long-text full pass
was stopped after visual audit exposed floating-carrier reward hacking. The v2
flat-carrier correction then passed its canary and completed all 80 training
epochs after the constant-decimal advantage bug was fixed. Its frozen OCR,
carrier-integrity, Luna, and blinded visual gates found no supported gain.
Section 10.13 records the final decision; no training reward is treated as an
improvement claim.

### 10.1 Why OCR comes before anatomy

Anatomy is still an important Anima weakness, but the current tools do not
measure anatomical correctness:

- `imgutils.detect_hands` supplies boxes, counts, and confidence. It can expose
  gross duplication, but it cannot tell a correct five-finger hand from a fused
  or malformed one. A count reward can also be maximized by hiding or cropping
  hands.
- MediaPipe failed on the relevant anime close-ups, while the luna hand rubric
  remained noisy even after crops filled the input image.
- Two additional one-shot candidates were rejected. HandCraft produced almost
  no usable detections on Anima images. MUSIQ was run on all 62 extracted hand
  crops, but ranked crop scale, sharpness, and composition rather than missing
  or fused fingers. It is a generic image-quality metric, not an anatomy metric.

OCR instead has a typed target string and deterministic edit-similarity math
conditional on a fixed PaddleOCR transcript/runtime. The base probe already
showed both headroom and usable within-group signal:
8 held-out prompts x 8 rollouts scored **0.547842** mean, with **44/64** nonzero,
**21/64** exact, and all **8/8** prompt groups active. That is enough to test the
RL path without training against an unqualified judge.

The rejected one-shot assets are not long-term dependencies. Their conclusions
are recorded above so the following temporary paths can be removed together:
`/tmp/vrl-hand-iqa-probe`, `/tmp/vrl-pyiqa-probe-env`,
`/tmp/vrl-pyiqa-cache`, `/tmp/vrl-handcraft-probe-env`,
`/tmp/vrl-handcraft-yolo-2d8a52f.pt`, and `/tmp/Ultralytics`.

### 10.2 Dataset and leakage boundary

`datasets/anima/ocr_short_v1` is deterministically derived from the vendored
Flow-GRPO OCR corpus. The recorded upstream source is
[Flow-GRPO commit 879042c](https://github.com/yifan123/flow_grpo/commit/879042cf5707f8b90daa98d147d7deac2317c5da).
That revision is inherited source metadata, not a fresh upstream attestation by
the builder. The auditable inputs are the vendored train/test SHA-256 digests
and the derived manifest digests in `dataset_spec.json`.

| split | normalized target length | rows | purpose |
|---|---:|---:|---|
| `train_stage1.jsonl` | 3--6 | 144 | equal-count initial curriculum (36 per length) |
| `train_short.jsonl` | 3--8 | 937 | later curriculum expansion |
| `heldout_short.jsonl` | 3--8 | 71 | frozen evaluation |

The derivation keeps one prompt per normalized target, accepts only ASCII
letters/digits/spaces, balances target lengths, and records source row, hashes,
selection seed, and source revision. More importantly, the builder excludes
**every** normalized target present anywhere in the vendored upstream test
split from both training manifests. Separately, `dataset_spec.json` reports
overlap **0** between each training curriculum and the selected 71-row held-out
manifest.

### 10.3 Exact reward and mandatory replay parity

The training recipe disables Flow-GRPO's image-only substring shortcut and uses
full normalized Levenshtein similarity. This prevents outputs such as `D007`
from receiving full credit for target `007`. The expected text comes from typed
manifest metadata; the reward never reparses the natural-language prompt.
Reward debug output stores every rollout, including zero-score samples, with
the recognized text and score for reward-hacking review.

Replay correctness is a trainer invariant, not an optional diagnostics flag.
`trainer.replay_parity.max_abs_logprob_diff=1e-6` gates the first real optimizer
update even when `debug.first_step=false`. Both the two-update canary and the
GAS1 probe passed with `max_abs_diff=0.0` and
`pre_update_active_clip_fraction=0.0`.

This recipe is the repository's first-order flow-matching GRPO path with ratio
clipping. It is not classical TRPO: there is no natural-gradient solve,
conjugate-gradient step, or line-search rollback in this path.

### 10.4 Canary result: healthy training, no descriptive held-out gain

The canary used four prompts per update and eight rollouts per prompt. Its
training rewards were `0.2902` and `0.7068`, but the pre-protocol descriptive
paired probe rejects that apparent improvement under its own generation policy:

| protocol | base | checkpoint-2 | delta | prompt-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| 32 held-out prompts x 2, seed 20260930 | 0.601228 | 0.591704 | **-0.009524** | [-0.059375, +0.047117] |

Exact outputs changed **27 -> 26**; nonzero outputs changed **45 -> 48**.
Image wins/ties/losses were **8/44/12**. With numerical-zero tolerance, prompt
wins/ties/losses were **5/16/11**; the raw report counted one `-5.55e-17`
roundoff as a loss. This v1 probe also used the generator's non-empty convenience
negative prompt, whereas training and the formal protocol use an empty negative
prompt. Two updates therefore establish execution health only. They do not
support an OCR improvement claim or constitute formal qualification.

### 10.5 GAS1 bottleneck probe

Changing `actor.gradient_accumulation_steps` from 4 to 1 was tested as a way to
collapse repeated model handoffs. A console-timed probe, whose raw console log
was not retained as a long-term artifact, measured consistently from recipe
start to checkpoint: GAS4 took **287.86 s** and GAS1 took **282.80 s**, only
**5.06 s (1.76%)** faster. The same observation put an all-zero-group handoff at
about **1.50 s**, while a group that actually trains added about **60 s** of
replay/backward work. Treat these as recorded operational observations rather
than independently reproducible benchmark artifacts. They locate training, not
handoff count, as the dominant cost.

GAS1 is therefore not a meaningful speed optimization. The formal recipe keeps
GAS4 to reduce host-resident rollout state. The outer `/usr/bin/time` value for
the GAS1 probe was 301.64 s including load/setup; it is intentionally not mixed
with the start-to-checkpoint comparison above.

The first formal launch also exposed why the documented Anima parking bound is
part of the execution contract. With the generic 256 MiB residual allowance,
the third GAS4 handoff failed before any optimizer step: CuMem still released
the full 5.66 GiB model pool, but device-wide residual reached 291.38 MiB above
the one-time pre-load baseline. The second and third handoffs had the same
`loaded - residual` bytes, so this is not evidence of a shrinking/leaked model
pool. The failed attempt is retained under
`outputs/probes/anima_ocr_short_stage1_parking256_failure_20260830`.

The restart uses `VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB=1024`, the existing bounded
Anima/RTX 5090 protocol already validated for a five-update run. This does not
disable parking validation: a leaked 5.66 GiB model pool remains far outside the
1 GiB bound. GAS1 alone is not a fix because its second update would still use
the same build-time device-wide baseline after replay/backward initialized
additional process-lifetime CUDA pages.

### 10.6 First formal result: healthy but no held-out improvement

The completed run was a 20-update, rank-32 LoRA stage-1 curriculum run, not
full-parameter Anima training. Training remained numerically healthy, but the
registered paired evaluation did not show a clear OCR gain:

| arm | mean OCR | exact | nonzero |
|---|---:|---:|---:|
| base | 0.618494 | 61 / 142 | 108 / 142 |
| checkpoint-20 | 0.625394 | 66 / 142 | 105 / 142 |

The paired mean delta was **+0.006900**, with a prompt-cluster bootstrap 95% CI
of **[-0.034633, +0.050050]**. Image wins/ties/losses were **23 / 94 / 25**;
the report therefore emitted `clear_improvement: false` and
`clear_regression: false`. This is an honest null result, not evidence that the
objective is untrainable.

The independent 32-prompt general-quality guard also found no Luna gain and no
edge-detail, saturation, or color-diversity regression. It did detect a small
systematic brightness increase of **+0.003897** (95% CI
**[+0.002383, +0.005702]**), which is why later runs still require a general
quality check rather than relying on OCR alone.

The registered promotion contract requires the complete 71-prompt held-out
manifest with two ordered samples from one prompt-batch generator stream,
base seed `20260930`, 512 x 512 output, 20 denoising steps, CFG `4.5`, BF16,
an empty negative prompt, no row limit, and `checkpoint-final` at step 20. The
two samples are correlated draws from the same prompt seed, not two independent
seeds; the uncertainty calculation therefore bootstraps prompt clusters.

Only the 20-update budget and launch config were fixed before execution. The
promotion contract was registered after the run started and after checkpoint 5,
but before the final checkpoint and before any 71 x 2 formal evaluation. It is
therefore a fixed mid-run promotion gate, not a fully prospective preregistration.

This budget is not a full pass over the 144-row stage-1 manifest. The current
`random_without_replacement` sampler draws a fresh permutation for each
four-prompt update, so its guarantee is batch-local rather than run-wide. An
exact replay from the implementation's implicit default seed 0 contains 80
prompt selections but only 54 unique rows. The committed recipe now states
`trainer.seed: 0` explicitly for future launches; the active run's resolved
config predates that clarification. A negative result therefore applies to this
fixed update budget and must not be described as failure after complete
stage-1 coverage.

The run also predates the final audit-artifact patch by several minutes: it
started at 11:48 PDT, while the reward debug writer was changed at 11:54 PDT.
The launched reward actor consequently used the launch-time debug text layout. The
edit added typed sample provenance and did not change exact OCR score math, so
it does not confound the paired base/final evaluation. However, checkpoint
metadata does not bind the training source tree, resolved config, train-manifest
hash, or reward-policy hash. The formal report can bind evaluation inputs,
generation policy/runtime, adapter hashes, checkpoint progress, and base/final
pairing; it cannot cryptographically prove the LoRA's complete training lineage.

Evidence paths:

- [dataset_spec.json](../../datasets/anima/ocr_short_v1/dataset_spec.json)
- [dataset builder](../../vrl/scripts/data/anima_ocr.py)
- [qualification protocol](../../datasets/anima/ocr_short_v1/qualification_protocol.json)
- [formal recipe](../../vrl/config/presets/experiment/anima_preview3/online_grpo_ocr_short_stage1_20update.yaml)
- [completed resolved config](../../outputs/anima_ocr_short_grpo_stage1_20update/resolved_config.yaml)
- [completed formal metrics](../../outputs/anima_ocr_short_grpo_stage1_20update/metrics.csv)
- [completed paired report](../../outputs/probes/anima_ocr_short_heldout71x2_final_20260830/ocr_exact_qualification_v4/report.json)
- [completed general-quality guard](../../outputs/anima_ocr_short_grpo_stage1_20update/heldout_quality_v1_luna_final/summary.json)
- [base OCR summary](../../outputs/probes/anima_ocr_short_base_20260830/ocr_exact_summary.json)
- [canary metrics](../../outputs/anima_ocr_short_grpo_canary/metrics.csv)
- [canary replay gate](../../outputs/anima_ocr_short_grpo_canary/training_debug.jsonl)
- [checkpoint-2 paired held-out report](../../outputs/probes/anima_ocr_short_heldout32_ckpt2_20260830/ocr_exact_report_v1/report.json)
- [GAS1 probe metrics](../../outputs/probes/anima_ocr_short_accum1_probe_20260830/metrics.csv)
- [GAS1 replay gate](../../outputs/probes/anima_ocr_short_accum1_probe_20260830/training_debug.jsonl)
- [256 MiB parking failure verdict](../../outputs/probes/anima_ocr_short_stage1_parking256_failure_20260830/run_verdict.json)

### 10.7 Why the grounded Luna guard was not promoted

The next experiment tried to prevent an OCR shortcut: the generated word had
to belong to the requested physical carrier instead of appearing as a floating
caption, collage tile, or unrelated duplicate. The reward was deliberately
atomic (`spelling_score * binary_guard`) so a high spelling score could not
compensate for a failed semantic guard. The corrected two-update canary was
numerically healthy, but a 64-image human gold audit found that the guard itself
was not reliable enough. The positive class below is **visually valid**; the
human gold contains 30 valid and 34 invalid images:

| guard | TP / FP / FN / TN | precision | recall | accuracy |
|---|---:|---:|---:|---:|
| current mirrored AND | 25 / 2 / 5 / 32 | 92.6% | 83.3% | 89.1% |
| fact-schema v2 safe aggregation | 26 / 1 / 4 / 33 | 96.3% | 86.7% | 92.2% |

The fact-schema v2 safe aggregation means: at least two of three complete
verdicts pass, all three calls say the carrier relation is exact, and all three
say synthetic overlay/composite is absent. This definition matters because the
same raw calls give different results under ordinary complete-verdict majority
voting.

The current mirrored guard accepted a floating-text overlay and a vertically
composited scene in both orders. It also rejected five valid images. More
importantly, it killed a valid `Skywalker` image with OCR score 0.889 while
leaving a weaker 0.111 image as the only positive winner in that prompt group.
Final pass/fail agreement across forward and reverse order was only **49/64
(76.6%)**. A three-order fact-schema probe improved the aggregate numbers, but
still made a stable false acceptance and had already been iterated on the same
64-image canary. It would also add another Luna call per image.

This route is therefore retained as an experimental framework boundary, not a
formal training reward. The large semantic rubric stays in YAML; the thin
`GroundedOCRReward` registry adapter and atomic reward owner stay because they
are real framework/invariant boundaries. The fact-schema probe is not promoted
to Python, and Luna remains an offline audit tool.

### 10.8 v1 correction: one requested visible token

v1 replacement 把 online semantic judge 从 reward path 移除，并把 text
contract 收窄为一个 requested token：每个 prompt 要求一个 front-facing text
carrier，且不能出现其他 readable text。OCR 部分因此可以确定性执行：选出
最佳完整 OCR line、计算 normalized Levenshtein similarity，并拒绝额外的高
置信度 alphanumeric line。但这一步只让 **text contract** 可判定，没有让
physical-carrier semantics 可判定；Section 10.9 正是这个遗漏的实测后果。

`datasets/anima/ocr_single_text_long_v1` contains:

| split | rows | unique targets | target lengths | templates |
|---|---:|---:|---|---:|
| train | 320 | 80 | 7--10, balanced | 8 |
| internal held-out | 40 | 40 | 7--10, balanced | 4 |
| external upstream test | 20 | 20 | 7--10 | 4 |

All three target sets are disjoint. The train manifest is arranged into 80
four-row windows, each containing one target of length 7, 8, 9, and 10. The
`sequential_window` sampler therefore consumes rows 0--319 exactly once in 80
updates. This fixes the earlier 20-update run's incomplete 54-row unique
coverage without inventing more optimizer steps than one manifest pass.

The reward now pins the locally qualified `ppocrv6_medium` engine as part of
its semantics. The two-update canary produced:

| update | reward mean | reward std | grad norm | advantage zero | parity / clip |
|---|---:|---:|---:|---:|---|
| 0 | 0.8252 | 0.2558 | 0.003842 | 0.0% | all zero |
| 1 | 0.8889 | 0.1885 | 0.002299 | 28.1% | all zero |

All four first-update groups had usable within-group signal. One second-update
group (`WHIZBANG`) was saturated, while the other three still ranked spelling
errors. Human review of those 64 saved rollouts found no collage, floating
caption, or carrier ambiguity; group winners were the more accurately spelled
images. This qualified execution only. It did **not** qualify all eight carrier
types against reward hacking, and the larger v1 run below exposed exactly that
blind spot.

### 10.9 Rejected v1 full pass: healthy mechanics, wrong objective

在 v1 formal optimizer run 启动前，internal held-out base grid 已冻结为
**40 prompts x 3 ordered samples = 120 images**，base seed 为 `20260930`。
generation config 和 120 个样本都通过 v5 qualification validator；protocol
绑定 manifest hash、base model identity、sampling geometry、BF16、empty
negative prompt、PP-OCRv6、exclusive-line policy、candidate label 和 final
global step 80。这套 frozen evaluation 仍然有效，v2 不改它。

同一 policy 下的 base rescore 为 **0.934315 mean**、**89/120 exact**、
**120/120 nonzero**。40 个 prompt cluster 中只有 14 个有 seed-dependent
score，仍有 **31/120** 个可见 spelling failure。也就是说，目标有真实
headroom，但最大可得 mean gain 很小，最终必须做 paired uncertainty gate。

v1 formal 本身在执行机制上完全健康。它写完 epoch `0--12` 共 **13 rows**
后由操作者发送 `SIGINT`；`run_verdict.json` 正确记录 `signal=2`，最后一个
完整 checkpoint 是 `checkpoint-10`。这 13 rows 的 reward mean 为
`0.8449--0.9861`，grad norm 为 `0.001408--0.009030`；replay parity、
pre-update clipping、PPO clipping、TIS clipping 和 mismatch KL 全部为零。
因此这不是 replay bug、数值发散或 optimizer 没有更新。

判废来自对前 10 updates、**320 张**持久化 rollout 的逐图审查。bottle
template 的图经常根本没有 bottle，而是把 token 放在漂浮矩形 label、
capsule 或尖片上；32 张该类样本中有 **17/32** 出现这种广义 carrier
collapse，其中 **14/17** 仍拿到 reward `1.0`。更严格的 definite failure
有 7 张，其中 6 张 reward `1.0`，另一张为 `0.889`。

根因是 objective 不完整：OCR reward 读取 target、recognized lines、edit
distance 和 extra-line veto，却从不读取 prompt 所要求的 physical carrier。
LoRA 因而找到了比“画好瓶子上的字”更容易的路径——“画一个 OCR 容易读的
漂浮 label”。v1 run 是**健康地优化了错误目标**，所以已经判废，绝不
resume，也绝不拿 `checkpoint-10` warm-start v2。它的 output 只保留为
reward-hacking audit evidence。

### 10.10 v2 data correction: flat carriers and pinned vocabulary

v2 没有再加一个不可靠的 Luna semantic guard，而是收缩未被 reward 覆盖的
生成自由度。八个 train template 全部改成低语义、正面、平坦的矩形载体：
enamel sign、paper poster、chalkboard sign、canvas placard、wooden plaque、
metal nameplate、cardstock card 和 acrylic panel。每个 prompt 只要求一个
token 和一个 surface；这不是新增了 carrier correctness scorer，而是让
carrier 本身不再成为高风险、未计分的复杂对象。

训练词来自 pinned [ESDB/SCOWL `rel-2026.02.25`](https://github.com/en-wl/wordlist/releases/tag/rel-2026.02.25)，
commit `7e99edab8e32f9f9ea2b15f249ca8d4d67237410`。固定 extraction policy
产出 35,685 个 7--10 letter lowercase words；再从不改变 frozen internal
eval 排名的安全 rank window 中人工复核并选出 **80 个**常用 target，每个
长度 20 个。每个 target 配四个 scene variant，得到 **320 rows**；每个
连续四行仍为 7、8、9、10 字符，供 `sequential_window` 一次完整消费。

v2 自带的 `eval_internal.jsonl` 只是 regeneration audit mirror，除了
`metadata.dataset` 外与 v1 row semantics 一致。formal training 和 paired
qualification 明确继续引用 byte-frozen v1 eval manifest，SHA-256 为
`09535f784e60f40065aedfcb0df3664d59dc360c7a21860a73a0eb35d855cf83`。
train、frozen eval 和 upstream external test 的 normalized target overlap
均为零。

本次 launch 的 data 边界如下；这些 hash 记录的是实际文件，不靠文档手抄
target list 充当 source of truth：

| artifact | SHA-256 |
|---|---|
| pinned ESDB `words.txt` | `481f04a74dfc3ebe6832eeb7a3db6e5f3e09aeec1fba971d6dfb803ec270033c` |
| reviewed target asset | `df4f3394f8eaf601e4c9d4599fe4ecea3f2009abaafd09ac1f008044ff542d06` |
| augmented source train | `fa10911eee992887cee6f541e76510d847b4717545c7105e940ed06a8cd2a497` |
| prompt templates | `5534a889a4cc489ed775b9713cc27fa9ff46bfe423a151b6c49aaf1c004c8aed` |
| v2 train manifest | `2a660e613d8d03ad493684e5ce96a75c5c929980d07d893a655d8fcc908afe54` |
| v2 eval regeneration mirror | `5860720709af108dafb8f5abea4bfd61cd9b4dc3947eaa3b5f002cbb52bc51e9` |
| v2 upstream external test | `eb771daff4032a0c06afee0eda50429c23df6ea5a87b272e3525324c12254c72` |
| v2 dataset spec | `c81373da8adbd34c58ee28f6853bf03cfa5aa17be3dd1feb4e8834dbbd26d4c9` |
| frozen v1 eval used by formal | `09535f784e60f40065aedfcb0df3664d59dc360c7a21860a73a0eb35d855cf83` |
| corrected frozen v1 qualification protocol (v6) | `3b37742fa05a45930178f0b6e1c134713c6fcd9b485aedeb846c8457ec8105db` |
| corrected frozen v1 external protocol (v6) | `c88a6029cfb9f5868b2f34ec204f5f3e3340113fc763f25184141947012aef3d` |

config 和对应 regression tests 也单独冻结记录：

| artifact | SHA-256 |
|---|---|
| v2 dataset config | `eefef646887b840524cbb3b815d7ce1cac253c14535c32bde5cb867c1298fd95` |
| v2 canary config | `93d3b6e86d4257ade3464229b221d2fa288b05fa9b1d842191b698e189947597` |
| v2 formal config used by failed first launch | `617bfeb9488fe82a0b84d49ca4fca22413f8a4e3910878c0ec915d111a37487d` |
| failed first-launch resolved config | `172458d5d6f58276bca9a924084463a785be8756af2327beff1ea86dbf8222af` |
| clean-relaunch formal config | `a44d0f4aab562f63361d7831208e08a8b139ad635e1a095e6447dd241a3c01d2` |
| clean-relaunch resolved config | `14af84b32a7d6c4390a272c87dbf12b219e1e3b99a3b3abb245397cbc8152472` |
| dataset builder/manifest tests | `61b32dda76b70180b7cb9fc7870b9db699635745f21396cc8f5dc942fd79410c` |
| prompt dataset config tests | `a73ddc8c377dfdb1c1801e00a88cacee57d4f4fa86c3f29aa0bf9f08c46cd55a` |
| all-experiment config test | `6e8f557fe108352763dad8ef2ae597f66d16084a01f7be47efacd663f6067520` |

相关三组 tests 实跑结果为 **75 passed**。测试覆盖 80 个 reviewed target 与
train manifest 精确相等、每个 target 出现四次、8 个 template 各 40 行、
每个 update 的 7/8/9/10 ordering、v2/v1 eval semantic equality、frozen eval
hash，以及 dataset/experiment config load and validation。

### 10.11 v2 canary qualified; first formal attempt quarantined

v2 canary 先跑两步，再从 `checkpoint-2` 续一小步以覆盖全部 template；它
总计 **3 updates、12 prompt groups、96 images**。对全部 96 张逐图复核后，
8/8 flat-carrier templates 都通过，没有重现 v1 的 floating-label、capsule
substitution 或 complex-carrier collapse。三步数值如下：

| update | reward mean | reward std | pre-clip grad norm | advantage zero | parity / clip |
|---|---:|---:|---:|---:|---|
| 0 | 0.9903 | 0.0200 | 0.004303 | 50.0% | all zero |
| 1 | 0.9512 | 0.0421 | 0.003403 | 25.0% | all zero |
| 2 | 0.8559 | 0.0461 | 0.658817 | 50.0% | all zero |

reward 没有完全饱和。例如 `bilingual` 八张的 mean 为 **0.611111**，范围
`0.444444--0.888889`，仍能提供明确的 group-relative ordering。第三步
记录的 `0.658817` 是 clip 前的 grad norm；`actor.max_norm=0.1` 已执行真实
gradient clipping。adapter 的相邻 checkpoint L2 delta 为
`1 -> 2: 0.022606`、`2 -> 3: 0.020771`，第三步并没有产生异常大的参数
jump。初始 launch 和 resume 的 replay parity gate 都是 `max_abs_diff=0.0`。

final canary checkpoint 的 metadata 为 schema v2，`global_step`、
`trainer_step`、`completed_epoch`、`next_epoch` 全为 3，checkpoint 文件为
`330937519` bytes。它证明 v2 目标能健康执行且保留 reward headroom，仍然
不是 held-out improvement claim。

The first 80-update v2 formal attempt started independently from the untouched
base and did not inherit either the rejected v1 checkpoint or the canary
adapter. It completed seven metric rows and 224 complete rollout images, then
failed during the first group of epoch 7 before publishing any checkpoint. A
review of the first 224 images found no missing carrier, floating-label collapse,
or global image collapse. That supports the v2 objective correction, but it is
not a formal training result.

The immediate Ray symptom was not an OOM or native actor crash. The worker
released the complete 5.66 GiB CuMem model pool, stopped before publishing the
usual physical-parking snapshot, and was then intentionally killed by driver
cleanup. Kernel and systemd logs contain no OOM event. The automatic retry
deleted the first verdict and did not persist its console traceback, so the
exact residual value is unrecoverable. The remaining evidence supports a
high-confidence diagnosis that the launch omitted the existing Anima/RTX 5090
`VRL_CUDA_RESIDUAL_BYTES_LIMIT_MIB=1024` contract and hit the generic 256 MiB
device-wide residual bound. This is the same failure mode already isolated in
section 10.5; it is not evidence of a shrinking or leaked model pool.

The supervisor then made a fresh second attempt in the same directory because
no complete checkpoint existed. That start replaced `metrics.csv`,
`resolved_config.yaml`, and the first verdict, while OCR debug records continued
from `000232`. The operator stopped it after record `000255`. The mixed directory
is quarantined at
`outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass_failed_mixed_attempts_20260830`
and must never be resumed or qualified.

The clean relaunch keeps the same rank-32 LoRA, PP-OCRv6 reward, learning rate
`1e-5`, KL coefficient `0.004`, four prompts x eight rollouts, 20 denoising
steps, seed, manifests, and 80 sequential updates. Only the operational protocol
changes: the documented 1 GiB bounded residual gate is explicit, checkpoints
publish every five updates, the supervisor permits one attempt, and combined
stdout/stderr is persisted outside the run directory. The 1 GiB bound remains
far below the 5.66 GiB model pool and therefore still fails closed on a real
parking leak. The relaunch is not described as successful until its final
checkpoint and all downstream gates exist.

The clean single-attempt relaunch started from base on 2026-08-30. Its console
stream is persisted at
`outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass.supervisor-attempt-01.log`;
the output directory was absent at launch, and the resolved config contains no
`resume_from`. This records launch provenance only, not a success claim.

The first five clean updates were also compared against the first attempt's
corresponding 160 saved rollouts. Targets matched `160/160`, but PNG hashes
matched `0/160`: `trainer.seed=0` fixes prompt and trainer-side ordering, while
the current generation worker intentionally draws fresh per-batch entropy. The
clean run is therefore an independent stochastic repetition, not a bitwise
crash replay. Full visual review found `0/160` missing carriers and `0/160`
global/pixel collapses in both arms. Mean OCR reward was `0.921300` clean versus
`0.924871` archived, and 20 group means had Pearson correlation `0.939`. This
supports protocol consistency without pretending that stochastic training
rollouts are reproducible images; promotion remains bound to the explicitly
seeded frozen paired evaluation.

### 10.12 Constant decimal rewards exposed a shared advantage bug

The clean relaunch reached epoch 15 before monitoring found an algorithmic
invariant violation at epoch 14. The `meditating` group contained eight exact
Python rewards of `0.9`, so a group-relative estimator must return eight zero
advantages. The previous implementation computed mean and standard deviation in
separate float32 reductions:

```text
stored reward = 0.8999999761581421
mean          = 0.9000000357627869
std           = 5.960464477539063e-08
advantage     = -1.0 for every sample
```

Because the false standard deviation exceeded `eps=1e-8`, clamping did not
protect the group. Epoch 14 therefore recorded `advantage_mean=-0.25`,
`policy_loss=0.5`, and `grad_norm=0.002584`; the bad gradient completed an
optimizer step. This was not an OOM, replay mismatch, or logging artifact.
`checkpoint-15` is contaminated and must not be evaluated or resumed.

The root fix uses one `torch.var_mean(..., correction=0)` reduction for each
group and uses its mean for centering even when the denominator is a global
standard deviation. fp16/bf16 statistics are promoted to fp32 so `eps=1e-8`
cannot underflow to zero. The focused regression covers both `global_std`
branches and fp16, bf16, and fp32; the algorithms plus online-advantage suite
passes **170 tests**. CPU and RTX 5090 CUDA reproductions return finite, exact
zero advantages for `0.9 x 8` in fp16, bf16, fp32, and fp64.

An offline replay of all 16 completed update batches found the O(1) discrepancy
only at epoch 14. For epochs 0--13, the largest old-versus-fixed advantage
difference was `2.15e-6`; epoch 14 changed by exactly `1.0`. Under the fixed
implementation, the three constant groups at epoch 14 are exact zero and only
the genuinely non-constant group remains trainable. This establishes
`checkpoint-10` as the last persisted clean resume point. The failed run,
including `checkpoint-15`, 520 debug pairs, explicit SIGINT verdict, and console
log, is quarantined under the path linked below.

The corrected canonical lineage copies only checkpoints 5/10, metrics history
through epoch 9, and debug records 0--319. It resumes from its local
`checkpoint-10`, and the checkpointed sampler/RNG continues at epoch 10. The
post-fix VRL Python-tree identity is
`7f40df862bcacd6a9841ccfa40f96639d8681436d648790ae82c72dafd3472e8`.
That identity differs from the previously frozen base generation run, so formal
qualification must regenerate the base arm under the fixed runtime rather than
rewriting provenance or weakening the validator.

The completed final checkpoint correctly records `completed_epoch=80`,
`trainer_step=80`, and `next_epoch=80`, but only `global_step=78`: two fully
filtered epochs advanced the training/data cursor without attempting an
optimizer update. The original v5 qualification schema incorrectly required all
four counters to equal one `candidate_global_step`. After both formal pixel arms
were frozen and their paired generation contract passed, the internal and
external protocols migrated to v6. The new typed checkpoint requirement pins the
final label plus exact completed epoch and trainer step, requires the resume
cursor to equal the completed epoch, and accepts only an integer
`0 <= global_step <= trainer_step`. Manifest, seeds, sampling, model identity,
OCR engine, and scoring policy are unchanged. Schemas v1--v5 retain their exact
legacy progress rule rather than being reinterpreted retroactively.

若 final checkpoint 完成，positive result 仍必须同时通过：

1. 在 exact frozen 40 x 3 grid 上生成 candidate。
2. 在同一进程按 v6 protocol 重打两臂，报告 prompt-cluster bootstrap CI、
   exact counts 和 wins/ties/losses。
3. blind-review paired images，检查 extra text、text-shaped artifacts 和
   carrier collapse。
4. 跑独立 general-quality guard，排除 brightness、saturation、edge-detail
   或 diversity regression。

Training reward alone cannot satisfy this gate.

Additional evidence paths:

- [rejected v1 verdict](../../outputs/anima_ocr_single_text_long_v1_ppocrv6_fullpass/run_verdict.json)
- [rejected v1 metrics](../../outputs/anima_ocr_single_text_long_v1_ppocrv6_fullpass/metrics.csv)
- [v2 dataset spec](../../datasets/anima/ocr_single_text_long_v2/dataset_spec.json)
- [reviewed ESDB targets](../../datasets/anima/ocr_single_text_long_v2/scowl_train_targets.json)
- [pinned vocabulary manifest](../../datasets/vocabularies/esdb_en_us_50_2026_02_25_7_10/vocabulary.json)
- [frozen v1 qualification protocol](../../datasets/anima/ocr_single_text_long_v1/qualification_protocol.json)
- [frozen held-out base](../../outputs/probes/anima_ocr_single_text_long_v1_base_ppocrv6_20260830/run_config.json)
- [frozen held-out base headroom](../../outputs/probes/anima_ocr_single_text_long_v1_base_ppocrv6_20260830/ocr_ppocrv6_headroom.json)
- [v2 canary config](../../vrl/config/presets/experiment/anima_preview3/online_grpo_ocr_single_text_long_v2.yaml)
- [v2 canary metrics](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_canary/metrics.csv)
- [v2 canary replay gates](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_canary/training_debug.jsonl)
- [v2 canary final metadata](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_canary/checkpoint-final/checkpoint_meta.json)
- [v2 formal config](../../vrl/config/presets/experiment/anima_preview3/online_grpo_ocr_single_text_long_v2_fullpass.yaml)
- [quarantined first formal evidence](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass_failed_mixed_attempts_20260830/FAILURE_README.md)
- [quarantined constant-advantage evidence](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass_failed_constant_advantage_20260830/FAILURE_README.md)
- [corrected resume supervisor log](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass.supervisor-advfix-resume10-attempt-01.log)
- [corrected v2 formal resolved config](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass/resolved_config.yaml)

### 10.13 Formal v2 endpoint: qualified execution, rejected model

The corrected v2 run completed as one rank-32 LoRA lineage. This was continuous
Flow-GRPO over the Anima diffusion flow (`rollout.sde.type: flow_grpo`), not
token GRPO, TRPO, or full-parameter training. The training cursor reached epoch
80, while `global_step=78`: epochs 48 and 54 contained no nonzero advantage and
correctly skipped optimizer and weight-sync work. `checkpoint-80` and
`checkpoint-final` are byte-identical. The final checkpoint SHA-256 is
`475da5f203d8a983251c862e3a07bc783323213d33b136a787eab87a98e747c5`.

Execution health is fully qualified: 80 metric rows, 2,560 rollout PNGs, 2,560
OCR records, finite model and optimizer tensors, no OOM, no replay mismatch, no
clipping, and no monotonic memory growth. All 143 constant-reward groups now
produced exact-zero advantages. The 80-update reward history did not show a
learning trend:

| phase | updates | aggregate OCR mean | raw similarity mean | exact / 640 |
|---|---:|---:|---:|---:|
| 0 | 0--19 | 0.909281 | 0.924175 | 419 |
| 1 | 20--39 | 0.915949 | 0.928877 | 425 |
| 2 | 40--59 | 0.917405 | 0.924826 | 422 |
| 3 | 60--79 | 0.909942 | 0.928584 | 416 |

The phase-0-to-phase-3 aggregate delta was **+0.000662**, with a target-cluster
bootstrap 95% CI of **[-0.012878, +0.013677]**. Only one of 80 targets had a
strictly increasing aggregate trajectory, and its exact count remained 0/8 in
every phase. Zero targets had a strictly increasing exact-count trajectory.
The training reward therefore supplies no evidence of learning.

The formal frozen qualification rescored both arms in one PP-OCRv6 runtime on
40 unseen targets, three fixed draws per target:

| arm | mean OCR | exact | nonzero | multiple lines | rejected extra line |
|---|---:|---:|---:|---:|---:|
| base | 0.934315 | 89 / 120 | 120 / 120 | 2 / 120 | 0 / 120 |
| checkpoint-80 | 0.927235 | 88 / 120 | 119 / 120 | 3 / 120 | 1 / 120 |

The paired delta was **-0.007080**, with a prompt-cluster 95% CI of
**[-0.026753, +0.005268]** and image wins/ties/losses of **4/113/3**. The report
emitted both `clear_improvement=false` and `clear_regression=false`. The
post-hoc length split was `-0.042857`, `+0.004167`, `+0.003704`, and `+0.006667`
for lengths 7, 8, 9, and 10 respectively; it cannot rescue the preregistered
7--10 endpoint.

A separate 240-cell carrier-integrity review was completed while the per-prompt
arm mapping remained hidden. It found no evidence that the candidate recovered
OCR by moving text onto an easier surface:

| criterion | base pass | checkpoint-80 pass |
|---|---:|---:|
| requested carrier present | 100 / 120 | 100 / 120 |
| target grounded on requested carrier | 97 / 120 | 98 / 120 |
| no floating or duplicate target | 106 / 120 | 107 / 120 |
| image integrity | 119 / 120 | 120 / 120 |
| all criteria | 97 / 120 | 98 / 120 |

The all-criteria comparison had one candidate win and 119 ties. This is enough
to rule out the v1-style carrier collapse, but one discordant image is not an
improvement claim.

The independent 32-prompt, 64-image-per-arm Luna quality guard also remained
null. Base scored **0.743641**, checkpoint-80 scored **0.742563**, and the paired
prompt delta was **-0.001078**, with a 95% CI of **[-0.014000, +0.011031]** and
wins/ties/losses of **7/18/7**. Edge detail, saturation, pixel diversity, and
color diversity had no supported regression. Mean brightness increased by
`0.000865` on the normalized 0--1 scale; its CI excluded zero, but the magnitude
is less than 0.1% of full scale and was not visually meaningful.

Three isolated Codex visual reviewers then inspected the 32 contact sheets
before the blind key was opened. After unblinding, checkpoint-80 was preferred
on 4 prompts, base on 7, with 21 ties (exact two-sided sign test `p=0.548828`).
This is an automated visual audit, not an independent human review. It found
shared severe failures on 12/32 prompts: missing requested actions, malformed or
extra limbs, wrong object interactions, duplicated subjects, and montage-like
scene fragmentation. Those failures were present in both arms and are evidence
for the next target search, not damage caused by the OCR LoRA.

**Decision: do not promote checkpoint-80.** Keep base as the default and retain
the checkpoint only as reproducible negative evidence. More OCR epochs are not
justified: the base is already near the ceiling on smooth edit similarity,
about 45% of samples have zero advantage, the remaining signal is dominated by
seed-dependent spelling and an extra-line veto, and 80 updates did not improve
either training progression or frozen held-out performance.

At the OCR endpoint, the next gate was **reliable measurement of structural and
relational scene quality**. The shared blind failures showed real headroom in
multi-person actions, hand-object contact, subject count, and single-scene
coherence. Existing generic Luna and hand-crop rubrics were too noisy to train
against directly. Section 11 records that search and supersedes this historical
next-step statement.

Final evidence paths:

- [final checkpoint metadata](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass/checkpoint-final/checkpoint_meta.json)
- [final training metrics](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass/metrics.csv)
- [frozen base generation](../../outputs/probes/anima_ocr_single_text_long_v1_base_ppocrv6_runtime7f40df86_20260830/run_config.json)
- [frozen candidate generation](../../outputs/probes/anima_ocr_single_text_long_v2_flat_carrier_checkpoint80_ppocrv6_20260830/run_config.json)
- [formal OCR report](../../outputs/probes/anima_ocr_single_text_long_v2_flat_carrier_checkpoint80_ppocrv6_20260830/ocr_exact_qualification_v6/report.json)
- [locked carrier review](../../outputs/probes/anima_ocr_single_text_long_v1_blind_carrier_review_checkpoint80_20260830/review_completed.jsonl)
- [unblinded carrier summary](../../outputs/probes/anima_ocr_single_text_long_v1_blind_carrier_unblinded_checkpoint80_20260830.json)
- [Luna quality summary](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass/heldout_quality_v1_luna_final/summary.json)
- [Codex blind visual review](../../outputs/anima_ocr_single_text_long_v2_flat_carrier_ppocrv6_fullpass/heldout_quality_v1_luna_final/codex_blind_visual_review.json)

## 11. First structural target: exact four/five-person adherence

The OCR endpoint above left one useful next question: can a narrow, directly
observable structural failure provide enough headroom and a sufficiently
precise reward for Anima RL? Exact total person count passed frozen
qualification far enough to justify a staged training run, but Section 11.10
shows that the CountGD reward failed after policy shift and no checkpoint passed
the model-improvement gate.

### 11.1 Fresh base audit established real headroom

A new frozen audit used eight natural multi-person prompts and eight fixed
rollouts per prompt. Manual visible-person counts marked only **36/64** images
as exactly matching the requested four or five people, an exact-adherence rate
of **56.25%**. Every eight-image prompt group contained both correct and
incorrect samples. This is materially different from a nearly saturated target:
the base has real failures, and each audited group contains within-prompt
contrast that a group-relative algorithm could in principle use.

The labels count people actually depicted in the image, including cropped and
background people. They do not infer intended characters from the prompt and do
not exclude a person merely because the face is hidden. This keeps the target
structural instead of quietly turning it into an occlusion or portrait-quality
rubric.

### 11.2 Precision, not raw recall, selected the reward

The first RTMLib dual-YOLOX one-shot probe was rejected on the same 64 manually
labeled images:

| scorer | TP | FP | FN | TN | precision | recall | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| RTMLib dual YOLOX | 32 | 3 | 4 | 25 | 0.9143 | 0.8889 | reject |
| Luna, two-pass AND | 22 | 0 | 14 | 28 | 1.0000 | 0.6111 | provisional; later reject |

RTMLib had higher recall, but its three false positives would reward an image
whose human count is wrong. That is an unsafe error for RL: it sends a gradient
in the wrong direction. The detector probe was a one-shot feasibility artifact;
the matrix is retained here, while the frozen images and human labels remain
the reproducible source material.

The provisionally retained Luna protocol assigns stable candidate IDs, scores the same eight
images once in forward order and once in reverse order, and emits an observed
integer count plus an ambiguity flag. Reward is `1` only when both passes are
unambiguous and both observed counts equal the typed
`metadata.expected_people` target. The generation prompt is conditioning, not
the source of the expected count. On the human audit this conservative AND
rule yielded **22/64 positives** and reward variance in **7/8 groups**. Its
false negatives reduce sample efficiency, but they do not positively reinforce
the wrong subject count.

Two attempts to recover recall were explicitly rejected:

| variant | scope | TP | FP | FN | TN | reason rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| four-by-four grid | 16 hard images | 4 | 0 | 3 | 9 | recall remained only 0.5714 |
| third layout alone | all 64 | 32 | 3 | 4 | 25 | reintroduced three false positives |
| three-pass majority | all 64 | 32 | 1 | 4 | 27 | one false positive still rewards a wrong image |

The majority rule looked better by aggregate recall (**0.8889**) and activated
all eight groups, but the remaining false positive was not a marginal label:
prompt 7 sample 7 contained six people for a target of five, while two of three
layouts returned five. More judge calls therefore did not remove the relevant
failure mode. At this stage the two-pass AND protocol was the precision-first
candidate; Sections 11.5--11.7 supersede that decision after the exact same
pixels failed independent reward retest.

### 11.3 The long-term dataset removes the prompt shortcut

The reviewed corpus contains **192 training prompts** and **64 held-out
prompts**, balanced equally between targets four and five. The concrete
location, action, role, object, and concept IDs are disjoint between train and
eval. All 256 action-location concepts are unique; only high-level scene-family
strata are shared so the evaluation measures count generalization rather than
an unrelated domain shift.

Ordering is part of the data contract. The trainer consumes four prompts per
update, and every consecutive four-row window contains exactly two target-four
prompts, two target-five prompts, and four distinct scene families. Prompt text
states the total naturally once, but does not enumerate the cast, arrange people
in a counting row, demand visible faces, or add negative counting scaffolding.
`taxonomy.json` is the semantic source of truth; the generated manifests and
dataset spec must not be hand-edited.

### 11.4 The three-update canary qualified execution, not improvement

This run is the repository's actual **flow-matching GRPO** path with
`rollout.sde.type=flow_grpo`, a rank-32 LoRA, eight rollouts per prompt, four
prompts per update, and full denoising-step replay. It is neither TRPO nor
full-parameter training.

| update | reward mean | reward std | grad norm | replay max diff | clip fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4062 | 0.4713 | 0.004186 | 0.0 | 0.0 |
| 2 | 0.2188 | 0.4074 | 0.006110 | 0.0 | 0.0 |
| 3 | 0.2188 | 0.2293 | 0.003447 | 0.0 | 0.0 |

All three updates completed without OOM or non-finite values, the parity and
drift guards remained clean, and the supervisor emitted `success`. Across 12
prompt groups and 96 rollouts, the positive counts were
`[5, 3, 2, 3, 2, 1, 2, 2, 5, 2, 0, 0]`: **27/96 positives** and **10/12 active
groups**. The final two groups were all zero, and the last update's advantage
zero rate reached 0.5. The canary therefore proves that the real Flow-GRPO LoRA
path can consume this reward safely; it does not prove that the reward is dense
enough for a durable model change.

### 11.5 Checkpoint 3 did not survive exact reward retest

The frozen count evaluation used eight held-out prompts. Each prompt's eight
correlated samples were generated together from one deterministic prompt-batch
seed, and base/candidate reused that same batch seed:

| arm | exact reward | rate | active groups | target four | target five |
|---|---:|---:|---:|---:|---:|
| base | 15/64 | 0.234375 | 4/8 | 15/32 | 0/32 |
| checkpoint 3 | 19/64 | 0.296875 | 6/8 | 17/32 | 2/32 |

The first paired point estimate was **+0.0625** (eight improvements, four
regressions, 52 ties), but its prompt bootstrap 95% CI was
**[-0.046875, +0.171875]** and the two-sided sign-test result was
**p=0.387695**. It was never evidence of held-out improvement. More
importantly, a clean rerun of the same production reward on the exact same
pixels and typed targets reversed the direction:

| scoring pass | base | checkpoint 3 | paired delta | prompt-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| first | 15/64 | 19/64 | +0.0625 | [-0.046875, +0.171875] |
| exact rerun | 17/64 | 15/64 | **-0.03125** | [-0.125000, +0.046875] |

Across the two executions, the final binary reward agreed on only **58/64
(90.6%)** base images and **56/64 (87.5%)** checkpoint images. Even inside one
execution, the complete forward/reverse verdict agreed on only **53.1% / 51.6%**
of base/checkpoint images in the first pass and **48.4% / 45.3%** in the rerun.
The original human-label qualification already contained only **39/64 (60.9%)**
forward/reverse observed-count agreement; the zero-false-positive AND rule hid
that instability by converting disagreements to zero. It made the reward
conservative, but not reproducible.

The old four-image apparent gain is therefore smaller than judge retest noise
and is invalidated. The exact-count evaluator was retained because it exposed
this failure by reusing the production scorer rather than copying its logic.

The independent general-quality guard covered 44 prompts with two fixed paired
samples from one deterministic batch seed per prompt, 88 images per arm.
General-quality Luna moved from **0.862943** to **0.864625**;
the paired delta was **+0.001682**, with a 95% CI of
**[-0.002227, +0.005955]** and wins/ties/losses of **4/38/2**. Edge energy moved
by **-0.0000185**, with CI **[-0.0000834, +0.0000488]**. Saturation decreased by
**0.000170** on a normalized 0--1 scale, and its CI excluded zero, but the
magnitude is only 0.017 percentage points and was not visually meaningful.
Brightness, pixel diversity, and color diversity showed no supported
regression. The sampled blind sheets showed no pixel collapse or visible
softening, but formal independent human blind review remains pending; the Luna
guard is not a substitute for it.

### 11.6 The checkpoint-12 stage was stopped after the reward failed

The initial result justified a bounded checkpoint-12 stage rather than a blind
48-update launch. The continuation resumed checkpoint 3 correctly and remained
numerically healthy, but the reward retest completed before the stage did. The
run was stopped with SIGINT as soon as the direction reversal was verified.
The supervisor recorded `terminated`, not a crash; the GPU workers exited and
released their memory.

Two additional optimizer updates completed. Their reward means were **0.2500**
and **0.0312**, with advantage-zero rates **0.25** and **0.75** respectively.
Only `checkpoint-4` is durable because the configured save interval is four;
the fifth update was not promoted to a checkpoint, and the following update
was interrupted during rollout collection. No Luna-reward checkpoint-12 result
exists and none should be inferred.

At the Luna endpoint, the immediate gate became reward repeatability rather
than training duration. A replacement person-count reward first had to pass
four gates on frozen pixels: at least **95%** exact reward agreement across
independent reruns, **zero false-positive rewards** against human counts, recall
of at least **22/36**, and reward variance in at least **7/8 prompt groups**.
Sections 11.7--11.10 supersede that intermediate decision with the later
CountGD policy-shift result.

### 11.7 Conventional deterministic detectors removed judge noise but failed precision

The Luna failure was not accepted as evidence that exact person count itself is
untrainable. The same 64 frozen, human-counted images were used to qualify five
deterministic candidates. A scorer had to reproduce every observed count,
emit no positive reward on a human-negative image, retain at least Luna's
`22/36` positive recall, and activate at least seven of eight prompt groups.
Parameters could be selected on a calibration split, but the untouched split
owned the decision.

| scorer and protocol | TP | FP | FN | TN | recall | active groups | repeat | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| RTMLib YOLOX-M, built-in `score > 0.3` | 32 | 3 | 4 | 25 | 0.8889 | 8/8 | 64/64 | reject |
| imgutils person detector v1.1/m, fixed default protocol | 31 | 4 | 5 | 24 | 0.8611 | 8/8 | 64/64 | reject |
| Grounding DINO tiny, calibration-selected `0.33` + NMS `0.5` | 31 | 2 | 5 | 26 | 0.8611 | 8/8 | 64/64 | reject |
| Florence-2-large-ft, fixed open-vocabulary `person` detection | 29 | 1 | 7 | 27 | 0.8056 | 8/8 | 64/64 | reject |
| CountGD, official text-only `person .` protocol at `0.23` | 27 | **0** | 9 | 28 | **0.7500** | **8/8** | **64/64** | **pass** |

RTMLib was also tested as two-model M/X AND/OR, confidence consensus,
box-IoU matching, and horizontal-flip consensus. None satisfied zero false
positives, Luna-level recall, and seven active groups together. A full-data
threshold scan could manufacture an apparent M/X-AND pass at thresholds
`0.0482313/0.5784553`, but a prompt-group calibration/held-out split exposed
the overfit: recall fell from **12/19 (63.16%)** on calibration to **10/17
(58.82%)** on held-out. The standard M threshold also cannot be changed through
the old wrapper configuration: these ONNX exports already include NMS, and
RTMLib 0.0.16 hard-codes `final_scores > 0.3` for that output branch. The old
`score_thr` and `nms_thr` fields were no-op knobs for these checkpoints.

Grounding DINO was exactly reproducible across two executions. The best rule
selected using the 32 even-index calibration samples still produced two false
positive rewards on the 32 odd-index held-out samples. An exhaustive diagnostic
scan found no full-audit rule with zero false positives, recall at least 0.5,
and seven active groups; this scan is negative evidence, not a deployable
threshold search.

Florence-2 used the predeclared task `<OPEN_VOCABULARY_DETECTION>`, text
`person`, `do_sample=false`, and `num_beams=3`. Two complete executions matched
on all counts, generated text, and parsed boxes. Median steady-state latency was
**81.6 ms/image**, p95 was **92.1 ms/image**, and peak allocated GPU memory was
**1.878 GB**. Its one false positive was not a disputed human label:
`anima_0002_01.png` contains five visible people for target four, while Florence
returned four. The same image was one of RTMLib M and X's shared high-confidence
false positives. Detector voting therefore preserves this systematic miss
rather than removing it.

This establishes two distinct reward failures. Luna can change its answer on
identical pixels; the first four conventional detectors answer deterministically
but can confidently miss a partially occluded person and positively reinforce
the wrong image. Neither failure is repaired by more Flow-GRPO epochs. CountGD
is the frozen-distribution exception in Section 11.8, but Section 11.10 shows
that it also fails after policy shift. The stopped Luna run remains stopped and
was never resumed with a different reward.

### 11.8 CountGD passed the frozen gate and gained a reproducible service boundary

CountGD was evaluated only after its protocol was fixed from the authors'
single-image implementation: the repository inference transform at short side
800 and maximum side 1333, no visual exemplars, caption `person .`, the
four-argument GitHub model forward, and `sigmoid(logits).max(-1) > 0.23`. No
threshold, NMS rule, SAM pass, or exemplar was selected from the Anima labels.
The source revision is
`b6f362b3f5cd20db4a171faa410dfed8f2f466d8`, the author Space revision is
`6e82e59569a84ee5c6aafa35d396f2d2bee57be2`, and the trusted checkpoint SHA-256
is `c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126`.

Two complete CUDA passes over the frozen 64-image human audit agreed on every
observed count and binary reward. Both yielded `TP=27, FP=0, FN=9, TN=28`,
precision 1.0, recall 0.75, and reward variance in all eight prompt groups. The
same canonical installation was then run in its legacy Python 3.12 CPU
environment; all 64 observed counts matched the qualified CUDA output. That
environment explicitly inherits system site packages, so this is behavioral
parity evidence rather than an isolation claim. CPU placement is a resource
decision, not a different reward protocol. The production HTTP path also
scored a mixed two-image batch as `(1.0, 0.0)` and advertised
`generation_overlap_safe=true` without reserving the sole GPU.

The integration deliberately keeps three existing boundaries:

- `CountGDPersonCountModel` owns the pinned external-model protocol and reads
  the target only from typed `metadata.expected_people`.
- `CountGDPersonCountReward` remains a thin `DiskArtifactRewardFunction`
  framework/HTTP adapter; flattening it would break the uniform reward registry
  shape without removing complexity.
- The existing reward-service wire format, artifact integrity checks, runtime,
  and multi-reward combiner remain unchanged. CountGD does not introduce a new
  detector abstraction or a second service protocol.

The module-level revision, checkpoint, and schema constants remain because they
are external source/checkpoint/protocol boundaries. Prompt taxonomy and dataset
vocabulary remain in the dataset assets rather than being duplicated as Python
ALL_CAPS data.

The ignored external installation is reproducible rather than machine folklore.
`python -m vrl.scripts.rewards.install_countgd install --python <python-3.12.2>`
downloads the pinned GitHub archive and Space assets, verifies every remote
artifact, applies the two compatibility patches with exact replacement counts,
builds an isolated environment in a sibling staging directory, and atomically
publishes only after the production verifier succeeds. The Python 3.12.2 Linux
x86-64 lock contains 76 exact artifacts plus the pinned `pip==24.0` bootstrap;
its digest is
`ab7569182368d3bdfd9ec9a75de679220b59f71780e8a4103cea4b5d8bd76a1f`, and
verification rejects missing, changed, or unexpected distributions. A clean
temporary rebuild had no system-site inheritance, passed `pip check`, model
imports, and real HTTP `/info` plus `/score` smoke tests, and reproduced all 133
runtime files with runtime-tree SHA-256
`e41c4fd64148a0a55a4d5bda3e0f5f8da6297811d3f7648506761742ac04b450`.
It matched the retained legacy canonical environment on all 64 frozen observed
counts and all 64 binary rewards. The current canonical environment is accepted
only through an exact legacy-manifest compatibility gate; that does not claim
it satisfies the new isolation contract. Model bytes remain ignored; the
installer, semantic protocol, and environment lock are the versioned sources
of truth.

### 11.9 The CountGD three-update canary was healthy but held-out neutral

The replacement canary used the same qualified Flow-GRPO LoRA recipe: four
prompts per update, eight rollouts per prompt, generation and replay microbatch
one, 20 denoising steps, and learning rate `1e-5`. It completed all three
updates on the first supervised attempt. Replay parity and active clipping were
zero throughout; gradients were finite (`0.012597`, `0.005342`, `0.007238`).
The 12 training prompt groups contained 35 positive rewards across 96 images,
and 11/12 groups had both positive and negative samples. This qualified the
runtime signal but did not estimate generalization because each update used
different training prompts.

The frozen evaluation instead used every one of the 64 concept-disjoint eval
prompts with two fixed paired samples generated together from one deterministic
batch seed per prompt, giving 128 images per arm. All 128 cells matched on
`prompt_index`, `sample_index`, prompt text, prompt-batch seed, and typed target.
The result was:

| arm | exact reward | rate | active 2-sample prompts | target four | target five |
|---|---:|---:|---:|---:|---:|
| base | 44/128 | 0.343750 | 28/64 | 25/64 | 19/64 |
| checkpoint 3 | 41/128 | 0.320312 | 19/64 | 22/64 | 19/64 |

There were six improved cells, nine regressed cells, and 113 ties. The paired
delta was **-0.023438**, the prompt-cluster bootstrap 95% CI was
**[-0.085938, +0.031250]**, and the two-sided sign-test result was `p=0.607239`.
Target four moved by -0.046875; target five was unchanged. Mean absolute CountGD
count error moved from 1.585938 to 1.515625, but exact adherence is the declared
objective and neither metric has supported evidence of improvement. Blind
inspection of representative reward flips found actual people-count changes,
not blank images, blur, or an obvious detector shortcut.

The correct interpretation is narrow: checkpoint 3 is not promoted, but three
updates cover only 12 of 192 training prompts and do not invalidate a scorer
that passed the frozen reward gates. The next bounded stage resumes the exact
checkpoint/RNG state through update 12 and keeps checkpoints 6, 9, and 12. It
must pass the same all-64-prompt paired evaluation before a full pass is
justified.

### 11.10 Checkpoint 12 exposed a post-training detector failure, not an improvement

The bounded continuation resumed the exact checkpoint-3 optimizer, scheduler,
and RNG state and completed updates 4--12 on the supervisor's first attempt.
Its nine continuation rows cover epochs 3--11. Reward means ranged from 0.1875
to 0.5000 and gradient norms from 0.003075 to 0.012563. Every value was finite;
rollout/replay max log-prob difference, pre-update clipping, active clipping,
and training clipping were zero throughout. Checkpoints 6, 9, 12, and final are
complete. This establishes a healthy Flow-GRPO LoRA execution path, not model
quality.

Checkpoint 12 was the predeclared endpoint; checkpoints 6 and 9 were not scored
and retrospectively selected. It generated the same complete 64-prompt by
2-sample grid as base under the same pixel-affecting protocol: 512 square, 20
steps, CFG 4.5, maximum sequence length 128, the same negative prompt, and the
same prompt-batch seed formula. The broad repository runtime hash necessarily
differs because reward/evaluator-only Python files were added after base
generation; none of those files participate in image generation.

The persisted v2 report rejected incomplete grids, revalidated the generation
and anchor schemas, paired every prompt/sample/seed/target cell, and bound every
PNG SHA-256 into a canonical grid digest. A later audit hardened the reusable
evaluator to v3: it also requires equal sampling, negative prompt, execution,
Python/core generator package versions, generation policy, and underlying
base-model identity, and checks every PNG's dimensions against the declared
sampling contract before scoring. The existing 128-by-128 source pair passed
that stricter preflight retrospectively. The numbers below remain the persisted
v2 report; the v3 audit was a stricter protocol/dimension preflight and did not
write a replacement report. Both arms were rescored from pixels, and the base
result repeated all 128 prior observed counts and rewards exactly. The automatic
endpoint was:

| arm | exact reward | rate | target four | target five | mean absolute count error |
|---|---:|---:|---:|---:|---:|
| base | 44/128 | 0.343750 | 0.390625 | 0.296875 | 1.585938 |
| checkpoint 12 | 50/128 | 0.390625 | 0.421875 | 0.359375 | 1.453125 |

The apparent delta was +0.046875: 13 improved cells, seven regressed cells, and
108 ties. It did not pass the statistical gate. The prompt-cluster bootstrap
95% CI was `[-0.031250, +0.117188]`; the prompt-cluster two-sided sign-test
result was `p=0.167068`. Therefore the evaluator correctly reported
`clear_improvement=false`.

The more important failure appeared in the predeclared reward-hacking review.
Two reviewers split all 19 prompts containing the 20 non-zero reward flips.
They recorded visible human counts from blinded A/B contact sheets before the
arm key or CountGD rows were opened. Nineteen of 20 flipped cells had unchanged
human exact-count status. The remaining cell, prompt 46 sample 0, became
potentially exact only because distant people collapsed into white/black
fragments. Prompt 13 sample 0 is the clearest detector false positive: both
arms contain detached giant hands and pens with zero visible people, while
CountGD changed from three to the requested five and awarded the checkpoint.

The frozen base audit's zero false-positive rewards therefore did not survive
policy shift. CountGD is deterministic, but deterministic does not mean robust
to optimized outputs. The `44 -> 50` gain is dominated by detector instability
on nearly identical occluded scenes and includes a concrete quality-damaging
shortcut. Checkpoint 12 must not replace base, and a longer 48-update pass is
not justified. Future detector rewards require two gates: frozen-distribution
qualification before training and blinded human review of every reward flip
after the bounded policy-shift canary.

The first problem to solve next is therefore reward robustness under policy
shift, not prompt volume or training duration. A replacement must be qualified
on anime-domain base images plus hard negatives produced by an optimized
policy, and it must reject malformed-person/fragment shortcuts before another
bounded Flow-GRPO canary is justified. The balanced 192-train/64-eval prompt
corpus remains suitable conditioning data; it cannot repair a reward that
changes meaning on policy outputs.

### 11.11 The frozen robustness benchmark rejected detector voting

The two human audits are now one independent qualification asset rather than a
set of paths under disposable experiment outputs. It contains 140 unique RGB
PNG images: 64 frozen base images and 38 complete base/checkpoint pairs (76
images) covering every non-zero checkpoint-12 CountGD reward flip. The copied
media totals 58,486,513 bytes and is bound by both per-image SHA-256 and a
manifest digest. The tracked annotation file preserves 137 point counts, three
count ranges, and three explicit `fragmented` findings. Unannotated integrity
is stored as `not_assessed`, never silently promoted to intact.

The benchmark deliberately keeps count and integrity separate. Human count
intervals imply positive, negative, or ambiguous target status; an explicit
fragment finding then forces the qualification label negative. Under the count
labels, CountGD has `TP=42, FP=11, FN=23, TN=62`, with two target-ambiguous
records. Applying only the three evidence-backed fragment vetoes changes this
to `TP=41, FP=12, FN=23, TN=62`. All 20 CountGD policy reward flips retain the
same final qualification status, making the reward-hacking failure executable
as a regression gate. The asset is marked `qualification_only`; using those
same 140 images to train a replacement critic would invalidate that gate.

A fixed-protocol bake-off then tested whether independent structure detectors
could repair CountGD without learning a new reward. No thresholds were scanned.
The candidates were DeepGHS anime head `v2.0_s`, anime person `v1.1_m`, the
newer person `v1.3_s`, and RTMLib's default performance-mode whole-body stack.
The confusion table below uses the frozen 64 exact labels and the 73
point-labeled policy images; the three count-range annotations are excluded so
it remains directly comparable to the earlier audit.

| reward rule | frozen TP/FP/FN/TN | frozen precision/recall | policy TP/FP/FN/TN | policy precision/recall | original flips left |
|---|---:|---:|---:|---:|---:|
| CountGD | 27/0/9/28 | 1.000 / 0.750 | 15/11/14/33 | 0.577 / 0.517 | 20 |
| CountGD AND head | 25/0/11/28 | 1.000 / 0.694 | 8/6/21/38 | 0.571 / 0.276 | 9 |
| CountGD AND person v1.1 | 24/0/12/28 | 1.000 / 0.667 | 10/6/19/38 | 0.625 / 0.345 | 13 |
| CountGD AND RTMW | 25/0/11/28 | 1.000 / 0.694 | 13/4/16/40 | 0.765 / 0.448 | 14 |
| CountGD AND all three | 20/0/16/28 | 1.000 / 0.556 | 5/1/24/43 | 0.833 / 0.172 | 4 |
| CountGD AND person v1.3 | 23/0/13/28 | 1.000 / 0.639 | 8/2/21/42 | 0.800 / 0.276 | 7 |

The strict intersection still falsely changes reward for prompts 9, 12, 28,
and 40 while cutting frozen recall from 0.750 to 0.556. Prompt 40 sample 1 is a
policy false positive shared by every detector. Prompt 13 also disproves the
stronger pose-completeness hypothesis: RTMW hallucinates one pose on the
detached hands and pens, and all four shoulder/hip confidence values pass the
predeclared 0.3 torso check. On prompt 46, the head and newer person detectors
reject the shortcut only because both arms are undercounted, not because the
models recognize malformed fragments. Their errors are correlated around
small, occluded, and crowded people, so voting cannot supply the missing
integrity concept.

This closes the fixed-detector branch. CountGD, anime detectors, and RTMW remain
useful diagnostic features, but none is promoted to an online reward. The next
reward candidate must be learned from a separate training set with explicit
count and integrity labels, then pass this untouched benchmark and a fresh
post-canary blind review before Flow-GRPO resumes.

The fixed protocols and all 140 per-image detector observations are persisted
in `detector_bakeoff.json`; the six aggregate rows above are independently
recomputable from that file instead of existing only in this narrative.

### 11.12 Intermediate decision: label reward semantics before training again

This section records the intermediate decision that led to the 384-image pilot.
Section 11.14 completes and rejects that pilot's data gate.

The first problem at that stage was the reward's meaning under policy shift,
not Anima's optimizer, prompt volume, or update count. The existing 140-image
benchmark is not training data: it has only three explicit fragment findings, 137
`not_assessed` integrity values, and has already been used to reject reward
candidates. Treating `not_assessed` as `intact` would manufacture labels and
invalidate the only policy-shift gate.

The first concrete deliverable at that stage was an independently reviewed
Anima-domain annotation pool with visible-person counts, coherent-person boxes,
malformed-fragment boxes, and explicit integrity labels. Only after those labels
exist is it meaningful to train a dense count-and-integrity critic. Detector
outputs may be used to prioritize disagreements, but they remain features or
label proposals; the bake-off proves they cannot supply the truth through
voting.

A 384-image pilot pool was then prepared at
`data/external/anima/person_count_critic_pilot_v1`: all 192 training concepts,
one base image and one rejected checkpoint-12 image per concept, paired by the
same prompt and seed. All images are unique 512x512 RGB PNGs and total
146,676,895 bytes. The tracked SHA-256 manifest binds every image. Both arms use
the exact same generation protocol and PEFT 0.19.1; an earlier base staging arm
under PEFT 0.18.1 produced identical hashes for all 192 images but was not
published. The pilot was initially marked `unlabeled`. Section 11.14 records
its later image-level Codex screen; it remains `training_use_allowed=false`.

The intermediate decision order was:

1. annotate the independent pilot and measure how many genuine fragments it
   contains;
2. reject the target if the fragment class is too sparse to test learnability,
   or generate additional policy/seed hard negatives;
3. train a count-and-integrity critic without prompt-target shortcut inputs;
4. freeze its weights and thresholds, run the existing 140-image rejection
   gate, then run a fresh untouched qualification set;
5. only a passing critic earns a three-update Flow-GRPO canary and blinded
   review of every new reward flip.

Section 11.14 completed steps 1--2 and rejected this pool. The current
deliverable is a targeted hard-negative corpus spanning multiple policies and
seeds, with over-count/crowded outputs, instance boxes, and blind adjudication;
it is not additional labeling of the same one-seed pilot.

Until step 4 passes, longer RL, full-parameter Anima training, detector voting,
and another Luna/Codex online judge run are all non-actions: they do not address
the demonstrated failure.

### 11.13 Automated teachers and synthetic labels do not replace domain truth

Three additional attempts tested whether the annotation step could be skipped.
All three failed before any replacement reward was integrated.

First, general vision-language judges remained deterministic but semantically
wrong on the frozen 140-image benchmark. Qwen3-VL-2B parsed all 140 responses
after stripping markdown fences, but produced `TP=51, FP=13, FN=14, TN=60` for
the requested-count decision, a point-count MAE of `0.7007`, and 11 errors of at
least three people. More importantly, it called all 140 images `intact`, so its
fragment recall was `0/3`; ten of its reward decisions flipped across the 38
policy pairs even though the blinded human qualification status flipped zero
times. Qwen2.5-VL-7B repeated the same integrity failure on a 12-image smoke, so
the full run was stopped. A single-image Luna smoke also missed all three
explicit fragments and put the human count inside its returned interval on only
six of 12 images. These models remain annotation suggestions, not label truth.

Second, stronger fixed anime detectors did not repair the policy-shift problem.
An Apache-2.0 RT-DETRv4-X checkpoint trained on Manga109-s reached frozen
body-count precision `0.9375` and recall `0.8333`, but fell to policy precision
`0.5484`. Intersecting it with CountGD removed frozen false positives only by
cutting frozen recall to `0.6944`; on policy images it still created 13 reward
flips, all false relative to unchanged human qualification. The available
AnimeInstanceSegmentation RTMDet-L checkpoint was weaker: at its predeclared
calibration threshold it reached only `TP=14, FP=0, FN=22, TN=28` on frozen
images and `TP=3, FP=3, FN=25, TN=43` on policy images. This closes the
detector-combination branch rather than merely rejecting one threshold.

Third, an intentionally small synthetic-domain probe measured whether exact
compositing labels alone could teach the missing count concept. Eight dedicated
rendered full-body Anima characters and eight empty Anima scene plates were
generated successfully; U2Net preserved all eight silhouettes well enough for
feasibility composites. A frozen DINOv2-S backbone (weights SHA-256
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`)
then embedded 980 on-the-fly training composites and 280 composites on two held
out backgrounds, balanced over counts zero through six. The fixed head used the
official one-layer DINO feature shape, class token concatenated with mean patch
token, followed by one 128-unit hidden layer. It achieved `0.8786` train and
`0.7857` synthetic held-out accuracy. Without benchmark-driven tuning, the
same head then reached only `0.1857` count accuracy and interval MAE `1.7214`
on the untouched 140 real Anima images. Its requested-target confusion was
`TP=9, FP=10, FN=56, TN=63` (precision `0.4737`, recall `0.1385`). CountGD on
the same records remained much stronger at `0.6000` count accuracy, MAE
`0.6571`, and `TP=42, FP=11, FN=23, TN=62`, despite already being rejected for
reward hacking. The synthetic head therefore failed the domain-transfer kill
gate. Programmatic composites may warm-start a later critic, but scaling the
same sticker-domain generator is not a substitute for real Anima labels.

The public integrity datasets have the same role limitation. Distorted-5K was
downloaded at immutable revision
`29a752335154346102c82b0fa85f1c88bd91f87b`; its 5,757,218,448-byte archive
matches SHA-256
`6ae44e4fdc85cb51e45ab05f5496723547be9a9eff87d2ac4016d50d15efd1a1`.
The release contains 4,989 image/JSON pairs, not 5,000, and the annotations are
three reviewers' point clicks rather than the withdrawn paper's polygon masks.
Only 2,235 images have a same-category spatial consensus from at least two
reviewers, while only 253 have complete three-reviewer empty annotations that
can serve as clean negatives. It can initialize a weak integrity representation
but supplies no person count, person box, or fragment-to-person association.

HumanRefiner was downloaded at immutable revision
`15b408cfde8c33cf6bb5b25114206c8b78b3e571`; its train and validation ZIPs
match SHA-256
`3379813309acd6a3d4a088900486aa015ca1f43be34227659bb342d4ed55b39f`
and `de1f6451d699648205b761729e11ef48c0f563edaee034515fdb9530828ad46c`.
The 56,444 images contain 147,573 YOLO part labels, but abnormal-hand boxes
alone make up `54.98%` of all rows. All 11,654 normal-part boxes come from the
real acrobatics subdomain, while only 61 of 122,348 abnormal boxes do, creating
a direct source/style shortcut. It also has no prompt, person count, coherent
person box, fragment label, or part-to-person association. Class 9 is
non-human and must remain an abstention/presence negative; treating it as
normal would explicitly reward deleting people. HumanRefiner can warm-start a
region representation only; its final 18-way head cannot become the reward.

Together these tests sharpen step 1 rather than changing it: the first scarce
asset is an independently reviewed Anima-domain label pool. External data and
synthetic composites are auxiliary pretraining inputs only, and the frozen 140
images remain qualification-only.

### 11.14 The first real-image critic pool failed its data gate

The 384-image pilot was globally shuffled with seed `2026083141` and exposed
only as blind IDs. Three isolated blind Codex primary-review shards partitioned
the pool into 128 images each, so each image received one primary review; the
reviewers did not open the arm/prompt key. The rule explicitly excluded
occlusion, back views, faceless distant people, and ordinary edge crops from the
fragment class. It included detached limbs, fused or duplicated bodies, and
collapsed person-shaped residue.

The screen found substantial exact-count headroom but almost no integrity
positives:

| arm | images | target in count range | final qualified | fragmented |
|---|---:|---:|---:|---:|
| base | 192 | 75 (39.06%) | 75 (39.06%) | 2 |
| checkpoint-12 | 192 | 81 (42.19%) | 80 (41.67%) | 4 |
| **total** | **384** | **156 (40.63%)** | **155 (40.36%)** | **6** |

The six fragments come from only four prompt groups. Their prevalence is
`1.56%` (Wilson 95% CI `[0.72%, 3.37%]`), only 18.75% of the predeclared
minimum of 32 fragment images. The descriptive base/checkpoint result is also
null:
11 qualification gains, 6 losses, 69 both-pass, and 106 both-fail across 192
pairs (`p=0.3323`, exact two-sided McNemar test). The rejected checkpoint is
not rescued by this larger blind screen. This cannot be interpreted as a strict
causal policy comparison: the two archives share package versions and sampling
settings, but their recorded VRL Python-tree hashes differ (`20cf8d...` versus
`696119...`). The replacement grid must regenerate every arm under one frozen
runtime.

The label process exposed a separate semantic risk. Forty-four crowded images
each received one separate blind Codex re-review. Exact point-count agreement
was 33/44 (`75.0%`), and count ranges overlapped on 36/44 (`81.8%`). Integrity
labels also agreed on 36/44, but this headline hides a prevalence failure: each
pass marked five images fragmented, while only one positive was shared. Cohen's
kappa was only `0.097`. Thirteen records required visual adjudication. A further
48 label-aware sampled audits brought the total second-look set to 89 unique
images, but they were not independent reviews and cannot replace instance-level
annotation.

This rejects critic training from this pool rather than the exact-count weakness
itself. The current labels are useful screening evidence, not training truth:
there are no coherent-person or fragment boxes, too few positive prompt groups,
and positive localization is reviewer-sensitive. `training_use_allowed` and
`qualification_use_allowed` therefore remain false. No `anima_person_critic`
model, reward adapter, or Flow-GRPO canary was added.

The next data asset must change the sampling distribution rather than scale the
same one-seed pool: add several policies and seeds, deliberately retain
over-count/crowded outputs, then collect coherent-person and malformed-fragment
boxes with independent adjudication. The existing Flow-GRPO implementation,
advantage estimator, CountGD diagnostic service, and frozen 140-image gate stay
unchanged. They are valid boundaries; the missing component is trustworthy
domain supervision.

### 11.15 Detector disagreement failed; close interaction passed the source probe

The larger hard-negative v2 generation contained 3,840 unique images and froze
a 768-image annotation selection before labels were opened. The selection mixed
uniform coverage with CountGD/RT-DETR disagreement mining and matched controls.
Annotation was stopped after 352 valid lane records covering 308 unique images:

| selection stratum | selected pool | reviewed unique | provisional fragments |
|---|---:|---:|---:|
| uniform core | 384 | 191 | 2 |
| detector-disagreement mined | 192 | 56 | 0 |
| matched control | 192 | 61 | 0 |
| **total** | **768** | **308** | **2** |

Both provisional positives are counted conservatively. One was labeled
fragmented in lane A but intact in lane B, and the other had only one completed
lane. Six additional images retained unresolved person evidence but did not
become fragment positives. On the 44 images common to both completed lanes,
count ranges overlapped on 40/44 (`90.91%`) and integrity labels matched on
43/44 (`97.73%`). The integrity headline is misleading under extreme class
imbalance: fragment micro-F1 and Cohen's kappa are both zero because the lanes
shared no positive.

A finite-population diagnostic gives a worst conditional-SRS tail probability
of `2.1916371e-5` for a pool containing at least 32 fragments, maximized at the
stratum allocation `(6, 21, 5)`. This is deliberately not reported as a formal
p-value. The contact-sheet scheduler was prompt-balanced rather than an
image-level simple random sample, the reviewed shard sequence was adaptively
stopped, and fragment sensitivity is not identified. The defensible conclusion
is narrower: continuing the remaining 460 reviews is not a useful engineering
investment, and detector disagreement did not enrich malformed topology.

The replacement probe changed the generation condition rather than another
detector threshold. Sixteen natural prompts requested 6--12 adults in
medium-wide close interactions. Base and checkpoint-12 used identical prompts,
sampling settings, and paired seeds, producing 64 unique images. A blind
high-recall screen marked five candidates. Prompt-aware adjudication excluded
four source images from two prompt groups before computing prevalence: one
prompt explicitly requested a training mannequin and another requested dress
forms, so detached human-shaped parts were correct scene objects rather than
malformed people. The three remaining candidates were all confirmed:

| source prompt | confirmed failure |
|---|---|
| eight martial-arts students | two independent arms emerging from one shoulder |
| six acrobats | fused torso boundary with duplicated limbs |
| eight volleyball players | a third arm emerging from the upper torso |

This yields 3/60 (`5.0%`) confirmed malformed images across three prompt groups,
versus the conservative 2/308 (`0.65%`) in hard-negative v2: `7.7x` enrichment.
All three happened to be from base, so this small probe is not a policy
comparison. It establishes only that close multi-person interaction is the
right source axis to scale.

The resulting long-term prompt asset contains 96 source prompts, split by
prompt identity into 72 train and 24 calibration groups, plus 32 untouched eval
prompts. Counts 6/8, eight scene families, and four interaction types are
balanced in every split. Train, calibration, and eval location/action/concept
IDs do not overlap. Human-shaped props and elongated limb-like objects are
excluded at the prompt-contract level. Generation still precedes labels, and
two blind review lanes plus adjudication remain mandatory.

The reward architecture remains the joint three-class Faster R-CNN rather than
`CountGD AND integrity classifier`. An integrity-only veto cannot correct the
many intact/occluded policy-shift count flips already demonstrated by CountGD.
The joint critic instead detects coherent people and malformed fragments from
pixels, derives a count interval and integrity state, and compares the external
requested count only after inference. CountGD remains a diagnostic and mining
reference, not an online reward.

This result authorizes generation and box-data qualification only. It does not
authorize critic fitting or Flow-GRPO. The unchanged minimum gate is 32
adjudicated fragment images across 16 prompt groups, two policy families, and
three seed waves, including at least 24 train and 8 calibration fragments. The
registered box agreement, count-branch, hard-intact, frozen-140, calibration,
and fresh-qualification gates also remain unchanged. If the new source misses a
gate, the response is a complete new seed wave, not a lower threshold or a
moved split.

Exact-person-count evidence paths (repository root
`/home/mingfeiguo/Desktop/VRL`):

- [frozen human-audit generation](../../outputs/probes/anima_person_count_m_frozen_audit_20260831/generation/run_config.json)
- [frozen human labels and earlier Codex qualification](../../outputs/probes/anima_person_count_exact_count_reward_qualification_20260831/qualification.json)
- [four-by-four hard-case probe](../../outputs/probes/anima_person_count_exact_count_grid4_probe_20260831/summary.json)
- [third-layout and three-pass-majority probe](../../outputs/probes/anima_person_count_exact_count_third_order_probe_20260831/summary.json)
- [dataset contract](../../datasets/anima/person_count_4_5_v1/dataset_spec.json)
- [dataset rationale and regeneration contract](../../datasets/anima/person_count_4_5_v1/README.md)
- [exact-count reward preset](../../vrl/config/presets/reward/codex_image_qa_anima_person_count.yaml)
- [Flow-GRPO canary configuration](../../vrl/config/presets/experiment/anima_preview3/online_grpo_codex_person_count_4_5_canary.yaml)
- [canary training metrics](../../outputs/anima_codex_person_count_4_5_canary/metrics.csv)
- [canary supervisor verdict](../../outputs/anima_codex_person_count_4_5_canary/run_verdict.json)
- [checkpoint-3 count evaluation](../../outputs/anima_codex_person_count_4_5_canary/heldout_count_exact_count_eval/summary.json)
- [checkpoint-3 exact reward rerun](../../outputs/anima_codex_person_count_4_5_canary/heldout_count_exact_count_eval_rerun_20260831/summary.json)
- [reusable exact-count evaluator](../../vrl/scripts/eval/anima_exact_count_checkpoint_eval.py)
- [checkpoint-3 general-quality evaluation](../../outputs/anima_codex_person_count_4_5_canary/heldout_luna_general_quality_base_vs_canary3/summary.json)
- [staged full-pass configuration](../../vrl/config/presets/experiment/anima_preview3/online_grpo_codex_person_count_4_5_fullpass.yaml)
- [stopped stage verdict](../../outputs/anima_codex_person_count_4_5_fullpass/run_verdict.json)
- [legacy CountGD installation manifest](../../data/external/countgd/install_manifest.json)
- [reproducible CountGD installer and verifier](../../vrl/scripts/rewards/install_countgd.py)
- [CountGD environment lock](../../vrl/scripts/rewards/countgd_environment_lock.py)
- [fresh-isolated versus legacy 64-image parity report](../../outputs/probes/anima_person_count_countgd_install_parity_20260831/qualification.json)
- [CountGD production model adapter](../../vrl/rewards/models/countgd_person_count.py)
- [CountGD HTTP reward adapter](../../vrl/rewards/functions/countgd_person_count.py)
- [CountGD reward preset](../../vrl/config/presets/reward/countgd_person_count_http.yaml)
- [CountGD service configuration](../../vrl/config/reward_service/countgd_person_count.yaml)
- [CountGD canary configuration](../../vrl/config/presets/experiment/anima_preview3/online_grpo_countgd_person_count_4_5_canary.yaml)
- [CountGD canary metrics](../../outputs/anima_countgd_person_count_4_5_canary/metrics.csv)
- [CountGD canary supervisor verdict](../../outputs/anima_countgd_person_count_4_5_canary/run_verdict.json)
- [all-64-prompt CountGD checkpoint-3 evaluation](../../outputs/anima_countgd_person_count_4_5_canary/heldout_all64_countgd_checkpoint3_eval/summary.json)
- [checkpoint-12 continuation metrics](../../outputs/anima_countgd_person_count_4_5_stage12/metrics.csv)
- [checkpoint-12 supervisor verdict](../../outputs/anima_countgd_person_count_4_5_stage12/run_verdict.json)
- [base 64-by-2 generation protocol](../../outputs/anima_countgd_person_count_4_5_canary/heldout_all64_base/run_config.json)
- [checkpoint-12 64-by-2 generation protocol](../../outputs/anima_countgd_person_count_4_5_stage12/heldout_all64_checkpoint12/run_config.json)
- [all-64-prompt CountGD checkpoint-12 evaluation](../../outputs/anima_countgd_person_count_4_5_stage12/heldout_all64_countgd_checkpoint12_eval/summary.json)
- [checkpoint-12 blinded policy-shift review](../../outputs/anima_countgd_person_count_4_5_stage12/heldout_all64_countgd_checkpoint12_eval/human_blind_review.md)
- [frozen reward-robustness annotations](../../datasets/anima/person_count_reward_robustness_v1/annotations.jsonl)
- [reward-robustness dataset contract](../../datasets/anima/person_count_reward_robustness_v1/dataset_spec.json)
- [reward-robustness rationale and use contract](../../datasets/anima/person_count_reward_robustness_v1/README.md)
- [reward-robustness validator](../../vrl/scripts/data/anima_person_count_reward_benchmark.py)
- [fixed detector bake-off evidence](../../datasets/anima/person_count_reward_robustness_v1/detector_bakeoff.json)
- [critic pilot contract](../../datasets/anima/person_count_critic_pilot_v1/dataset_spec.json)
- [critic pilot annotation and regeneration notes](../../datasets/anima/person_count_critic_pilot_v1/README.md)
- [critic pilot final image-level screen](../../datasets/anima/person_count_critic_pilot_v1/screening_annotations.jsonl)
- [critic pilot screen report](../../datasets/anima/person_count_critic_pilot_v1/screening_report.json)
- [critic pilot blind review evidence](../../datasets/anima/person_count_critic_pilot_v1/screening_reviews/protocol.json)
- [Distorted-5K immutable release](https://huggingface.co/datasets/xgklndsgkl/Distorted-5K/tree/29a752335154346102c82b0fa85f1c88bd91f87b)
- [Distorted-5K author release caveat](https://github.com/TheRoadQaQ/Predicting-Distortion/blob/ec4b3dd146de33ce17a6430c4823b778a9b9b684/README.md)
- [HumanRefiner immutable release](https://huggingface.co/datasets/Enderfga/HumanRefiner/tree/15b408cfde8c33cf6bb5b25114206c8b78b3e571)
- [HumanRefiner official class mapping](https://github.com/Enderfga/HumanRefiner/blob/d13975c722d384f5ec2e78207417dae49568ca20/example.jpg)
- [hard-negative v2 selection report](../../outputs/probes/anima_person_count_critic_hardneg_v2_generation_20260831/selection_report.json)
- [hard-negative v2 completed review shards](../../data/external/anima/person_count_critic_hardneg_v2_reviews/raw)
- [source-feasibility decision record](../../outputs/probes/anima_person_integrity_source_feasibility_20260831/summary.json)
- [interaction-source prompt contract](../../datasets/anima/person_integrity_interactions_v1/dataset_spec.json)
- [interaction-source rationale](../../datasets/anima/person_integrity_interactions_v1/README.md)
- [interaction-source prompts](../../datasets/anima/person_integrity_interactions_v1/source_prompts.jsonl)
- [untouched interaction eval prompts](../../datasets/anima/person_integrity_interactions_v1/eval_prompts.jsonl)
- [Manga109-s RT-DETRv4-X checkpoint](https://huggingface.co/tori29umai/rtdetrv4-x-manga109s_v2/tree/864c3bfb837a03ecc62557d5152a5ade5566489b)
- [CountGD official repository](https://github.com/niki-amini-naieni/CountGD)
- [CountGD project page](https://robots.ox.ac.uk/~vgg/research/countgd/)
- [DeepGHS anime head detector](https://huggingface.co/deepghs/anime_head_detection/tree/06604feee81983792a57c21081e539c0ae229833)
- [DeepGHS anime person detector](https://huggingface.co/deepghs/anime_person_detection/tree/e39c744c22432ad01f91dd254fe2b02c8d878b8c)
- [RTMLib](https://github.com/Tau-J/rtmlib)

### 11.16 The complete interaction population is frozen; critic fitting is still gated

The replacement source generation completed under one frozen runtime. The
matrix contains four policy arms (`base`, `countgd-6`, `countgd-12`, and
`quality-16`), three seed waves, and all 96 source prompts, for 1,152 images.
The untouched 32-prompt evaluation manifest was not used by any indexed
archive. A new corpus validator reopened every archive and rechecked the
prompt rows, metadata, sampling recipe, runtime, model identity, LoRA assets,
seed formula, anchor manifest, exact PNG set, encoded bytes, and decoded RGB
pixels. Its qualified population is:

| dimension | verified count |
|---|---:|
| train | 864 |
| calibration | 288 |
| expected people = 6 | 576 |
| expected people = 8 | 576 |
| each policy arm | 288 |
| each seed wave | 384 |
| unique PNG SHA-256 | 1,152 |
| unique decoded-RGB SHA-256 | 1,152 |

The corpus manifest SHA-256 is
`93b587422ea992fdcbaf194b1cb13dfb70d6a74a0ce027154e7025afeca91b45`;
the corpus report SHA-256 is
`4e75fce4ed40f27e41d2f8fad26730a5ef15da888093b7a0edc5e81345650153`.
Both an initial publish and a full rebuild-and-verify pass succeeded.

One generator process outlived its orchestration session after writing 70
images in `countgd-6/seed-20271031`. The partial directory was isolated and the
canonical archive was regenerated from an empty directory because the
generator has no resume contract. All 70 overlapping PNGs were byte-identical,
the two run-config hashes were identical, and the completed 96-image archive
passed the typed loader. The recovery receipt records this as an orchestration
interruption rather than a model, CUDA, or determinism failure.

Screen-judge qualification first corrected a false ground-truth label. Q1 had
inherited the pilot's `fragmented` label because dark objects under the shelves
were mistaken for legs without torsos. The source prompt explicitly asks four
shoemaking artisans to organize boot uppers and soles around open material
shelves. The image shows exactly four people behind the shelves and stored
boots below them, not detached human legs. Q1 is therefore an intentionally
difficult hard clean negative that guards against repeating this false
positive. The frozen sentinel set contains the five topology positives Q2--Q6
and the one clean negative Q1. Its manifest SHA-256 is
`f6ddee6e1dd476f25a5a25db50fe25411506efd46340c45651d23ca30035c2a6`,
and its dataset-spec SHA-256 is
`56f11897bbcb8d78c261f1beeb6ad07309a5d082128ad5db44eabdac7e32d237`.

Formal blind qualification used the same stepwise identity-and-limb tracing
rubric, each full image, and four quadrant zooms. All three lanes classified Q1
as `clean`:

| lane | positive hits | Q1 | locked submission SHA-256 |
|---|---:|---|---|
| A | 2 / 5 | clean | `e381b6c7fbc78513c8f71a71ac1fbfd586493b05d631180dc9f4b3928ef58c40` |
| B | 5 / 5 | clean | `a95e73a2855e597ac7e86f5a210d5150fd60cc6d9ee28235fd5fd3d8470b68c7` |
| J | 5 / 5 | clean | `c69849a3f82df93c984f0db8eb5f857dccba39d840120c895e7f620342685906` |

A/B candidate union recall is 5/5. J achieved 6/6 exact accuracy: 5/5 positive
hits and zero false positives on the clean sentinel. The qualification-package
SHA-256 is
`7d46beeb1e5856bd550c933674a616dcaed70b555fa71e854e527a195c0590f5`,
and the qualification-report SHA-256 is
`a10586f57713bd460f3ba3ef6346fb6b9886b8a31593bcfb0824edb610e5a718`.
A/B union specificity deliberately remains ungated; independent J's Q1 result
constrains clean specificity. All reviewers are Codex processes on one host, so
this establishes procedural independence rather than independent human ground
truth.

An earlier revision reported Luna `0/6`; both the attribution and denominator
are withdrawn. A root-level one-shot judge did return `clean` for Q1--Q6. Under
the corrected truth, the observed positive recall is `0/5`, while Q1's clean
decision is correct. However, its prompt/schema/result artifacts did not lock
the model identity, command, or invocation receipt. The result therefore
cannot be attributed rigorously to `gpt-5.6-luna` and is excluded from formal
qualification evidence. A separate lane-B raw response failed in the sandbox
before reading any image, and all six rows lack a valid candidate type; it is
wholly invalid. Those one-shot files were removed after this decision was
recorded. Luna has not passed this topology qualification and cannot be used as
the topology reviewer or online reward. The formal conclusion comes only from
the hash-locked A/B/J submissions above.

The formal screen now has narrower roles. A and B each review the complete
1,152-image population independently; their union provides only high-recall
diagnostic evidence. J no longer receives only that union. It independently
reviews the same complete population under fresh HMAC blind IDs without seeing
any prior decision, and only J's full-population verdict produces the final
candidate set. This covers every A/B union candidate while eliminating the
prior leakage that would arise merely from membership in a union-only J queue.
A/B/J overlap is computed only as a final diagnostic and never affects J's
decision.

Qualification and provenance now fail closed throughout the workflow:

1. The package loader verifies the protocol, corpus report, rubric copies,
   canonical paths, source/copy hashes, quadrant locks, and exact file
   inventory. Missing or extra files and changed rubric/source bytes fail.
   Staging packages are loaded completely before the atomic rename.
2. A locked submission binds the package report, rubric, reviewer identity,
   each shard's SHA-256 and row count, blind-ID order, and crosswalk. The full
   package is reloaded before locking, and a newly written invalid lock is
   rolled back.
3. Downstream stages cannot trust a `qualified` field alone. They reopen the
   qualification package and A/B/J locked submissions, recompute the 5/5 union
   recall and J's 5/5 hits plus zero false positives, and only then permit
   packaging or finalization.
4. The adjudication parent binds the population package, qualification report,
   A/B submission hashes, and union digest. Finalization also requires three
   distinct reviewer identities and verifies that J's queue equals the complete
   corpus exactly.

Four implementation seams were corrected before labels are opened:

1. The annotation consumer now reads a hash-pinned full-population source
   manifest. Detector disagreement and the old `uniform/mined/control`
   workspace remain only as an explicit legacy exporter; they no longer own
   annotation semantics.
2. `expected_people` and `source_stratum` are data, not 4/5-person or
   mined/control branches. The hard-intact strata are declared by protocol.
3. The draft critic policies now match 864 training and 288 calibration rows,
   with 24 calibration prompt clusters and a 144/144 target distribution for
   six and eight people.
4. Frozen calibration observations are parsed once from the exact hash-verified
   bytes into immutable typed detections. Non-finite scores, invalid geometry,
   out-of-bounds or undersized boxes, score-floor violations, output-order
   changes, and detection-cap violations fail closed; calibration labels must
   also match the locked pixel dimensions.

The complete-population screen is now final and it is a formal `stop`. J
reviewed all 1,152 images and marked 19 candidates across 10 prompt groups,
three policy families, and all three seed waves. The train split contains only
10 candidates across seven prompt groups; calibration contains nine across
three prompt groups. The image, prompt-group, train, and calibration-prompt
gates therefore fail. The locked J submission SHA-256 is
`4a1c540df1f1b2d201ba4fe895ce1da39c8fdd6792a9aaa200a620a6c6e0980f`;
the recomputed stop/go report SHA-256 is
`13ff68b12d655522834ee0db6c8020710e86f360b8e561fb264d92520c1f828d`.

This result does not authorize annotation, critic training, or Flow-GRPO. The
pre-registered failure action is binding: add one complete four-arm seed wave
under an append-only protocol amendment, then independently screen that entire
leaf and recompute the gates over the canonical-ID union. Gates, split
assignments, and the old evidence remain immutable. A passing combined screen
will authorize full-population dual-lane box annotation; it will never select a
positive-enriched training subset.

New evidence paths:

- [generation protocol](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/protocol.json)
- [recovery receipt](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/recovery_report.json)
- [qualified corpus manifest](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/corpus_manifest.jsonl)
- [qualified corpus report](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/corpus_report.json)
- [corpus validator](../../vrl/scripts/data/anima_person_integrity_corpus.py)
- [screen protocol](../../datasets/anima/person_integrity_interactions_v1/screen_protocol.json)
- [screen sentinel contract](../../datasets/anima/person_integrity_screen_sentinels_v1/dataset_spec.json)
- [screen sentinel rationale](../../datasets/anima/person_integrity_screen_sentinels_v1/README.md)
- [screen sentinel manifest](../../datasets/anima/person_integrity_screen_sentinels_v1/sentinels.jsonl)
- [screen rubric](../../datasets/anima/person_integrity_screen_sentinels_v1/rubric.md)
- [Q1 source prompt](../../datasets/anima/person_count_4_5_v1/train_prompts.jsonl)
- [qualification package report](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/qualification_package/package_report.json)
- [reviewer qualification report](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/reviewer_qualification.json)
- [locked A review](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/qualification_reviews/lane-A/submission.json)
- [locked B review](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/qualification_reviews/lane-B/submission.json)
- [locked J review](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/qualification_reviews/lane-J/submission.json)
- [locked full-population J review](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/adjudication_reviews/lane-J/submission.json)
- [formal source stop/go report](../../outputs/probes/anima_person_integrity_interactions_v1_generation_20260831/screening/source_stop_go.json)
- [screen package and evidence-chain implementation](../../vrl/scripts/data/anima_person_integrity_screen.py)
- [generic annotation source and review lifecycle](../../vrl/scripts/data/anima_person_count_annotations.py)
- [critic calibration manifest lifecycle](../../vrl/scripts/data/anima_person_critic_manifest.py)

### 11.17 The first append-only wave is valid; automated adjudicators are not

The pre-registered failure action was executed without changing the source
gate. Seed wave `20272031` added one complete 96-prompt archive for each of
`base`, `countgd-6`, `countgd-12`, and `quality-16`. The extension contains 384
images: 288 train, 96 calibration, 192 requested-six, and 192 requested-eight.
All 384 encoded PNG hashes and decoded RGB hashes are unique. Generation used
the same 512-square, 20-step, CFG 4.5 recipe and one frozen runtime; every arm
completed without OOM, retry, or archive drift. A full typed replay passed
after publication.

The create-only extension evidence is bound by these SHA-256 values:

| artifact | SHA-256 |
|---|---|
| generation protocol | `94f8eb24bc463c0107ad1603720cf46a46fa2de97ca0b9d0f2bf4ff546830fba` |
| corpus manifest | `aaca7a05808916f66a403ec0cbd22182ad22d2f7de9bb078e643ec09a1cb7c6c` |
| corpus report | `ce6cb7fecb2e97caa351e26a050e823090fab56e2d03caec34d9eb862353aad3` |
| completion marker | `a9419ea80907af960273d3f4f569793f76d0305c74d9415a2f06dd98bdbcef0a` |
| extension screen protocol | `d6a6f60d91a5ff9337a9ec71c080227dc48f2819b14fedac15cc40d15a27d4ba` |
| qualification package | `6843515139d619b9741cbcb22c36c8d11a7345bf1b96f776d0beccd491d326a0` |

The new leaf did not proceed to population review because the independent J
qualification failed six times. A and B remained qualified: their union hit
all five positive sentinels, with locked submission SHA-256 values
`fa3d8472fdb4ad91948e76d912ec7ef28a9d69b6fff63c75f87d4b75800afb04`
and `799e7863a5964a0db627dbeb1dc419ee29d18d82503c67b4dc7cdc077a1344aa`.
The independent J outcomes were:

| attempt | reviewer policy | positive hits | clean false positives | qualification report SHA-256 |
|---|---|---:|---:|---|
| 1 | isolated Codex process | 4 / 5 | 1 | `9d3f1b7962e12de7f6e1cd74ba8f152df19effbadd9459781b51a0b65f6c7cec` |
| 2 | isolated Codex process | 2 / 5 | 0 | `1c58a5174f9790d702b565f4d39f9f6a9ac4e8cf1a4c440b1ed42427e24bcb96` |
| 3 | isolated Codex process | 2 / 5 | 0 | `41fdb93a9a71d06c663f190c93a7c25e00a9defc694386da40044691d9d0674e` |
| 4 | isolated Codex process | 2 / 5 | 0 | `489c0053711113887ddd79cfbefd7a2bad6f678fecfd78a8c94bd840f0e1c01b` |
| 5 | Claude Opus 5 | 1 / 5 | 0 | `812d5853c9c43aa2bed2d1e387016cff0eabc64af14bec861880d3f348c82c39` |
| 6 | Claude Opus 5, one fresh context per sentinel | 1 / 5 | 0 | `e9c134717732e37a4ec7b89a0051fe3a235bc08bc7b65283763383e419be720d` |

Attempt 5 is the first external VLM attempt in this line with an attributable
invocation receipt. Claude Code 2.1.252 resolved the requested `opus` alias to
`claude-opus-5`, inspected only the public J package under a Read/Glob/Write
tool allowlist, used no web requests, and cost USD 1.561889. Its receipt
SHA-256 is
`a6e970bf93e96573df088447b669beba838c855d1ccc4ff9e0d0cd9dd9690584`.
Attempt 6 pre-registered six independent calls, fixed blind-ID order, no
retries, no partial-result inspection, one full image plus four quadrants per
context, and a USD 0.45 per-call cap. Every call resolved to
`claude-opus-5`; total cost was USD 1.509662. The policy, prompt, orchestrator,
raw responses, copied-asset hashes, locked submission, and rejected report are
archived together. Its policy SHA-256 is
`514fd81dadc90afac6630655f04fabec27c4c932068de977f6f39d3c1f3b76f8`,
invocation-receipt SHA-256 is
`2e25ab2dece12222ac31bcf2980b474a5013246fcd177ca75146026d481279d1`,
and locked-submission SHA-256 is
`04e877faf12cfaf63f2d44f5046641d8468ddad71a854f398ae0373ab3877aa6`.
The unchanged 1/5 result rejects cross-image attention dilution as the cause
of attempt 5's low recall; more prompt shaping or Opus budget is not justified
by this evidence.

All six rejected reports and submissions remain immutable under
`screening/rejected_qualification_attempts`; the canonical J/report paths are
empty. Decisions were never edited in response to sentinel feedback.

This closes another-Codex-thread, Claude, Luna, and the already rejected Qwen-VL
branches as automatic topology ground truth. A/B union cannot be copied into J
because that would remove adjudicator independence. Old leaf sentinel answers
also cannot be translated to the new blind IDs: the implementation could not
detect that contamination, but the submission would falsely claim that prior
decisions were unseen. The current leaf therefore requires a fresh external
human reviewer who has not seen class counts or prior outcomes. Until that
reviewer passes 5/5 positive recall and zero clean false positives, population
screening, composition, annotation, critic fitting, and RL remain prohibited.

Preparation that does not consume labels is complete. A 72-row train-only
manifest is the byte-exact projection of `source_prompts.jsonl[0:72]` and has
SHA-256
`cf08328bdca77f8472ac60e3929f06beb109142f31e049b209598cd3f1699bdb`;
the 24 calibration rows are no longer reachable through the online dataset
preset. The bounded Anima canary composes the shared `flow_matching_dppo`
recipe, not basic GRPO: LoRA, four prompts by eight rollouts, two PPO epochs,
full-step SDE, non-streaming retained replay, and three outer rollout/trainer
iterations. This is an asymmetric Gaussian-KL Flow-DPPO trust region, not
classical TRPO. Two PPO epochs permit at most six optimizer-step attempts over
those three iterations; zero-advantage or scaler skips can reduce the committed
count, so a completion artifact must report attempts and commits rather than
calling this "exactly three optimizer updates."

The bounded-canary execution boundary has now landed as one complete path. It
does not weaken or reuse the production deployment guard. The dedicated
entrypoint freezes a content-addressed critic bundle, dataset spec, 72-row train
manifest, and untouched 32-row eval manifest; runs the canonical qualification;
receives a process-local, non-serializable, single-use qualification receipt;
and activates only the canary reward model. The public YAML still cannot mint
authority. The production reward continues to require separate deployment
qualification.

The implementation also closes the evidence gaps found by red-team review:

1. Training parses authenticated manifest bytes once and injects immutable
   `PromptExample` values into the online runner. It never reopens the path.
   The generic prompt default no longer hard-codes `text_to_video`; Anima now
   derives the correct `text_to_image` task from its family registry entry.
2. Qualification replay recomputes every decision and every registered gate
   from the published raw observations before issuing the receipt and again
   during completion verification. The receipt carries the exact staged report
   bytes that passed replay and never reopens the writable published path. A
   hash-consistent invented green report is insufficient.
3. Trusted producer code loads each `checkpoint.pt` once and emits a strict JSON
   receipt. Public completion verification validates that receipt, metadata,
   hashes, and LoRA safetensors without invoking the unsafe
   `weights_only=False` checkpoint loader. `checkpoint-final` must match
   checkpoint 3's actual adapter hashes, not merely its step and model label.
4. All 96 retained rollouts are bound to exactly three reward requests, twelve
   eight-sample groups, the first twelve sequential train prompts, their typed
   metadata, the critic bundle ID, generic request/result debug streams, and the
   configured artifact root. Completion verification reruns the frozen critic
   on every retained pixel tensor and requires exact agreement with the stored
   reward replay. JSON assets and compact JSONL streams use separate encoders.
5. A completion report states explicitly that held-out evaluation was not run,
   reports the actual BF16 optimizer attempts/commits, remains
   `deployment_authorized=false`, and publishes a read-only exact inventory.
   The evidence is a trusted local producer attestation, not a signed
   third-party cryptographic proof.

This prepares the real repository mechanism: Flow-DPPO with an asymmetric
Gaussian-KL trust region. It is not classical TRPO and is not renamed as such.
No formal canary has been launched. The blocker remains unchanged and is now
narrow: a fresh external human reviewer must first qualify, the combined source
screen must then pass, and the resulting dual-lane/adjudicated boxes must produce
a critic bundle that passes the frozen qualification gates. Until those facts
exist, bundle/deployment inputs remain intentionally unresolved and RL remains
prohibited.

New evidence and preparation paths:

- [extension generation protocol](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/protocol.json)
- [extension corpus report](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/corpus_report.json)
- [extension screen protocol](../../datasets/anima/person_integrity_interactions_v1_extension_seed_20272031/screen_protocol.json)
- [rejected qualification attempts](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/rejected_qualification_attempts)
- [Claude Opus 5 invocation receipt](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/rejected_qualification_attempts/attempt-005/invocation_receipt.json)
- [independent per-item Opus attempt](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/rejected_qualification_attempts/attempt-006)
- [train-only prompt manifest](../../datasets/anima/person_integrity_interactions_v1/train_prompts.jsonl)
- [train-only dataset preset](../../vrl/config/presets/dataset/anima_person_integrity_interactions_v1.yaml)
- [Anima Flow-DPPO canary](../../vrl/config/presets/experiment/anima_preview3/online_flow_dppo_person_integrity_canary.yaml)
- [dedicated canary entrypoint](../../vrl/scripts/anima_person_critic_canary.py)
- [process-local canary authority](../../vrl/rewards/anima_person_critic_canary.py)
- [qualification replay](../../vrl/scripts/eval/anima_person_critic_qualification.py)

### 11.18 Human-J provenance is now an enforced protocol boundary

The first external-human handoff exposed a protocol contradiction before any
labels were accepted. The extension's historical v1 screen protocol freezes
all reviewers as same-host procedural Codex processes. A human J response
therefore cannot be attached to that package without falsifying the protocol,
and a six-sentinel human qualification cannot be followed by a machine J on
the 384-image adjudication queue. Reviewer identity is part of the evidence,
not an interchangeable implementation detail.

A create-only v2 protocol and fresh qualification package now encode the
actual policy: qualification A/B and population A/B remain procedural Codex,
while qualification J and adjudication J require the same isolated external
human. Historical v1 protocols, packages, submissions, rejected attempts, and
replay behavior remain unchanged. The frozen v2 identities are:

| artifact | SHA-256 |
|---|---|
| human-J screen protocol | `4e22d7830209c3613e403a82c7136df030d898acce2a407704cdda4230c04021` |
| fresh qualification package | `166b31b20f7b871d0c5eeca1d908dcc92bdd8492f21b5c13357e7f94122003c9` |
| lane-A key fingerprint | `8b52f7ff277727a97d5b20664d9aab412cab8ce91fc0140d495babd1325d8220` |
| lane-B key fingerprint | `fbbb4803ab3409e2890f671b84758afe09e7df7a4baef578bb544b07c02cf742` |
| lane-J key fingerprint | `62c5c576dbf5e4345318445d776a0e57dd7274a41556247b02437af912dfc80d` |

The provenance direction is deliberately acyclic:

```text
v2 protocol
  -> fresh qualification package
    -> public-only handoff ZIP
      -> human attestation + response shards
        -> response receipt
          -> v2 J submission
            -> v2 reviewer qualification
              -> population/adjudication packages
                -> stop/go, composition, boxes, critic, and canary
```

The importer authenticates one ZIP snapshot, compares every J blind ID, full
image, and quadrant with the operator's fresh package locks, validates the
attestation and response bytes, and publishes the handoff, attestation,
receipt, shards, and submission together. Downstream qualification replay
reopens and rehashes that chain. A normal `lock-review` cannot create either a
qualification or adjudication J submission under v2 without the receipt. The
receipt explicitly records `qualification_performed=false` and
`training_use_allowed=false`; successful format validation alone cannot mint
training authority.

The reviewer instructions are a packaged protocol asset rather than a large
workflow constant. Their registered SHA-256 is
`ce746684d9b5669af1fc26373e9ea22a40f6027d82ec4a95150be87f6a50c354`,
and a built wheel was inspected to confirm the Markdown asset is present. The
small contract module remains separate because schema strings are a public
producer/consumer boundary. ZIP parsing, response validation, import, and
submission replay also remain separate narrow functions because each is a
security or protocol boundary; flattening them would obscure which bytes each
stage authenticates.

Release acceptance passed all 71 tests under `tests/scripts/data`, targeted
Ruff checks, `py_compile`, real A/B submission replay, and wheel asset
inspection. The final derived-structure audit also removed three validated but
otherwise unread fields (`ScreenPackage.corpus_report_sha256` and
`ValidatedSubmission.lane`/`review_origin_receipt`). Their persisted inputs
remain validated at load or replay time; only the redundant in-memory copies
were removed. No P0, P1, or P2 issue remained after that audit.

Fresh isolated A and B qualification reviews are complete and hash-locked:

| lane | candidates | positive hits | clean false positives | submission SHA-256 |
|---|---:|---:|---:|---|
| A | 4 | 4 / 5 | 0 | `bf55077e811ba18082082fdb430eb9e5234e9ebfe2ae1750e76324d65207ce79` |
| B | 2 | 2 / 5 | 0 | `7fab9d402c70731abed68d0032ad0dbfaa88cda678512e7b5d64e14d5ef45056` |
| A-or-B union | 5 | 5 / 5 | 0 | n/a |

The formal public-only human ZIP is deterministic, contains 36 sorted
read-only stored members, and has no private crosswalk, A/B material,
canonical IDs, expected labels, prompts, requested counts, seeds, or prior
outcomes. Its SHA-256 is
`37775b74f34e378fd15f52fb6f0f1248eff330c6d8a603128c1a05563a0652c3`.
This establishes procedural isolation only: the operator must send only the
ZIP under an ordinary filename, transmit the digest through a separate
channel, and use a human who has no repository, label, package-private,
other-lane, prior-outcome, or image-search access. The attestation is not a
cryptographic proof that a reviewer told the truth.

No J response exists yet. Consequently no reviewer-qualification report,
population package, adjudication queue, source stop/go, composed labels,
critic bundle, or Flow-DPPO run has been created. This is the intended hard
stop. The next evidence-producing action is the external human's six-sentinel
response; if it passes all five positives with zero clean false positives, the
same reviewer must later inspect the 384-image J queue. Training before that
point would optimize another unqualified automated judge and repeat the root
failure this sprint identified.

New protocol and handoff paths:

- [human-J v2 screen protocol](../../datasets/anima/person_integrity_interactions_v1_extension_seed_20272031_human_j_v2/screen_protocol.json)
- [fresh v2 qualification package](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/human_j_v2/qualification_package/package_report.json)
- [locked v2 lane-A review](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/human_j_v2/qualification_reviews/lane-A/submission.json)
- [locked v2 lane-B review](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/human_j_v2/qualification_reviews/lane-B/submission.json)
- [external-human qualification ZIP](../../outputs/probes/anima_person_integrity_interactions_v1_extension_seed_20272031_generation_20260901/screening/human_j_v2/external_handoffs/qualification-j.zip)
- [external-review runbook](../anima_person_integrity_extensions.md)
- [human-review handoff implementation](../../vrl/scripts/data/anima_person_integrity_review_handoff.py)
- [human-review protocol schemas](../../vrl/scripts/data/anima_person_integrity_review_contract.py)

### 11.19 The planned critic population has an append-only policy revision

The parent generated corpus and its complete extension wave do not fit the
original critic policy population. Their corpus reports contain 864/288 and
288/96 pre-exclusion train/calibration images respectively, so a complete
four-arm, four-wave population contains 1152 training images and 384 reserved
calibration images. The calibration split has 24 prompt clusters, with 16
images per cluster, 96 images per arm, and 192 images for each requested-count
stratum. These are population-shape facts, not a claim that external review or
the final data gate passed.

A new packaged v2 policy pair records only those changed
population facts. Both JSON schema identifiers remain `/v1`: the document
shape, network, fixed 12-epoch optimization recipe, threshold search,
constraints, and minimum-denominator gates did not change. Directory identity
and exact policy bytes distinguish the revision; changing the schema would
misrepresent a data-population revision as a protocol-format migration. The
frozen identities are:

| policy | SHA-256 |
|---|---|
| v1 training, replay-only | `bb185c92ad5028659960545cd60fb79827641be35623eadb69afaf5bb472e0dc` |
| v1 calibration, replay-only | `e773bf195167023df390852c3e64f4eb6fd5d844475e4b22005f63f957060c1b` |
| v2 training, create-new default | `dfea31e6120cfde49fc9602162e9d59241df1cfb1e300c553dc571604baf5ff1` |
| v2 calibration, create-new default | `403cc7cbbcbc86f2a248c74019b15cce8bea32880934ee227be8570ca433b1a1` |

The policy registration is a paired dataclass because training and calibration
are one versioned invariant. New training and calibration publications accept
only the current v2 pair. Historical v1 training artifacts and complete v1
bundles remain replayable, while a rehashed v1-training/v2-calibration bundle is
rejected. This also closes a pre-existing hole: bundle loading previously
authenticated the training policy but merely hash-bound arbitrary calibration
policy bytes.

The four JSON documents now live under `vrl.rewards.assets` as their only source
of truth and are included in built wheels. Registry lookup verifies each asset
against the frozen SHA-256 above before comparing embedded bytes. The previous
repository-root `datasets/` location worked only from a source checkout and made
installed bundle replay fail because those files were absent from the wheel.

What intentionally remains unchanged is the v1 policy bytes, all persisted
bundle/run/report schemas, the bundle inventory, the critic architecture and
optimizer, the threshold grid, and the minimum denominators. Those thin
artifact loaders and schema constants remain protocol boundaries rather than
being flattened for line-count reduction. No reward config consumes policy
counts directly. This revision is now reserved for a future qualified
full-factorial v2 population; it does not authorize reusing the old marginally
balanced population, bypass external-human review, create annotations, produce
a critic bundle, launch Flow-DPPO, or claim held-out improvement.

New policy paths:

- [v2 training policy](../../vrl/rewards/assets/anima_person_critic_protocols/v2/training_policy.json)
- [v2 calibration policy](../../vrl/rewards/assets/anima_person_critic_protocols/v2/calibration_policy.json)
- [paired policy registry](../../vrl/rewards/anima_person_critic_bundle.py)

### 11.20 The first blocker is a count-by-interaction confound, not training length

A joint-distribution audit found that the v1 prompt corpus was only marginally
balanced. Every six-person row used `paired_guidance` or `linked_motion`, while
every eight-person row used `mutual_support` or `shared_reach`. Train,
calibration, and eval therefore each contained four populated joint cells and
four empty cells. The existing test checked each marginal count and four-row
window diversity, so it could not detect this confound.

This invalidates separate claims about requested-count or interaction
generalization. A reward or policy can learn the four observed count-action
bundles and still fail on every missing combination. Extending training on
those bytes would only make the confounded estimate more precise.

The append-only `person_integrity_interactions_v2` corpus fixes the source of
the problem without increasing the GPU budget:

| split | joint cells | rows per cell | total rows |
|---|---:|---:|---:|
| train | 8 | 9 | 72 |
| calibration | 8 | 3 | 24 |
| eval | 8 | 4 | 32 |

The deterministic builder alternates requested count in consecutive four-row
windows. Each individual window retains two six-person and two eight-person
prompts plus all four interactions; each two-window cycle contains all eight
`expected_people x interaction_type` cells exactly once. Changed six/eight and
three/four wording remains natural, and every action/concept identity is
count-qualified. The builder verifies the frozen v1 dataset-spec SHA before
rendering, publishes create-only, and can reproduce every checked-in byte.

Frozen v2 identities are:

| asset | SHA-256 |
|---|---|
| dataset spec | `d2cd61f3c94519ace28048c21d2cb4a63a98ab196649809e3bf9bdab54b7f4f1` |
| source manifest | `84ab67f23d0dc646432690a1473fddbee48e76a74146edd8ed62a3fc3bf7f941` |
| train projection | `7ee5d6c238d466e88cecf988a1aaca2f0df6570b1ee5cec7d4721582e93eae60` |
| eval manifest | `cc817b1f82bf3508bb1411b87234f0a00750a8e282941b8da26075f619fe851c` |

The bounded canary now registers this v2 dataset spec and composes the v2
dataset preset. Its frozen 4/5 critic benchmark remains only a historical
reward-hacking rejection gate replayed in a fresh process; it is not described
as fresh 6/8 qualification.

No old image or review evidence was renamed. v1 generation, screen,
qualification, annotation, and human-J packages bind different prompt bytes
and semantic IDs, so they remain historical pilot evidence. The v2 corpus still
needs a complete fresh four-arm/four-wave image population, new blinded labels,
a qualified critic bundle, an untouched fresh 6/8 critic qualification, and a
paired full-population post-canary review before bounded continuation. No GPU
training should start before those gates exist.

New dataset paths:

- [v2 rationale and lifecycle](../../datasets/anima/person_integrity_interactions_v2/README.md)
- [v2 dataset specification](../../datasets/anima/person_integrity_interactions_v2/dataset_spec.json)
- [deterministic factorial builder](../../vrl/scripts/data/anima_person_integrity_factorial.py)
- [v2 dataset preset](../../vrl/config/presets/dataset/anima_person_integrity_interactions_v2.yaml)
- [v2 reproducibility test](../../tests/data/test_anima_person_integrity_interactions_v2.py)

---

## 11. 目标 C：NSFW 场景下的 prompt 遵循（2026-09-03，训练进行中）

用户在 A（压制）/ B（条件控制）/ C（能力）中选了 **C**。硬约束：**仅成年角色**——
数据层用 `adult` 标签 + 剔除 `school_uniform` 等青少年语境标签写死（2200 行中剔除
298 行；第一次正则审计漏掉了下划线形式的标签，教训：**必须扫描下划线标签**）。

### 11.1 兑现半边已饱和，遵循半边有真实空间

200 条 adult prompt（explicit 100 / questionable 100），base 生成，客观测量：

| 半边 | 度量 | base | 结论 |
|---|---|---|---|
| 兑现 | Falconsai P(nsfw)>0.35 触发率（explicit 档） | **97%**（真失败 1/100，灰区 2） | **饱和**。分类器双峰（≥0.93 或 ≈0），组内奖励恒定→无梯度。只能当护栏 |
| 兑现 | 同上（questionable 档） | 79% | 内衣/泳装本就在裸露分类器阈值下方，是口径错位不是失败 |
| **遵循** | WD14 对 prompt 自带标签的召回（白名单） | **0.913，完全正确 61%** | **有空间**（SFW anatomy 上是 0.980 / 81%） |

丢得最多的白名单标签（missed/asked）：`lingerie` 12/20、`ribbon` 11/32、`bent_over`
6/13、`bottomless` 5/8。锐度基线（Laplacian var ×1e-3，200 张）：mean 19.07 / median
17.10 / p10 4.97。

### 11.2 奖励设计

- **驱动**：`tag_adherence`——WD14 SwinV2 v3 直接经 onnxruntime 进程内加载（`.venv`
  已有 onnxruntime，零新依赖），对 `metadata.adherence_tags` 的召回。确定性、零判官噪声、
  CPU 毫秒级。**与 imgutils 逐位一致**（一致性门：8 张图 max |Δp| = 0.0000，≥0.35 检出集
  0/278 差异），因此训练奖励与基线测量是同一个量。
- **v1 白名单**：发型/瞳色/服装与内衣/身体状态/姿势取景/表情/场景。**排除**性行为与
  显式解剖类标签——tagger 在生成图上对这些标签的准确率未验证，不能拿未验证的量当奖励。
  113/1649 训练行只剩 1 个目标标签（二值奖励），v1 接受，记为可选 `min_tags` 过滤。
- **护栏**：`image_sharpness` 0.25。v1 **不用 PickScore**：单卡上它只能跑 CPU（CLIP-H，
  每次 256 图更新 10+ 分钟），且驱动是语义信号而非锐度本身，锐度文档警告的"被噪声骗"
  不适用；退化由评测端兑现率 + 锐度 + 肉眼兜底。
- **合并**：`normalized_sum`（[0,1] 召回与锐度尺度不同，避免方差支配）。
- **数值**：继承 §1 平价配置（compile off、生成/replay 批 1）、KL 0.004、256 样本/更新，
  supervisor 挂 grad_norm>0.5 中止 + 首步平价门。

落地：`vrl/rewards/{models,functions}/tag_adherence.py`、注册表、`reward/tag_adherence.yaml`、
`dataset/anime_safety_c_adherence.yaml`、`experiment/anima_preview3/online_grpo_tag_adherence_nsfw.yaml`、
`datasets/danbooru/safety/{train,eval}_c_adherence.jsonl`、13 个单元测试（`tests/rewards` 401 全绿）。
基线固化：`outputs/nsfw_compliance_base200/baseline_c_adherence.json`。

### 11.3 预定判据（训练前写死）

成功 = held-out（170 行）白名单召回显著 > 0.913 **且** 完全正确 > 61%，**同时**
explicit 兑现 ≥ 95%、锐度 median ≥ 15、肉眼抽查无退化。任一护栏破 = 不算成功。
评测脚本：`vrl/scripts/eval/anima_tag_adherence_eval.py`（训练奖励 == 评测打分器）。

### 11.4 结果：零结果，但根因是运行长度不足（2026-09-04）

Held-out 170 行，**seed 逐行对齐**（下方 11.5 记录了这个陷阱）：

| 臂 | 召回 | 完全正确 | explicit 兑现 | 锐度 median |
|---|---|---|---|---|
| **base** | **0.9232** | **67.1%** | **100%** | **16.08** |
| ck5 | 0.9194 | 62.9% | 97.6% | 15.61 |
| ck10 | 0.9178 | 64.1% | 97.6% | 15.15 |

- ck5 − base：−0.0038（z=−0.54），W/T/L **10/144/16**，自举 CI95 [−0.018, +0.010]
- ck10 − base：−0.0054（z=−0.64），W/T/L **17/139/14**，自举 CI95 [−0.023, +0.010]

按 §11.3 判据：**不成功**（两者都略负且不显著）。

**但这不是对目标的证伪，是功率不足。** 三个数字互相印证模型基本没动：

| 证据 | 值 |
|---|---|
| 实际优化器更新 | **10 次**（每个 metrics epoch = 1 次更新：16 prompt × 16 样本，ga=4 拆 4 个累积切片） |
| 见过的 prompt | 160 / 1649 = **9.7%** |
| 最终 KL 惩罚 | **0.000287** |
| held-out 打平 | 139–144 / 170 |

Flow-GRPO 在 SD3.5 上是数百至上千次更新；10 次对 2B 模型等于没训。

**机制侧全部通过**，这轮的正资产：平价 10 epoch 恒 0.0000；grad_norm 0.0016–0.0055（早停线 0.5 从未触发）；组内展布 0.135→0.205 且 `adv_zero_rate=0`（奖励确实产生梯度，避开了 OCR 线的常数-advantage 坑）；`tag_adherence` 与 imgutils **逐位一致**（max |Δp|=0.0000，≥0.35 检出集 0/278 差异）；评测脚本在 base 图上复现基线到 0.913 / 103 行完全正确。

### 11.5 新踩到的陷阱：eval seed 随 manifest 行数错位

`anima_fixed_eval._generate` 用 `seed = args.seed + index`，index 是**在被评测行列表中的位置**。首次比较时 base 图来自 200 行 manifest、ck5 来自 170 行过滤 manifest，**同一条 prompt 两边拿到不同 seed**，配对比较失效。

影响是实质性的：未对齐时 base 召回 0.9127，seed 对齐后 **0.9232**（+0.011），比任何 checkpoint 的效应都大，且**方向相反**（未对齐时 ck5 看起来 +0.0067，对齐后是 −0.0038）。

**规则：任何 before/after 必须用同一份 manifest 生成两臂**（`--images-dir` 复用旧图只在行数与 manifest 完全一致时才安全）。这与 §9 的「held-out 也要校准」是同一条纪律的另一种形态。

### 11.6 下一步：提速后重跑长 run

50 分钟/更新的构成：生成 256 张逐张约 8 分钟（平价要求 batch=1），**replay 256 样本 × 20 去噪步 + backward 约 40 分钟**——瓶颈在后者。

| 杠杆 | 效果 | 代价 |
|---|---|---|
| **`timestep_fraction` 1.0 → 0.25**（`timestep_selection=random` 去相关） | replay/backward 降 4 倍 → 约 15 分钟/更新 | 每次更新只对 1/4 去噪步给梯度（DanceGRPO 标准做法） |
| 减小 batch（16×16 → 8×8） | 再降 4 倍 | advantage 更噪——正是 luna run 的失败模式，不建议 |
| 恢复批量生成 | 只省 8 分钟 | 重新引入 0.2 log-prob 漂移，**不可接受** |

建议：`timestep_fraction=0.25` + 保持 256 样本/更新，一夜可跑 60–80 次更新（比本轮高一个量级），不牺牲 advantage 质量或数值正确性。

**已启动（2026-09-04 18:08）**：从 `checkpoint-10` 续训（保留已有 10 次更新），只改一个变量：

```
trainer.resume_from=outputs/anima_tag_adherence_nsfw/checkpoint-10
actor.timestep_fraction=0.25  actor.timestep_selection=random
trainer.total_epochs=50  trainer.save_freq=10
```
其余与 §11.2 相同（supervisor 健康门、首步平价门）。目标 = 累计 50 次更新，
checkpoint 20/30/40/50。评测规则：每个 checkpoint 对 `base_matched`（同一份 170 行
manifest 生成，§11.5）做配对比较；同时看 KL 是否随更新次数上升——**KL 不动**则下一个
杠杆是 `clip_ratio`（当前 3e-3 极紧，可能限制了每次更新能移动的幅度），而不是再加步数。

## Worktree cleanup (2026-09-04)

This audit examined the uncommitted changes, their consumers, tests, and existing
experiment artifacts. It did not commit changes, launch training, or alter the
running experiment. Earlier experiment results above remain historical evidence,
not an endorsement of every implementation or causal interpretation.

### Removed and archived

- Removed the unlaunched person-critic continuation, fresh-qualification,
  held-out, and blind-policy-review implementation, its private policies,
  qualification dataset, presets, and tests: 20 files. The chain had no deployed
  artifact consumer and failed its own end-to-end contracts (review identity
  mismatch and missing held-out completion evidence). Retaining incomplete
  machinery would not establish that the critic is qualified.
- Removed six one-shot hand, natural-language, and FRELAN probe manifests after
  checking that no live recipe, source module, or test referenced them. Their
  generated images and crops were moved to the external recovery directory,
  preserving the same lifecycle for inputs and outputs (approximately 38 MB).
- In total, 26 files containing 11,970 lines were removed. Removed files were
  untracked additions, so the reduction is not visible in `git diff --stat`.
- Removed OCR qualification v3/v4 compatibility branches: no real archived
  protocol inputs used them. Retained v1/v2/v5/v6 and explicit rejection tests.
- Archived stale ignored build output after detecting that setuptools included
  deleted modules in a wheel built over the old `build/` tree. A clean rebuild
  excludes every retired path and retains the four remaining critic policy assets.

Recovery root: `/home/mingfeiguo/Desktop/vrl-cleanup-backup-Zb2c57`.
`uncommitted-before.tar.gz` preserves all 352 scoped modified/untracked files
from before cleanup. `retired/` preserves the moved probe outputs; `README.md`
in the recovery root explains selective recovery. The original FRELAN archive
was not changed. Do not restore the snapshot over the current worktree wholesale.

### Retained boundaries and correctness fixes

- Kept the critic data collection, annotation/review handoff, training,
  calibration, bundle, and existing canary path. These have real consumers.
  Deployment qualification remains mandatory and fail-closed; removing the
  incomplete extension does not make a critic eligible for production or a
  longer training run. Fresh qualification still needs a complete future design.
- Kept multi-objective advantage/config integration, image SFT target support,
  Anima model/generation fixes, reward deployment configuration, and sample
  identity propagation. These are functional changes, not disposable probes.
- Fixed streaming training bypassing the initial precision-drift guard when
  precision correction was enabled. Ordinary and streaming updates now share
  `_check_initial_precision_drift` and enforce it before backward/optimizer work.
  Regression cases cover allowed drift, excessive drift, and filtered batches.
- Static preset validation now uses the existing static-validation loader for
  templates with mandatory artifact identities; runtime validation is unchanged.
- Kept `resolve_clean_target` as a shared producer/consumer identity boundary,
  but replaced its `Any`/defensive attribute reader with the actual typed input
  union and direct field access. Its dataclass fields both have runtime consumers.
- Kept schema/version constants, the strategy dispatch table, optional-import
  adapters, public reward facades, and consistent family constructors. Their
  boundary or consumer justifies their existence. No blanket helper inlining,
  constant relocation, or formatting sweep was performed for line-count reduction.

### Verification

- Core/config/model/trainer audit suite: 885 passed.
- OCR report suite after retiring v3/v4: 51 passed.
- Retained critic chain: 67 passed; deployment-negative/config suite: 66 passed.
- Data, data scripts, create-only publication, and SFT-loading suite: 146 passed.
- SFT target encoding/regularizer checks after the typing cleanup: 15 passed.
- These are scoped, partly overlapping suites, not a repository-wide test total.
  Ruff checks and formatting checks passed for touched Python files;
  `git diff --check` passed. Source/test searches found no retired-path consumers.
- Clean wheel construction passed. Archive inspection confirmed retired paths
  absent, the inspected runtime modules byte-identical to source, and all four
  retained critic protocol assets present and byte-identical.
- No GPU training or reward-quality claim was validated by this cleanup.
