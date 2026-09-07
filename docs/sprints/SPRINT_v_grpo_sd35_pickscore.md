# SPRINT: V-GRPO 接入 + SD3.5-M LoRA PickScore 复现（对照 FLUX DiffusionNFT）

**日期**: 2026-09-07  **状态**: 算法与 preset 落地、CPU gate 全绿；真机曲线待 GPU 空出（§5）
**触发**: 用户 "does this algorithm look promising… help me implement and 拿论文的
SD3.5-M LoRA 配置在本仓库复现一条 PickScore 曲线，和现有
flux/online_diffusion_nft_pickscore_validation 对比 NFE 和奖励"
**论文**: V-GRPO, arXiv:2604.23380（Tang / Zhang / Wang / Mao / Schmidt / Yeung-Levy,
2026-04-25，代码 github.com/tang-bd/v-grpo，Apache-2.0）
**证据来源**: 论文 §4.4–4.5、Alg. 1、附录 A.1/A.2 的公式与超参表；仓库
`vrl/algorithms/diffusion_nft.py`、`vrl/trainers/online/trainer.py` 的 replay 分支。

---

## 0. 算法一句话

GRPO 不变；把"策略似然"从逐步 SDE log-prob 换成最终样本的 flow-matching 损失
（ELBO 替代）：`log π(o|c) ← −L̂(θ|o,c)`，`ρ = exp(−L̂(θ) + L̂(θ_old))`，再套
`min(ρA, clip(ρ)A)`。rollout 变成普通 ODE 生成，训练只要最终 latent、prompt
条件和几对 `(t, ε)`。

## 1. 落到仓库的形状（与 trainer 的映射）

| 论文要素 | 仓库实现 |
| --- | --- |
| `N_MC` 对 `(t, ε)`，分层采 t | trainer 每个 replay index 一对：`actor.timestep_selection: stratified`（新增第三种模式：把 rollout 网格切成 `count` 段、每段均匀抽 1 个、每次 update 重抽）+ `timestep_fraction = N_MC / num_steps` |
| 组内共享 `(t, ε)` | `t` 天然共享（同一 update 同一 index）；`ε` 由 `(update 计数, group_id, index)` 播种，同组同噪声、跨 update 刷新 |
| 自归一化 x-pred 权重（Eq. 14） | `normalized_mse(x_pred, x0)`（与 NFT 同一函数） |
| `θ_old` | 冻结的 `previous` LoRA adapter，`after_optimizer_step` 后同步（同 NFT）。`ppo_epochs: 1` 时严格等于论文的 rollout 策略 |
| ratio 的 `exp(·)` 里对 `N_MC` 求均值 | **逐对 ratio**（trainer 逐 index backward）。on-policy（ρ≡1）时二者梯度相同；开 clip 时逐对形式更保守。写在模块 docstring 里 |
| ε clip / 简单 KL（Eq. 16）/ 优势软裁剪（Eq. 17） | `clip_ratio` / `kl_coef`（`‖x_θ − x_θold‖²`，同一对上）/ `adv_soft_clip`（`η·tanh(A/η)`），各自可关 |
| rollout 用 ODE（DPMSolver++） | `rollout.denoise_mode: native`（家族 scheduler 的确定性步；SD3.5 是 FlowMatchEuler） |

需要的模型面 = NFT 已有的两件：`diffusion_nft_prepare_transformer_input`（对加噪
clean latent 的原始 transformer kwargs）+ `previous` adapter。本 sprint 把
attach / sync 从 flux 搬进 `LoraModelMixin`（`model.nft_previous_adapter: true`
opt-in），给 sd3_5 补了 hook、`latents_clean` 导出和 `SD3_5ModelSection`。

## 2. 落地文件

| 路径 | 内容 |
| --- | --- |
| `vrl/algorithms/v_grpo.py` | `VGRPO` / `VGRPOConfig`；lr=0 不变式 `loss(A) = −loss(−A)`（`first_step_invariant_check`） |
| `vrl/trainers/online/trainer.py` | `_train_timestep_indices(..., "stratified")` |
| `vrl/models/steps/denoise/common/lora.py` | `attach_previous_policy_adapter` / `sync_previous_policy_adapter` 进 mixin；flux 去重 |
| `vrl/models/families/sd3_5/{config,model}.py` | `SD3_5ModelSection.nft_previous_adapter`、forward-process hook、`latents_clean` |
| `vrl/config/{schema,algorithm}.py`、`vrl/scripts/common/{factory,online}.py` | `kind: v_grpo`；无 evaluator 的目标不建 ref model |
| `presets/base/algorithm/v_grpo.yaml`、`recipe/online/v_grpo.yaml`、`experiment/sd3_5/online_v_grpo_pickscore.yaml` | 三层 preset |
| `tests/algorithms/test_v_grpo.py`（18）、`tests/models/families/sd3_5/test_forward_process_interface.py`（4） | 真 LoRA 双 adapter 上的方向 / ratio / clip / KL / 共享噪声 / 不变式；SD3.5 hook 与 `forward_step` 条件分支逐位一致 |

## 3. 复现配置：论文 Stage-1 → 单卡 5090

| | 论文 SD3.5-M Stage 1 | `experiment/sd3_5/online_v_grpo_pickscore` |
| --- | --- | --- |
| 奖励 | HPSv2.1 + PickScore + CLIP，Pick-a-Pic | PickScore（`pickscore_sfw`） |
| LoRA / lr | r32 α64 / 3e-4 | 同（`model/sd3_5/medium`） |
| 组 × prompt / 迭代 | 24 × 48 | 16 × 8（与 `sd3_5/online_grpo_pickscore` 同预算） |
| rollout | 40 步 ODE，无 CFG，720² | 20 步 ODE（native），无 CFG，512² |
| `N_MC` / 采样 | 4，分层，组内共享 | 4 of 20（`timestep_fraction 0.2`，stratified） |
| 梯度步 / 迭代 | 1（ρ≡1） | 1（`ppo_epochs 1`，`gradient_accumulation_steps 4`） |
| 控制 | η=3，无 clip，无 KL | 同 |
| 迭代数 | 150 | `trainer.total_epochs=150` |

## 4. NFE 对账（每样本每迭代，解析值；墙钟等真机）

| preset | rollout 前向 | 训练前向 | 合计 |
| --- | --- | --- | --- |
| `sd3_5/online_v_grpo_pickscore` | 20（ODE，单分支） | `N_MC × 2`（θ_old 无梯度 + θ）= 8 | **28** |
| `flux/online_diffusion_nft_pickscore_validation` | 10（SDE，单分支） | `⌊0.99×10⌋=9` 个 index × 3（previous / default / 关 adapter 的 ref）= 27 | **37** |
| 论文口径（SD3.5-M） | 40 | 4 × 2 ≈ 6.9（按 π_old/π 计） | 46.9 vs DiffusionNFT 80 |

同一 rollout 预算下 V-GRPO 的训练前向是 NFT 的 30%（8 vs 27），而且少一个
reference 前向（NFT 的 KL 走冻结基座，V-GRPO 的 KL 走 θ_old）。但注意两条线不是
同一个模型：FLUX 12B vs SD3.5-M 2B，绝对墙钟不可比，可比的是"训练前向 / rollout
前向"的比值和奖励曲线形状。

## 5. 待做（GPU 空出后，两条命令）

```bash
CUDA_VISIBLE_DEVICES=0 vrl-train --config experiment/sd3_5/online_v_grpo_pickscore \
  trainer.total_epochs=150
CUDA_VISIBLE_DEVICES=0 vrl-train --config experiment/flux/online_diffusion_nft_pickscore_validation \
  trainer.total_epochs=30 trainer.save_freq=9999
```

记录：`debug.first_step` 的 antisymmetry 不变式是否通过；`reward_mean` 曲线
（PickScore，固定 eval manifest）；每迭代墙钟与 `phase_times`；显存峰值。
写回本文件 §6 后再判断是否推到 FLUX（hook 已在）和视频家族（wan / minimax_h3
需各加一个 forward-process hook）。

阻塞原因：2026-09-07 晚 GPU 上有用户的 anima 训练（12.6 GB）+ 渲染任务，
SD3.5-M 的 T5-XXL + MMDiT + 16 样本 rollout 装不进剩余显存，且不能挤占正在
跑的任务。

## 6. 结果（待填）
