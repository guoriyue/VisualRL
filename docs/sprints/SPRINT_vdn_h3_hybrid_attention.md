# SPRINT: VDN-H3 接入 — 把 hybrid window-softmax / linear attention 嫁接到已有的 minimax_h3 家族

**日期**: 2026-09-07  **状态**: code-complete，CPU tiny-real 全绿；真权重未验证（见 §5）
**触发**: 用户 "can i add support for this model" → "please don't worry add full support
for minimax h3 and this linear attention"
**上游**: VDN-H3 / VideoDeltaNet，github.com/OpenVDN/vdn-minimax-h3（代码 Apache-2.0），
weights huggingface.co/OpenVDN/vdn-minimax-h3，blog openvdn.github.io
（UC Berkeley / Impossible Inc. / UT Austin）
**证据来源**: 上游源码逐文件阅读（`src/inference/assemble.py`、`src/models/hybrid_transform.py`、
`hybrid_attention.py`、`sequence_layout.py`、`checkpoints/loader.py`、`models/factory.py`、
`inference/lora.py`、`paths.py`）+ released artifact 的 `model_spec.json`。

---

## 0. 它是什么

MiniMax-H3 的派生：**不改 backbone 权重**，把每个 DiT block 的 attention 换成两支
——近帧的滑窗 softmax + 长程的双向 delta-rule 线性分支（Video Delta Attention）——
再加一个线性分支输出投影、若干 gate，和两个在加载时折进 backbone 的 LoRA。
上游数字：14.4 s 768p 视频，8×B200 上 11.23 秒，8 NFE。

## 1. 三个决定形状的事实

1. **是模块交换，不是 fork。** `apply_hybrid_attention_transform` 做的是
   `block.attn = HybridAttention(block.attn, ...)`，原 attention 留在 `attn.orig`，
   transformer 自己的 forward 签名不变。所以 `minimax_h3` 家族已有的一切（packed
   layout、row-timestep plan、双调度器、音频侧流、replay 契约）原样继承，本家族只多两件事：
   加载时嫁接，每次 forward 前交一次几何。
2. **所谓 "patched diffusers" 不是 fork。** `scripts/setup_diffusers.sh` = 上游 diffusers
   pin 到 `3a2f35d` + 两个补丁：AdaLN-SiLU 的 fp32 钉，和 NFE 计数
   （`linspace(1,0,n+1)` 而非 `n`）。第二个我们**这边早就补偿过**：
   `MiniMaxH3Model.set_num_steps(n)` 调的是 `set_timesteps(n+1)`。所以不需要换掉整个 venv 的
   diffusers，我们的 0.40.0 直接用。
3. **推理 kernel 是 forward-only。** `set_inference_mode` 装的是 no-grad 融合体
   （上游原话："the CALLER states no graph will be built"）。RL 要穿过它反传，所以我们
   **永远不开**，只暴露 `softmax_backend` 这一个仍然可导的运行时开关。

## 2. 落地方式：pinned submodule，不转写数学

按仓库 `third_party/README.md` 的惯例（git submodule + editable wrapper）接：

- `third_party/vdn-minimax-h3`，pin 在 `57edaf69`，`.gitignore` 加白名单，
  `third_party/pyproject.toml` 加一条 `where`/`include`。
- 上游包名就叫 `src`（它自己的 pyproject 就是 `include = ["src*"]`）。**保留这个名字**：
  改目录名会打断它自己的 `from src.models...` 绝对导入，而 verbatim 才是 submodule 的意义所在
  ——hybrid attention 是新数学，转写进 `vrl/` 就是给自己埋一个无法对照的错误。
  两道围栏：wrapper 的 `include` 只从这个根白名单 `src*`；`vrl/` 侧所有使用收敛到
  `vrl/models/families/vdn_h3/vendor.py` 一个文件。

在我们的环境（torch 2.11+cu130、triton 3.6、diffusers 0.40、**没有 flash_attn**）里，
这套 model 子树可导入：FA4 不是模块级依赖，triton 只有 `ops/fp8_linear.py` 和
`ops/temporal_conv.py` 两个文件用到。

## 3. 文件

| 路径 | 内容 |
| --- | --- |
| `third_party/vdn-minimax-h3` | 上游 submodule（pin `57edaf69`） |
| `vrl/models/families/vdn_h3/vendor.py` | 唯一碰 `src` 名字的地方；缺 submodule 时给 `make setup` 提示 |
| `vrl/models/families/vdn_h3/model.py` | `VDNH3Model(MiniMaxH3Model)`：`install_hybrid_attention`（顺序照抄 `assemble.py`：base → transform → branch 权重 → 折 LoRA）、`set_hybrid_layout`、`forward_step`；`VDNH3ReplayModel` |
| `vrl/models/families/vdn_h3/{config,runtime}.py` | `vdn_checkpoint` / `softmax_backend` 两个键 + 双调度器 replay builder + batch=1 executor |
| presets | `model/vdn_h3/{8nfe,50nfe}`、`sampling/denoise/8_step_no_cfg`、`experiment/vdn_h3/online_grpo_kling_video_reward` |
| tests | `tests/models/families/vdn_h3/test_hybrid_attention.py`（8 个）+ `TINY_VDN_H3_TRANSFORM_CONFIG` 夹具 |

## 4. CPU 上真验证了什么

夹具用的是 **released 8-NFE artifact 自己的 transform config**
（`delta_rule: vdn_solve`、`bridge: alpha`、`anchor_frames: both`、`short_conv: [k,v]`、
`enable_text_state: true`），只把 `linear_head_dim` 缩到 tiny 模型的 head dim。所以 CPU 测试
跑的是真权重会跑的那套语义。钉住的：

- transform 换掉每个 block 的 attention，`attn.orig` 仍可达；
- `forward_step` 一次联合前向，输出形状/有限性/fp32/负号约定不变；
- 几何交接：`SequenceLayout` 到每个 block，且 `seq_len`/帧数/每帧 token/网格/video 区间/text 区间
  与 packed layout 一致；
- **嫁接确实改变预测**，且 `teacher_mode=True` 能逐值还原 dense 基线（最强的一条）；
- **梯度到达线性分支的输出投影**（RL 要穿过它训）；
- `inference_mode` / `hybrid_inference_mode` 保持关闭；
- 缺 `vdn_checkpoint` 时报错并指向 `minimax_h3`。

scheduler log-prob parity 套件加了 `vdn_h3` 的 pin fixture（沿用 H3 的 shift 12 视频档）。

## 5. 没验证 / 拦路的

1. **真权重**：`h3-base` 约 72 GB（transformer + 两个 VAE），还不含 32B 条件器；上游基准是
   8×B200。本机 5090 32 GB 装不下，没有下载 82 GB 的权重。所以 README 标 🔌 Integrated。
2. **环境**：上游 pin torch 2.13 + FlashAttention 4 + triton 3.7.1；我们是 torch 2.11 /
   无 flash_attn / triton 3.6。`softmax_backend=ref` 处处可跑；`flex`/`decomposed` 与 fp8
   路径在本机未验证（`decomposed` 是 sm100 专用，5090 是 sm120）。
3. **许可（重要）**：上游 NOTICE 明写"运行本仓库的代码即运行 MiniMax H3 或其派生，
   因而受该协议约束"，而协议的 Applicable Territory 是"全球，**排除** 欧盟、英国、韩国、
   **美国**"，并称在该territory之外的使用"未获授权"。代码是 Apache-2.0（vendor 合规，
   NOTICE 已随 submodule 带入），**运行**是另一件事，由你决定。

## 6. 下一步

多卡环境上：`python -m vrl.scripts.generation.full_sequence_denoise_probe --family vdn_h3
--path MiniMaxAI/MiniMax-H3 --dtype bf16 --check-replay`，再跑
`experiment/vdn_h3/online_grpo_kling_video_reward` 的 smoke，把 README 提到 Runnable。
