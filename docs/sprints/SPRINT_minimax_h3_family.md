# SPRINT: MiniMax-H3（Hailuo 3.0）家族接入 — 视频潜变量是动作，音频是确定性侧流

**日期**: 2026-09-07  **状态**: 代码完成（CPU tiny-real 全绿），真机验证被硬件阻塞（见 §5）
**触发**: 用户 "minmax recently has a cool video model can you find it and support it for me"
**证据来源**: diffusers 0.40.0 源码（`modular_pipelines/minimax_h3/*`、
`transformer_minimax_h3.py`、`scheduling_minimax_h3.py`）逐块阅读；
HF `MiniMaxAI/MiniMax-H3` 的 `model_index.json`；仓库 denoise seam
（`vrl/math/denoise/flow_matching.py::sde_step_with_logprob`、
`vrl/generation/steps/denoise/loop.py`）对照。

---

## 0. 模型是谁

MiniMax-H3 = Hailuo 3.0 开源权重（2026-08-03），diffusers 0.40 内置为 modular
pipeline（`MiniMaxH3ModularPipeline`，不带 `DiffusionPipeline` 版本）。
一个 repo 两个分区：`transformer/`（t2va / fl2va）和 `transformer_ref/`
（ref2va）。本 sprint 只接 **t2va**（纯文本 → 视频+音频），任务登记为 `t2v`，
奖励看视频。

| 组件 | 类 | 规模 |
| --- | --- | --- |
| transformer | `MiniMaxH3Transformer3DModel` | 33B，50 层，hidden 5376 |
| 文本条件 | `Qwen3VLForConditionalGeneration`（读 `hidden_states[50]`） | 32B |
| 视频 VAE | `AutoencoderKLMiniMaxH3` | f16、17 帧→5 潜帧、24 通道 |
| 音频 VAE | `AutoencoderKLMiniMaxH3Audio` | 40 Hz 潜变量、32 通道、立体声 |
| 调度器 | `MiniMaxH3Scheduler` ×2 | shift 12（视频）/ 3（音频） |

## 1. 与仓库 seam 的三处不匹配，以及各自的处理

### 1.1 一次前向同时去噪视频和音频，而 loop 只认一个 latent + 一个 scheduler

`run_denoise_loop` 只读写 `state.latents`，没有 side-stream 钩子。做法
（`vrl/models/families/minimax_h3/model.py`）：

- **视频潜变量是 RL 动作**：走共享 SDE 步和 log-prob，与其他 flow 家族一致。
- **音频行是确定性侧流**：`forward_step(i)` 消费第 i 步的音频行，前向后用
  checkpoint 自己的 Euler（`eta=0`，音频 schedule）算出第 i+1 步的行，惰性在
  下一次 `forward_step` 推进。每步输入的音频行记进 `audio_rows_by_step`，
  replay 从中取第 i 步的行喂 transformer → rollout/replay 前向逐位一致
  （测试：`test_replay_restores_every_step_bit_exactly_from_the_exported_tensors`）。
- 同一步重复调用（`cache_ref_noise_pred` 的冻结参考前向）看到相同音频行；
  跳步（TeaCache）直接 `RuntimeError`，不静默漂移。

### 1.2 速度符号与时间约定相反

H3：`x0 = x_t + sigma * v`，`t = 1 - sigma`；仓库 SDE 推导自 `x0 = x_t - sigma * v`。
`forward_step` 返回 `-v`；`MiniMaxH3FlowScheduler.step` 再取一次负，所以
`denoise_mode: native` 也复现参考采样器（测试
`test_flow_scheduler_step_reproduces_the_reference_sampler_from_the_negated_velocity`）。
sigma 表在 `[0, 1]`，`sde_step_with_logprob` 自动判为 flow 域。

注意 H3 的 `set_timesteps(n)` 给 n 个 sigma、n-1 次模型评估；家族的
`set_num_steps(n)` 对两个调度器都调 `set_timesteps(n + 1)`。

### 1.3 序列无 padding、布局随 prompt 长度变

transformer 的 batch 轴只是复制轴，一个 layout 服务整批。所以：

- executor 钉死 `samples_per_generation_batch=1`（`runtime.py`，同 cosmos3）。
- `prompt_embeds` 以 `sampling.max_sequence_length` 宽度零填充后导出，另存
  `num_text_tokens`；`restore_eval_state` 先切掉填充再建 layout，transformer
  从不见到 pad 行。replay micro-batch 混不同 prompt 长度会被拒绝 →
  experiment preset 里 `actor.train_batch_size: 1`。
- VAE 几何（`clip_length / tokens_chunk_size / spatial_ratio`）随
  `batch_context.vae_geometry` 走，replay 端无 VAE 也能重建 layout。

## 2. 落地的文件

| 路径 | 内容 |
| --- | --- |
| `vrl/models/families/minimax_h3/model.py` | `MiniMaxH3Model` / `MiniMaxH3ReplayModel` / `MiniMaxH3Components`（组件壳） / 布局与 plan 复用 diffusers 的 block statics |
| `vrl/models/families/minimax_h3/runtime.py` | `MiniMaxH3BatchExecutor`（batch=1，默认 124 帧 @ 24 fps）、双调度器 replay builder |
| `vrl/config/sampling_schema.py::MiniMaxH3SamplingSection` | `guidance_scale` 只接受 1.0（蒸馏），带 `max_sequence_length` |
| `vrl/models/families/registry.py` / `names.py` | `minimax_h3`，别名 `minimax-h3` / `hailuo_3` / `hailuo3` |
| `vrl/config/presets/model/minimax_h3/h3.yaml` 等 | 模型 / `sampling/video/h3_768p_124f` / `experiment/minimax_h3/online_grpo_kling_video_reward` |
| `tests/models/steps/denoise/fixtures.py` | tiny-real：transformer ~9K 参数、两个 VAE、2 层 Qwen3-VL + 真 tokenizer/processor |
| `tests/models/families/minimax_h3/` | backbone parity（14）+ loading / replay builder / executor（3） |
| `pyproject.toml` / `uv.lock` | diffusers `>=0.40.0,<0.41`（`uv lock --upgrade-package diffusers`，只动这一项）；miniconda base 同步 `--no-deps` 到 0.40 |

## 3. 验证了什么、没验证什么

验证（全部 CPU、真对象）：patchify 行序与 diffusers 一致且可逆；prompt 编码
= `hidden_states[layer]`、无特殊 token、按 `max_sequence_length` 截断；布局行序
`[text | audio | video]`、音频 channel-major、首步只有一个 distinct timestep；
几何拒绝（非 17n+5 帧、非 32 倍画布、fps≠24、guidance≠1）；音频 Euler 与
`MiniMaxH3Scheduler.step` 逐位相等；rollout→replay 三步 `noise_pred` 逐位相等；
`decode_latents` 两层反归一化 + `[B, C, T, H, W]`；`decode_audio` 出立体声波形。

**没验证**：真权重的任何数字。见 §5。

## 4. 明确的范围边界（不是遗漏）

- 音轨不进 RL 产物：`decode_denoise_result` 只解码视频；`decode_audio` /
  `final_audio_rows` 给评估脚本用。要把音频进奖励需要 trajectory 多输出，
  超出本 sprint。
- fl2va（首/末关键帧）和 ref2va（omni-reference）未接：需要 keyframe 条件行
  + `transformer_ref` 分区，属于另一个 family entry 或 task。
- TeaCache 不支持（§1.1）。

## 5. 阻塞与风险（给用户的决定）

1. **硬件**：t2va 分区 ≈ 144 GB bf16（33B + 32B 条件器）。RTX 5090 32 GB 单卡
   装不下任何一半；"Runnable" 的 `full_sequence_denoise_probe --check-replay`
   和 preview 都做不了。需要多卡 FSDP（README 表标 🔌 Integrated）。
2. **许可**：MiniMax-H3 Community License 对美国 / 欧盟 / 英国 / 韩国的本地部署
   有限制（需申请）。部署前请自行核对许可文本。
3. **文本编码器版本**：`model_index.json` 指向 transformers 的
   `Qwen3VLForConditionalGeneration`（本地 5.13.0 有），无需额外升级。

## 6. 下一步（按顺序）

1. 多卡环境上：`python -m vrl.scripts.generation.full_sequence_denoise_probe --family minimax_h3 --path MiniMaxAI/MiniMax-H3 --dtype bf16 --check-replay`。
2. 通过后跑 `experiment/minimax_h3/online_grpo_kling_video_reward` 的 smoke，
   把 README 从 Integrated 提到 Runnable。
3. 如需音轨奖励：给 trajectory 加第二个 reward view（音频）。
