# SPRINT: reward 侧 tiny-real 仓库，以及 optional 车道的第一批真成员（done）

状态：**done（2026-09-05 落地 RW-01/02/03/04-A/B/C/06/07/08/11；2026-09-06 GPU 空出后补齐 RW-04-D 与 vLLM 依赖清单，见 §11.5）**。
原计划状态：planned / CPU-only（两个 opt-in 测试需要 `--optional`，一个需要 `--extra reward`）。
基线：main @ `812cc3cf`。分层判据见
`docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`
（Tier Policy：T1 / T2 / T2-PIPE / T3）。
前身：[[SPRINT_test_suite_tiny_real_and_fake_audit]]（done, `84584d23`）——本 sprint **推翻**它对 kling `_Fake*` 的 KEEP 裁定，理由见 §6.1。

> 本文所有数字都是本机实测（`.venv/bin/python -m pytest -p no:randomly`，`/usr/bin/time`），不是估算。凡与任务简报口径不一致的，§7 逐条列出。

---

## 0. 一句话

把 reward 侧五处「造一个假模型再断言假模型被告知的返回值」的测试，换成**在 `tmp_path` 上现造一个真仓库、让真 loader 去读它**：一个 108K 参数的真 CLIP 仓库（配真 `CLIPProcessor`）让 aesthetic 第一次真正加载仓库自带的 3.7 MB LAION 头、PickScore 的分数第一次能被独立的 `cosine_similarity` 当 oracle 校验；一个 31,824 参数的真 Qwen2-VL Kling reward model 让 checkpoint key remap 从字符串比较升级成对活模型的 `strict=True` 加载；一个真 byte-level BPE 让 VideoScore2 的 marker 双拼写 / 多 token 分支第一次被执行。同时给 `optional` 车道送进它今天为止的**头两个真成员**（真 RAFT-small 静止 vs 运动判别、真 VideoScore2 tokenizer 保真钉），并顺手修掉一个今天就会崩的生产 bug。

**代价：`tests/rewards` 5.50 s → 6.21 s，默认车道 +0.71 s**（对 189 s 全套是 +0.4%）。新增 39 个测试（37 个默认车道 + 2 个 optional 车道），改写 7 个，净 +32。

---

## 1. 一览表

实测口径：`tests/rewards` 基线 4 次平均 **5.50 s**（228 passed / 2 skipped）；替换后 3 次平均 **6.21 s**（258 passed / 4 skipped / 7 deselected）。

| 测试路径 | 今天假的是什么 | 变成 | 实测成本 |
|---|---|---|---|
| `tests/rewards/test_clip_reward_models.py:9-21,24-68` | `_FakeClip` + `_FakeProcessor` + `torch.load` → 全零的手写 aesthetic head | **T2**：真 `CLIPModel` 仓库 + 真 `CLIPProcessor` + 真 3.7 MB LAION 资产 | 会话 fixture 一次 ~1.18 s（其中 ~1.0 s 是本文件今天已经在付的 `CLIPModel` 首次 import），每测试 ~20 ms |
| `tests/rewards/test_clip_reward_models.py:117-154` | `_FakeModel(logit_scale=0)` + `pooler_output=eye(B)`，直接赋 `model._module` | **T2**：真 CLIP + `logit_scale_init_value=log(26)`，断言对独立 `cosine_similarity` | +~40 ms（复用同一 builder，另一组参数） |
| `tests/rewards/kling_video_reward/test_model_loading.py`（周边空洞） | 无假替身——`KlingQwen2VLRewardModel.forward` / `_reward` 全仓**零测试** | **T2**（新增文件）：31,824 参数真模型，三条 pooling 分支 + z-score 算术 | 建模 5.8 ms，首次 forward 140 ms（SDPA dispatch warmup），热 forward 0.42 ms |
| 同上，`load_kling_video_reward_checkpoint` / `_create_model_and_processor` | 全部被 `_FakeModel`（body 只有 `def eval`/`def to`）替掉 | **T2**：真 peft 包裹 + `strict=True` 加载 + 真 `Qwen2VLProcessor` 往返 | strict load 2.3 ms；`_create_model_and_processor` 全流程 195 ms |
| `_prepare_batch`（同文件） | 无覆盖 | **T1 + T3**：抽出纯字典装配段 → T1；剩余 decode/chat-template → optional 车道 + 诚实缺口 | T1 段 ~0 ms |
| `tests/rewards/functions/test_future_reward.py:15-29`（周边空洞） | 无假替身——`MotionDynamicsModel._dynamic_degree` 全仓零测试 | **T1**（新增文件）：banded flow 精确算术 8 条 + **optional 车道第一个真成员**（真 RAFT-small） | 默认 8 条全部 < 5 ms；optional 那条 0.59 s |
| `tests/rewards/functions/test_ocr.py:32-39,86-117` | `_FakePaddleOCR` 手抄 PaddleOCR **2.x** 的嵌套返回格式（extra 钉的是 >= 3.5.0） | **保留但双协议参数化**（(c) 类外部引擎边界）+ 收紧断言 + 真引擎门控测试 | 6 个参数化测试合计 ~0.03 s |
| `tests/rewards/videoscore2/test_parsing.py:93-114,125-186` | `_FakeTokenizer.encode` = `text.strip()` 查表，`' quality'` 与 `'quality'` 塌成同一个 id | **T2**：真 `PreTrainedTokenizerFast` + 真 byte-level BPE + **optional 车道第二个真成员** | builder 0.56 ms；optional 那条 1.79 s |
| `tests/nn/kernels/test_vllm_paged_attention_real_ops.py:20-23` | 不是假替身——是**全仓唯一的真 vLLM kernel 测试在推荐环境里恒 skip** | 基础设施修复：区分「没装 vLLM」与「装了但不完整」 | 默认车道 0 ms |

---

## 2. 逐条

### 2.1 RW-01 — aesthetic：全零的手写 head 让 `isfinite` 断言恒真（→ T2）

**今天断的是什么**（`tests/rewards/test_clip_reward_models.py:65-68`）：

```python
scores = model._module([object(), object()])
assert scores.shape == (2,)
assert torch.isfinite(scores).all()
```

`_aesthetic_head_state_dict()`（同文件 `:9-21`）把每一层权重和 bias 全填 `torch.zeros`，所以生产的 `_MLP` 对**任何**输入都精确返回 0.0。`isfinite(0.0)` 是空断言。真正过掉的只有一件事：如果代码读了 `.last_hidden_state`（rank 3）而不是 `.pooler_output`（rank 2），形状会炸。仓库自带的 3.7 MB 资产 `vrl/rewards/assets/sac+logos+ava1-l14-linearMSE.pth` 一次都没被碰过。

`_aesthetic_head_state_dict()` 本身还是 AGENTS.md 点名的反模式：它手抄了 `vrl/rewards/models/aesthetic.py:35-44` 的 `_MLP` 几何（`nn.Linear(768, 1024)` 起头的五层）。资产布局一漂移，测试不会红。

**换成什么**：新增 `tests/rewards/fixtures.py`（对齐 `tests/models/steps/denoise/fixtures.py` 的既有先例），提供

```python
def build_tiny_clip_repo(root, *, projection_dim, logit_scale_init_value, seed=0) -> Path
```

- vocab 用 `transformers.convert_slow_tokenizer.bytes_to_unicode()` 的 byte-level 字母表（514 项）+ `<|startoftext|>` / `<|endoftext|>`，`merges.txt` 只写 `#version: 0.2`（零 merge 合法，实测可 tokenize 任意 prompt）；
- `CLIPModel(CLIPConfig(...)).save_pretrained(root)` + `CLIPProcessor(CLIPImageProcessor(...), CLIPTokenizer(...)).save_pretrained(root)`；
- **`projection_dim` 从 shipped 资产推导，不硬编码**——这正是本条自己引用的那条 AGENTS.md 规则：

```python
asset = resources.files("vrl.rewards.assets").joinpath("sac+logos+ava1-l14-linearMSE.pth")
projection_dim = int(torch.load(asset, map_location="cpu", weights_only=True)["layers.0.weight"].shape[1])
```

实测该值为 **768**，与 `_MLP` 首层入宽一致。

**种子**：`with torch.random.fork_rng(devices=[], device_type="cpu"): torch.manual_seed(seed)`。实测 0.81 ms 且 `torch.cuda.is_initialized()` 全程保持 `False`；**不要**用裸 `torch.random.fork_rng()`，实测 19 ms 并且会在 CPU 车道里初始化一个 CUDA context。权重构造后写盘、由 `from_pretrained` 读回，断言时不再发生任何随机初始化。

**断言换成假货物理上做不到的那一条**：

```python
assert tuple(head.weight.shape) == (1024, 768)
assert float(head.weight.detach().abs().sum()) > 0.0      # 真资产真的加载了
assert black["aesthetic"] != white["aesthetic"]           # 整条链路真的串起来了
assert model._module([img_a, img_b]).shape == (2,)        # 保住原测试的 .squeeze(1) 契约
```

实测（本机、跨进程稳定）：全黑 `torch.zeros(3,12,12)` → 5.348966，全白 → 5.968630，灰阶 ramp → 5.612262。**只断不等，绝不断具体数值**——数值取决于 tiny CLIP 的随机初始化和 torch 版本的 `Linear` init。零权重头永远返回 0.0，现有测试在物理上不可能做这个断言。

**顺带删掉 `_aesthetic_head_state_dict()`**：它有两个调用点（`:60` 和 `:103`）。实测 `:103`（revision 用例）的 `torch.load` monkeypatch 可以直接删——真资产 ~1.8 ms 加载进真 `_MLP`，两个参数化用例照过（实测 2 passed / 1.60 s，其中大头是进程启动）。删掉后 `_aesthetic_head_state_dict()` 零调用者，整个删除。

**验证**：

```bash
.venv/bin/python -m pytest tests/rewards/test_clip_reward_models.py -q -p no:randomly
```

---

### 2.2 RW-02 — PickScore：`1/26` 是被工程出来的常数（→ T2 + 一个生产 bug 修复）

**今天断的是什么**（`:154`）：

```python
assert score == pytest.approx(1.0 / 26.0)
```

这个数是机械地掉出来的：`eye(2)` 本来就是单位范数，归一化是 no-op；`exp(0.0) == 1.0`，logit_scale 是 no-op；`eye @ eye.T == eye`，矩阵乘是 no-op；`diag().mean() == 1.0`。剩下真正被钉住的只有字面量 `/26` 和 `.pooler_output` 解包。真 CLIP 投影、真 processor 的 resize/normalize、真归一化、以及 `score_media` 的 tensor→numpy→PIL 派发（`vrl/rewards/models/pickscore.py:61-80`）全部未测。

**换成什么**：同一个 builder，另一组参数——`projection_dim=16, logit_scale_init_value=math.log(26.0)`。实测 `logit_scale.exp() == 25.999998`，于是生产公式 `logit_scale * (text @ image.T) / 26` 塌成**匹配对余弦的均值**，是一个有界、可用独立实现校验的不变量：

```python
score = pickscore._score("a green square", images)
with torch.no_grad():
    expected = torch.nn.functional.cosine_similarity(txt, img, dim=-1).mean()
assert score == pytest.approx(float(expected), abs=1e-6)
```

实测 `score = 0.036835171`，oracle `0.036835179`，差 **7.45e-09**。这是 differential 断言不是镜像：`cosine_similarity` 与生产手写的 norm → matmul → diag → /26 是两套实现，把 `/26` 改成 `/13` 分数翻倍、测试红。

**一起修一个今天就会崩的生产 bug（不 xfail）**。`vrl/rewards/models/pickscore.py:66-68`：

```python
elif arr.ndim == 5:
    mid = arr.shape[1] // 2
    arr = arr[:, mid].transpose(0, 2, 3, 1)
```

对本仓的**规范视频布局** `[B,C,T,H,W]`（`vrl/rewards/models/media.py::decode_artifact_frames` 接受 channel-first `[C,T,H,W]` / `[1,C,T,H,W]`）这行切的是**通道轴**。实测：

```
score_media(media=torch.rand(2,3,5,12,12)) -> TypeError: Cannot handle this data type: (1, 1, 5), |u1
score_media(media=torch.rand(2,5,3,12,12)) -> {'pickscore': 0.135}      # [B,T,C,H,W] 侥幸正确
```

`vrl/rewards/models/nsfw_safety.py:186-191` 已有正确的嗅探写法，直接抄（同时也满足 AGENTS.md 的「贴合本地既有 pattern」）：

```python
if arr.ndim == 5:
    if arr.shape[2] in (1, 3, 4):        # [B,T,C,H,W]
        arr = arr[:, arr.shape[1] // 2]
    elif arr.shape[1] in (1, 3, 4):      # [B,C,T,H,W] -- 本仓规范布局
        arr = arr[:, :, arr.shape[2] // 2]
```

实测该嗅探在 `[B,C,T,H,W]` / `[B,T,C,H,W]` / 单通道 / 两轴都 ∈ (1,3,4) 的歧义情况下全部落到 4 维、通道数合法；歧义时 `[B,T,C,H,W]` 分支优先，与 nsfw_safety 的既有顺序一致。

**另外两条几乎免费的新覆盖**：`score_media(media=torch.rand(2,3,12,12))` 走 NCHW→PIL 派发；`score_media(media="not-media")` 走 `pickscore.py:79-80` 的 else 分支返回 `{"pickscore": 0.0}`（今天零覆盖）。

**必须记进文档、不许假装覆盖的限制**：`_score` 对每张图重复同一个 prompt，所以文本行逐字节相同、分数矩阵秩亏——实测 `diag().mean()` 与全矩阵 mean 精确相等。**`.diag()` 的选择通过这个入口不可测**。如果它以后重要了，它属于带 per-image prompt 的真 e2e，不属于这里。

---

### 2.3 RW-03 — Kling 打分数学全仓零测试（→ T2，纯新增覆盖）

这不是一个假替身问题，是一个洞。`KlingQwen2VLRewardModel.forward`（`vrl/rewards/models/kling_video_reward.py:309-393`）和 `KlingVideoRewardModel._reward`（`:249-281`）**全仓没有任何测试**。最接近的那个（`test_model_loading.py:131-138`）以

```python
assert captured == {"dtype": ..., "disable_flash_attn2": True, "local_files_only": True,
                    "checkpoint_dir": root, "model_eval": True, "model_device": "cpu"}
```

收尾——断言的是「假货记下了别人递给假货的 kwargs」。那是 loader 接线，不是打分。

与此同时无人看守的算术包括：三条 pooling 分支、`sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1` 这个 off-by-one 加上它后面的 `% input_ids.shape[-1]` 回绕、以及 `_reward` 里三个轴各自的 z-scoring 和 `Overall = VQ + MQ + TA`。这里改错一个符号或调换一次 mean/std，训练信号被静默重标定。

**新增 `tests/rewards/kling_video_reward/fixtures.py`**（本仓已有 5 个同名同位置的 fixtures.py 先例）：

```python
def build_tiny_kling_reward_model(*, output_dim=4, reward_token="last",
                                  special_token_ids=None, pad_token_id=0,
                                  vocab_size=64, seed=0)
```

`Qwen2VLConfig(text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64, bos_token_id=None, eos_token_id=None, rope_parameters={"rope_type":"default","rope_theta":10000.0,"mrope_section":[2,1,1]}), vision_config=dict(depth=1, hidden_size=16, embed_dim=16, num_heads=2, in_chans=3, spatial_patch_size=14, temporal_patch_size=2, out_hidden_size=16), bos_token_id=None, eos_token_id=None)` 然后 `config.pad_token_id = pad_token_id`。

实测：**31,824 参数、5.8 ms 构建**。三处必须注意（简报和上游提案都漏了）：
1. `bos_token_id` / `eos_token_id` 必须在**顶层和 `text_config` 里都**设成 `None`，否则每次构建打两行 `must be between 0 and 63, got 151643` 警告；
2. `pad_token_id` **不能**当 `Qwen2VLConfig(...)` 的 kwarg 传（实测 `AttributeError: 'Qwen2VLConfig' object has no attribute 'pad_token_id'`），必须构造后赋值；
3. 第一次 forward 付 ~140 ms 的 SDPA dispatch warmup，之后 0.42 ms。这笔一次性开销在本 sprint 的总账里已计入。

**参考实现（不要用 `output_hidden_states`，那条路子在 tiny 上取不到想要的张量）**：

```python
def _head_logits(model, input_ids, attention_mask):
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)[0]
        return model.rm_head(hidden)
```

**八条测试，全部实测通过**：

| 测试 | 钉住什么 | 实测 |
|---|---|---|
| `test_last_pooling_reads_the_final_non_pad_row` | `sequence_lengths` 算术，`torch.equal` 精确 | ids `[[3,4,5,0,0],[7,8,9,10,0]]` → `pooled[0]==ref[0,2]`、`pooled[1]==ref[1,3]` 均 True |
| `test_last_pooling_wraps_to_the_final_row_when_no_pad_is_present` | `% input_ids.shape[-1]` 回绕 | 无 pad 时 argmax 返 0、减 1 得 -1、模 5 得 4 → 落在 `ref[0,4]`，True。这是「天真地修掉那个 off-by-one」就会炸的那一半 |
| `test_mean_pooling_drops_the_final_real_token_and_ignores_padding` | padding 不变性 **+ 实际行为** | 不变性差 2.98e-08；`unpadded[0] == ref[0,:2].mean(0)` True；`!= ref[0,:3].mean(0)` True——第三行是重点，它把「最后一个真 token 被排除」这件事写死，读者不会误以为是「valid prefix 的均值」 |
| `test_special_pooling_takes_the_diagonal_in_sequence_order` | 布尔 mask 索引按 **sequence 顺序**返回，不按 `special_token_ids` 顺序 | 用打乱的第二行 `[7,1,6,2,5,3,0]`，期望 `stack([ref[1,0],ref[1,2],ref[1,4]]).diagonal()`，True。同时断言 `model.reward_token == "special"`——传 `special_token_ids` 会**无条件覆盖** `reward_token` 实参，是个值得钉住的静默改写 |
| `test_special_pooling_with_output_dim_one_yields_three_axes` | shipped checkpoint 的真实形态 | 实测 shipped `model_config.json` 是 `output_dim=1` + `reward_token="special"` + `use_special_tokens=true`；此时 `view(B,3,-1)` → 不走 diagonal → `view(B,-1)` = **(B,3)**。这正是 `_reward` 读 `reward[0..2]` 当 VQ/MQ/TA 的前提，今天没人钉 |
| `test_pooling_rejects_unpadded_batches_and_unknown_reward_tokens` | 两条 fail-fast | 分别实测 raise `Cannot handle batch sizes > 1 if no padding token is defined.` / `Invalid reward_token` |
| `test_reward_z_scores_each_axis_and_sums_overall` | 三轴 z-score + Overall | `KlingVideoRewardModel.__new__` + `model.model = tiny` + stub `_prepare_batch`（沿用本文件 `:56` 的既有写法）。三个轴用**互不相同**的 mean/std（1.0/2.0、-0.5/4.0、0.25/0.5），调换轴或调换 mean/std 都过不了。永远与当场重算的 logits 比，**不硬编码浮点数**（线程数漂移 1.5e-8） |
| `test_reward_without_norm_keeps_the_raw_head_logits` | `use_norm=False` 分支 | `inference_config` 缺失时会静默改变量纲的那条分支，免费 |

**成本**：八条合计实测 < 0.15 s call time，落在 `tests/rewards` 的 run-to-run 噪声里。**无新依赖**——`vrl/rewards/models/kling_video_reward.py:20` 已在模块作用域 import `Qwen2VLForConditionalGeneration`，这笔 transformers import 已由 `tests/rewards/test_model_protocol.py` 付过。

---

### 2.4 RW-04 — Kling loader / remap / `_create_model_and_processor`（→ T2；`_prepare_batch` → T3）

#### A. checkpoint key remap：从字符串比较升级为对活模型 strict load

`test_kling_video_reward_remaps_qwen2vl_checkpoint_keys`（`:279-297`）断的是四个 `_remap_qwen2vl_key(str) == str`。这只覆盖了函数的一半——另一半在 `_remap_qwen2vl_state_dict`（`:661-666`）：

```python
remapped = {_remap_qwen2vl_key(str(key)): value for key, value in state.items()}
return remapped if set(remapped) == set(target_state) else state
```

`set(remapped) == set(target_state)` 是**与活模型比对**，后面紧跟 `model.load_state_dict(state, strict=True)`。transformers 升级把 `language_model` 嵌套改掉的那天，唯一会红的就是这个比对，而今天没人跑它。

**新测试**（实测全流程通过）：真 `peft.get_peft_model(tiny, LoraConfig(r=2, target_modules=["q_proj","v_proj"]))` → `torch.save(state_dict)` 到 `checkpoint-11352/model.pth` → `load_kling_video_reward_checkpoint` 断言 `step == "11352"`；再把 key 改回 transformers-5 之前的布局（实测 39 个 key 中 37 个会变）→ 再次 strict load 通过 → 断言 `set(_remap_qwen2vl_state_dict(legacy, live)) == set(live)`。实测 strict load **2.3 ms**；把 target 布局故意改坏时确实 `RuntimeError: Missing key(s) in state_dict`，说明这条断言是敏感的。

#### B. `_create_model_and_processor` 真跑

实测可以完全离线跑通，**195 ms**：手造一个 tiny `Qwen2VLProcessor` 存到 `tmp_path`（`Qwen2Tokenizer` + `Qwen2VLImageProcessor` + `Qwen2VLVideoProcessor`，byte-level 字母表 vocab，实测 build+save **1.1 ms**、`AutoProcessor.from_pretrained` **2.9 ms**），tiny 模型 `save_pretrained` 到同一目录，然后 `local_files_only=True` 走真路径。断言：

```python
assert model.config.pad_token_id == processor.tokenizer.pad_token_id
assert model.config.tokenizer_padding_side == "right"
assert len(base.special_token_ids) == 3            # add_special_tokens + resize_token_embeddings 真跑了
targets = _find_target_linear_names(base, lora_namespan_exclude=["lm_head", "rm_head", "embed_tokens", "visual"])
assert targets and not any(bad in n for n in targets for bad in ("rm_head", "embed_tokens", "visual"))
```

最后一条不是编的：shipped `model_config.json` 的 `peft_lora_config.lora_namespan_exclude` 就是 `["lm_head","rm_head","embed_tokens","visual"]`。LoRA 误命中 `rm_head` 是静默的打分腐化，今天完全没防线。

> **`Qwen2VLProcessor` 必须传 `video_processor=`**，否则 `TypeError: Received a NoneType for argument video_processor`。这是实测踩到的。

**这条同时推翻 [[RW-09 非目标]] 的一个理由**：那份裁定说手工离线造 Qwen2-VL processor 等于「vendoring the vendor's tokenizer/image-processor file format」，触犯 AGENTS.md 的「reimplementing a dependency's internals」。不成立——我们用的是 transformers 自己的类和它自己的 `save_pretrained`，格式由库写、由库读，我们一个字节都没手抄。手抄的是 `_FakePaddleOCR` 那种（见 §2.6），那条裁定用对了地方、这条用错了。

#### C. `_prepare_batch`：拆成能测的一半和诚实缺口的一半（根因修法）

`_prepare_batch`（`:185-247`）今天零覆盖，且它第一行就是 `from qwen_vl_utils import process_vision_info`。实测本机 `qwen_vl_utils` / `decord` / `cv2` **全部 ModuleNotFoundError**（它们在 `[reward]` extra 里，默认环境不装）。

**不要**整块打成 T3。里面有一大段是纯字典装配、零依赖：`max_pixels` 回落到 `data_config.max_frame_pixels`、`min_pixels` 仅在非 None 时写入、`nframes` 与 `fps` 互斥、`build_kling_video_reward_prompt` 拼接。把它抽成

```python
def _build_chat_payload(video_paths, prompts, *, max_pixels, min_pixels) -> list[list[dict]]
```

抽出来之后立刻是 T1，测试成本 ~0 ms，而这些分支直接决定 reward 是否 in-distribution。剩下的 `process_vision_info` + processor 调用才是真外部边界 → 见 §5。

#### D. e2e：真 2B 的数值行为（优先级低于 A/B/C）

`tests/e2e/test_real_checkpoint_rl.py:697` 把 `worker_config.model_factory` 覆盖成 `tests.e2e.test_real_checkpoint_rl:build_tensor_mean_model`（`:683`，score = 张量均值）。那个 case 证明的是 reward **传输**真的端到端跑通（`reward_components` 有 `kling_video_reward`、`reward_artifacts/manifest.jsonl` 与 `reward_debug/kling_video_reward_results.jsonl` 都在盘上）——**这很有价值，一个字都不要动**。但它不证明 `vrl/rewards/models/kling_video_reward.py` 里工厂字符串之后的任何一行。

如果要加真模型 case（本 sprint 列为可选、优先级最低），必须补上三件事，否则写出来是坏的：

1. `_local_reward_overrides(tmp_path)` 参数化成 `_local_reward_overrides(tmp_path, model_factory: str | None)`，`None` 时不发 `model_factory` 覆盖；同时补 `worker_config.local_files_only=true`。
2. 除了 `CheckpointField(cfg_path="reward.kwargs.kling_video_reward.worker_config.model_path", repo_id="KlingTeam/VideoReward", required_files=("model_config.json",))`，**还要**加一条无 cfg_path 的守卫：base 模型名 `Qwen/Qwen2-VL-2B-Instruct` 写死在 snapshot 的 `model_config.json` 里（实测已确认），没有任何 cfg path 能重定向它，而 runner 在 `:564` 设了 `HF_HUB_OFFLINE=1` —— 不加守卫就是 ERROR 而不是 skip。
3. `min_cuda_memory_gib` 从 28.0 抬到 ~36.0（bf16 2B ≈ 4.4 GB 权重 + `min_frame_pixels=200704` 上采样后的视觉激活）。

成本：**仅热缓存机器，且需 `--extra reward`**。本机两份快照都在（KlingTeam/VideoReward 4.8 GB + Qwen2-VL-2B-Instruct 4.2 GB ≈ 9 GB）；冷机器在 `HF_HUB_OFFLINE` 下 skip 而非下载。

**A/B/C 用 < 0.4 s 覆盖了 loader / pooling / remap / LoRA 目标的全部结构性风险；D 只额外覆盖真权重的数值行为。**

---

### 2.5 RW-06 — motion_dynamics：反静止塌缩守卫全仓零测试（→ T1 + optional 车道第一个真成员）

模块 docstring 说这个 reward 的「single job」是给静止 / 模糊 / 时间均值塌缩钉一条硬地板。全仓提到 `MotionDynamicsReward` 的测试只有两条：`assert reward.inference_runtime is not None` 和 `pytest.raises(TypeError, match="unknown_knob")`。**没有一条碰打分。**

无人看守的算术：interpolate 到 `flow_size` → 映射到 [-1,1] → 沿 flow 轴 `torch.linalg.vector_norm` → 除以帧对角线 `sqrt(2)*flow_size` → `topk(k = numel * top_fraction)` → mean → `min(1.0, max(0.0, raw * magnitude_scale))`。`vrl/rewards/models/motion_dynamics.py:50-53` 的注释解释 `magnitude_scale=50` 是调出来的（DROID 运动落 ~0.37、静止 ~0.03），这个常数今天没人守。

#### 默认车道：banded flow，不是 constant flow

**不要用恒定流场**：恒定场下每个 `top_fraction` 都给同一个数，topk 完全不受力。用 `_BandFlow`——上 25% 行给 5 px 的 `(3,4)`，其余为 0。实测（`flow_size=8`，`unit = 5/sqrt(2*8²) = 0.44194174`）：

| `top_fraction` | 实测 raw | `raw / unit` |
|---|---|---|
| 0.25 | 0.4419417381 | 1.000000 |
| 0.5 | 0.2209708691 | 0.500000 |
| 1.0 | 0.1104854345 | 0.250000 |

三个 k 三个不同的数。断言用 `pytest.approx(unit * expected, rel=1e-6)`，**不要用 `==`**。

八条默认车道测试（全部实测 < 5 ms，文件整体 0.58 s 含 import）：对角线归一化、上表三条参数化、`magnitude_scale=50` 精确饱和到 `{"motion_dynamics": 1.0}`、`magnitude_scale=1.0` 时线性（这条才真正守着那个 50.0）、单帧 clip 返回 `{"motion_dynamics": 0.0}` 而不是崩（这是 reward 存在的理由本身，绝不能杀掉 reward worker）、`flow_size=100` raise `divisible by 8`。

饱和 / 线性 / 单帧三条走真实的 `__call__` → `decode_artifact_frames` 路径，用内存里的 `RewardInferenceArtifact(path="", media_type="video", media=<[C,T,H,W] uint8>)`——`__post_init__` 在 `media` 非空时允许空 path，所以不需要 tmp 文件、不需要 ffmpeg。

#### optional 车道：真 RAFT-small 判别（**本 sprint 的旗舰交付**）

`_load_module`（`:62`）硬编码 `raft_small(weights=Raft_Small_Weights.DEFAULT)`。实测权重 **4.0 MB**（不是简报说的 20 MB），本机已缓存，加载 0.42 s；`flow_size=128` 下每对 ~13-17 ms（128 是 RAFT 相关金字塔的真实下限）。

```python
@pytest.mark.optional
def test_real_raft_separates_a_static_clip_from_a_panning_clip() -> None:
    ...
    torch.manual_seed(0); base = torch.rand(1, 96, 96, 3)
    static = model._dynamic_degree(clip(shift=0))
    moving = model._dynamic_degree(clip(shift=6))
    assert static < 0.002
    assert moving > 20 * static
    assert min(1.0, static * 50.0) < 0.1
    assert min(1.0, moving * 50.0) > 0.9
```

实测：`static raw = 0.001175`（scored 0.0587），`moving raw = 0.063542`（scored 1.0000），比值 **54x**。阈值留了 2x 以上余量，权重由 `Raft_Small_Weights.DEFAULT` 的 hash 钉死，不 flaky。整条 0.59 s。取不到权重时 `pytest.skip`，离线干净机器降级为 skip 不是 fail。

**这是全套里唯一能抓住「算术还对但感知坏了」的断言**——RAFT 输出取错 index（`[-1]`）、丢掉 [-1,1] 映射、帧对顺序调换，stubbed-flow 那八条对这三种全瞎。所以这里**不该**打诚实缺口标签，该交付一个真测试。

**唯一要保留的反面直觉**：不要用随机初始化的 RAFT 冒充判别。`raft_small(weights=None)` 实测 0.01 s 离线可建、990,162 参数，诱人得很——但随机权重的流场不携带运动信号，那样的测试什么都不断言。**要么用训练权重，要么不做。**

**车道意义**：`tests/conftest.py:64-69` 明写 `optional` 车道全仓零成员、作为保留脚手架「Do not delete as dead code」。这条是它的第一个真成员——把一个保留脚手架变成活覆盖，这本身就是不该在这里发缺口标签的第二个理由。

---

### 2.6 RW-07 — OCR：手抄的是 PaddleOCR **2.x** 的格式，而 extra 钉的是 >= 3.5.0（保留 + 双协议）

`_FakePaddleOCR.ocr()` 返回 `[[(None, (text, 1.0))]]`——手抄依赖库的返回布局，是死代码审计第 5 形态（reimplementing a dependency's internals）落在测试替身上：格式一变，生产坏、测试绿。

诚实地说，这两条 fake-engine 测试本身**不差**：`assert score == pytest.approx(1.0)` 和 `assert 0.0 < score < 1.0` 都是生产代码从假货的文本算出来的，不是复述。问题在安全网：唯一碰真引擎的 `test_ocr_reward_paddleocr_core_scoring_behaviors` 是 `pytest.importorskip("paddleocr")`，而 paddleocr 在可选的 `[ocr]` extra 里、本机没装——实测它就是 `tests/rewards` 两个 skip 之一。所以引擎边界**实际零真覆盖**，唯一钉住 PaddleOCR 返回格式的是测试文件里的一个字面量。

**动作一：保留 `reward._engine = ...` 这个 seam，不要改成 `worker_config`。** 它是 `vrl/rewards/models/ocr.py:9-10` 文档化、`vrl/rewards/functions/ocr.py:58-63` property 支撑的设计接缝。给 `OCRReward.__init__` 加 `worker_config` 是独立的生产改动，不是测试清理，且不增加任何覆盖。

**动作二（真正加覆盖的那一条）：把 in-process double 按两种引擎协议参数化。**

```python
class _PaddleOCR2x:   # .ocr(frame, cls=False) -> [[(box, (text, score))]]
class _PaddleOCR3x:   # .predict(frame) -> [{"rec_texts": [...], "rec_scores": [...]}]
```

`vrl/rewards/models/ocr.py::_run_paddle_ocr` 用 `hasattr(engine, "predict")` 派发，3.x 那条路径今天**零覆盖**，而 `[ocr]` extra 钉的正是 `paddleocr>=3.5.0`。实测两种协议下三条断言全部一致通过。

**动作三：收紧断言。**

- `assert 0.0 < score < 1.0` → `assert score == pytest.approx(0.875)`。推导写进 docstring：一次插入，`len("freewifi") == 8`，`frame_interval=4` 下 `[3,8,64,64]` 采到 2 帧取均值。松界抓不住「除以 `len(text)` 而不是 `target_len`」（那会给 0.888，仍在界内）。
- **补一条缺失的不变量**：full-credit 捷径是 image-only（`models/ocr.py:119` 的 `single_image and target_text in text`），所以帧里逐字包含目标文本的**视频**必须拿不到满分。实测更锋利——它拿 **0.0**：`distance("cafefreewifiopen", "freewifi") == 8`，`min(8, target_len=8) = 8`，reward `1 - 8/8 = 0`，且 0 不进 `frame_rewards` 于是均值为 0。与图片路径的 1.0 正好相反，是一条强断言。
- 补 `_join_ocr_texts` 的 `zip(..., strict=False)` 静默截断：实测 `_extract_ocr_text([{"rec_texts":["ab","cd"],"rec_scores":[1.0]}]) == "ab"`。
- 把三条复述函数名的 docstring 换成不变量本身（例如「flow_grpo 的子串捷径是 image-only：视频里逐字出现目标文本仍按编辑距离算分」）。

**动作四：真引擎测试按本仓惯例门控，并断言 SHAPE 而不是满分。** 门控 = `pytest.importorskip("paddleocr")` **加** `WM_RUN_REAL_MODEL_TESTS=1`（`tests/ci_envs.py`，兄弟先例 `tests/rewards/inference/test_in_process_runtime.py:436-441`），因为首次运行会联网下载 OCR 权重。断言：`_build_paddle_ocr()` → 渲染文本 → `_extract_ocr_text(_run_paddle_ocr(engine, frame))` 非空且经 `_normalize_ocr_text` 后包含渲染的文本。**那才是钉住字面量的东西。** 后面再跟一条差分：匹配目标的分数严格高于不匹配的，而不是 `== 1.0`。渲染必须用显式的 `ImageFont.truetype(path, 48)` 并在字体缺失时 skip；**绝不用** PIL 默认位图字体（"HELLO" 只有 33x8 px）。确定性来自固定字体 / 字号 / 字符串 / 黑底白字，无 RNG。

**动作五：把这条真相写进模块 docstring**（本文件 `:27-29` 已有散文注记的先例）：fake 的 `[[(None,(text,1.0))]]` 字面量钉的是 PaddleOCR 2.x，而 `[ocr]` extra 钉的是 >= 3.5.0；两种形状都是手写字面量，只由 opt-in 真测试校验。

---

### 2.7 RW-08 — VideoScore2：假 tokenizer 把一条生产分支定义掉了（→ T2）

`_FakeTokenizer.encode` 是 `key = text.strip(); return [self._vocab[key]] if key in self._vocab else [99]`。它 `strip()` 掉前导空格，于是 `_marker_token_ids`（`vrl/rewards/models/videoscore2.py:321-331`）刻意搜索的**两个变体** `marker_word` 与 `f" {marker_word}"` 塌成同一个 id、去重恒命中、`" " + marker` 那条分支**永不执行**；它又永远返回单元素列表，于是 `_find_subsequence` 的多 token needle 路径（`marker_ids` 之所以是 `list[list[int]]` 的全部理由）也从不被走。破掉多 token 或空格前缀 marker 的回归，这套测试全绿。

软打分测试本身很强（`test_soft_scores_anchor_last_marker_not_cot_mention` 复现了一次真实的线上错位：CoT 提到 marker 后编号列表的 "1" 被读成分数，实测 soft 1.0 vs hard 3），所以**一条断言都不能删**——只换驱动它们的 tokenizer。

**新增 `tests/rewards/videoscore2/fixtures.py::build_tiny_marker_tokenizer()`**：真 `PreTrainedTokenizerFast` 包一个真 byte-level BPE，离线构造、零下载。关键是 merges 的优先级——**空格前缀形式排在前面**：

```python
for word in ("Ġquality", "Ġalignment", "Ġconsistency", "quality", "alignment", "cons", "istency"):
```

实测结果（**必须写进 docstring 说明为什么顺序重要**，而不是复述函数名）：

| marker | 变体 | 是否多 token |
|---|---|---|
| `quality` | `[[47], [14]]` | 否 / 否 |
| `alignment` | `[[55], [27]]` | 否 / 否 |
| `consistency` | `[[58, 64], [41]]` | **是** / 否 |

digits 1-5 全是单 token。**构建 0.56 ms，无种子——固定 vocab / merges、零随机初始化。**

> BPE 的 merges 需要所有中间前缀都在 vocab 里，只放「单字符 + 最终词」会 `Error while initializing BPE: Token 'Ġq' out of vocabulary`。这是实测踩到的。

**保真性已用真 tokenizer 校验过**（`AutoTokenizer.from_pretrained("TIGER-Lab/VideoScore2")`，本机缓存 2.44 s）：

| marker | 真 tokenizer 变体 | 多 token |
|---|---|---|
| `quality` | `[[10473], [4271]]` | 否 / 否 |
| `alignment` | `[[44598], [17189]]` | 否 / 否 |
| `consistency` | `[[6254, 47094], [28137]]` | **是** / 否 |

digits → `{1:16, 2:17, 3:18, 4:19, 5:20}`，全单 token。**tiny fixture 的形状与真 tokenizer 逐项一致。**

**三条现有软打分测试改成喂真文本**：把不透明的 `generated_ids` 整数列表换成 `tok.encode("quality (1) quality: 3 (2) alignment: 4 (3) consistency: 4", add_special_tokens=False)`。既保住每一条断言，又让原本不可读的 setup 读起来就是评委的真实输出，并且让真 BPE 而不是测试自己去决定 id。实测三条全过（soft = 3.0 / 4.0 / 4.0）。

**两条补上死分支的新测试**：

- `test_marker_search_covers_both_spacings_and_multi_token_markers`：三个 marker 各 `len(...) == 2`，且 `consistency` 至少有一个 `len(v) > 1` 的变体。
- `test_unprefixed_multi_token_marker_still_anchors`：软打分 `"(1) quality: 3, (2) alignment: 4, (3)consistency: 5"`——`consistency` 没有前导空格，**只有两 token 的 needle 能匹配**。实测 `physical_common_sense == 5.0`。

**optional 车道的保真钉**（第二个真成员）：tiny tokenizer 编码的是我们**对 Qwen2 的信念**，只有真 tokenizer 测试能抓住这个信念过期。一条 `@pytest.mark.optional`，断言上表右侧那三行 + `set(_resolve_digit_token_ids(real_tok)) == {1,2,3,4,5}`（软路径在这条不成立时会静默关闭自己）。实测 1.79 s；冷机器约 15 MB 下载（tokenizer.json 11M + vocab.json 2.7M + merges.txt 1.6M）——**不碰 7B 权重**。

---

### 2.8 RW-11 — 基础设施：全仓唯一的真 vLLM kernel 测试在推荐环境里是黑的

`tests/nn/kernels/test_vllm_paged_attention_real_ops.py:20-23`：

```python
try:
    kernels = VllmPagedAttentionKernels(VllmPagedAttentionConfig(family="janus_pro"))
except ARAttentionUnavailable as exc:
    pytest.skip(f"vLLM paged-attention internals are unavailable: {exc}")
```

本机实测：`import vllm` **成功**，`vllm.device_allocator.cumem` **成功**，`importlib.metadata.version("vllm")` = 0.21.0；但 `vllm.v1.worker.block_table` **失败**，于是这个 `@pytest.mark.gpu` 测试在一台有 RTX 5090、装了 vLLM 的机器上恒 skip。

**根因**：README:234 推荐 `pip install "vllm>=0.21.0,<0.22" --no-deps`（`pyproject.toml:114` 的注释同样这么写），为的是隔离 ABI。代价是 vLLM 自己声明的依赖一个都不装。

**必须纠正简报的说法**：这不是「漏了 cbor2」一个包。实测枚举 vLLM 的非-extra `Requires-Dist`，**48 个缺失**。到 `block_table` 的 import 链先撞 `cbor2`，补上之后立刻撞 `gguf`（`vllm/transformers_utils/gguf_utils.py:9`），而且 `gguf` 不能用桩——vllm 读它的 `gguf.constants` 符号。所以「补进 extra」不是一行改动，需要先确定一个能让内部 API import 成功的最小集合。

**本 sprint 交付的是那一半确定能做对的：让门控说真话。** 判据现成：

```python
if importlib.util.find_spec("vllm") is None:
    pytest.skip("vLLM is not installed")     # 真正的能力缺席 -> skip 合法
# 装了 vLLM 却 import 不到内部 API = 环境坏了，不是能力缺席 -> 让它红
kernels = VllmPagedAttentionKernels(VllmPagedAttentionConfig(family="janus_pro"))
```

这样一台按 README 装出来的机器会**明确报出**「vLLM 装了但不完整」，而不是无声跳过全仓唯一的真 kernel 覆盖。补依赖清单是紧随其后的独立动作（要改 `pyproject.toml` + README，见 §8 的协作风险）。

**诚实声明**：简报还提到「补一个 CPU-real 测试」。`get_kv_cache_shape` / `new_block_table(device=cpu)` / `compute_slot_mapping` 原则上都不需要 CUDA，但**本机无法验证**——上面那条 import 链在补齐依赖之前根本走不到。所以本 sprint **不承诺**那个 CPU-real 测试；它是环境修复落地之后的后续项，写在 §5 的诚实缺口里。

> **范围提示**：本条严格说不属于「reward 侧」。它被并进来是因为它挡着本轨道的旗舰 gpu 测试。它与其余各条零耦合，可以单独 merge。

---

## 3. 共享机械（跨条目复用的部分）

新增两个 fixtures 模块，都对齐 `tests/models/steps/denoise/fixtures.py` 的既有先例：

| 文件 | 提供 | 被谁用 |
|---|---|---|
| `tests/rewards/fixtures.py` | `build_tiny_clip_repo(root, *, projection_dim, logit_scale_init_value, seed=0)` | RW-01（`projection_dim` 从资产推导 = 768，`logit_scale` 任意）、RW-02（`projection_dim=16`，`logit_scale=log(26)`） |
| `tests/rewards/kling_video_reward/fixtures.py` | `build_tiny_kling_reward_model(...)`、`build_tiny_qwen2vl_processor(root)` | RW-03、RW-04 |
| `tests/rewards/videoscore2/fixtures.py` | `build_tiny_marker_tokenizer()` | RW-08 |

**关于「aesthetic 与 PickScore 共用 session fixture」这个说法要收窄**：两边要的 `projection_dim` 不同（768 vs 16，前者由 LAION 头的入宽决定、不可协商），所以是**一个参数化的 builder、两个 session-scoped 的仓库**，不是一个共享仓库。真正被摊薄的是 `CLIPModel` 类的首次 lazy import（实测第一次建仓 949-1006 ms，第二次 8.4 ms）——而这笔 import 本文件今天已经在付（现有 `test_aesthetic_model_reads_transformers_5_projected_image_features` 单条 call time 就是 1.32 s）。

仓库尺寸实测：aesthetic 版 108,769 参数 / 465 KB；pickscore 版 60,641 参数 / 337 KB。均在 `HF_HUB_OFFLINE=1` 下工作。

**已知无害噪声**：`CLIPProcessor.save_pretrained` 会打一行 `The OrderedVocab you are attempting to save contains holes for indices [1]`；`CLIPTokenizer` 会打一行 `Deprecated in 0.9.0: BPE.__init__ will not create from files anymore`。`pyproject.toml` 没有 `filterwarnings = error`，不会导致失败。

---

## 4. 非目标（刻意保留的替身，每条附理由）

> 本节是强制章节。这些替身**不**被本 sprint 改动。

### 4.1 (c) 类 wire boundary：六个 revision 转发用例

`tests/rewards/test_clip_reward_models.py:71-114`（aesthetic ×2 参数）与 `:157-215`（pickscore ×4 参数）用 recorder 替掉 `from_pretrained`，断言收到了哪个 revision kwarg。

**保留理由（实测）**：本地目录**没有 revision 概念**。我用 tiny 仓库实测 `CLIPModel.from_pretrained(<local dir>, revision="does-not-exist-anywhere")` —— **正常返回一个 CLIPModel**，revision 被静默忽略。所以 tiny 本地仓库在物理上无法表达这个边界，只能靠拦截调用。这是正当的 (c) 类保留。

**唯一改动**：删掉 `:103` 的 `torch.load` monkeypatch（实测删掉后两个用例照过，真资产 1.8 ms 加载），并在 docstring 里写明「这是 hub 线边界断言，真实对位在 opt-in lane 的 `tests/rewards/inference/test_in_process_runtime.py:436-485`」。

> 同形状的 recorder 在全仓还有至少 8 处（`tests/e2e/test_real_checkpoint_rl.py:1016-1025`、`magi_1/test_subprocess_runtime.py:554`、`causvid/test_replay_and_loading.py:370,405`、`echo/test_model_loading.py:204`、`cosmos/anima/test_artifact_resolution.py:54`、`wan_2_1/test_model_loading.py:629`、`scripts/eval/test_sana_aesthetic_checkpoint_eval.py:682`）。它们属于同一裁定，本 sprint 不动。

### 4.2 (b) 类环境模拟：`local_files_only` 传播

`tests/rewards/kling_video_reward/test_model_loading.py:182-208` 和 `:211-276`。

**保留理由（实测）**：把 `HF_ENDPOINT` 指向一个死地址后，不带 `local_files_only` 调用 `snapshot_download` **仍然在 0.005 s 返回缓存快照**（hf_hub 1.23 在连接错误时回落到缓存）。所以「它到底走没走离线」在进程内不可观测，只有真实的网络可达性能区分。这一条断言配得上拦截。

`:211-276`（`_create_model_and_processor` 的 offline kwargs）在 §2.4-B 之后**部分**转成了真跑，但断言「传给 hub 的是 `local_files_only=True`」这一句仍需 recorder——保留原测试，新增的 tiny-real 测试与它并存、不替代。

### 4.3 e2e 的 `build_tensor_mean_model`

`tests/e2e/test_real_checkpoint_rl.py:683-703`。它是 reward **传输**测试（artifact 落盘、manifest、reward_components 汇总），形状与它要测的东西完全匹配。真模型 case 是**并列新增**，不是替换。

### 4.4 `_FakePaddleOCR`（升级为双协议，但仍保留为替身）

PaddleOCR 在可选 `[ocr]` extra 里、默认环境没装（实测 ModuleNotFoundError），是真正的外部引擎边界。改造见 §2.6：保留替身，但让它同时覆盖 2.x 与 3.x 两种协议，并给它一个真正会断言的真引擎对位。

### 4.5 不在本轨道范围内的 reward 侧替身

- `tests/rewards/service/test_service.py:91-105` 的 `_FakeRuntime`：假的是**打分模型**不是**线**（该文件起真 aiohttp server、走真 127.0.0.1、真 subprocess CLI + SIGTERM，实测 29 tests / 0.35 s）。真 `InProcessRewardInferenceRuntime` 另有真覆盖。属于另一条轨道。
- `tests/rewards/inference/test_in_process_runtime.py` 的 `_FakeCumemAllocator` 一族：涉及 vLLM CuMemAllocator 的进程级残留会计，另有独立裁定，本 sprint 不碰。

### 4.6 机制层面的非目标：本 sprint **不**引入 `real_cover` marker

> **现状修正（2026-07-30）：** 轨道一已经注册 `real_cover` 并落地 AST 守卫。
> 以下文字保留的是本计划写作时的基线；实际施工时应直接使用当前 marker 契约，
> 不再等待 `SPRINT_test_tiers`。

`grep -rn "real_cover" tests/ vrl/ docs/ pyproject.toml` **零命中**，而 `pyproject.toml:203` 开着 `--strict-markers`——直接贴上去是**收集期 ERROR**，不是警告。

本 sprint 因此**不依赖尚未落地的 marker**，改用两样已经存在的东西：
1. `@pytest.mark.optional`（`tests/conftest.py:81-85`，marker 已在 `pyproject.toml:210` 注册）；
2. 模块 docstring 里的散文缺口声明（`tests/rewards/functions/test_ocr.py:27-29` 已有先例）。

§5 的清单就是这批缺口的登记册。如果轨道一之后落地了 `real_cover` + AST meta-test，本文即
`tracked_in=` 的目标，届时把 §5 逐条转成 marker 是一次机械改写。**这样本 sprint 可以独立
merge 而不炸 `--strict-markers`。**

---

## 5. 诚实缺口（本 sprint 明确标为「进程内测不了」的东西）

| 缺口 | 阻塞证据（具体到行/实测） | 真实对位 |
|---|---|---|
| `KlingVideoRewardModel._prepare_batch` 的 decode + chat-template 段 | `kling_video_reward.py:191` `from qwen_vl_utils import process_vision_info`；实测 `qwen_vl_utils` / `decord` / `cv2` 均 ModuleNotFoundError（都在 `[reward]` extra，CI 只装 `[cosmos]`） | **有**：新增一条 `@pytest.mark.optional` + `pytest.importorskip("qwen_vl_utils")` + opencv 写 8 帧 mp4 + 真 processor 的 **CPU** 测试。比 2-4 分钟的 GPU e2e 便宜两个数量级，覆盖的正是 decode + chat template 这一段。纯字典装配段已由 §2.4-C 抽成 `_build_chat_payload` 转 T1 |
| 真 Kling 2B 权重的数值打分 | e2e 唯一的 kling case 覆盖了 `model_factory`；`test_real_checkpoint_rl.py:697` 把它换成 `build_tensor_mean_model` | **今天没有**。§2.4-D 描述了新 case 的形状与三处必补项。在它落地前，「`tests/e2e/test_real_checkpoint_rl.py` 覆盖 `online_grpo_kling_video_reward`」这句话的诚实版本是：**checkpoint 是真的，reward model 不是** |
| 真 PickScore_v1 权重 + 真 ViT-H/14 processor 的 resize/normalize 常数 | tiny 仓库证明的是接线、张量数学、媒体派发，不是 reward 的语义质量 | **今天没有**。属于 `WM_RUN_REAL_MODEL_TESTS=1` 车道，与 `tests/e2e/test_real_checkpoint_rl.py` 并列 |
| PickScore `_score` 里 `.diag()` 的选择 | `_score` 对每张图重复同一 prompt，文本行逐字节相同 → 实测 `diag().mean()` 与全矩阵 mean 精确相等 | **无法通过这个入口测**。要测它需要 per-image prompt，那属于真 e2e。记录，不假装 |
| 真 PaddleOCR 引擎的返回格式 | `paddleocr` 在 `[ocr]` extra，实测本机未装、该测试 skip | **有**：§2.6 动作四的 `importorskip("paddleocr")` + `WM_RUN_REAL_MODEL_TESTS=1` 门控测试，断言 `_extract_ocr_text` 能从真引擎输出里读出渲染的文本 |
| 真 vLLM paged kernel（CUDA） | 实测 `vllm.v1.worker.block_table` import 失败：`cbor2` 缺失，补上后 `gguf` 缺失；vLLM 声明的非-extra 依赖共 48 个缺失 | **有但今天是黑的**：`test_vllm_paged_attention_writes_real_cuda_kv_cache`。§2.8 让它从「无声 skip」变成「明确报错」；补齐依赖清单是后续独立动作 |
| vLLM paged kernel 的 CPU-real 部分 | 同上——依赖链不通，本机无法验证任何 CPU-real 断言 | **今天没有，且本 sprint 不承诺**。环境修复落地后再评估 |
| `_find_subsequence` 返回值的 off-by-one（`i + 1` vs `i + len(needle)`） | 被 8 token 的 `_DIGIT_PROXIMITY_WINDOW`（`videoscore2.py:89`）吸收 | **今天没有**。新旧测试都抓不住。记录为已知未覆盖边缘 |
| `mean` pooling 排除最后一个真 token 是意图还是上游 VideoAlign 的 bug | 实测行为确实是 `ref[0,:2].mean()` 而非 `ref[0,:3].mean()` | 无。§2.3 的测试把这件事**写死使其可判定**——不静默「修正」它，也不假装它是「valid prefix 的均值」 |

---

## 6. 与既有裁定的冲突（必须显式重判的部分）

### 6.1 推翻 [[SPRINT_test_suite_tiny_real_and_fake_audit]] 对 kling `_Fake*` 的 KEEP

该 sprint（§KEEP，~30 条）把 kling `_Fake*` 列为合法边界替身，理由是「多 GB Qwen2-VL，无 tiny Hub repo」。

**推翻**：「没有 tiny Hub repo」与「无法 config-init」是两件事，而 T2 的定义恰恰是后者。实测 `Qwen2VLConfig(...)` → `KlingQwen2VLRewardModel(...)`：**31,824 参数、5.8 ms、纯 CPU、零下载、零缓存权重**。整条 `_create_model_and_processor` 离线跑通 195 ms。当时的裁定看的是调用点（「要一个 2B 模型」），没有读构造签名——正是 AGENTS.md「Dead Code Audit — 五种形态」里点名的那种误判方式。

保留有效的部分：`_FakePaddleOCR`（外部引擎）、fake Ray ref、只需 `.decode` 形状的 fake VAE，那些裁定仍然正确。

### 6.2 推翻 [[RW-09 非目标]] 对 tiny Qwen2-VL processor 的「vendoring」定性

见 §2.4-B：用 transformers 自己的 `Qwen2Tokenizer` / `Qwen2VLImageProcessor` / `Qwen2VLVideoProcessor` + `save_pretrained`，格式由库写由库读，实测 4 ms 往返。这不是 reimplementing a dependency's internals。

（该非目标的**其余部分仍然成立**并被本 sprint 采纳：真 `snapshot_download` 在 0-byte 缓存 fixture 上 0.1 ms 解析 revision，是可行的；`local_files_only` 才是真正的 (b) 类保留。）

### 6.3 不采纳「给 8 处无断言测试统一加注释」式的机制

同理适用于本 sprint：**不新增没有消费者的手写约定**。本 sprint 的缺口登记册是 §5，它的消费者是本文档本身以及未来的 `real_cover` AST meta-test；docstring 里写的是不变量与推导，不是「这一行不抛就是断言」这类第三次复述。

---

## 7. 与任务简报口径的偏差（逐条纠正）

| 简报说 | 实测 | 影响 |
|---|---|---|
| 默认车道 +1.5 s | **+0.71 s**（5.50 → 6.21 s，各 3-4 次取平均） | 预算减半；仍建议按 +1.0 s 立预算留余量 |
| 77K 参数的 CLIP 仓库，aesthetic 与 PickScore 共用 | 两个不同 `projection_dim` 必须建两个仓库：**108,769**（aesthetic）/ **60,641**（pickscore）参数 | 共享的是 builder 和 `CLIPModel` 首次 import，不是仓库实例 |
| 全黑 5.5437 vs 全白 5.3218 | 本机配置下 **5.348966 vs 5.968630**（含灰阶 5.612262） | 数值随 tiny 配置与 torch 版本的 `Linear` init 变化 → **断言必须是「不相等」，不是字面值** |
| 32K 参数的 Qwen2-VL Kling reward model | **31,824** | ✓ |
| RAFT moving/static ≈ 38x；下载 ~20 MB；35 ms/pair | **54x**（本机纹理）；**4.0 MB**；**~13-17 ms/pair @128px** | 阈值（`static < 0.002`、`moving > 20 * static`）余量更大；128 px 相关金字塔下限属实 |
| `pyproject:114` 的 `--no-deps` 漏了 vLLM 声明的 cbor2 | 漏的是 **48 个**非-extra 依赖；到 `block_table` 的链先撞 `cbor2` 再撞 `gguf`（且 `gguf` 不能桩掉） | 「补进 extra」不是一行改动；本 sprint 只交付门控修复，依赖清单是后续项 |
| 工作树在 `vrl/rewards/`、`vrl/models/families/registry.py`、`pyproject.toml` 有未提交编辑 | 本次读取时**这三处全部干净**；HEAD 已从 `12f35438` 前进到 `812cc3cf`；当前脏文件是 `tests/generation/execution/test_execute_request_pipelined.py`、`tests/models/families/flux/test_diffusion_nft_interface.py` + 一个未跟踪的 `test_zzscratch_probe_real.py` | 与本 sprint 零重叠。但 `pyproject.toml` 是共享面（§2.8 若要补 vLLM 依赖会碰它），落地前先 `git status` 复核 |

---

## 8. 保持不变

- 不删任何测试的覆盖。7 个被「改写」的测试，其断言全部保留或加强；`_aesthetic_head_state_dict()` 是唯一被整体删除的东西，它是零调用者的手抄常量。
- 不为整洁而重命名或搬文件。新增文件都落在既有的同目录 `fixtures.py` / `test_*.py` 惯例上。
- 不给 `OCRReward.__init__` 加 `worker_config` 参数（那是独立的生产改动，且不增加覆盖）。
- 不动 e2e 的 `build_tensor_mean_model` case。
- 不引入 `real_cover` marker、不引入 `xfail`/`skipif` 作为缺口标注手段。
- 不用随机初始化的 RAFT 冒充运动判别。
- 不硬编码任何浮点分数字面量作为断言目标（线程数漂移实测 1.5e-8，torch 版本 init 差异更大）。
- 不把 `_aesthetic_head_state_dict` 的几何搬进 fixture——`projection_dim` 从 shipped 资产推导。

---

## 9. 验收

### 9.1 分条验证

```bash
# RW-01 / RW-02（含 pickscore 视频布局修复）
.venv/bin/python -m pytest tests/rewards/test_clip_reward_models.py -q -p no:randomly

# RW-03 / RW-04 A+B+C
.venv/bin/python -m pytest tests/rewards/kling_video_reward -q -p no:randomly

# RW-06 默认车道
.venv/bin/python -m pytest tests/rewards/functions/test_motion_dynamics.py -q -p no:randomly

# RW-07
.venv/bin/python -m pytest tests/rewards/functions/test_ocr.py -q -p no:randomly

# RW-08
.venv/bin/python -m pytest tests/rewards/videoscore2 -q -p no:randomly

# optional 车道的两个真成员（今天为止该车道的全部成员）
.venv/bin/python -m pytest tests/rewards -q -p no:randomly --optional -k "real_raft or real_videoscore2_tokenizer"

# 需要 --extra reward 的两条（本机默认环境会 skip）
.venv/bin/python -m pytest tests/rewards -q -p no:randomly --optional -k "prepare_batch"
WM_RUN_REAL_MODEL_TESTS=1 .venv/bin/python -m pytest tests/rewards/functions/test_ocr.py -q -p no:randomly

# RW-11：在一台按 README 装了 vLLM 的机器上，这条现在必须 FAIL 并指出缺哪些依赖，而不是 skip
.venv/bin/python -m pytest tests/nn/kernels/test_vllm_paged_attention_real_ops.py -q -p no:randomly -rs
```

### 9.2 区域回归 + 预算核对

```bash
/usr/bin/time -f "%e s" .venv/bin/python -m pytest tests/rewards -q -p no:randomly
```

**验收门槛：< 6.5 s**（基线 5.50 s，实测替换后 6.21 s，留 ~5% 余量）。超过就说明某条转换比预算贵，回去查是不是漏了 session fixture 或重复付了 warmup。

### 9.3 全套

```bash
.venv/bin/python -m pytest tests -q -p no:randomly
```

预期 189 s → ~190 s。**skip 数不是指标**（它是环境的函数）——用 marker/gate 清单核对：本 sprint 后 `optional` 车道应从 0 个成员变成 2 个。

### 9.4 Ruff（仅限本任务触碰的文件）

```bash
.venv/bin/ruff check --fix <touched>  && .venv/bin/ruff format <touched>
.venv/bin/ruff check <touched>        && .venv/bin/ruff format --check <touched>
```

---

## 10. 顺序

条目之间基本无依赖，但共享 fixture 决定了顺序：

1. **`tests/rewards/fixtures.py::build_tiny_clip_repo`**（RW-01 与 RW-02 的共同前置）→ RW-01 → RW-02（RW-02 带一个生产 bug 修复，独立可 review）。
2. **`tests/rewards/kling_video_reward/fixtures.py`** → RW-03（纯新增，零风险）→ RW-04 A/B（loader 与 remap）→ RW-04 C（`_build_chat_payload` 抽取，是唯一改动 kling 生产文件结构的一步）。
3. **RW-06**（独立，且交付 optional 车道第一个真成员——建议早做，它给车道机制一个真实的验证）。
4. **RW-08**（独立）。
5. **RW-07**（独立，纯测试侧）。
6. **RW-11**（与全部其余条目零耦合，可先可后；补 vLLM 依赖清单要碰 `pyproject.toml`，与其它 agent 的共享面，落地前复核 `git status`）。
7. **RW-04 D**（e2e 真模型 case）——优先级最低，只在 A/B/C 全绿之后做，且必须先把 `_local_reward_overrides` 参数化和 Qwen2-VL-2B 守卫补上。

风险：**medium**。风险集中在两处：RW-02 的生产 sniff 修改（有兄弟实现可抄、四种布局实测通过）、RW-04 C 的 `_prepare_batch` 拆分（改的是生产文件结构，但抽出的是纯字典装配、无行为变化）。其余全部是测试侧新增或等价替换。

---

## 11. 执行记录（2026-09-05）

### 11.1 Re-baseline：8 月 9 日基线到当天代码的差异

| 条目 | 基线说 | 当天核对 |
|---|---|---|
| RW-02 生产 bug | `pickscore.py:66-68` 切通道轴 | **已在 28210cf6 单独修掉**（嗅探写法照抄 nsfw_safety），本 sprint 只剩测试转换 |
| §4.6 `real_cover` | 不引入 marker | marker 与 AST 守卫已落地；本次直接给保留的替身贴 `real_cover` 标签（aesthetic revision → in-process 真 CLIP gate；pickscore revision → `None` + `tracked_in` 本文；两条 kling recorder → 新的 tiny-real `_create_model_and_processor` 测试） |
| `tests/rewards/test_model_protocol.py` 已付 transformers import | 该文件在 test-slop 批次 6 删除 | 不影响：`kling_video_reward.py` 模块级 import 仍由 `tests/rewards/kling_video_reward/test_function.py` 触发 |
| RW-07 环境 | `paddleocr` 未装、真引擎测试 skip | 本机装的是 **2.9.1（2.x）**，权重已缓存；旧的 `importorskip` 测试在默认车道里**真跑过引擎**，本次按动作四改为 `WM_RUN_REAL_MODEL_TESTS=1` 门控 |
| RW-11 环境 | `vllm.v1.worker.block_table` import 失败（缺 cbor2/gguf） | 本机 `cbor2` 已装，内部 API import **成功**；门控修改仍按 §2.8 落地（只有 vLLM 缺席才 skip） |
| 依赖 | `qwen_vl_utils` / `decord` / `cv2` 缺席 | 三者本机均在，`_prepare_batch` 的 optional 测试可以真跑 |
| 实测数字 | tiny CLIP 108K/60K 参数；RAFT 54x | 本次 tiny 配置更小（34,369 / 10,305）；RAFT 本机纹理 **38x**（static 0.001175，moving 0.0448）；阈值不变仍留 ~2x 余量 |

### 11.2 落地内容

- **RW-01/02**（`tests/rewards/fixtures.py::build_tiny_clip_repo`，`shipped_aesthetic_projection_dim` 从资产推导 = 768）：aesthetic 断言 head 形状、非零权重、黑白图分数不等、批量形状；PickScore 用 `logit_scale=log 26` 对 `cosine_similarity` oracle 做 differential 断言，另加 NCHW 派发与非媒体返回 0.0 两条。`_aesthetic_head_state_dict()` 与两处 `torch.load` monkeypatch 删除。
- **RW-03**（`tests/rewards/kling_video_reward/fixtures.py` + `test_scoring.py`）：31,824 参数真模型；八条测试按 §2.3 表逐条落地，期望值全部由同一模型的 `head_logits` 当场重算。
- **RW-04 A/B/C**：`test_checkpoint_loader_strict_loads_a_live_model_in_either_key_layout`（真 peft 包裹 + 当前与 legacy 两种 key 布局的 `strict=True` 加载 + `_remap_qwen2vl_state_dict` 集合比对）；`test_create_model_and_processor_runs_offline_on_a_tiny_repo`（真 `Qwen2VLProcessor` 往返、special token 三个、pad/padding_side 进 config、`lora_namespan_exclude` 真的挡住 rm_head/embed_tokens/visual）；`_prepare_batch` 的纯字典段抽成模块级 `_build_chat_payload(video_paths, prompts, *, data_config, max_pixels, min_pixels)`（比 §2.4-C 多一个 `data_config` 参数，因为 fps/nframes/eval_dim 都来自它），T1 测试覆盖 budget 回落、min_pixels 条件写入、nframes 优先于 fps、非 uniform 拒绝。
- **§5 第一条缺口已补上**：`test_prepare_batch_decodes_a_real_clip_through_the_real_processor`（`@pytest.mark.optional` + `importorskip("qwen_vl_utils")`）用 `write_mp4` 写 8 帧 mp4，走真 decode + 真 chat template + tiny 真处理器，tiny 模型前向得到 `(1, 3)` logits。tiny 处理器带一个测试拥有的最小 chat template（视频占位 + 文本），不是 hub 文件的拷贝。
- **RW-06**（`tests/rewards/functions/test_motion_dynamics.py`）：banded flow 三条参数化 + 饱和 + 线性 + 单帧 + `divisible by 8`（默认车道 7 条）；`test_real_raft_separates_a_static_clip_from_a_panning_clip` 是 optional 车道第一个真成员。
- **RW-08**（`tests/rewards/videoscore2/fixtures.py::build_tiny_marker_tokenizer`）：真 `PreTrainedTokenizerFast` + byte-level BPE，marker 形状与真 tokenizer 逐项一致（`consistency` 无空格形式两 token）；三条软打分测试改为喂真文本；新增双拼写 / 多 token 与无前缀多 token 两条；optional 车道第二个真成员 `test_real_videoscore2_tokenizer_has_the_marker_shape_the_fixture_assumes`。
- **RW-07**：`_PaddleOCR2x` / `_PaddleOCR3x` 双协议参数化（substring 满分、视频编辑距离、视频不享受 image-only 捷径三条）；视频编辑距离断言收紧为 `0.875`；新增视频逐字包含目标仍得 0.0 的不变量；列长度不一致的 `zip` 截断单独成测；真引擎测试按 `WM_RUN_REAL_MODEL_TESTS=1` 门控、显式 TrueType 字体、断言形状与差分。`OCRReward.__init__` 未加参数，`_engine` seam 保留。
- **RW-11**：`find_spec("vllm") is None` 才 skip，装了却 import 不到内部 API 直接红。

### 11.3 未做（2026-09-05 时点；两项都已在 11.5 补齐）

- **RW-04-D**（真 2B 权重的 e2e case）：当时 GPU 被占。
- 补 vLLM 依赖清单进 `pyproject.toml`（§2.8 的后续项）。

### 11.4 验收数字

| 命令 | 结果 |
|---|---|
| `pytest tests/rewards tests/nn/kernels/test_vllm_paged_attention_real_ops.py` | 322 passed / 6 skipped，**6.49 s**（§9.2 门槛 < 6.5 s） |
| `--optional -k "real_raft or real_videoscore2_tokenizer or prepare_batch"` | 3 passed（0.12 s / 0.48 s / 0.41 s） |
| `WM_RUN_REAL_MODEL_TESTS=1 ... -k real_paddleocr` | 1 passed（真 PaddleOCR 2.9.1 读出 HELLO，匹配目标分数 > 不匹配） |
| `tests/architecture/test_real_cover_labels.py` | 10 passed（新标签全部解析） |
| 全量 CPU 套件 | 见提交信息 |

### 11.5 2026-09-06 补齐：RW-04-D 与 vLLM 依赖清单

**vLLM 依赖清单**（commit `3fad24ab1`）：§2.8 说的"48 个缺失"是按 `--no-deps` 推算的上界。
实测法：在装了 vllm 0.21.0 的 base 解释器里 import VRL 真正用到的四个入口
（`vllm.v1.worker.block_table`、`vllm.device_allocator.cumem`、`vllm._custom_ops`、
`vllm.v1.attention.backends.utils`），把新进入 `sys.modules` 的顶层包映射回发行版，与 vLLM 的
非-extra `Requires-Dist` 求交：**30/69** 个真的被 import；其中只经 vllm 进入 lock、VRL 主闭包
自己不带的是 `cbor2`、`gguf`、`mistral-common[image]`（连带 `tiktoken`）、`openai`、
`openai-harmony`、`cloudpickle`、`py-cpuinfo`。这 7 个写进 README:234 的 `--no-deps` 命令
与 pyproject:114 注释；不动 uv.lock（它们经 `ar-vllm` extra 早已在锁里）。

**RW-04-D**（commit `d1fb9f812`，predict2 修复 `3a9f9865a`）：`tests/e2e/test_real_checkpoint_rl.py` 新增
`cosmos_predict2_kling_real_reward`，由 transport case `replace()` 派生：`reward_model_factory=None`
保留 preset 的真工厂；`CheckpointField(cfg_path="…worker_config.model_path", repo_id="KlingTeam/VideoReward")`
+ 一条 `cfg_path=None` 的 `Qwen/Qwen2-VL-2B-Instruct` 门控（§2.4-D 第 2 条：基座名写死在
`model_config.json`，没有 cfg 路径能重定向，只能门控）。`_local_reward_overrides(tmp_path, model_factory)`
按 §2.4-D 第 1 条参数化。实测（单卡 32 GiB）：

| | transport case | real reward case |
|---|---|---|
| 结果 | passed | passed |
| wall | 20.1 s | 38.7 s |
| GPU 峰值 | — | **16.4 GiB** |
| reward_debug 里的分数 | tensor-mean | 真 `KlingTeam/VideoReward@main`：VQ/MQ/TA/overall 四个键，两条样本分数不同 |

§2.4-D 第 3 条（把 `min_cuda_memory_gib` 抬到 ~36）**没有成立**：实测 16.4 GiB，因为 harness 现在会在
reward 加载前把 11B 文本编码器与 transformer 挪到主机内存（见下）。两个 case 的门槛改为 24.0。

**顺手发现 cosmos_predict2 这个 GPU-only case 已经烂了五处**（CPU 车道全绿，只有真跑才看得见）：

1. `sampling.max_sequence_length=64` 不是 predict2 sampling section 的字段（`VideoSamplingSection` 没有它；
   只有 `TextEncoded*` 有）——typed sampling 之后配置解析直接炸。删掉，并加
   `test_every_case_config_parses_without_a_gpu`（15 个 case 全部逐个解析）。
2. reference image 现在从 `GenerationInput.reference_image` 读（executor 不再收构造参数）；
   harness 改喂 `PromptExample(prompt, reference_image)`，走 `generation_input()`。
3. reward model 调用协议是 `model(artifact)`；tensor-mean 替身改为 `decode_artifact_frames` 后取均值，
   返回 Kling 的公开键 `overall_reward`。
4. collector 没拿到 lifecycle 计划 → reward 加载前没人让出 GPU → 真 2B 直接 OOM。现在传本机真实的
   `ResolvedDistributedResources.lifecycle`；`_DirectExecutorGenerationRuntime.activate/offload` 在 trainer
   自己的模块上扮演 colocated rollout worker 的 wake/sleep（trainer park → runtime 把模块搬回 GPU 生成 →
   reward 前搬走 → phase 结束 trainer restore）；shutdown 走 `rollout_schedule.lifecycle.shutdown_collector_runtime()`，
   让 reward 的 residual 门看到已 park 的 trainer。
5. preset 2026-06-10 改成全参微调（"sized for multi-GPU"）：2B 的 fp32 master + Adam 矩 + 11B T5 在 32 GiB
   上放不下（实测 optimizer.step 时 OOM）。case 改走 LoRA（`to_q/to_k/to_v/to_out.0`，仓库自己的 predict2
   LoRA 实验同一组 target）；`min_cuda_memory_gib=28` 是 LoRA 时代的遗留数字。
6. 打分后 artifacts 默认 `release`；`manifest.jsonl` 已不存在。override 加 `retain_artifacts=true`，断言改为
   "每个样本一个保留的 mp4 + debug results 行"。

另有一个生产 bug 被同一条车道抓出：**predict2 关 CFG（`guidance_scale=1.0`）时 replay 恢复
`KeyError: 'uncond_mask'`**——`prepare_latents` 在 CFG 关时不返回 uncond mask/indicator，导出为 None，
trajectory builder 丢掉 None，`restore_eval_state` 走 `replay_tensor()` 的 batch_context 回退就炸。
修法与 `negative_prompt_embeds` 一致（`.get`），并加 CPU 往返测试
（`test_predict2_restore_tolerates_the_absent_uncond_bundle_when_cfg_is_off`）。

### 11.6 2026-09-07：其余 e2e case 的巡检（`e624a0cc1`）

把 `CASES` 里剩下的十个 cached case 各跑一遍（GPU 当时被一条 anima 训练 + 一个渲染任务共用，约 13 GB 已占）：

| case | 结果 | 结论 |
|---|---|---|
| wan_2_1、sd3_5 ×4 | 先 `GenericDiffusionBatchExecutor` 缺 `family`/`task`（harness 位腐，已修：按 worker.py 的构造方式）；修后 OOM | OOM 是显存被共用造成，**未得出结论**，GPU 空出后重跑 |
| cosmos_anima ×2 | 先 `_SyntheticDiffusionReplayCollector` 缺 `supports_reward_generation_overlap`（已补）；修后 `PrecisionDriftError`：bf16/bf16 期望 1e-6 精确一致，实测 7.2e-4 | **真发现，未修**：synthetic collector 用 `restore_eval_state`+`forward_step`+`sde_step_with_logprob` 自算 old_log_prob，与 trainer 的 evaluator 路径有 7e-4 的差，需要在 GPU 上对照两条路径的 dtype/autocast 才能定位 |
| janus_pro | `janus.models` import 失败：`VisionConfig(PretrainedConfig)` 里 `params: dict = {}`，transformers 5.x 把 `PretrainedConfig` 子类转成 dataclass 后拒绝可变默认值 | **外部包问题**：`~/Desktop/Janus` 上游要 `field(default_factory=dict)`，不在本仓修 |
| cosmos_predict2_5 | skip：`nvidia/Cosmos-Predict2.5-2B` 快照不在缓存 | — |
| nextstep_1 | skip：门槛 64 GiB | — |
