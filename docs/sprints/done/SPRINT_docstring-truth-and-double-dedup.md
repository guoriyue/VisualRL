# SPRINT：让 docstring 陈述不变量，让每个替身只有一个 owner

状态：**done（2026-09-07 落地，见文末「落地记录」）**。原基线 main @ `812cc3cf`。本文全部数字均在本机实测（`.venv/bin/python -m pytest ... -p no:randomly`），不是估算。

序号：**轨道六 / 共六轨。风险 low。必须最后执行。**

> **为什么排最后**：前面四个转换轨（rewards / trainers+nn / models+generation / config+ray）会给它们改到的每一个测试重写 docstring。扫描先跑，等于把同一批 docstring 改两遍，而且后续轨的新 docstring 会被本轨的机械 gate 反复挑战。等它们全部落地后，本轨一次扫干净，并落一个防复发的 AST 守卫。

关联：[[SPRINT_test_suite_tiny_real_and_fake_audit]]（前置审计，本文第 6 节对其两条陈旧引用做修正）、[[SPRINT_deadcode_rewards]]（格式范本）、[[SPRINT_weak_test_cleanup]]。

---

## 0. 一句话

本轨零转换、零新增覆盖，只做三件事：**（1）** 把 347 条只是把函数名重念一遍的 docstring 改成陈述不变量或删掉，并顺手修掉 1 条**内容写错**的 docstring（写 `BCHTW`，而函数名、production 的 `permute` 和文件头 docstring 全都说 `BCTHW`）；**（2）** 把 4 组重复手写的测试替身各收给一个 owner，并把 4 张手写家族表改从 registry 派生；**（3）** 删 3 处证据充分的死物。

**运行时代价：全套 188.88s → 约 189.3s（+0.40s，+0.21%）**，全部来自新增的 AST 守卫（实测 0.40s call / 385 个文件）。其余改动为零运行时：替身收编后构造的仍是同一批对象，家族表派生后断言覆盖不变，删除项净减 1 个测试函数。

**不删任何覆盖**，唯一的例外是 §4.1 那条经三重证据证明「provably asserts nothing at all」的墓碑测试。

---

## 1. 总表

| 位置 | 今天的问题 | 变成什么 | 层级 | 成本 |
|---|---|---|---|---|
| 全仓 347 条 test docstring / 98 文件 | 剥掉前导动词与停用词后，剩余 token 是函数名 token 的子集——零信息 | 改写成不变量（默认）或删除（名字已完整且体内已有注释） | clarity-only | 0 |
| `tests/models/steps/denoise/common/test_decode_layout_parity.py:79` | docstring 写 `BCHTW`，函数名/production/文件头全说 `BCTHW`——**内容错误** | 改写：说明 `_VideoProcessor` 与 `ChunkedLatentDecoder` 两次 `permute(0,2,1,3,4)` 互相抵消，所以断言的是 round-trip 恒等 | clarity-only | 0 |
| 新增 `tests/architecture/test_docstring_truth.py` | 无守卫，`1e693da1` 那种批量复述可以随时重来 | AST 守卫（不 import 测试模块，沿用 `tests/architecture/` 已有的 AST 先例） | T1 | **+0.40s** |
| 4 处 loader placement 替身（cosmos-predict2 / sd3_5 / wan_2_1 / model_base） | 4 份近乎逐字重复的 `_FakeModule` / `_LoadedModule` | 收编进 `tests/models/steps/denoise/fixtures.py`（已有 26 个 importer），**保持 plain object、不做 `torch.device` 归一化** | T1 | 0 |
| `tests/trainers/` 5 个 toy 类 × 2 份 + `cpu_process_group` × 3 + `_free_port` × 6 | 重复手写 | 新建 `tests/trainers/_strategy_policies.py`（对齐已有的 `_state_dict_helpers.py`）；`_free_port` 收编到 6 个站点共用 | T1 | 0 |
| 4 处 `GenerationRuntimeLaunchContract` 最小样板 | 逐字重复的 4-kwarg 构造 | `worker_launch_contract()` helper；**范围收窄到 contract，不收编 `GenerationWorkerCore` 构造** | T1 | 0 |
| `tests/models/interfaces/test_replay_model_contract.py` 的 4 张手写家族表（27 个条目） | 手维护，加家族必漏 | 从 `FAMILY_REGISTRY` + `policy_semantics.generation_regime` 派生（24 家族 / 31 实例） | T1 | −0.02s |
| `tests/rollouts/collector/_collect.py` | 全 git 历史零 import 的重复实现（born dead） | 删除整个文件 | — | −0 |
| `vrl/scripts/eval/cosmos_predict25_kling_eval.py:414-416` | 3 个私有别名，其中 1 个纯为测试而活 | 删 3 个别名；2 个测试搬到共享模块自己的测试文件 | T1 | 0 |
| `tests/architecture/test_memory_policy_boundaries.py:52-60` | 断言两个**在 vrl/ 全部 1258 个 commit 里从未存在过**的字符串 | 删除（例外条款）；把它想说的那句话搬进真正有断言的测试的 docstring | — | −0.00s |

---

## 2. 提交一：复述型 docstring 清扫

### 2.1 先统一匹配器——六份调查给出六个数字，因为用了六种匹配器

六份区域调查各自独立发现同一缺陷，报出 61 / 334 / 364 / 35 / 98 / 13 六个互不相同的数字。本轨的第一件事是**只留一个匹配器**，并把它连同脚本一起写进本文，让数字可复现而不是靠断言。

判定规则：

1. 取 docstring 第一行；
2. 剥掉前导动词 `{check(s), verify/verifies, ensure(s), test(s), assert(s)}`；
3. 两侧都按非字母切词、去停用词、粗暴单数化；
4. **复述** iff `doc_tokens - name_tokens == ∅` 且 `|name_tokens - doc_tokens| ≤ 2`。

实测结果（`.venv/bin/python _scratch_docstring_audit.py count`，扫 385 个文件、单次 288–364 ms）：

```
slack=0: 343 hits
slack=1: 347 hits
slack=2: 347 hits          <- 本轨采用
slack=3: 347 hits
docstrings starting with "Checks ": 484
```

**347 条 / 98 个文件。** 与任务简报给的 346–364 / ~100 一致。

### 2.2 关键在于**保住**前缀匹配会一并摧毁的 137 条

484 条 docstring 以 `"""Checks ` 开头。其中 347 条是复述——**剩下 137 条是有信息量的**。一个「删掉所有 `Checks ` 开头的 docstring」的 sed 会把它们一起摧毁。实测样本：

```
tests/algorithms/test_diffusion_nft.py:533  Checks EDM-scale timestep grids cannot silently pass the /1000 heuristic.
tests/config/test_load_all_experiments.py:388  Checks the Kling reward recipe matches the paper RL batch geometry.
tests/config/test_load_all_experiments.py:1265  Checks a typo fails fast instead of silently running the default arm.
tests/config/test_load_all_experiments.py:1273  Checks the knob is refused where it could have no effect.
```

这四条讲的是「为什么这条必须成立」，不是函数名的回声。**词集匹配器和前缀匹配器的差别就是这 137 条。**

### 2.3 缺陷的单一来源

`git show --stat 1e693da1` → **118 files changed, 536 insertions(+), 0 deletions**，且新增行**全部**是 docstring（`git show 1e693da1 | grep '^+' | grep -v '^+++' | grep -vc '"""'` → `0`）。commit message 是 "Document test intent with simple docstrings"。整个 347 条的population 出自这一次批量改动，动机是好的、执行是机械的。这也解释了为什么修复必须是逐条人工，不能是另一次机械替换。

### 2.4 处置是三种，不是两种

**默认走「改写」。** 实测 347 条里只有极少数函数体自带注释可以承载 WHY；93% 的体内零解释性注释，直接删 docstring 会让不变量彻底无处可读。

| 处置 | 判据 | 实测规模 |
|---|---|---|
| **删** | 函数体是短 `pytest.raises` 且 `match=` 已经带上消息、函数名准确覆盖断言 | 少数 |
| **写不变量** | 名字覆盖不全断言 | **默认** |
| **提升** | 真正的解释已经以注释形式存在于函数体里，只是被 `1e693da1` 插在它上面的字符串盖住了 | 实测 **35** 条带非分隔线解释性注释；其中 **3** 条是注释**紧贴**复述句（严格相邻） |

> **对简报「13 处已确认」的更正**：我用两个口径量了。严格相邻（注释就在 docstring 正下方或 `def` 正上方）只有 **3** 条；放宽到「函数体内任意位置有非分隔线的解释性注释」是 **35** 条。13 这个数字我复现不出来，本文按实测的 3 / 35 走。三条严格相邻的是：
> - `tests/rewards/functions/test_ocr.py:69` —— 注释解释了「真引擎是 `score()` 里懒构建的，所以 gate 打在 production runtime 真正 import 的那个依赖上」
> - `tests/rollouts/orchestration/continuous/test_staleness.py:18` —— 注释解释了「Mechanism-only boundary: production continuous config requires >= 1」，即为什么测试可以用 `max_stale_policy_versions=0` 这个生产里不合法的值
> - `tests/data/test_setup.py:175` —— `# NOTE:` 说明了 `download_danbooru_images` 那条路径归 `test_danbooru.py` 管

**「名字覆盖不全」的三个确认样本：**

- `tests/rollouts/orchestration/continuous/test_queue.py:141`
  ```python
  def test_stats_shape() -> None:
      """Checks stats shape."""
      queue = ContinuousRolloutQueue(max_items=8)
      queue.put(_item(group_key=0, version=1, nbytes=4))
      stats = queue.stats()
      assert stats["ready_items"] == 1.0
      assert stats["ready_groups"] == 1.0
      assert stats["ready_bytes"] == 4.0
  ```
  名字说「shape」，断言钉的是三个 key 名**和**它们的精确值。同文件 `:151` 就是正例：`"""Queue stats count group slots; version selection belongs to the scheduler."""`

- `tests/rollouts/orchestration/continuous/test_staleness.py:18` `test_too_stale_and_future` —— 见上，属「提升」。

- `tests/models/steps/denoise/common/test_decode_layout_parity.py:78` —— 见 §2.5，属**真错**。

### 2.5 差一个 token 的「近似命中」必须单独复核——那里藏着写错的 docstring

匹配器把 `|doc_tokens - name_tokens| == 1` 的条目单列出来（实测 12 条），逐条人工看。其中一条是真错：

```python
def test_wan_decode_latents_preserves_bcthw_layout() -> None:
    """Checks Wan decode latents preserves BCHTW layout."""
```

函数名说 `bcthw`，docstring 说 `BCHTW`。谁对？三方证据一致指向 `BCTHW`：

- 文件头 docstring（`test_decode_layout_parity.py:4-5`）："then layout normalization (**permute to ``B,C,T,H,W``**, frame matching)"
- production：`vrl/models/steps/denoise/common/latent_decode.py:46` `return output.permute(0, 2, 1, 3, 4)`
- 测试里的 `_VideoProcessor.postprocess_video`（`:53-56`）也是 `video.permute(0, 2, 1, 3, 4)`

两次 permute 互相抵消，所以 `assert video.shape == latents.shape` 才成立。**这是「近似命中要人工复核」这条规程的全部理由**：完全命中的那 347 条只是没信息，差一个 token 的这一条是在说假话。改写后的 docstring 必须写明「两次 permute 相抵，因此断言的是 round-trip 恒等」，否则读者会以为 shape 相等证明了没有 layout 变换。

### 2.6 施工必须写进规程的一个坑（我自己踩到过）

`tests/architecture/test_generation_rollout_boundaries.py:164-169`：

```python
retired_regime_paths = (
    VRL_ROOT / "generation" / "bindings" / ("joint" + "_denoise"),
    VRL_ROOT / "generation" / "bindings" / ("causal" + "_token"),
    VRL_ROOT / "generation" / "composition" / ("caus" + "al"),
    VRL_ROOT / "scripts" / "generation" / ("joint" + "_denoise.py"),
)
```

字符串拼接不是手滑，是**为了让全仓 `grep joint_denoise` 返回 0 命中**，从而「这个退役名字真的没了吗」可以只靠 grep 回答。实测：

```
joint_denoise: 0
causal_token:  0
```

改写这条 docstring 时，只要在散文里把那个字面量拼出来，命中数当场 0 → 1，**你就在文档化不变量的同时破坏了它**。

**规程**：`tests/architecture/` 下的任何 docstring 改写，以及任何函数体刻意回避字面量的测试，改完必须重跑对应 grep 验证。可直接采用的成品：

```python
    """Deleted layering packages must not reappear under their old names.

    The retired regime path names are built by concatenation on purpose: it
    keeps this assertion from being the single hit a repo-wide grep for a
    retired name returns, so "is that name really gone?" stays answerable by
    grep alone. Do not inline the literals — and do not spell them out in this
    docstring either, which defeats the same trick.
    """
```

### 2.7 先修 `tests/architecture/` 自己的 11 条

`tests/architecture/test_generation_rollout_boundaries.py` 一个文件贡献 11 条复述（`:13, :105, :156, :197, :231, :246, :260, :280, :310, :334, :345`）。守卫住在这个目录，它自己先干净是最低要求。

**同文件内部就有两个可直接照抄的正例**（不需要去别的文件找模板）：

- `:123` `test_families_registry_stays_import_light` —— 写清了「为什么只走 `tree.body` 不走 `ast.walk`：否则会把故意的函数级懒加载 import 扫进来误报」
- `:173` `test_retired_routing_paths_have_no_python_source` —— 写清了「为什么断言无 `*.py` 而不是 `.exists()`：后者会被残留 `__pycache__` 误伤」

这两条正是「陈述不变量 + 说明为什么用这种断言方式」的标准形态。

### 2.8 各区分布（供分批施工）

| 区 | 复述条数 | 文件数 |
|---|---|---|
| `tests/models` | 61 | 22 |
| `tests/rollouts` | 38 | 15 |
| `tests/config` | 36 | 3 |
| `tests/rewards` | 31 | 10 |
| `tests/trainers` | 29 | 7 |
| `tests/generation` | 29 | 13 |
| `tests/data` | 28 | 5 |
| `tests/algorithms` | 24 | 5 |
| `tests/ray` | 23 | 2 |
| `tests/nn` | 16 | 6 |
| `tests/scripts` | 13 | 5 |
| `tests/architecture` | 11 | 1 |
| `tests/trajectory` | 6 | 2 |
| `tests/math` / `tests/e2e` | 各 1 | 各 1 |
| **合计** | **347** | **98** |

单文件 top 5：`tests/ray/test_resources.py`(18)、`tests/config/test_schema.py`(17)、`tests/config/test_load_all_experiments.py`(16)、`tests/algorithms/test_grpo_token.py`(12)、`tests/architecture/test_generation_rollout_boundaries.py`(11)。

> **对 RW-12 / CRD-07 两份既往调查的口径统一**：RW-12 报 `tests/rewards` 是「35 条 / 11 文件 / 共 28 个测试文件」，统一匹配器实测是 **31 条 / 10 文件**。CRD-07 报 `tests/{ray,config,architecture,data}` 是「98 条 / 6 文件」，统一匹配器总数确实是 **98**，但分布是 **11 个文件**，且 CRD-07 列进优先级的 `test_chunk_dispatch.py` 属于 `tests/generation`、不在那个区里。差异全部来自匹配器不同，不是树变了。**以本文的 347 / 98 为准。**

### 2.9 一条 RW-11：名字本身在撒谎，不只是 docstring

```python
def test_video_reward_config_accepts_ray_runtime() -> None:
    """Checks video reward config accepts Ray runtime."""
    validate_reward_config(_video_reward_config())
```
（`tests/rewards/kling_video_reward/test_function.py:163-165`）

函数体里没有任何 Ray。实测 `vrl/rewards/` 下大小写不敏感 grep `ray`，只剩四条讲历史的注释（`inference.py:373` "the former Ray reward worker"、`functions/registry.py:191` "Ray reward pool was removed"），没有任何 Ray 代码。名字是一条已删传输层的化石，今天在**主动误导**：有人 grep「Ray reward 的覆盖在哪」，会找到一个完全没有 Ray 的测试。

它守的真不变量恰恰**相反**——不是「校验了 kling kwargs」，而是「配置层**故意不**校验 per-reward kwargs，任意嵌套的 `worker_config` 必须原样穿透」。这句话在 production 里写得明明白白（`vrl/config/schema.py:55-61`）：

```python
class RewardConfig(ConfigBase):
    # reward names are user-chosen — open by design
    components: Annotated[dict[str, Any], OPEN]
    # each reward's kwargs contract is owned and validated by the reward class
    # itself at construction (vrl/rewards/), same as model families — the
    # config layer does not duplicate per-reward knowledge
    kwargs: Annotated[dict[str, Any], OPEN] = Field(default_factory=dict)
```

**替换**（成品，已实测通过）：

```python
def test_reward_schema_passes_through_unvalidated_kling_kwargs() -> None:
    """RewardConfig.kwargs is open by design (vrl/config/schema.py:55-61): a
    reward's kwargs contract belongs to the reward class at construction, not to
    the config layer. An arbitrary nested worker_config must survive validation
    byte-for-byte — tightening this schema (adding a sub-model, extra='forbid')
    would silently break every reward's kwargs passthrough at once.
    """
    cfg = _video_reward_config()
    worker_config = cfg.reward.kwargs.kling_video_reward.worker_config

    parsed = validate_reward_config(cfg)

    assert parsed.kwargs["kling_video_reward"]["worker_config"] == worker_config
```

关键改动是**加上返回值断言**。今天 `validate_reward_config(_video_reward_config())` 丢弃返回值，只证明「没抛异常」——这正是它退化成空测试的根因。实测：返回 `RewardConfig`，round-trip 断言为 `True`，耗时 **0.18 ms**。

同时把 fixture 里虚构的 `worker_config` `{"reward_model_version": "unit-test"}` 换成出厂预设真正会透传的形状（`vrl/config/presets/reward/kling_video_reward.yaml:24-31`）：`{"reward_model_name": "KlingTeam/VideoReward@main", "dtype": "bfloat16", "min_frame_pixels": 200704}`，这样「透传」测的是真的会被透传的东西。**`_video_reward_config` 实测只有这一个 caller**（`grep -n '_video_reward_config' tests/rewards/kling_video_reward/*.py` → 定义 `:70` + 调用 `:165`），改它不会波及兄弟用例。

---

## 3. 提交二：一个替身一个 owner

### 3.1 4 处 loader placement 替身 → `tests/models/steps/denoise/fixtures.py`

四份近乎逐字重复的 placement 记录器：

| 位置 | 类名 | 差异 |
|---|---|---|
| `tests/models/families/cosmos/predict2/test_model_loading.py:38` | `_FakeModule` | `to()` 返回 `self` |
| `tests/models/families/sd3_5/test_model_loading.py:12` | `_FakeModule` | `to()` 返回 `None` |
| `tests/models/families/wan_2_1/test_model_loading.py:44` | `_FakeModule` | `dtype` 为 keyword-only，`device` 有默认值 |
| `tests/models/steps/denoise/common/test_model_base.py:160` | `_LoadedModule` | 无 `self.dtype` 字段 |

共同体：

```python
class _FakeModule:
    def __init__(self) -> None:
        self.dtype: torch.dtype | None = None
        self.requires_grad_enabled: bool | None = None
        self.to_calls: list[tuple[Any, torch.dtype | None]] = []

    def requires_grad_(self, enabled: bool) -> None:
        self.requires_grad_enabled = enabled

    def to(self, device: Any, dtype: torch.dtype | None = None) -> _FakeModule:
        self.to_calls.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype
        return self
```

**两条必须写进新 owner 的设计约束（不写清楚就会被后人「优化」掉）：**

1. **故意做成 plain object，不是 `nn.Module`。** 断言钉的是 `"cuda:0"` / `torch.device("cuda:1")`：
   ```
   cosmos/predict2 :166   assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
   sd3_5           :85    assert encoder.to_calls == [("cuda:0", torch.float16)]
   wan_2_1         :349   assert transformer.to_calls == [(torch.device("cuda:1"), torch.bfloat16)]
   model_base      :211   assert pipeline.vae.to_calls == [("cuda:0", torch.float32)]
   ```
   真 `nn.Module` 的 `super().to("cuda:0")` 在没有 CUDA 的默认车道上会直接炸；改成 `nn.Module` 等于把这四个文件全部变成 CUDA-only。（本机有 1 张 CUDA 卡，所以本地跑不出这个失败——这正是必须写下来的原因。）
2. **不做 `torch.device` 归一化。** 现有断言混用裸字符串 `"cuda:0"`、`torch.device("cuda:1")` 和 `None`（wan 的 `:214-216` 断言 `(None, torch.float32)`）。归一化会要求同时改五处以上断言，把一次纯收编变成一次语义改动。新 owner 原样 append 收到的东西，**五个文件现有断言一字不改**。

**Owner 选址**：`tests/models/steps/denoise/fixtures.py`。它已经是跨家族共享 fixture 的既定住址（实测 **26 个 importer**，其中就包括本次四个站点之一的 `tests/models/steps/denoise/common/test_model_base.py:13`），文件头 docstring 已经在讲「tiny real / 不下载 / 可复现」这套规矩。新符号名 `RecordingModule`（与同文件已有的 `record_forward_calls` / `stamp_model_precision` 同一命名家族）。

**签名合并**：`def to(self, device: Any = None, dtype: torch.dtype | None = None) -> RecordingModule`。既接受 wan 的 keyword 形式，也接受另外三家的 positional 形式；返回 `self` 对返回 `None` 的两处调用点无影响（无人读返回值——已逐个 grep 确认）。

**不动**：`tests/models/steps/denoise/test_frozen_offload.py:19` 的 `_RecordingModule`（真 `nn.Module`，断言 `to_devices == ["cpu"]`，是不同概念）、`tests/rollouts/orchestration/test_driver_frozen_offload.py:21` 的 `_FakeDriverModel`（带 `move_frozen_components` 钩子探测，也是不同概念）、`tests/models/steps/denoise/common/test_lora_fp8_build.py:24` 的 `_TrackingTransformer`（记录的是事件**顺序**，不是设备）。

### 3.2 `tests/trainers/_strategy_policies.py`

**实测的重复情况，与提案有出入，按实测走：**

| 符号 | 实测重复 | 处置 |
|---|---|---|
| `_Block`（`Linear(4,4)` + relu） | `test_fsdp.py:87` / `test_ddp.py:54` **逐字相同** | 收编 |
| `_ToyTransformer`（`blocks` + `head`） | `test_fsdp.py:96` / `test_ddp.py:63` **仅差 `_no_split_modules` ClassVar** | 收编（带上 ClassVar） |
| `_FakePolicy` | `test_fsdp.py:112` / `test_ddp.py:75` **仅差 docstring** | 收编 |
| `_DualStagePolicy` | `test_fsdp.py:128` / `test_ddp.py:91` **逐字相同** | 收编 |
| `_Bundle` | `test_fsdp.py:145` / `test_ddp.py:112` **逐字相同** | 收编 |
| `_ToyTransformer`（`transformer_blocks`，`_Block` = 3×`Linear(16,16)`） | `test_fsdp_gather_distributed.py:50` | **不收编** |
| `_Block`（`Linear(4,4,dtype=bfloat16)`） | `test_fsdp_fp32_master.py:47` | **不收编** |

> **对「三份 toy policy」的更正**：实测是 **2 份逐字重复 + 1 份故意不同**。`test_fsdp_gather_distributed.py` 那份用的是 `transformer_blocks`（diffusers DiT 的真属性名，gather 路径正是按这个名字走的）和 `Linear(16,16)` 的 q/k/v 三件套，几何和命名都是为了 gather 测试专门选的。把它折进共享类会抹掉那个命名，测试就不再镜像真模型。`test_fsdp_fp32_master.py` 的 `_Block` 是 bf16，那是 fp32-master 测试的全部要点。**这两个是刻意的形状差异，不是重复。**

`_no_split_modules` ClassVar 加到共享类上对 DDP 是惰性的：实测 `grep -rn '_no_split_modules' vrl/` 只命中 `vrl/trainers/fsdp.py`（`:6, :133, :163, :165, :169, :172`），DDP 路径从不读它。

**外加提案没看到、但重复更严重的两项：**

**（a）`cpu_process_group` fixture × 3**（`test_ddp.py:117`、`test_fsdp_fp32_master.py:20`、`test_fsdp.py:150`），三份都是 `scope="module"`，三份体内都内联了同一段自由端口探测 + 5 个 `setenv` + `if not dist.is_initialized()` 守卫 + `destroy_process_group` 拆除，**仅变量名 `mp` / `monkeypatch` 不同**。

**（b）`_free_port()` × 6，不是 3**：

```
tests/scripts/test_online_metrics.py:27                          (context-manager 变体)
tests/trainers/test_wan_fsdp_distributed.py:58                   (context-manager 变体)
tests/trainers/test_fsdp_fp32_master.py:169                      (显式 close 变体)
tests/trainers/test_fsdp_gather_distributed.py:63                (显式 close 变体)
tests/trainers/online/test_skip_backward_agreement_distributed.py:56  (显式 close 变体)
tests/utils/test_model_diagnostics.py:43                         (显式 close 变体)
```

两种写法、六份拷贝，语义完全相同。三份被 `cpu_process_group` 内联的实现算第七、第八、第九处。

**机制验证（我实际跑了，因为这是本条的承重点）**：把 `@pytest.fixture(scope="module")` 定义在一个普通模块里、再 `from ... import cpu_process_group  # noqa: F401` 进 N 个测试模块，pytest 会**按请求模块各建一次**——scratch 原型里两个测试模块得到 `len(CALLS) == 2`，与今天「三个模块各自一个 group」的语义逐字相同。所以不需要新建 `tests/trainers/conftest.py`（全仓目前只有 `tests/conftest.py` 一个 conftest，不打破这个格局），走显式 import 即可，和已有的 `tests/trainers/_state_dict_helpers.py`（被 `test_fsdp.py:23` / `test_fsdp_gather_distributed.py:28` 显式 import）完全同构。

跨模块的 `dist.is_initialized()` 守卫本来就在每份拷贝里，收编后原样保留，行为不变。

**基线**：`pytest tests/trainers/test_fsdp.py tests/trainers/test_ddp.py` → **69 passed in 2.76s**。

### 3.3 `worker_launch_contract()` —— 收窄到 contract，不收编 core 构造

四处逐字相同的最小样板：

```python
GenerationRuntimeLaunchContract(
    family="sd3_5",
    model_build={},
    expected_model_identity={"schema": "test"},
    policy_version=<n>,
)
```

| 位置 |
|---|
| `tests/generation/execution/test_execute_request_pipelined.py:47` |
| `tests/generation/execution/test_chunk_memory_shadow.py:163` |
| `tests/generation/execution/test_chunk_memory_shadow.py:354` |
| `tests/generation/ray/test_runtime_lease_sleep.py:160` |

另有两处是同一样板 + 一个额外 kwarg，收编后传参即可：`test_worker_versioned_slots.py:96`（`versioned_weight_sync=`）、`test_worker_sleep.py:149`（`family=` / `sleep_offload=` / 变体 `model_build`）。

**范围收窄的理由**：不要做提案里的 `build_worker_core()`。那个 helper 是 8 个 kwarg 的一比一透传，而且这六个站点里**每个文件唯一不同的那个旋钮**（`core._uses_versioned_slots`、`core.executor` 装的是哪种 executor、`versioned_weight_sync`、`sleep_offload`）都会被搬进读者看不见的地方。contract 是纯数据、真的相同；core 的组装是每个文件的测试意图所在，必须留在原地可见。

**⚠️ 排序约束**：`tests/generation/execution/test_execute_request_pipelined.py` **当前有未提交的 in-flight 改动**（`git status` → ` M`）。那次改动正在把该文件从 `SimpleNamespace` + `GenerationWorkerCore.__new__` 改成真 `GenerationRequest` + 真构造，并且**恰好新造出**上面这份 contract 样板。本条必须**排在那次改动落地之后**，否则会与它冲突。同目录还有一个未跟踪的 scratch 探针 `tests/generation/execution/test_zzscratch_probe_real.py`（GR-06 的一次性产物），**不属本轨，不动**。

### 3.4 派生而非手维护：4 张手写家族表

`tests/models/interfaces/test_replay_model_contract.py` 里有四张手写的 `@pytest.mark.parametrize("family", [...])`：

| 行 | 条目数 | 内容 |
|---|---|---|
| `:143` | 4 | `["emu3", "glm_image", "llamagen", "nextstep_1"]` |
| `:161-183` | 17 | `["cogvideox", "cosmos-predict2", …, "wan_2_1_i2v"]` |
| `:197` | 3 | `["causvid", "janus_pro", "janus_pro_r1"]` |
| `:206` | 3 | 同上 |

讽刺的是**同一个文件的 `:224` 已经在做派生**：`@pytest.mark.parametrize("family", sorted(registered_replay_model_classes()))`，而 `tests/models/interfaces/__init__.py` 就是为此存在的 registry 派生 fixture 模块。四张手写表是同一个文件里的反例。

**派生的判别式（实测精确匹配，不是近似）**：`FAMILY_REGISTRY[family].policy_semantics.generation_regime`

- `== "full_sequence"` → 恰好 **17** 个家族 = `:161-183` 那张表，一个不多一个不少（这些有真实的 timestep 轴，所以只查 segment guard）
- `!= "full_sequence"`（`token_autoregressive` / `chunk_autoregressive`）→ 恰好 **7** 个家族 = `{emu3, glm_image, llamagen, nextstep_1} ∪ {causvid, janus_pro, janus_pro_r1}` = `:143` ∪ `:197`（这些没有 timestep 轴，所以额外查 zero-timestep guard）

注意 `causvid` 是 `DenoiseFamilyBuild` 但 `chunk_autoregressive`，`janus_pro_r1` 是 `token_autoregressive` 但 `multisegment_token` layout——**按 `TokenFamilyBuild` / `DenoiseFamilyBuild` 分会错放 3 个家族**，必须按 `generation_regime` 分。

**成品（已实测通过：31 passed in 0.15s）**：

```python
def _replay_families() -> list[str]:
    return sorted(registered_replay_model_classes())


def _no_timestep_axis_families() -> list[str]:
    """Families whose replay has no denoise-step axis, read off the registry.

    ``generation_regime`` is the source of truth, not the build class: causvid is
    a DenoiseFamilyBuild but chunk-autoregressive, so a TokenFamilyBuild split
    would misplace it (and janus_pro_r1) into the wrong guard.
    """
    return sorted(
        family
        for family in registered_replay_model_classes()
        if FAMILY_REGISTRY[family].policy_semantics.generation_regime != "full_sequence"
    )


@pytest.mark.parametrize("family", _replay_families())
def test_every_replay_family_rejects_an_unsupported_segment_selection(family: str) -> None:
    replay_cls = registered_replay_model_classes()[family]
    # ``__new__`` without ``__init__``: the guard must fire before any model state
    # is touched. Uniform across families — the AR trunks reach a bound mixin
    # method (``_resolve_image_token_replay``), so a bare ``object()`` self is not
    # enough for them the way it is for the denoise families.
    model = replay_cls.__new__(replay_cls)
    with pytest.raises(ValueError, match="supports segments"):
        replay_cls.replay_forward(
            model, object(), timestep_idx=0,
            request=ReplayRequest(segment_names=("unsupported",)),
        )


@pytest.mark.parametrize("family", _no_timestep_axis_families())
def test_replay_without_a_timestep_axis_rejects_a_nonzero_index(family: str) -> None:
    replay_cls = registered_replay_model_classes()[family]
    model = replay_cls.__new__(replay_cls)
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        replay_cls.replay_forward(model, object(), timestep_idx=1)
```

**覆盖账（诚实版）**：今天 4 张表 = 27 个参数化实例；派生版 = 24 + 7 = 31 个。但**断言覆盖完全相同**（24 条 segment + 7 条 timestep）——实例数变多只是因为 `:143` 那张表把两种检查塞进了同一个函数体。所以简报说的「派生版更宽」在**今天**不成立，实测两者相等；派生版的真价值是**不会漂移**：加第 25 个 replay 家族时，手写表静默漏掉，派生表自动收进来。

**成本**：当前等价子集 `pytest -k "unsupported or nonzero"` → 30 passed in **0.17s**；派生版 31 tests / **0.15s**。净 **−0.02s**。

---

## 4. 提交三：三处删除，各自带证据

> 与简报的「两个提交」分法有出入：删除是功能性改动，与 docstring 清扫和替身收编的 reviewer 关注点完全不同，混在一起会让「这个删除的证据是什么」淹没在 347 条 docstring 里。按 AGENTS.md 的 diff 纪律独立成一个提交。

### 4.1 `tests/architecture/test_memory_policy_boundaries.py:52-60` —— 墓碑测试

```python
def test_runtime_interface_does_not_parse_model_memory_sections() -> None:
    """ModelBuild is a data contract, not a model.memory parser."""

    text = (VRL_ROOT / "models" / "interfaces" / "runtime.py").read_text(encoding="utf-8")

    assert "model_memory_config_from_cfg" not in text
    assert "memory_policy_config_from_cfg" not in text
```

docstring 说的是一条真不变量，断言检查的是**两个虚构的名字**。三条独立证据：

1. **这两个字符串在全部 1258 个 commit 的 `vrl/` 下从未出现过。** `git log --all -S'model_memory_config_from_cfg' -- vrl/` 和 `-S'memory_policy_config_from_cfg' -- vrl/` 都是空。全仓 grep 只命中这两行断言自己。所以它守的不是一条删除记录，是两个从未存在的名字。
2. **变异测试：对它声称防御的行为完全失明。** 往 `vrl/models/interfaces/runtime.py` 注入一个真的 `_parse_memory(cfg)` 解析器，重跑 → `3 passed in 0.01s`。（已 `cp` 备份并还原，`git diff --stat vrl/models/interfaces/runtime.py` 为空。）
3. **它 docstring 里那条真不变量已由 production 的 `raise` 强制**：
   ```python
   # vrl/models/interfaces/runtime.py:206-209
   if self.model_config is not None and "memory" in self.model_config:
       raise ValueError(
           "ModelBuild.model_config must not carry model.memory; "
           "resolve it into generation_memory",
       )
   ```
   并被 `tests/models/steps/denoise/common/test_vae_decode_memory.py:97-102` 的 `pytest.raises(ValueError, match=r"must not carry model\.memory")` 直接覆盖。

**唯一的真解析点**在 `vrl/models/families/registry.py:317-337`（`model_memory.model_fields_set` → `GenerationMemoryPolicy`），而它的三条不变量已被 `tests/rollouts/runtime/test_family_registry.py` 逐条覆盖：

```python
:108  assert build.generation_memory == GenerationMemoryPolicy(
          vae_decode=VaeDecodeMemory(tiling=False, slicing=False))     # rollout 有
:123  assert replay_build.generation_memory is None                    # replay 无
:124  assert "memory" not in (replay_build.model_config or {})         # raw key 不漏给下游
```

**所以不需要新增任何 registry 级测试**——净覆盖变化为零。**动作**：删这一个函数。同文件 `:11` 与 `:29` 两条 `_forbidden_text` 源码扫描是真机械 gate（`enable_tiling(` / `.to("cpu")` 等，都是当前真会出现的字符串），**不动**。

**顺带（成本 0）**：把这条测试想表达的那句话搬进真正有断言的地方——`tests/rollouts/runtime/test_family_registry.py:49` `test_model_build_projects_typed_sections_without_losing_falsy_presence` 的名字只讲了 falsy-presence 那一半，实际它还是 `model.memory` 单点解析边界的唯一守卫。在它的 docstring 里写明：「`model.memory` 只在 `resolve_model_build` 一处被解析成 typed `GenerationMemoryPolicy`；rollout 才有、replay 必须为 `None`；raw `memory` key 不得漏给下游」。

### 4.2 `tests/rollouts/collector/_collect.py` —— born dead

全文件 17 行，定义一个 `collect_scored`。

- **全 git 历史零 import**：`git log --all -S'collector._collect'` 和 `-S'_collect import'` 都是空。**它从来没有被任何 commit 引用过**——是 born dead，不是 went dead。
- **它是怎么进来的**：`git log --diff-filter=A` → 唯一添加者 `f2600071`，commit message 自述 "**Working-tree snapshot** of the in-flight precision/rollout work"，一次 174 文件 / 8312 行的快照提交。它是被卷进来的，不是被写进来的。
- **同目录有活的同名实现**：`tests/rollouts/collector/_helpers.py:12` 的 `collect_scored`——带完整类型标注（`RolloutCollector` / `Mapping` / 具名 keyword）、带解释「为什么 production 不提供合并入口」的 docstring，被 `tests/rollouts/collector/test_runtime.py:12` import 并在 7 处调用。死的那份是它的弱化无类型副本。

**动作**：`rm tests/rollouts/collector/_collect.py`。

### 4.3 `vrl/scripts/eval/cosmos_predict25_kling_eval.py:411-416` —— 为测试而活的别名

```python
# Generation helpers live in the shared denoise_video_generation module (wan reuses
# generate_one_video too). Keep the private names as aliases so the call sites above
# and the pinned test refs (eval_script._seed_for / _video_to_cthw) keep resolving.
_seed_for = seed_for
_generate_one_video = generate_one_video
_video_to_cthw = video_to_cthw
```

**是三个别名，不是两个**（简报说两个）。逐个 grep 后的读者账：

| 别名 | production reader | test reader | 判定 |
|---|---|---|---|
| `_seed_for` | `:377` | `test_cosmos_predict25_kling_eval.py:49` | 无意义私有重命名 |
| `_generate_one_video` | `:390` | — | 无意义私有重命名 |
| `_video_to_cthw` | **无** | `test_cosmos_predict25_kling_eval.py:70` | **纯 test-only** |

注释自己写明了动机：「so … the pinned test refs … keep resolving」——为了让测试的 import 继续解析而保留。这正是 AGENTS.md 死代码五形态里的第一形（test-only reader 即死）。

**同时暴露的真问题**：这两个测试测的根本不是 cosmos 脚本，是共享模块 `vrl/scripts/eval/denoise_video_generation.py`（`seed_for` / `generate_one_video` / `video_to_cthw`，`__all__` 三个都在），而**那个共享模块至今没有自己的测试文件**（`tests/scripts/eval/` 下只有 sana ×3 和 wan ×1）。测试通过一个家族脚本的私有别名去够共享实现，是所有权错位。

**动作**：
1. 删 `:414-416` 三个别名和 `:411-413` 那段解释别名存在理由的注释；`:377` / `:390` 两个调用点直接调 `seed_for` / `generate_one_video`（`vrl/scripts/eval/denoise_video_generation.py:28` 已经把它们 import 进来了）。
2. 把 `test_seed_grid_is_identical_across_checkpoints`（`:41-63`，docstring 已经很好，原样搬）和 `test_video_to_cthw_accepts_btchw_layout`（`:65-73`，docstring 是复述型，按 §2 改写）搬进新建的 `tests/scripts/eval/test_denoise_video_generation.py`，直接 `from vrl.scripts.eval.denoise_video_generation import seed_for, video_to_cthw`。**覆盖不减，并且共享模块第一次有了自己的测试。**
3. **同名兄弟陷阱**：`vrl/scripts/eval/wan_robotics_checkpoint_eval.py:597` 有一个**签名完全不同**的本地 `_seed_for(*, base_seed, row_index, sample_index, seed_stride)`，被 `:231` 和 `tests/scripts/eval/test_wan_robotics_checkpoint_eval.py:101,107` 使用。**不要连坐**。共享模块的 `seed_for` 签名是 `(*, base_seed, prompt_index, sample_index, samples_per_prompt)`。

---

## 5. 提交四（可选，纯 gate）：AST 守卫防复发

新增 `tests/architecture/test_docstring_truth.py`：遍历 `tests/**/*.py` 的 AST，对每个 `test_*` 函数应用 §2.1 的匹配器，命中即失败并列出全部 offender。

- **为什么用 AST 不用 import**：import 全部 302 个 test 模块又慢又有副作用；`tests/architecture/` 已经立了 AST 的先例（`test_generation_rollout_boundaries.py:123` 的 `tree.body` 扫描）。
- **实测成本**：**0.40s call**（385 个文件）。占 188.88s 全套的 **0.21%**、占 104.41s fast lane 的 0.38%。
- **实测行为**：在今天的树上运行，正确失败并列出 347 条。§2 全部落地后转绿。
- **误报边界（必须写进它自己的 docstring）**：这个启发式依赖「前导动词 + 与函数名重词」两个特征。改写后的 docstring 合理复用函数名里的名词是安全的（`extra_doc` 非空即放行），但**如果改写后的第一行仍然只由函数名里的词构成**，它会命中——这是设计意图，不是误报。
- **不要为此开 ruff 的 D 系列**：`pyproject.toml` 当前 `select` 是 E/W/F/I/UP/B/SIM/TCH/RUF，没有 D，而且 ruff 也没有「docstring 复述函数名」这条规则。守卫必须是本仓自己的 AST 测试。

---

## 6. NON-GOALS（本区刻意不动的替身与不做的事）

**A. 刻意的形状差异，不是重复——不收编：**

- `tests/trainers/test_fsdp_gather_distributed.py:50` 的 `_ToyTransformer`：用 `transformer_blocks`（diffusers DiT 的真属性名）+ `Linear(16,16)` 的 q/k/v 三件套。gather 路径正是按这个名字走的；折进共享类会抹掉测试镜像真模型的那一点。
- `tests/trainers/test_fsdp_fp32_master.py:47` 的 `_Block`：`Linear(4, 4, dtype=torch.bfloat16)`。bf16 就是 fp32-master 测试的全部要点。
- `tests/models/steps/denoise/test_frozen_offload.py:19` `_RecordingModule`（真 `nn.Module`，记 `to_devices`）、`tests/rollouts/orchestration/test_driver_frozen_offload.py:21` `_FakeDriverModel`（探测 `move_frozen_components` 钩子）、`tests/models/steps/denoise/common/test_lora_fp8_build.py:24` `_TrackingTransformer`（记事件顺序）：三个不同概念，名字像而已。

**B. 环境模拟（T3-ENV），本轨不碰、也不给它们贴标签：**

> **现状修正（2026-07-30）：** 轨道一已经注册 `real_cover` 并落地 AST 守卫。
> 以下段落保留的是本计划写作时的基线；实际施工若触及这些替身，应使用当前 marker
> 契约，而不是继续依赖散文缺口。

本轨是 0 运行时的清扫轨，**不引入 `real_cover` marker**。实测 `pyproject.toml:203` 有 `addopts = ["--strict-config", "--strict-markers"]`，而 `:204-211` 的 markers 列表里没有 `real_cover`——现在贴等于**整个文件收集期硬报错**，不是良性注解。marker 注册与「诚实缺口登记册」属于 foundation/infra 轨的范围，本轨不依赖它、也不制造对它的依赖。

具体地，以下几类在本轨**明确保持原样**，理由是它们模拟的是无法按需制造的条件，而不是可以构造的对象：

- `tests/trainers/test_fsdp.py:861` / `test_ddp.py:203`：物理 rank 3 of 4 + nccl，需要 4 个进程和 CUDA。
- `tests/trainers/test_distributed_training.py:44`：本机**有** 1 张 CUDA 卡，不打这个 patch 时 `resolve_training_context` 会走 `visible_device_count == 1` 分支返回 `cuda:0`，测试断言的 `cuda:1` 会失败。你没法「un-have」这张卡。
- `tests/math/test_denoise_flow_matching.py:21` / `test_token_flow_matching.py:20`：这两个不是环境阻塞，是**解析式 oracle 替身**——手配的 EDM/flow sigma 表本身就是被测的同构关系，`0.1*x` 的头是为了闭式解可手算。给它们贴「被阻塞」标签是假文档。

**C. 不做的事：**

- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function。
- 不为整洁而改名或搬文件。§4.3 搬两个测试是**所有权修正**（测试搬到它真正测的模块旁边，并让那个模块第一次有测试），不是整洁。
- **不加 ruff 的 docstring 规则**，理由见 §5。§2.1 的脚本是一次性分析工具（按仓库规范命名 `_scratch_docstring_audit.py`，扫完复核一次即弃），只有 §5 那个 AST 测试是长期资产。
- **不修 `tests/conftest.py:64-69`**。那段注释写着 `distributed`/`optional` "no current members"，但实测 `tests/trainers/test_wan_fsdp_distributed.py:576,612` 已经带 `@pytest.mark.distributed`（2 个装饰点）。这是一条**真的陈旧注释**、也确实属于「让注释陈述事实」的主题，但它已经被分配给 infra/tier 轨（marker 与 lane 的 owner）。本轨只记录，不动手，避免两轨改同一段。
- **不动 `tests/generation/execution/test_zzscratch_probe_real.py`**（未跟踪的 GR-06 一次性探针，属另一轨的生命周期）。

---

## 7. HONEST GAPS（本轨明确标注为「进程内没覆盖」的东西）

本轨**不新增**任何 T3 缺口，因为它不转换任何替身。需要如实记下的只有一条**已存在的缺口**，是 §4.3 顺带暴露的：

- **`vrl/scripts/eval/denoise_video_generation.py` 的 `generate_one_video` 在进程内没有真覆盖，且本轨不解决它。** 该模块的三个导出里，`seed_for`（纯算术）和 `video_to_cthw`（纯 tensor 变换）在 §4.3 之后会有直接的 T1 测试；`generate_one_video` 需要一个真 diffusion pipeline 才能跑，进程内做不到。**目前没有任何 e2e case 覆盖它**——`tests/e2e/test_real_checkpoint_rl.py` 的 `CASES` 是 online-RL 路径，不走 eval 脚本。这是一个真缺口，本轨把它写下来而不是假装它被覆盖了；补齐属于 eval-lane 的工作。

另外两条属于「本轨修正了既往文档里的假记录」：

- `docs/sprints/done/SPRINT_test_suite_tiny_real_and_fake_audit.md` 的 **`:274`、`:282`、`:704`** 三处仍在引用 `tests/rewards/ray/test_runtime.py`——**该文件与其目录内容已不存在**。（注意：同文档 `:368`、`:548`、`:675` 指的是另外的 `test_runtime_model_contract.py` / `generation/ray/test_runtime_config.py`，**不要误改**。）随 §2.9 一并把这三处标注为已失效。
- `tests/rewards/ray/` 目录本身还在磁盘上，但**只剩 `__pycache__`**（`test_runtime` / `test_resource_lifecycle` 的陈旧 `.pyc`），且 `git ls-files tests/rewards/ray` 返回 **0**——是未跟踪的孤儿。与 Ray reward pool 移除同源同生命周期，随 §2.9 一并 `rm -rf`。

---

## 8. 验证与排序

### 8.1 排序（硬约束）

1. **本轨整体排在轨道一~五之后。** 前面四个转换轨会给它们改到的测试写新 docstring；扫描先跑会被改两遍甚至冲掉。
2. **§3.3（`worker_launch_contract`）排在 `tests/generation/execution/test_execute_request_pipelined.py` 那笔 in-flight 改动落地之后。** 该文件当前 ` M`，那笔改动正在新造出本条要收编的样板。
3. 提交内部顺序：**提交一（docstring）→ 提交二（替身 owner + 派生）→ 提交三（删除）→ 提交四（AST 守卫）**。守卫最后落，因为它在 §2 完成前是红的。

### 8.2 施工约束

- **绝不对以下文件跑 `ruff format`**（实测在**纯净树上就已经**不通过 `ruff format --check`，让格式化器碰它们会把既有风格债折进一个 docstring-only / 收编-only 的 diff，违反 AGENTS.md 的 diff 纪律）：
  ```
  tests/algorithms/test_dpo.py                       tests/rewards/phymotion/test_phymotion.py
  tests/algorithms/test_global_std_distributed.py    tests/scripts/perf/test_reward_overlap_benchmark.py
  tests/nn/layers/test_attention_cache_rows.py       tests/scripts/test_wan_train.py
  tests/rewards/kling_video_reward/test_model_loading.py   tests/test_logging.py
  tests/trainers/test_ddp.py                         tests/utils/test_profiling.py
  ```
  其中 `test_model_loading.py`（§2 的 5 条目标）和 `test_ddp.py`（§3.2 的收编目标）**都在这个名单里**。这两个文件只做手工最小编辑，`ruff check` 可以跑（实测 `All checks passed!`），`ruff format` 不跑。既有风格债留给单独的格式化任务。
- 其余触及文件按常规：先 `ruff check --fix <files>`，再 `ruff format <files>`，然后 `ruff check <files>` + `ruff format --check <files>` 验证。
- **§2.6 的 grep 规程**：`tests/architecture/` 下每改一条 docstring，改完必须跑
  ```bash
  grep -rIn 'joint_denoise\|causal_token' --include='*.py' --include='*.yaml' --include='*.md' . | grep -v '\.git/' | wc -l   # 必须仍为 0
  ```

### 8.3 验证命令

```bash
# 提交一
.venv/bin/python /path/to/_scratch_docstring_audit.py count      # 期望 slack=2 -> 0 hits
.venv/bin/python -m pytest tests -q -p no:randomly                # 3816 passed（docstring 不改行为）
grep -rIn 'joint_denoise\|causal_token' --include='*.py' . | grep -v '\.git/' | wc -l   # 0

# 提交二
.venv/bin/python -m pytest tests/models/steps/denoise tests/models/families/sd3_5 \
    tests/models/families/cosmos/predict2 tests/models/families/wan_2_1 -q -p no:randomly
.venv/bin/python -m pytest tests/trainers -q -p no:randomly       # 基线 483 passed
.venv/bin/python -m pytest tests/generation -q -p no:randomly
.venv/bin/python -m pytest tests/models/interfaces/test_replay_model_contract.py -q -p no:randomly
#   派生前 62 passed / 0.17s -> 派生后 66 passed / 0.15s（27 -> 31 个参数化实例，断言覆盖不变）

# 提交三
grep -rn 'collect_scored\|_video_to_cthw\|_generate_one_video' vrl/ tests/ --include='*.py'
.venv/bin/python -m pytest tests/architecture tests/rollouts/collector tests/scripts/eval -q -p no:randomly

# 提交四
.venv/bin/python -m pytest tests/architecture/test_docstring_truth.py -q -p no:randomly --durations=3
#   期望 1 passed，call ≈ 0.40s

# 全套回归 + 时间账
.venv/bin/python -m pytest tests -q -p no:randomly                # 期望 188.88s -> ~189.3s
.venv/bin/python -m pytest -m "not e2e and not slow_test" -q      # 期望 104.41s -> ~104.8s
```

### 8.4 完成判据

- §2.1 的匹配器在全仓返回 **0** 命中；`tests/architecture/test_docstring_truth.py` 绿。
- 137 条有信息量的 `"""Checks …"""` docstring **一条都没被删**（改动前后各跑一次 §2.2 的统计，数字必须仍是 137 或更高）。
- `tests/models/steps/denoise/common/test_decode_layout_parity.py` 里不再出现 `BCHTW`。
- `_FakeModule` / `_LoadedModule` / `_FakePolicy` / `_DualStagePolicy` / `_Bundle` / `_free_port` / `cpu_process_group` 在 `tests/` 下各只剩**一个**定义（`grep -c 'class _FakeModule'` 等）；§6.A 列的三个刻意差异版**仍在原地**。
- `tests/models/interfaces/test_replay_model_contract.py` 里不再有手写家族名列表；新加一个 replay 家族时派生表自动收进来（可用一次 scratch 注册验证）。
- 全套 `pytest tests` 测试数：3816 → **3819**（−1 墓碑，+4 派生实例，+1 守卫，−1 kling 测试改名不影响计数；净 +3），零 fail、零新增 skip。

---

## References

- 统一匹配器脚本（一次性产物）：`_scratch_docstring_audit.py`（§2.1 全文规则，实测 288–364 ms / 385 文件）
- 缺陷来源：commit `1e693da1` "Document test intent with simple docstrings"（118 files, 536 insertions, 0 deletions, 全部为 docstring）
- docstring 正例模板：`tests/architecture/test_generation_rollout_boundaries.py:123`、`:173`；`tests/rollouts/orchestration/continuous/test_queue.py:151`；`tests/rewards/functions/test_multi.py:264`
- 拼接不变量：`tests/architecture/test_generation_rollout_boundaries.py:164-169`
- BCTHW 事实来源：`vrl/models/steps/denoise/common/latent_decode.py:46`；`tests/models/steps/denoise/common/test_decode_layout_parity.py:4-5,53-56,78-95`
- placement 替身：`tests/models/families/cosmos/predict2/test_model_loading.py:38,166-168`、`tests/models/families/sd3_5/test_model_loading.py:12,85-87`、`tests/models/families/wan_2_1/test_model_loading.py:44,214-216,349,624-626`、`tests/models/steps/denoise/common/test_model_base.py:160,208-232`；新 owner `tests/models/steps/denoise/fixtures.py`
- trainers 替身：`tests/trainers/test_fsdp.py:87,96,112,128,145,150`、`tests/trainers/test_ddp.py:54,63,75,91,112,117`、`tests/trainers/test_fsdp_fp32_master.py:20,47,169`、`tests/trainers/test_fsdp_gather_distributed.py:39,50,63`；既有同构范本 `tests/trainers/_state_dict_helpers.py`
- `_free_port` 六处：`tests/scripts/test_online_metrics.py:27`、`tests/trainers/test_wan_fsdp_distributed.py:58`、`tests/trainers/test_fsdp_fp32_master.py:169`、`tests/trainers/test_fsdp_gather_distributed.py:63`、`tests/trainers/online/test_skip_backward_agreement_distributed.py:56`、`tests/utils/test_model_diagnostics.py:43`
- launch contract 四处：`tests/generation/execution/test_execute_request_pipelined.py:47`（**in-flight**）、`tests/generation/execution/test_chunk_memory_shadow.py:163,354`、`tests/generation/ray/test_runtime_lease_sleep.py:160`；另两处 `tests/generation/execution/test_worker_versioned_slots.py:96`、`test_worker_sleep.py:149`
- 派生源：`vrl/models/families/registry.py`（`FAMILY_REGISTRY` / `policy_semantics.generation_regime`）、`tests/models/interfaces/__init__.py`（既有派生 fixture）、`tests/models/interfaces/test_replay_model_contract.py:143,161-183,197,206,224`
- 删除项证据：`tests/architecture/test_memory_policy_boundaries.py:52-60`（+ `vrl/models/interfaces/runtime.py:206-209`、`vrl/models/families/registry.py:317-337`、`tests/rollouts/runtime/test_family_registry.py:49,108,123,124`、`tests/models/steps/denoise/common/test_vae_decode_memory.py:97-102`）；`tests/rollouts/collector/_collect.py`（+ 活的 `tests/rollouts/collector/_helpers.py:12`、commit `f2600071`）；`vrl/scripts/eval/cosmos_predict25_kling_eval.py:377,390,411-416`（+ `vrl/scripts/eval/denoise_video_generation.py:13,25,81,97`、`tests/scripts/test_cosmos_predict25_kling_eval.py:41-73`、同名兄弟 `vrl/scripts/eval/wan_robotics_checkpoint_eval.py:597`）
- RW-11 依据：`vrl/config/schema.py:55-61`、`vrl/config/presets/reward/kling_video_reward.yaml:24-31`、`tests/rewards/kling_video_reward/test_function.py:70,163-165`
- 陈旧引用：`docs/sprints/done/SPRINT_test_suite_tiny_real_and_fake_audit.md:274,282,704`；`tests/conftest.py:64-69`（归 infra 轨）
- 配置：`pyproject.toml:203-211`（`--strict-markers` + 已注册 marker 清单，`real_cover` 不在其中）
- 当前 marker 契约：`docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`

## 落地记录（2026-09-07）

五个 commit，顺序与 §8.1 一致：

| 提交 | commit | 内容 |
|---|---|---|
| 一 | `ef9874daa` | 复述型 docstring 清扫。用 §2.1 的匹配器在当日树上重新量：**262 条 / 86 文件**（不是 347——前面的 test-slop 批次已经删掉一批带复述 docstring 的测试）。逐条读函数体后改写为不变量，或在函数体已有解释性注释 / 短 `pytest.raises(match=)` 时删除。§2.5 的两条近似命中一并修：`BCHTW`→ 说明两次 permute 相抵、"batch batches"。§2.9 的 kling 测试改名为 `test_reward_schema_passes_through_unvalidated_kling_kwargs`，fixture 换成 preset 真透传的形状并加返回值断言。118 条有信息量的 `Checks` docstring 一条未动；`joint_denoise` / `causal_token` 全仓 grep 在 `vrl/` + `tests/` 仍为 0 |
| 二 | `747aff824` | 一个替身一个 owner：`RecordingModule`（§3.1，plain object、不归一化 device，四处断言原样）；`tests/trainers/_strategy_policies.py`（§3.2：`ToyBlock`/`ToyTransformer`/`FakePolicy`/`DualStagePolicy`/`Bundle`/`free_port`，五处 `_free_port` 收编，gather_distributed 与 fp32_master 的刻意变体保留）；`tests/generation/execution/_helpers.py::launch_contract`（§3.3，五处纯数据字面量） |
| 三 | `aa5704a98` | 删 `tests/rollouts/collector/_collect.py`（§4.2） |
| 四 | `21a58d4d9` | `tests/architecture/test_docstring_truth.py`（§5，AST 扫描 0.40 s，两条边界用例钉住启发式） |
| 附 | `0accf019c` | bloat audit §8.2 的 models 项（sana `from_build`→super、wan I2V replay MRO、cosmos3 `set_num_steps`）——它的前置"tiny-model 回归"由本轨的 RecordingModule 与前一天的 cosmos3 tiny 真对象补齐 |

### 与计划的偏差

- **§3.2 的 fixture 放进了 `tests/trainers/conftest.py`**，不是按 §3.2 说的显式 import。显式 import 一个 fixture 名会让 ruff 在每个请求它的测试处报 F811（33 处），`# noqa` 只能压一半；conftest 是 pytest 的正解，且 module scope 语义不变（每个模块各建一次 group）。
- **§3.4 不需要做**：`test_replay_model_contract.py` 的四张手写家族表已不存在，文件现在只有 `registered_replay_model_classes()` 派生的一张。
- **§4.1 / §4.3 不需要做**：墓碑测试与 `cosmos_predict25_kling_eval.py` 的三个别名都已被更早的批次删除。
- 三个既有 docstring 改写又被 §3 收编触到的文件（sd3_5 model_loading、test_model_base、test_runtime_inputs）放在提交二里，没有拆 hunk。

### 验收

- 匹配器：262 → **0** 命中；`test_docstring_truth.py` 绿。
- 提交一后全仓（.venv）：3463 passed / 65 skipped，与改前相同（docstring 不改行为）。
- 提交二~附后全仓（.venv）：3469 passed / 65 skipped（+2 sana 加载测试、+1 cosmos3 `set_num_steps`、+1 wan I2V MRO 钉、+2 守卫）。
