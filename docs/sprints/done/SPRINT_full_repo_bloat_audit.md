# SPRINT：Full-repo bloat audit（全库肿胀与死代码审计）

状态：**done（审计 2026-08-22 完成；执行批次 0/1/2/3/4a 同日落地，2026-09-05 结清六项余项；
2026-09-06 归档）**。§8.2 表里剩下的八项是**逐条写明缓做原因的搁置项**（各自挂着一个前置：
包级真 Ray 集群、tiny-model 回归、config key 迁移裁决、KEEP 脚本逐个确认），不是待办清单，
再次扫描 planned 时不应重新评估它们；要重启时从 §8.2 的原因反查前置是否已满足。
审计本身：7 条并行审计道覆盖全部 vrl/ + tests/ + configs/，每条 finding 附 path:line 与
grep 证据，函数体已读后才下判定；6 条最高风险判定由主线复核确认。

审计方法：AGENTS.md 死代码五形式 + 参数级审计（每个参数问"有无非默认调用者/函数体是否
消费"）+ placement 四规则 + 测试审计（mock 自证 / 重复定理 / 无牙墓碑）。

## 0. 结论先行

全库 ~19 万行（生产 9.6 万 + 测试 9.4 万），**没有结构性腐烂**：注册表可达性完好
（25 个 model family 全部有 preset 可达，无整族死码）、schema known-key 全部派生、
display/provenance-only 标注纪律在多数包执行良好。肿胀是真实的但集中：

- **3 个已确认 bug + 2 个行为缺口**（§1，立即修）。
- **五种跨仓肿胀模式**（§2）解释了 90% 的 finding——逐条修 finding 不如按模式修根因。
- 可安全删除/合并约 **2.5–3k 行**（生产 ~1.6k + 测试 ~1k，约 1.5%），另有 ~1k 行
  是**搬家不是删除**（utils 错置）。行数不是重点；§1 的 bug 和 §2-C 的"会撒谎的
  防护栏"比行数重要。
- 各包健康度：config/utils ≈ models > math/nn > rollouts > generation（生产干净、
  测试双路径拖累）> rewards > trainers（横向重复最重）> scripts（死旋钮最多）。

## 1. P0 — 已确认 bug 与行为缺口（复核过函数体，立即修）

| # | 位置 | 问题 | 修复 | 
|---|---|---|---|
| B1 | `vrl/generation/ray/launcher.py:290` | 清理路径调用 `session.kill_workers()`，真实 `RayGenerationSession` 只有 `kill_engines()`（session.py:182）。测试绿是因为 fake session 自己定义了假方法——mock 自证。生产 resident 启动失败时会 `AttributeError` 吞掉根因 | 改调 `kill_engines()`；`_FactorySession` 同步；测试替身换成有真实方法面的 |
| B2 | `vrl/trainers/online/trainer.py:1451` | 流式路径"所有 microbatch 被过滤"分支置 `initial_replay = None`，下游 `metrics_io.py` 直接 `.clip_fraction` → `AttributeError`。全批路径孪生代码（:1782）写的是正确的 `InitialReplayStats()`——孪生漂移的直接恶果 | `initial_replay = InitialReplayStats()`；补该分支测试 |
| B3 | `vrl/scripts/data/pickapic.py:35` | bootstrap 生成给用户照抄的前门命令硬编码已 404 的 `yuvalkirstain/pickapic_v2`；`presets/dataset/pickapic_v1.yaml:3-10` 早已记录下架并指向镜像 `pickapic-anonymous/pickapic_v1`。dataset identity 的真源在 config，脚本留了旧值副本 | 从 cfg 下传 `data.dataset_name`（bootstrap 已持有 cfg）；至少改默认为镜像 |
| B4 | `vrl/models/families/cosmos/{predict2,predict2_5}/model.py` | 两族 `from_pretrained` 用 `**build.revision_kwargs` 而非 `build.pretrained_kwargs`，**静默忽略 `model.local_files_only`**（该 key 在 wan 实验 preset 中是活的） | 换 `pretrained_kwargs` |
| B5 | `vrl/rollouts/orchestration/continuous/types.py:59` | `estimate_batch_bytes` 的 `tensor_extras` 过滤只收顶层 Tensor，而 extras 唯一真实内容是 `reward_components`（dict of Tensor，core.py:277）→ 队列字节预算系统性低估。`trajectory_tensor_bytes` 本身就递归 Mapping，过滤反而是破坏 | 删过滤，直接传 `batch.extras` |
| A1 | `vrl/models/families/wan_2_1/model.py:272,296` vs `vrl/models/peft_adapter.py:134` | **待裁决**（非直接修）：wan 手抄的 `apply_lora` 单边传 `autocast_adapter_dtype=False`，共享路径默认 `True`。注释称这是 FSDP/rollout 字节兼容硬需求——若成立则对所有 LoRA family 成立（共享侧是 bug）；若不成立则 wan 注释在撒谎。二者必有一错，需要一次 weight-sync 数值裁决 | 跑一次双侧对比后统一，赢家进共享 mixin |

## 2. 五种跨仓肿胀模式（根因层——修模式，不逐条摸鱼）

### 2-A mock 伺服双路径（重灾区：generation）

生产代码为测试替身留的分支：`executor.py` 4 处 + `weight_sync.py` 1 处
`if hasattr(x, "remote") else 本地直调`、`worker.py:550` `completion_callback=None`、
`worker.py:316` `margin`/`knee_threshold` 纯测试旋钮、`rollouts/collector/config.py:144`
三分支只有第一支有生产 producer、`prompt_collection.py:47` `stats=None`（两个生产调用者
都传值）、`batch_builder.py:37` `device=None`（唯一生产构造点恒传 "cpu"）。
**B1 就是这个模式的必然恶果**：替身有真对象没有的方法面。
修法统一：分支删掉、参数改必填、替身补真实方法面（`executor.py:686`
`_remote_engine_methods` 的 raise 写法已是范式；`test_weight_sync.py:70` 的
`_RemoteMethod` 已是现成的合格替身）。

### 2-B 孪生漂移（同一逻辑 ≥2 份拷贝，已经或正在分叉）

| 位置 | 拷贝数 | 已发生的漂移 |
|---|---|---|
| `trainers/online/trainer.py` replay/backward 循环（:1371 vs :1792）+ 收尾（:1410 vs :1837） | 2 | **B2 崩溃** |
| reward 模型 tensor→PIL（pickscore/animereward/aesthetic/hpsv3/nsfw 各一份，`decode_artifact_frames` 已存在只有 2 个消费者） | 5 | pickscore 取中间帧 vs animereward 取 batch 0 全帧——**打分对象已不同**，合并前需 reward 数值回归 |
| `cosmos3_reasoner.py:164` vs `videoscore2.py:129` Qwen-VL judge 前奏 | 2 | `.to(self.model.device)` vs `.to(self.device)`（device_map 下前者才对） |
| trainers 三份 all-reduce 统计体（advantages.py:30 / trainer.py:82 / grpo/continuous.py:362）+ 三份 dtype-only 集合原语（trainer.py:507-543） | 6 | 尚未分叉 |
| wan `apply_lora`（:241）/ `WanI2VReplayModel`（:1194）手抄共享 mixin/兄弟类 | 2 | **A1 分歧** |
| cosmos 安全检查器 stub ×2、`set_num_steps` 恒等 override ×3、p2.5 `finalize_noise_pred` 与基类逐字节同 | 6 | 未分叉 |
| sana `from_build` 手抄共享 loader 序列（flux 已是 `super().from_build` 正确形态） | 1 | 未分叉 |
| collector/core 与 prompt_collection 双秒表（一套 `VRL_PROFILE` 门控）；三个 `write_jsonl` 语义相反（append vs overwrite） | — | 语义已相反 |
| 三份 lazy-export facade 机器（trainers/data、trainers/online、trajectory）；`_PUBLIC_EXPORTS` 值重复自己的键 | 3 | — |

### 2-C 会撒谎的防护栏（比没有更糟：读者以为有保护）

- `kling_video_reward.py:47` "生产锁定键集"锁的 6 个 key 全仓无人读（只 `model_factory` 活）。
- `nn/optimization/passes.py` 四个 `conflicts()` 恒返回空元组 + 不可达校验循环（:390）；
  `OptimizationReport`/`introduces_replay_drift` 整条链零生产消费者（活的 drift 门禁走
  `REQUEST_SCOPED_DRIFT_SOURCES`，与此链无关）。
- `rollouts/collector/config.py:7` docstring 说"never from a hand-maintained list"，55 行后
  就是手写 sde key 映射（`SdeConfig` 加字段 → 用户设置静默失效，最坏一类）。
- `_OFFLINE_DPO_*_FIELDS`（schema.py:620）与真实读取集合无机械联系；
  `model.lora.init`/`init_lora_weights` 双别名 knob（preset 全用前者，denoise 读后者）。
- 无牙墓碑测试：`test_removed_inline_eval.py` 等 2 文件断言"派生机制会拒绝不在 schema 的
  key"（`test_unknown_keys.py` 已系统性证明）；`test_memory_policy_boundaries.py:52` 断言
  两个已不存在符号的字符串缺席。

### 2-D 死旋钮（零 producer 的可配置面）

- **CLI**：约 28 个 flag 全语料（.md/.yaml/.sh/tests）零出现——scripts lane 清单见其报告
  §14；最有害的是 `--use-config-lora`（开启后刻意保留一个日志自称要禁用的坏状态）与
  encode/merge 分片链（`merge_target_latents.py` 整文件零调用者）。
- **config/worker_config**：`flow_kl_use_dt`（两个 preset 都设 false=默认）、offline DPO
  `v_prediction` 分支（零 producer）、`use_adafactor`（仅测试）、reward 侧 9 个
  `.get(key, default)` 无任何 yaml 设置的 key（其中 `allow_absolute_paths` 是卡在
  permissive 位的安全旋钮，须显式裁决）。
- **参数**：`copy_ema_to(store_temp=)`、`inspect_cluster(driver_node_ip=)`、
  `actor_scheduling_strategy(capture_child_tasks=)`、`timeout_s`（reward client）、
  `_validate_rank_gpu_ids(expected_gpu_ids=None)` 等——全部"唯一非默认调用者是测试或不存在"。

### 2-E 幽灵与错置（删除量最大、风险最低）

- `vrl/families/`：纯 `__pycache__` 幽灵目录（git 追踪数 0，源码 097c60df 已迁走）。`rm -rf`。
- 零引用资产：`kling_video_reward_http.yaml` 等 5 个 preset、`kling_overlap_gate.yaml`
  （其启动说明指向不存在的文件）、`wan_i2v_logprob_parity_probe.py`（答案已逐字记录于
  parked sprint）、`wan_phys_ab_sample.py`、`anime_anatomy_report` 集群（608 行，依赖
  `rtmlib` 从未进 pyproject——装不上的僵尸，删或补依赖二选一，不许保持第三态）、
  fp8 两个 1 行兼容 facade（所有文档都已用新名——历史 KEEP 判决的前提未经 grep 验证）。
- 零导入 facade：`math/denoise`、`math/token`、`nn/modules`、`nn/layers/attention`、
  `steps/denoise/common` 5 个 re-export、`trainers/core/__init__`（唯一引用是一个负向测试）、
  danbooru `__init__` 33 个 re-export 中 29 个零外部消费者。
- utils 错置（搬家不删）：`nsys_report.py`（756 行）单消费者 → `scripts/perf/`；
  `model_diagnostics.py`（227 行）单消费者 → `trainers/`；`precision.py:_select` 重写了
  `cfg_path`；`ema.py:to()` 零调用者（且调了会破坏 parking 一致性）。
- reward inference 映射 4 个构造点（builders/reward_inference/registry-fallback/schema 重复
  parse）——数据版 form-4，收敛到 `RewardRuntimeConfig.from_cfg` 单点。

## 3. 测试侧清单（~1k 行可回收）

1. 逐字复制的测试文件/助手：hpsv3 vs videoscore2 `test_function.py` 四份同构（参 registry
   parametrize 合并）；`_FakeRuntime` ×6；`_engine`/`_ResolvedRef`/`_parking_snapshot` 各 2-3
   份（conftest 已存在可承接）；fp4/fp8 对基类定理的双份断言（真正的 per-scheme 数值保留）。
2. mock 自证：fp8 facade 测试（monkeypatch 掉唯一一行再断言它被调用）；
   `test_export_rollout_state_matches_helper`（断言实现等于实现）；三份
   `test_*_exposes_trainer_replay_methods`（注册表遍历版严格更强）。
3. 重复定理：`test_resources.py` 三对同配置同断言 + 一处三重死断言；
   `test_precision_drift_guard.py` 三对；`test_schema.py` vs `test_load_all_experiments.py`
   三对（留真 preset 版）。
4. 断言 stdlib 行为：`test_iteration_types.py`、`test_validation_cache.py`。
5. 错层测试：`test_wan_dpo_config.py:28-131` 八个测试测的是 `config/builders.py` 的定理，
   搬 `tests/config/`。
6. 手写表 → 注册表派生：`_CUSTOM_REPLAY_MODEL_CLASSES` 与 `_UNREGISTERED_REPLAY_CLASSES`
   互为重复；`test_replay_model_contract.py` 4 张写死 family 名单可由 `policy_semantics` 派生。

## 4. 本 sprint 自产代码的回审结论（诚实记录）

continuous 遥测刚落地即被审出两条成立的 finding：`continuous.active_batches` 与
queue `ready_batches` 在当前"单批不变量"下是结构性常量（≤1）——按我们自己给
`groups_discarded` 定的先例，应当缓做而没缓。处置：**删除 gauge/CSV 列与 stats key，
Sprint 2 lookahead 使两批并存时再加**（那时它才携带信息）；`queue.stats()` 的 6-key dict
唯一生产消费者是一条超时 f-string，收敛为 3 字段。双秒表问题（`VRL_PROFILE` 门控的
collector 内层计时 vs prompt_collection 恒开计时）归入 2-B 一并处置。

## 5. KEEP 汇总（已审过、下轮不再重审）

各审计道的完整 KEEP 清单在本次审计记录中，重点公示防误删：

- **generation/ray**：`RayLifecyclePlan` 全部字段有非日志消费者；`actor_pool` 公平准入拆分；
  `rank_group` 多卡链路是"已实现未启用能力"（registry 有 installer）——保留。
- **models**：25 族全可达；跨族统一薄方法是刻意的 grepability 资产；llamagen `vendor/` 是
  upstream-verbatim 快照不许改；能力方法（`apply_generation_offload` 等）走 getattr 字符串
  派发，符号 grep 零命中≠死。
- **trainers**：`_all_ranks_have_work` 的 docstring 是 NCCL 死锁证明本体；
  `_UnshardedStateStrategy` 一行委托是 lazy-import 环边界；selective checkpointing 是有实测
  收益的用户旋钮（probe-only 现状已在此记录，不算静默漂移）。
- **config**：`unknown_keys.py` 纯派生机制是全仓范本；41 个 schema 字段逐一查过 reader，
  零空旋钮。
- **scripts**：`*_probe` 生命周期制度执行良好（b0a27a8d 一次清了 10 个）；
  `wan_i2v_base_sample.py` 是刻意保留的上游归因 adapter（有 sprint 判决）；
  `reward_overlap_benchmark` 的三个"断点续跑"flag 保留但须写进 docs/perf/README。
- **rollouts**：continuous 的 Schedule/Owner/Runtime 三层各是真边界；
  `_interval_overlap_seconds` 是命名的非平凡算法。
- **require 家族（第三遍，2026-08-22，三条新轴机械扫描后关闭）**：前两遍查的是
  五形态必要性（形态 2/3 抽查）；本遍补上前两遍未覆盖的三轴，83 个定义全量：
  ① **跨名同定理**（AST 归一化 body、剥错误文本、无行数阈值——短守卫不会漏）：
  唯一命中组是 4 个 Protocol `...` stub（声明层，构造性误报），实现层零重复；
  ② **异常类型学**：ValueError 32 / RuntimeError 12 / TypeError 4，每个稀有类型
  语义正确（KeyError=映射取键、ImportError=懒导入边界、FileNotFoundError=文件、
  NotImplementedError=不支持拓扑、domain error=wire 边界、SystemExit=CLI probe）
  ——**统一成单一类型反而是倒退**，上层按语义类 catch；
  ③ **消息悬空引用**：14 个 config 形 token，2 个标记均为误报（`ar_engine` 是
  request 载荷键、`pipeline_offload_mode` 是内部派生字段）。
  **除非新增定义，require 家族不再立审计。** 真正落地的统一是同批收养的四处
  same-theorem 合并（sufficient-stats 归约、标量 collective、process-group mixin、
  replay sweep），见对应提交。

## 6. 制度补丁（比单次清理更值钱的两条）

1. **兼容 facade 保留判决必须附 grep 证据**：本次两处历史 KEEP（fp8 facade、danbooru
   门面）都建立在"旧调用者仍存在"这个未验证前提上——这是"判 caller 不判 body"失误在
   兼容层上的变体。规则：sprint doc 里无"谁还在用旧名"的 grep 输出即默认删除。
2. **dead-flag lint 进 `make verify`**：argparse flag × 全语料 grep 的机械检查（本次审计
   脚本可直接沉淀），让旋钮腐烂在当天而非三个 sprint 后被发现。`vrl/config/lint.py` 的
   AST sweep 机制已现成，可扩展承接 `_OFFLINE_DPO_*_FIELDS` 的交叉校验。

## 7. Non-goals

- 不折叠跨族统一形状的薄方法（models）；不动 vendor 快照；不删"已实现未启用"能力
  （sequence_parallel/rank_group）。
- 不把 `tests/config` 的独立 capability matrix fixture 改成派生（注释已声明刻意独立）。
- 不为省行数合并断言不同定理的相似测试（fp4 对齐拒绝 vs fp8 blockwise 回退各自保留）。
- 2-B 的 reward tensor→PIL 合并**不得**顺手统一帧采样语义——采样策略变更是行为变更，
  须先出 reward 数值回归再定。

## 8. 执行批次与验收门

- **批次 0（P0）**：§1 六条。每条独立 commit；B1/B2 补失败路径测试；A1 出数值裁决记录。
- **批次 1**：2-E 幽灵删除 + 零导入 facade + 死旋钮（机械、低风险、量大）。
- **批次 2**：2-A 双路径拆除（generation 测试替身升级先行）。
- **批次 3**：2-B 孪生合并（trainers replay 循环、rewards judge/PIL——后者带 reward 回归门）。
- **批次 4**：2-C 防护栏修真 + §3 测试清理 + §6 制度补丁。
- 每批验收：`pytest`（受影响包全量）、`ruff check/format --check`（仅触碰文件）、
  config resolve 冒烟、`git diff --check`；删除项按"同源同生命周期"扩展清理并 grep import
  graph 确认无长期资产引用。

## 8.1 执行日志（2026-08-22，同日完成）

- **批次 0（P0）** `a6a21420`：B1 kill_engines、B3 pickapic 数据集标识下传、B4 cosmos
  pretrained_kwargs、B5 extras 全量计数。B2 上游已自修（`3bbd0097` 带回归测试），划掉。
  A1 静态裁决：**降级为"连贯的 opt-out"**——共享 helper docstring 点名 wan 并附实测证据，
  失败模式是响的（strict dtype check），其他 family 上 FSDP+LoRA 前不预防性泛化。
- **批次 1** `a6a21420`：全部幽灵/门面/死旋钮项。判定修正三条：`reward/phymotion.yaml` 与
  causvid `wan_1_3b_ar.yaml` 不删（各自是对应 reward/family 的唯一布线，删除等于变相杀
  family——那是独立决定）；`kling_overlap_gate.yaml` 不删（info sprint 明文"保留为对照"），
  改修悬空指针 + 进 package-data；`EMAModuleWrapper.to()` 复活——`load_state_dict` 经
  `self.to` 调它（审计 grep 模式漏了内部接收者）；reward client `timeout_s` 保留（三个真实
  服务集成测试的合法实例级 seam）。fp8 facade 测试文件中三个**真实校验**测试保留，只删两个
  mock 自证。附带：上游红着的架构测试顺手修复（idm_action_following 模块名对齐注册键）。
- **批次 4a** `b73ee28c`：conflicts()/OptimizationReport 链、kling 锁集收缩（保留
  model_factory + import_path——后者在 geneval 通用 loader 里有真读者，判定精化）、
  precision legacy 循环、三个无牙墓碑测试。
- **cosmos 接缝** `29a1908c`：两个常量 round-trip、runner 继承基类 + finalize/getattr 收敛、
  安检 stub 去重、p2.5 set_num_steps 删除（base docstring 明示为 pipeline-less replay 设计）、
  anima 双重 set_timesteps。判定精化：anima 的 set_num_steps override **必要**（其 base 是
  raise NotImplementedError）。
- **trainers 孪生** `db03886f`：`_run_replay_pass` 单体（只抽内层循环；两侧编排尾部差异是
  实质的，强行合并会造神助手）、`all_reduce_sufficient_stats` 单归约（推导留在各调用者）、
  标量集合原语参数化、`_ProcessGroupStrategy` mixin。
- **批次 2 安全半** `519b5981`：probe 旋钮 → 模块常量 + monkeypatch、completion_callback
  必填（替身补全契约面）、collect_prompt_groups stats 必填。
- **rewards 收缩** `0b3896c0`：9 个 DiskArtifact 类 → ClassVar 声明块（零行为变化：
  request_prefix 保留各自历史拼写）；geneval 走共享 import_from_path。
- **测试去重** `7be85123`：Ray 测试 fakes 三件套 → `_helpers.py`、resources 两对 + 三重死
  断言、drift-guard 组合冗余测试、strategy 自证测试、schema 三个手搭孪生。判定修正：
  drift-guard 的单元测试保留（单元 vs 集成非重复）、auto-placement 对保留（默认态 vs 显式
  态是两个定理）。

## 8.2 余项与缓做原因

| 项 | 缓做原因 |
|---|---|
| generation executor/weight_sync 的 remote-vs-local 双路径（2-A 核心） | 需要把 test_oom_split/test_engine 等替身迁移到 conftest 的包级真 Ray 集群上——独立的测试架构工程,不宜混入机械批次 |
| `retain_artifacts`、reward client `_execute`/`_revalidate` 脚手架 | 低值;前者三个测试消费者需改断言路径 |
| reward inference 映射 4→1 构造点 | `resources.py` 在 typed build 之前运行,需先确认调用序 |
| `_OFFLINE_DPO_*_FIELDS` 派生/交叉校验、`lora.init` 双别名、precision `_select`→`cfg_path` | 用户可见 config 面(别名删除是 key 迁移)、`None`-vs-缺失语义差需逐调用点核对 |
| 三份 lazy-export 表 → 共享工厂 | 公共 facade 行为须逐字节保持,含 torch-free import 契约测试 |
| kling / unified_reward_video 两份 reward function 测试文件 | 与 hpsv3/videoscore2 不同构（各带 release-after-success、empty runtime、rubric 断言），不并入参数化模块 |
| scripts 长尾死 CLI flag(wan_robotics/videophy 等)、`--vbench-*` 决定、`init-dirs` | 各挂着 KEEP 脚本,逐个确认;vbench 需先确认 extra 是否装过。`dead_flags` lint 当前报 350 个 flag 全部有消费者 |

### 2026-09-05 结清的余项

- **`_sampling_fields_for_cfg` 三分支收敛**：函数已不存在——`SPRINT_config_boundary_program`
  把 sampling 字段集改为 `type(root.sampling).model_fields` 派生，随后
  `SPRINT_generation_request_typed_sampling` 把 request 边界关闭。零额外动作。
- **rewards Qwen judge 基类合并、tensor→PIL 统一路由**（`41b38f64`）：数值门由
  `SPRINT_reward-tiny-real-and-optional-lanes` 建成后动手。`QwenVLVideoJudge`
  承接 VideoScore2 与 Cosmos3 reasoner 的加载 / chat turn / processor / generate / decode，
  漂移按 device_map 正确的一侧收敛（inputs 跟 `self.model.device`）；
  `pil_frames_from_media` 承接 pickscore / aesthetic / animereward 的 tensor→PIL，
  布局契约与 `decode_artifact_frames` 一致（`[C,H,W]` / `[C,T,H,W]` / `[B,C,T,H,W]`），
  帧选择仍各自决定（中间帧 / 三等分 / 窗口）。判定精化：pickscore 早先修复里接受的
  `[B,T,C,H,W]` 猜测随之删除——没有任何 producer。unified_reward_video 不并入：它把帧当
  图片送 processor，不走 `process_vision_info`，形状不同。
- **`robotics_discrimination.py` 迁往 `vrl/scripts/eval/`**，测试随迁 `tests/scripts/eval/`；
  `vrl/rewards/evaluation/` 包删除（唯一成员）。
- **`test_wan_dpo_config.py` 的 builder 定理迁往 `tests/config/test_offline_dpo_builders.py`**；
  留下的两条是 `train_wan_2_1_dpo` 的入口行为。
- **fp4/fp8 基类定理参数化**：`tests/nn/quantization/test_quantized_linear.py` 对
  `QUANTIZATION_SCHEMES` 参数化五条 master-weight 生命周期 / fail-before-mutation 定理，
  两个 per-scheme 文件只留数值与 targeting。
- **hpsv3 / videoscore2 两份逐字同构的 function 测试** → `tests/rewards/test_disk_artifact_reward_functions.py`。

### 已结清的 deferral

- **models 三项（2026-09-07，`0accf019c`）**：sana `from_build`→`super().from_build` + 两处 SANA 特有步骤；`WanI2VReplayModel(WanT2VReplayModel, WanI2VDiffusersModel)`，每个方法的 owner 由 `test_family_mro.py` 钉住；cosmos3 的 `set_num_steps` 与基类静态分支逐字相同（UniPC 无 dynamic shifting，pipeline 自己也是无 `mu` 的 `set_timesteps`），删除。前置的 tiny-model 回归由 `RecordingModule` 与 cosmos3 tiny 真对象提供。

- **制度补丁:dead-flag lint 进 `make verify`（2026-08-22 落地）。**
  `vrl/scripts/lint/dead_flags.py` + `tests/scripts/test_dead_flag_lint.py`，
  与 `vrl.config.lint` 并列进 gate。判据:flag 的 `dest` 在 `vrl/`+`tests/`
  中无 `args.<dest>` 读取，且其 option 字符串不出现在 docs / Makefile / 其他
  模块 → 死。当前 **350 个 flag 全部有消费者**（本次审计清理后的零态）。
  `DYNAMIC_CONSUMERS` 是 `getattr(args, …)` 这类无法 grep 证明的显式登记口，
  今天为空。
  **负控是这个门的承重测试**:向真实脚本注入一个未消费 flag 必须报红，
  否则一个永远绿的门等于没有门——测试里用真 AST 扫描验证过。

## 8.3 require* 家族审计（2026-08-22/23）

87 个 `require*` 定义的全量分类，以及"能不能消掉 require"的结论。

**不可行的两条路。**（a）统一成一个 `require(cond, msg)`：名字里的定理没了就 grep 不出来；
仓库里信息量最低的正是最泛化的那个 `require(cfg, path)`。（b）靠类型注解替代：`make verify`
只跑 ruff + config lint + dead-flags lint + config 测试，**没有任何类型检查器**，注解不被强制
——`require_*` 就是本仓库运行时的类型系统。

**可行的一条：在值诞生处验一次，而不是在每个使用处验。** 这不是新发明，是仓库已有的模式
（`ContinuousRolloutConfig.__post_init__`、`TrajectorySignalBatch.__post_init__`、
`WorkerMemoryParkingSnapshot.validate()`），只是有不变量没被纳入。

| 类别 | 数量 | 处置 |
|---|---|---|
| `Any` 收窄成类型 | 22 | 减：在接收处验一次；注解本身不解决问题 |
| `Optional` 收窄成值 | 3 | 可消：构造时钉住不变量 |
| 原始 config / 用户输入 | 6 | **保留**，真边界 |
| 已类型化、跨字段不变量 | 43 | 归属问题，逐个看 |
| `requires_*` 谓词（返回 bool 不 raise） | 13 | 不是校验器，与本议题无关 |

**已落地。** `require_tensor` 从 13 个调用点到 0，函数本身删除（`1aee1038` + `355d9339`）：
同一条 CFG 不变量原先有三种处理——8 个 uncond 分支致命拒绝、4 个生产端静默丢弃、还有直接
读的。`DiffusionBackboneInput.__post_init__` 归位主干，sd3_5 / cosmos-predict2 的 SamplingState
归位各自的家族尾巴（pooled embeds / uncond indicator），wan 的两处则是逻辑上无法触发的检查
（一处装箱自非可选字段，一处已被 `is not None` 守住）。另 `99edc1ad` 修正 Rule-1 身份穿线
（`owner=type(self).__name__` × 6 → 类属性声明），并查出 causvid 的零时间步检查因继承默认值
而空转。

**试过并被证据否决：`RuntimeBundle.model` 提前到 `__post_init__`。** 字段已声明为
`RuntimeModel`，两个消费点各自重验一次，看似该收敛。实测 14 个测试为**非重放关切**
（VAE 显存策略、LoRA 构建）构造 bundle，其 model 本就不需要重放契约——把最严格消费者的契约
提到构造时是过度约束。且与 CFG 那例不同：两个消费点处理方式一致、无分叉，检查本就发生在
运行早期。**保持现状**。

**剩余目标。** `DiffusionBackboneInput.extra: dict[str, Any]` 仍是"已类型化 state 字段 → 装箱
进无类型袋子 → 类型被抹掉"的往返（Rule 4）；43 个 D 类里应逐个问"这条不变量属于哪个类型"。
两者都不急，且都应遵循同一判据：**只有当不变量在构造处可判定、且现状存在分叉时才搬**——
没有分叉的重验，搬家换不来正确性。

## 9. References

- 审计道报告（7 份，本次会话产出，finding 坐标以本文档为准）
- `AGENTS.md`（死代码五形式 / placement 四规则 / one-shot vs long-term）
- 先例：`docs/sprints/done/SPRINT_deadcode_00_overview.md`、
  `SPRINT_deadcode_rollouts_trainers_ray.md`、`SPRINT_allcaps_constants_audit.md`、
  `SPRINT_homeless_function_placement.md`、`SPRINT_docstring-truth-and-double-dedup.md`
