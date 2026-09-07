# SPRINT: config 校验分层 — schema 只管形状，跨 section 规则与启动门各归一处

**日期**: 2026-09-07  **状态**: DONE（CPU gate 全绿）
**触发**: 用户 "help me better organize my validation logic and config logic
vrl/config/validation.py feels chaos independent function everywhere"
**证据来源**: `vrl/config/{schema,validation,builders,lint}.py` 全文、12 个 importer、
`tests/config/*` 与 `tests/trainers/test_activation_checkpointing.py`、
`git log -- vrl/config/validation.py`（S3–S6 系列 refactor 把 validator 逐个收进类型，
剩下的正是"不属于任何类型"的那些）。

---

## 0. 之前乱在哪

`validation.py`（413 行）里堆了四件互不相关的事：Kling 生产门（含 230 行读
manifest / source report 的文件校验）、`require_training_config` 入口、torch.compile
兼容矩阵、rollout 漂移守卫，外加一个历史遗留的 `RewardConfig` 转口（4 个测试和
builders 从这里 import）。而另一批"跨 section"规则（`kl_reward_coef` × kind、
`sft_weight` 归属、`rollout.sde` 存在性、nextstep / janus_r1 配对、offline DPO 面）
藏在 `schema.py` 的 `RootConfig._cross_field_validate` 里——同一类规则按"当时写在
哪"分成两处，没有原则。`trainers/activation_checkpointing.py` 还有一份
compile × checkpointing 的重复文案，docstring 说"也被 require_training_config 调用"
早已不真。

## 1. 现在的分层（每层一个模块、一个注册表）

| 层 | 模块 | 谁跑 | 需要什么 |
| --- | --- | --- | --- |
| 1 section 形状 | `schema.py`（pydantic） | `parse_config` | 该 section 自身 |
| 2 跨 section 规则 | `rules.py::CROSS_SECTION_RULES`（6 条 `rule_*`） | `RootConfig` 的 model_validator 转发，所以直接 `RootConfig.model_validate` 也触发 | 只读已解析的 root，不 import torch / 运行时模块 |
| 3 启动门 | `validation.py::TRAINING_GATES`（compile 矩阵、漂移守卫、生产 Kling 门） | `require_training_config`（只有训练启动付这个钱） | precision policy / 运行时模块 / 文件系统 |

生产门不再是 Kling 专属的一坨：`production.py` 只剩一个通用循环——对每个
`production.<reward>.enabled` 的组件，调 reward 类自己的
`validate_production_kwargs`（`DiskArtifactRewardFunction` 上的契约：media type、
artifact format、`production_task_types`、锁定的 worker_config 键），再调数据层的
`validate_dataset_provenance`（`vrl/trainers/data/provenance.py`：按 `data.task_type`
查 `PROVENANCE_SPECS`，用 config 声明的 loader 加载两份 manifest，走
`ArtifactManifestReport.from_examples` 做 artifact / 元数据检查，再用 `SourceReport`
对账 report.json）。奖励读的额外 artifact（DINO 的 `target_video`）由
`RewardFunction.required_prompt_artifacts` 声明，不再在 config 层硬编码奖励名。
原来 i2v 手写的 JSONL 解析器删掉了——它重复了 `ImageCaptionPromptDataset`。`compile_conflicts` 返回结构化的 `CompileConflict(feature, message)`，
`activation_checkpointing.validate_compile_checkpointing_compatible` 改为按
`feature == "gradient_checkpointing"` 复用矩阵，删掉重复文案。`RewardConfig` 转口
删除，importer 改从 `schema` 取。

行为不变：规则顺序与报错文案逐条保留；`tests/config` 328 项 + 全量 gate 通过。
新增 `tests/config/test_validation_tiers.py` 钉住三个接缝：规则/门的签名与命名、
直接构造 root 也触发第 2 层、`rules.py` 保持 import-light、启动门不进 `parse_config`。

## 2. 新增一个检查往哪放

只看它需要什么：只读两个 section → `rules.py` 加 `rule_<name>` 并 append 到
`CROSS_SECTION_RULES`；需要 precision / 运行时模块 / 文件 → `validation.py` 加
`gate_<name>(root, precision)` 并 append 到 `TRAINING_GATES`；需要已解析的运行时
对象（GPU 拓扑、schedule、reward parking）→ 留在产生该对象的 resolver 旁边，
`build_configs` 之后跑（`docs/CONFIGURATION.md` "Validation tiers" 一节）。

## 3. 没动的（有意）

- `schema.py` 里 family-selected 的 model / sampling 分发（`_parse_model_section` 等）
  是第 1 层的一部分，留在原处。
- `TrainerConfig.from_root`、`ResolvedDistributedResources.from_root`、
  `validate_rollout_schedule_topology`、`validate_reward_memory_parking` 是 resolution
  时校验，各自跟着 resolver；`tests/config/test_load_all_experiments.py` 手工串起
  这一组做静态预检，暂不再抽一个总入口。
