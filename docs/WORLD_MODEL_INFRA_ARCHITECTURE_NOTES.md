# World-Model Infra Architecture Notes

## 目的

这份笔记记录一组面向 `wm-infra` 的架构讨论结论，重点针对两个当前最重要的工作负载：

- `wan-video`：视频模型 / 多阶段生成路径
- `matrix-game`：video-model-based dynamics / learned-simulator / 有状态 rollout 路径

这不是在把仓库改写成一个泛化的“任意模型推理平台”。
目标是给 `wm-infra` 提供一套更适合世界模型、视频模型、以及 learned temporal simulation 的 infra 结构。

## 当前判断

`wm-infra` 同时面对三种不同层次的问题：

1. 产品语义
   - sample
   - episode
   - branch
   - rollout
   - checkpoint
   - artifact

2. 执行语义
   - batching
   - scheduling
   - queueing
   - worker placement
   - profiling

3. 状态语义
   - latent
   - embedding
   - recurrent / world state
   - fork / resume
   - checkpoint delta
   - residency / transfer

过去结构容易变乱，不是因为目录名字不好，而是因为这三类问题常常被混在一个模块里。

## 目标结构

推荐把未来结构收敛成下面这种层次：

```text
wm_infra/
  api/
  controlplane/
  backends/
  engine/
    execution/
    state/
    pipeline/
    rollout/
    env/
    workers/
```

注意：

- 这是目标结构，不代表仓库当前已经完全长成这样
- `backends/` 仍然保留，但它应该变薄
- `engine/` 是执行和状态内核，不是产品入口

## 每一层在解决什么

### `api/`

北向 HTTP 面。
只处理：

- protocol
- auth
- streaming
- artifact URL

它不应该拥有执行细节，也不应该决定 state 如何布局。

### `controlplane/`

持久实体和 lineage 面。
只处理：

- sample
- episode
- branch
- rollout
- checkpoint
- state_handle
- artifact metadata

它不应该变成执行热路径的一部分。

### `backends/`

产品后端入口。
保留：

- `wan-video`
- `matrix-game`
- `cosmos`

这层的价值是 northbound product adapter，而不是再长出一整套 backend 私有 scheduler/runtime/persist。

更合理的职责是：

- 把产品请求翻译成 stage graph / state 操作
- 把执行结果翻译成 product-facing artifacts / manifests
- 保留 backend-specific knobs

### `engine/execution/`

纯执行原语层。
它回答的是：

- 一批同构工作怎么 batch
- chunk 怎么形成
- scheduler metadata 怎么表达
- stage profiling 怎么记录
- queue / dispatch 怎么跑

这一层应该只关心 “how to run work efficiently”，不关心 sample、episode、artifact。

### `engine/state/`

统一的 state / latent / embedding residency 层。
这是世界模型 infra 里最重要的一层之一。

这一层应该拥有：

- paged latent pool
- embedding cache
- transfer planner
- copy-on-write fork
- checkpoint delta materialization
- state residency metadata

对世界模型来说，真正共享的底层问题往往不是“都是一个 tensor”，而是：

- state 在哪里
- state 能不能便宜地 fork
- checkpoint 是全量还是 delta
- 后续 step 能不能复用前序状态
- 哪些 state 值得常驻 GPU

### `engine/pipeline/`

多阶段执行编排层。
适合表达：

- `encode -> transition -> decode`
- `text_encode -> condition_encode -> denoise -> vae_decode`
- `prompt_prepare -> transition -> checkpoint -> persist`

它负责：

- stage graph
- stage config schema
- stage input / output contract
- per-stage scheduler binding
- stage-local worker placement

这层借鉴的是 vLLM-Omni / Dynamo 这类多阶段 multimodal serving 的思路，而不是单纯 token serving。

### `engine/rollout/`

直接 rollout engine。
它是比较垂直的一条引擎路径，负责：

- observation / latent 输入
- encode
- transition
- decode
- rollout result assembly

这层更像低层 bring-up / research engine。
它不是整个 learned simulator runtime 的总入口。

### `engine/env/`

learned simulator runtime。
负责：

- session / episode / branch 生命周期
- `step_many`
- `send/recv`
- trajectory persistence hooks
- collector-facing async runtime

这层借鉴的是 EnvPool 的 northbound contract，但底层执行走自己的 `engine/*`。

### `engine/workers/`

stage worker 接口层。
按 stage family 建 worker，而不是按模型名字建 worker。

例如：

- encoder worker
- diffusion / DiT worker
- decoder / VAE worker
- transition worker
- AR worker

这样 stage 后面可以挂不同实现：

- PyTorch eager
- TensorRT / Triton
- 外部 serving engine
- 本地 compiled kernel

## 这套结构为什么适合世界模型

核心原因不是“更整齐”，而是它把世界模型系统里三种不同性质的问题拆开了：

- 模型怎么跑
- 状态怎么活着
- 产品怎么持久化

如果不拆开，常见结果就是：

- diffusion backend 自己偷偷做缓存和阶段管理
- stateful backend 自己偷偷做 session / checkpoint
- API 和 control plane 夹带执行状态
- 每加一个模型就复制一套 `runner / scheduler / runtime / persist`

## 为什么它能同时支持视频模型和 learned simulation

### 对 diffusion-based 视频模型

这类模型更像多阶段 pipeline。

重点通常是：

- stage 多
- 中间 latent 大
- encode / decode 成本显著
- 不同 stage 适合不同 worker / device / scheduler

所以它最依赖：

- `engine/pipeline/`
- `engine/execution/`
- `engine/state/`

### 对 Genie 这类有状态 world model

这类模型更像持续推进的 simulator。

重点通常是：

- state 是一等公民
- step 之间不是独立请求
- fork / branch / resume 重要
- 吞吐常常受 state layout 和 state transfer 影响

所以它最依赖：

- `engine/env/`
- `engine/state/`
- `engine/execution/`

这正是这套结构最强的地方：

- diffusion 路径主要吃 `pipeline + execution + state`
- learned-simulator 路径主要吃 `env + execution + state`

共享的是执行和状态内核，不共享的是上层语义。

## `engine/state/` 为什么是护城河

对世界模型和视频模型 infra，最值钱的底层机制通常不是 API，也不是普通 scheduler，而是 state residency。

因为最贵的问题通常是：

- latent / state 有没有常驻
- fork 能不能 copy-on-write
- checkpoint 是不是全量复制
- 一个 episode 的连续 step 能不能复用前面的结果
- embedding / latent / decode-side state 是否共址
- 多 worker 之间是否在无意义搬状态

所以世界模型 infra 的“KV cache 等价物”更接近：

- latent/state pool
- state locality
- checkpoint delta
- embedding reuse
- transfer avoidance

## 如何把视频模型和 learned simulation 抽象成同一种 state

正确方法不是让它们都变成“同一种张量”，而是让它们遵守同一种 state contract。

统一的不是 payload 内容本身，而是下面这些系统属性：

- 身份
- 生命周期
- 驻留
- 依赖
- 可执行性

更合适的统一方式是：

```text
State = Identity + Semantic Payload + Execution Cursor + Residency + Provenance
```

### 1. Identity

例如：

- `state_handle_id`
- `owner_type`
- `owner_id`
- `branch_id`
- `parent_state_handle_id`
- `checkpoint_id`

### 2. Semantic Payload

payload 本身不必统一成同一个 tensor 结构。
应该统一“描述方式”，而不是统一“内容长相”。

例如：

- `payload_kind`
- `payload_refs`
- `payload_layout`
- `payload_schema_version`

视频 diffusion payload 可能包含：

- video latent
- text/image conditioning embeddings
- sampler timestep / sigma index
- RNG state

world-model payload 可能包含：

- world latent / hidden state
- recurrent memory
- prompt/history context
- action-conditioned carry state

### 3. Execution Cursor

这层是视频模型和 learned simulation 真正可以统一的重要部分。

统一字段可以包括：

- `stage_cursor`
- `logical_step`
- `ready_ops`
- `blocked_on`
- `terminal`

diffusion 模型的 cursor 表示：

- denoise 跑到第几步
- encode 是否完成
- 下一步该进哪个 stage

learned simulator 的 cursor 表示：

- env step 到哪里
- 下一步是 transition、checkpoint 还是 observation materialization

### 4. Residency

统一字段可以包括：

- `residency_tier`
- `device_id`
- `memory_region`
- `bytes`
- `hotness`
- `co_located_refs`
- `rebuild_cost`
- `transfer_cost`

### 5. Provenance

统一字段可以包括：

- `created_by_stage`
- `derived_from`
- `action_ref`
- `prompt_ref`
- `artifact_ref`
- `metrics_ref`

这样 branch / lineage / replay / export 才能共享一套机制。

## 统一的不是内部表示，而是操作协议

统一 state 的关键不是一个万能 `State` 类，而是一组统一操作：

- `materialize(handle, target_layout)`
- `fork(handle)`
- `advance(handle, op, inputs)`
- `checkpoint(handle)`
- `evict(handle)`
- `transfer(handle, target_device)`
- `decode_observation(handle)`
- `persist(handle)`

这组操作对两类模型都成立：

- 对 diffusion，`advance` 可以是一次 denoise step
- 对 Genie，`advance` 可以是一次 transition step
- 对 diffusion，`fork` 可以从某个 noisy latent 分出多条采样轨迹
- 对 Genie，`fork` 可以从某个 world state 分出多条 action rollout

所以统一的是状态机 contract，而不是模型数学形式。

## 为什么不能只剩 `execution/`

`execution/` 只解决：

- work item 怎么 batch
- chunk 怎么切
- scheduler 怎么排
- profiling 怎么记

它不解决：

- state 生命周期
- observation / latent / decode 语义
- episode / branch / checkpoint 语义
- trajectory persistence

所以：

- `execution/` 是 mechanism
- `pipeline/` 是 stage semantics
- `env/` 是 session semantics

`rollout/` 和 `env/` 都应该建立在 `execution/` 上，但不能塌缩成 `execution/`。

## 应该向哪些系统学习

### Madrona

最值得学的是：

- world state 作为一等对象
- ECS / SoA 式状态布局
- task graph 视角
- profiling 按 system / node 粒度做

它最像 learned simulator runtime 的长期方向。

### EnvPool

最值得学的是：

- RL-facing contract
- batched `reset / step / send / recv`
- 异步 collector 交互形式

它更像 northbound env API 的参考，而不是 GPU 内核参考。

### SGLang

最值得学的是：

- runtime decomposition
- scheduler / executor / cache-aware runtime 的分层
- 面向结构化流程的 serving runtime 思路

它比纯 token server 更接近“runtime system”。

### vLLM / vLLM-Omni / Dynamo

最值得学的是：

- paged memory
- worker / engine 边界
- cache accounting
- disaggregated stages
- multimodal stage config

对 `wm-infra`，vLLM-Omni / Dynamo 这类多阶段 multimodal serving 的启发通常比单纯 AR token serving 更直接。

## 总结

这套结构的优秀点在于，它不是把所有模型硬塞进同一种数学形式，而是把世界模型 infra 里最关键的三个问题拆开：

- 产品如何暴露和持久化
- 执行如何批处理和调度
- 状态如何驻留、迁移、分叉、检查点化

因此它既能支撑：

- diffusion-based 视频模型
- stateful world model / learned simulation

又不会逼着两者共享一套不自然的 backend 壳。

真正统一的不是“模型长什么样”，而是：

- 它们都依赖一个共享执行内核
- 它们都依赖一个共享状态协议
- 但它们在上层语义上保持不同

## 参考

- vLLM Architecture Overview  
  https://docs.vllm.ai/en/latest/design/arch_overview/

- vLLM-Omni Stage Configs  
  https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/

- NVIDIA Dynamo vLLM-Omni  
  https://docs.nvidia.com/dynamo/latest/user-guides/diffusion/v-llm-omni

- NVIDIA Dynamo Multimodal Model Serving  
  https://docs.nvidia.com/dynamo/user-guides/multimodal

- EnvPool Documentation  
  https://envpool.readthedocs.io/en/latest/

- Madrona Repository  
  https://github.com/shacklettbp/madrona
