  models/          ← 所有模型架构 (dynamics, cosmos, wan)
  engine/          ← 统一推理引擎 (调度, 显存, 批处理, 执行)
  controlplane/    ← 数据模型 + 持久化
  sim/             ← 环境模拟运行时
  workloads/       ← RL 等工作负载
  api/             ← HTTP 入口，直接调 engine
  config.py        ← 配置