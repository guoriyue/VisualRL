"""Trainer-facing RL experiment primitives built on wm-infra env sessions."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from wm_infra.controlplane import (
    EvaluationRunCreate,
    ReplayShardCreate,
    ReplayShardManifest,
    TemporalStatus,
    TemporalStore,
)
from wm_infra.workloads.reinforcement_learning.runtime import TemporalEnvManager


@dataclass(slots=True)
class ExperimentSpec:
    experiment_name: str = "toy-line-actor-critic"
    seed: int = 7
    num_envs: int = 64
    horizon: int = 16
    updates: int = 120
    gamma: float = 0.97
    learning_rate: float = 3e-3
    max_episode_steps: int = 12
    hidden_dim: int = 64
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    train_env_name: str = "toy-line-v0"
    train_task_id: str = "toy-line-train"
    eval_task_id: str = "toy-line-eval"
    eval_num_envs: int = 16
    eval_episodes: int = 16
    eval_interval: int = 10
    collector_auto_reset: bool = True
    replay_dir: str = ""
    temporal_root: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CollectedBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    entropies: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    successes: torch.Tensor
    trajectory_ids: list[str]
    transition_ids: list[str]
    policy_version: str
    collection_ms: float
    runtime_profile: dict[str, Any]


class Collector(ABC):
    @abstractmethod
    def collect(self, learner: "LearnerAdapter", *, policy_version: str) -> CollectedBatch:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class LearnerAdapter(ABC):
    @abstractmethod
    def act(self, observations: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: CollectedBatch) -> dict[str, float]:
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, learner: LearnerAdapter, *, policy_version: str) -> dict[str, Any]:
        raise NotImplementedError


class CategoricalPolicy(nn.Module):
    """Small categorical policy used by the toy RL experiment."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ValueNetwork(nn.Module):
    """Small critic used by the toy RL experiment."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class LocalActorCriticLearner(LearnerAdapter):
    """Minimal learner loop that consumes collected trajectory batches."""

    def __init__(self, *, obs_dim: int, num_actions: int, spec: ExperimentSpec) -> None:
        self.spec = spec
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        torch.manual_seed(spec.seed)
        self.policy = CategoricalPolicy(obs_dim, num_actions, hidden_dim=spec.hidden_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden_dim=spec.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=spec.learning_rate,
        )

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.policy(observations)
        dist = Categorical(logits=logits)
        action_idx = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        actions = torch.nn.functional.one_hot(action_idx, num_classes=logits.shape[-1]).to(dtype=observations.dtype)
        log_prob = dist.log_prob(action_idx)
        value = self.value_net(observations)
        entropy = dist.entropy()
        return actions, log_prob, value, entropy

    def update(self, batch: CollectedBatch) -> dict[str, float]:
        returns = []
        discounted = torch.zeros(batch.rewards.shape[1], dtype=self.dtype, device=self.device)
        for reward_tensor in reversed(batch.rewards):
            discounted = reward_tensor + (self.spec.gamma * discounted)
            returns.append(discounted)
        returns.reverse()
        returns_tensor = torch.stack(returns, dim=0)
        advantages = returns_tensor - batch.values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        policy_loss = -(batch.log_probs * advantages).mean()
        value_loss = (batch.values - returns_tensor).pow(2).mean()
        entropy_bonus = batch.entropies.mean()
        loss = policy_loss + (self.spec.value_coef * value_loss) - (self.spec.entropy_coef * entropy_bonus)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        total_reward = batch.rewards.sum(dim=0)
        success_rate = batch.successes.amax(dim=0).mean().item()
        num_transitions = float(batch.rewards.numel())
        collection_ms = float(batch.collection_ms)
        env_steps_per_sec = (num_transitions / (collection_ms / 1000.0)) if collection_ms > 0 else 0.0
        step_latency_ms = (collection_ms / num_transitions) if num_transitions > 0 else 0.0
        reward_stage_ms = float(batch.runtime_profile.get("reward_stage_ms", 0.0))
        trajectory_persist_ms = float(batch.runtime_profile.get("trajectory_persist_ms", 0.0))
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "mean_return": float(total_reward.mean().item()),
            "success_rate": float(success_rate),
            "collection_ms": collection_ms,
            "num_transitions": num_transitions,
            "env_steps_per_sec": env_steps_per_sec,
            "step_latency_ms": step_latency_ms,
            "chunk_count": float(batch.runtime_profile.get("chunk_count", 0)),
            "max_chunk_size": float(batch.runtime_profile.get("max_chunk_size", 0)),
            "avg_chunk_size": (
                float(sum(batch.runtime_profile.get("chunk_sizes", [])) / len(batch.runtime_profile.get("chunk_sizes", [])))
                if batch.runtime_profile.get("chunk_sizes")
                else 0.0
            ),
            "reward_stage_ms": reward_stage_ms,
            "reward_stage_latency_ms": (reward_stage_ms / num_transitions) if num_transitions > 0 else 0.0,
            "trajectory_persist_ms": trajectory_persist_ms,
            "trajectory_persist_latency_ms": (trajectory_persist_ms / num_transitions) if num_transitions > 0 else 0.0,
            "state_locality_hit_rate": float(batch.runtime_profile.get("state_locality_hit_rate", 0.0)),
            "auto_reset_count": float(batch.runtime_profile.get("auto_reset_count", 0)),
        }


class SynchronousCollector(Collector):
    """Synchronous rollout collector backed by batched env-session stepping.

    Auto-reset is collector-local. The northbound env/session API still keeps
    explicit reset semantics after terminal or truncated steps.
    """

    def __init__(self, manager: TemporalEnvManager, spec: ExperimentSpec) -> None:
        self.manager = manager
        self.spec = spec
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.env_ids: list[str] = []
        self.observations: torch.Tensor | None = None

    def _ensure_sessions(self) -> None:
        if self.env_ids:
            return
        sessions = [
            self.manager.create_session(
                env_name=self.spec.train_env_name,
                task_id=self.spec.train_task_id,
                seed=self.spec.seed + index,
                policy_version=f"{self.spec.experiment_name}:bootstrap",
                max_episode_steps=self.spec.max_episode_steps,
                labels={"split": "train", "component": "collector"},
                metadata={"experiment_name": self.spec.experiment_name, **self.spec.metadata},
            )
            for index in range(self.spec.num_envs)
        ]
        self.env_ids = [item.env_id for item in sessions]
        self.observations = torch.as_tensor([item.observation for item in sessions], dtype=self.dtype, device=self.device)

    def collect(self, learner: LearnerAdapter, *, policy_version: str) -> CollectedBatch:
        self._ensure_sessions()
        assert self.observations is not None

        started_at = time.perf_counter()
        observation_steps = []
        action_steps = []
        log_prob_steps = []
        reward_steps = []
        value_steps = []
        entropy_steps = []
        terminated_steps = []
        truncated_steps = []
        success_steps = []
        trajectory_ids: set[str] = set()
        transition_ids: set[str] = set()
        runtime_profile = {
            "chunk_count": 0,
            "chunk_sizes": [],
            "max_chunk_size": 0,
            "reward_stage_ms": 0.0,
            "trajectory_persist_ms": 0.0,
            "state_locality_hit_rate": 1.0,
            "auto_reset_count": 0,
        }

        for step_idx in range(self.spec.horizon):
            current_obs = self.observations.view(self.spec.num_envs, -1)
            actions, log_probs, values, entropies = learner.act(current_obs, deterministic=False)
            response = self.manager.step_many(
                self.env_ids[0],
                env_ids=self.env_ids[1:],
                actions=actions.detach().cpu().numpy().tolist(),
                policy_version=policy_version,
                checkpoint=False,
                metadata={
                    "experiment_name": self.spec.experiment_name,
                    "component": "collector",
                    "split": "train",
                    "rollout_step": step_idx,
                },
            )
            next_obs = torch.as_tensor(
                [item.observation for item in response.results],
                dtype=self.dtype,
                device=self.device,
            )
            runtime_profile["chunk_count"] += int(response.runtime.get("chunk_count", 0))
            runtime_profile["chunk_sizes"].extend(response.runtime.get("chunk_sizes", []))
            runtime_profile["max_chunk_size"] = max(
                int(runtime_profile["max_chunk_size"]),
                int(response.runtime.get("max_chunk_size", 0)),
            )
            runtime_profile["reward_stage_ms"] += float(response.runtime.get("reward_stage_ms", 0.0))
            runtime_profile["trajectory_persist_ms"] += float(response.runtime.get("trajectory_persist_ms", 0.0))
            rewards = torch.as_tensor([item.reward for item in response.results], dtype=self.dtype, device=self.device)
            terminated = torch.as_tensor([item.terminated for item in response.results], dtype=torch.bool, device=self.device)
            truncated = torch.as_tensor([item.truncated for item in response.results], dtype=torch.bool, device=self.device)
            successes = torch.as_tensor(
                [bool(item.info.get("success", False)) for item in response.results],
                dtype=self.dtype,
                device=self.device,
            )

            observation_steps.append(current_obs)
            action_steps.append(actions)
            log_prob_steps.append(log_probs)
            reward_steps.append(rewards)
            value_steps.append(values)
            entropy_steps.append(entropies)
            terminated_steps.append(terminated)
            truncated_steps.append(truncated)
            success_steps.append(successes)

            for index, item in enumerate(response.results):
                if item.trajectory_id:
                    trajectory_ids.add(item.trajectory_id)
                if item.transition_id:
                    transition_ids.add(item.transition_id)
                if item.terminated or item.truncated:
                    if not self.spec.collector_auto_reset:
                        continue
                    reset = self.manager.reset_session(
                        item.env_id,
                        seed=self.spec.seed + step_idx + index + 1,
                        policy_version=policy_version,
                        metadata={
                            "experiment_name": self.spec.experiment_name,
                            "component": "collector",
                            "split": "train",
                            "reason": "auto_reset",
                        },
                    )
                    runtime_profile["auto_reset_count"] += 1
                    next_obs[index] = torch.as_tensor(reset.observation, dtype=self.dtype, device=self.device)

            self.observations = next_obs

        return CollectedBatch(
            observations=torch.stack(observation_steps, dim=0),
            actions=torch.stack(action_steps, dim=0),
            log_probs=torch.stack(log_prob_steps, dim=0),
            rewards=torch.stack(reward_steps, dim=0),
            values=torch.stack(value_steps, dim=0),
            entropies=torch.stack(entropy_steps, dim=0),
            terminated=torch.stack(terminated_steps, dim=0),
            truncated=torch.stack(truncated_steps, dim=0),
            successes=torch.stack(success_steps, dim=0),
            trajectory_ids=sorted(trajectory_ids),
            transition_ids=sorted(transition_ids),
            policy_version=policy_version,
            collection_ms=(time.perf_counter() - started_at) * 1000.0,
            runtime_profile=runtime_profile,
        )

    def close(self) -> None:
        for env_id in list(self.env_ids):
            try:
                self.manager.delete_session(env_id)
            except KeyError:
                pass
        self.env_ids = []
        self.observations = None


class FixedTaskEvaluator(Evaluator):
    """Deterministic evaluator that runs a fixed task split against current policy."""

    def __init__(self, manager: TemporalEnvManager, spec: ExperimentSpec, temporal_store: TemporalStore) -> None:
        self.manager = manager
        self.spec = spec
        self.temporal_store = temporal_store
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def evaluate(self, learner: LearnerAdapter, *, policy_version: str) -> dict[str, Any]:
        eval_run = self.temporal_store.create_evaluation_run(
            EvaluationRunCreate(
                policy_version=policy_version,
                task_split="eval",
                metadata={"experiment_name": self.spec.experiment_name},
            ),
            status=TemporalStatus.ACTIVE,
        )
        sessions = [
            self.manager.create_session(
                env_name=self.spec.train_env_name,
                task_id=self.spec.eval_task_id,
                seed=self.spec.seed + 1000 + index,
                policy_version=policy_version,
                max_episode_steps=self.spec.max_episode_steps,
                labels={"split": "eval", "component": "evaluator"},
                metadata={"experiment_name": self.spec.experiment_name},
            )
            for index in range(self.spec.eval_num_envs)
        ]
        env_ids = [item.env_id for item in sessions]
        observations = torch.as_tensor([item.observation for item in sessions], dtype=self.dtype, device=self.device)
        running_returns = torch.zeros(len(env_ids), dtype=self.dtype, device=self.device)
        episode_returns: list[float] = []
        episode_successes: list[float] = []
        trajectory_ids: set[str] = set()
        transition_ids: set[str] = set()

        try:
            while len(episode_returns) < self.spec.eval_episodes:
                actions, _, _, _ = learner.act(observations.view(len(env_ids), -1), deterministic=True)
                response = self.manager.step_many(
                    env_ids[0],
                    env_ids=env_ids[1:],
                    actions=actions.detach().cpu().numpy().tolist(),
                    policy_version=policy_version,
                    checkpoint=False,
                    metadata={
                        "experiment_name": self.spec.experiment_name,
                        "component": "evaluator",
                        "split": "eval",
                    },
                )
                next_obs = torch.as_tensor([item.observation for item in response.results], dtype=self.dtype, device=self.device)
                rewards = torch.as_tensor([item.reward for item in response.results], dtype=self.dtype, device=self.device)
                running_returns += rewards

                for index, item in enumerate(response.results):
                    if item.trajectory_id:
                        trajectory_ids.add(item.trajectory_id)
                    if item.transition_id:
                        transition_ids.add(item.transition_id)
                    if not (item.terminated or item.truncated):
                        continue
                    episode_returns.append(float(running_returns[index].item()))
                    episode_successes.append(float(bool(item.info.get("success", False))))
                    running_returns[index] = 0.0
                    if len(episode_returns) < self.spec.eval_episodes:
                        reset = self.manager.reset_session(
                            item.env_id,
                            seed=self.spec.seed + 2000 + len(episode_returns) + index,
                            policy_version=policy_version,
                            metadata={
                                "experiment_name": self.spec.experiment_name,
                                "component": "evaluator",
                                "split": "eval",
                                "reason": "episode_complete",
                            },
                        )
                        next_obs[index] = torch.as_tensor(reset.observation, dtype=self.dtype, device=self.device)
                observations = next_obs
        finally:
            for env_id in env_ids:
                try:
                    self.manager.delete_session(env_id)
                except KeyError:
                    pass

        eval_metrics = {
            "mean_return": float(sum(episode_returns) / max(len(episode_returns), 1)),
            "success_rate": float(sum(episode_successes) / max(len(episode_successes), 1)),
            "episodes": float(len(episode_returns)),
            "transition_count": float(len(transition_ids)),
        }
        eval_run.task_split = "eval"
        eval_run.trajectory_ids = sorted(trajectory_ids)
        eval_run.metrics = eval_metrics
        eval_run.status = TemporalStatus.SUCCEEDED
        eval_run.completed_at = time.time()
        self.temporal_store.update_evaluation_run(eval_run)
        return {
            "eval_run_id": eval_run.eval_run_id,
            "policy_version": policy_version,
            "trajectory_ids": sorted(trajectory_ids),
            "transition_ids": sorted(transition_ids),
            **eval_metrics,
        }


def export_replay_shard(
    temporal_store: TemporalStore,
    *,
    replay_dir: str,
    policy_version: str,
    task_split: str,
    trajectory_ids: list[str],
    transition_ids: list[str],
    metadata: dict[str, Any],
) -> ReplayShardManifest:
    replay_root = Path(replay_dir)
    replay_root.mkdir(parents=True, exist_ok=True)
    trajectory_set = set(trajectory_ids)
    transition_set = set(transition_ids)
    payload = {
        "policy_version": policy_version,
        "task_split": task_split,
        "trajectories": [
            item.model_dump(mode="json")
            for item in temporal_store.trajectories.list()
            if item.trajectory_id in trajectory_set
        ],
        "transitions": [
            item.model_dump(mode="json")
            for item in temporal_store.transitions.list()
            if item.transition_id in transition_set
        ],
        "metadata": metadata,
    }
    output_path = replay_root / f"{policy_version.replace(':', '_')}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return temporal_store.create_replay_shard(
        ReplayShardCreate(
            policy_version=policy_version,
            task_split=task_split,
            trajectory_ids=sorted(trajectory_set),
            transition_ids=sorted(transition_set),
            uri=str(output_path),
            metadata=metadata,
        )
    )


def run_local_experiment(spec: ExperimentSpec | None = None) -> dict[str, Any]:
    """Run a local actor-critic experiment on top of env-session primitives."""

    cfg = spec or ExperimentSpec()
    temporal_root = Path(cfg.temporal_root) if cfg.temporal_root else Path("/tmp") / "wm_infra_rl" / cfg.experiment_name
    temporal_store = TemporalStore(temporal_root)
    collector_manager = TemporalEnvManager(temporal_store)
    evaluator_manager = TemporalEnvManager(temporal_store)
    collector = SynchronousCollector(collector_manager, cfg)

    try:
        collector._ensure_sessions()
        assert collector.observations is not None
        obs_dim = collector.observations.view(cfg.num_envs, -1).shape[-1]
        num_actions = collector_manager.action_dim_for_env(cfg.train_env_name)
        learner = LocalActorCriticLearner(obs_dim=obs_dim, num_actions=num_actions, spec=cfg)
        evaluator = FixedTaskEvaluator(evaluator_manager, cfg, temporal_store)

        metrics: list[dict[str, float]] = []
        last_eval: dict[str, Any] | None = None
        last_replay: ReplayShardManifest | None = None
        for update_idx in range(cfg.updates):
            policy_version = f"{cfg.experiment_name}:update-{update_idx}"
            batch = collector.collect(learner, policy_version=policy_version)
            train_metrics = learner.update(batch)
            metric_row = {"update": float(update_idx), **train_metrics}
            if update_idx == 0 or (update_idx + 1) % cfg.eval_interval == 0 or update_idx == cfg.updates - 1:
                last_eval = evaluator.evaluate(learner, policy_version=policy_version)
                metric_row["eval_mean_return"] = float(last_eval["mean_return"])
                metric_row["eval_success_rate"] = float(last_eval["success_rate"])
            metrics.append(metric_row)

            if cfg.replay_dir:
                last_replay = export_replay_shard(
                    temporal_store,
                    replay_dir=cfg.replay_dir,
                    policy_version=policy_version,
                    task_split="train",
                    trajectory_ids=batch.trajectory_ids,
                    transition_ids=batch.transition_ids,
                    metadata={"experiment_name": cfg.experiment_name, "update": update_idx},
                )

        result = {
            "config": asdict(cfg),
            "metrics": metrics,
            "experiment_name": cfg.experiment_name,
            "temporal_root": str(temporal_root),
            "backend_runtime": {
                "backend_family": collector_manager.backend_for_env(cfg.train_env_name),
                "runner_mode": (
                    collector_manager.genie_world_model.runner.mode
                    if cfg.train_env_name == "genie-token-grid-v0"
                    else "native"
                ),
            },
            "final_mean_return": metrics[-1]["mean_return"],
            "best_mean_return": max(item["mean_return"] for item in metrics),
            "final_success_rate": metrics[-1]["success_rate"],
            "last_evaluation": last_eval,
            "replay_shard": None if last_replay is None else last_replay.model_dump(mode="json"),
        }
        return result
    finally:
        collector.close()
