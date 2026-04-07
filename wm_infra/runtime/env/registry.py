"""Registration-based environment catalog for the learned env runtime.

Instead of hard-coding concrete world models and reward functions, external
code registers environments via :class:`EnvRegistry`.  The runtime substrate
only depends on :class:`LearnedEnvProtocol` and :class:`RewardProtocol` --
never on concrete implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from wm_infra.controlplane import EnvironmentSpec, TaskSpec, TemporalStore


# ---------------------------------------------------------------------------
# Protocols -- what the substrate requires from any concrete env
# ---------------------------------------------------------------------------

@runtime_checkable
class LearnedEnvProtocol(Protocol):
    """Minimal contract a world-model must satisfy for the runtime."""

    def predict_next(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class RewardProtocol(Protocol):
    """Minimal contract for a reward function."""

    def evaluate(
        self,
        next_state: torch.Tensor,
        goal_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: ...


@runtime_checkable
class InitialStateSampler(Protocol):
    """Produces (state, goal) tensors for env reset."""

    def __call__(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class EnvInfoProvider(Protocol):
    """Optional: rich info dict for session/transition responses."""

    def session_info(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]: ...

    def transition_info(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Registration record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RegisteredEnv:
    """One registered environment with all its resolved components."""

    env_name: str
    spec: EnvironmentSpec
    tasks: list[TaskSpec]
    world_model: LearnedEnvProtocol
    reward_fn: RewardProtocol
    action_dim: int
    initial_state_sampler: InitialStateSampler
    info_provider: EnvInfoProvider | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class EnvRegistry:
    """Registration-based catalog that replaces hard-coded LearnedEnvCatalog.

    Call :meth:`register` for each concrete environment at startup, then use
    :meth:`resolve` / :meth:`world_model_for_env` etc. from the runtime.
    """

    def __init__(self) -> None:
        self._envs: dict[str, RegisteredEnv] = {}
        self._tasks: dict[str, TaskSpec] = {}

    # -- mutation -----------------------------------------------------------

    def register(
        self,
        env_name: str,
        *,
        spec: EnvironmentSpec,
        tasks: list[TaskSpec],
        world_model: LearnedEnvProtocol,
        reward_fn: RewardProtocol,
        action_dim: int,
        initial_state_sampler: InitialStateSampler,
        info_provider: EnvInfoProvider | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredEnv:
        entry = RegisteredEnv(
            env_name=env_name,
            spec=spec,
            tasks=list(tasks),
            world_model=world_model,
            reward_fn=reward_fn,
            action_dim=action_dim,
            initial_state_sampler=initial_state_sampler,
            info_provider=info_provider,
            metadata=dict(metadata or {}),
        )
        self._envs[env_name] = entry
        for task in tasks:
            self._tasks[task.task_id] = task
        return entry

    def sync_to_store(self, temporal_store: TemporalStore) -> None:
        """Persist all registered specs/tasks into the temporal store."""
        for entry in self._envs.values():
            temporal_store.upsert_environment_spec(entry.spec)
            for task in entry.tasks:
                temporal_store.upsert_task_spec(task)

    # -- resolution ---------------------------------------------------------

    def resolve(self, env_name: str) -> RegisteredEnv:
        entry = self._envs.get(env_name)
        if entry is None:
            raise KeyError(f"Environment {env_name!r} not registered")
        return entry

    def resolve_task(self, task_id: str, *, env_name: str) -> TaskSpec:
        task = self._tasks.get(task_id)
        if task is None or task.env_name != env_name:
            raise KeyError(task_id)
        return task

    def default_task_for_env(self, env_name: str) -> str:
        entry = self.resolve(env_name)
        for task in entry.tasks:
            if task.split == "train":
                return task.task_id
        raise KeyError(f"No default (train-split) task for {env_name!r}")

    def world_model_for_env(self, env_name: str) -> LearnedEnvProtocol:
        return self.resolve(env_name).world_model

    def reward_fn_for_env(self, env_name: str) -> RewardProtocol:
        return self.resolve(env_name).reward_fn

    def action_dim_for_env(self, env_name: str) -> int:
        return self.resolve(env_name).action_dim

    def sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.resolve(task.env_name)
        return entry.initial_state_sampler(task, seed)

    def session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        entry = self.resolve(env_name)
        if entry.info_provider is not None:
            return entry.info_provider.session_info(env_name, goal, step_idx)
        return {"step": step_idx, "goal": goal.squeeze(0).detach().cpu().numpy().tolist()}

    def transition_info_for_env(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        entry = self.resolve(env_name)
        if entry.info_provider is not None:
            return entry.info_provider.transition_info(
                env_name, goal=goal, info_tensors=info_tensors, index=index, step_idx=step_idx,
            )
        info = self.session_info_for_env(env_name, goal, step_idx)
        info["success"] = bool(info_tensors.get("success", torch.zeros(1))[index].item() > 0)
        return info

    def list_environment_specs(self) -> list[EnvironmentSpec]:
        return [entry.spec for entry in self._envs.values()]

    def list_task_specs(self, env_name: str | None = None) -> list[TaskSpec]:
        tasks = list(self._tasks.values())
        if env_name is not None:
            tasks = [t for t in tasks if t.env_name == env_name]
        return tasks

    def backend_for_env(self, env_name: str) -> str:
        return self.resolve(env_name).spec.backend

    def __contains__(self, env_name: str) -> bool:
        return env_name in self._envs

    def __len__(self) -> int:
        return len(self._envs)


__all__ = [
    "EnvInfoProvider",
    "EnvRegistry",
    "InitialStateSampler",
    "LearnedEnvProtocol",
    "RegisteredEnv",
    "RewardProtocol",
]
