"""Environment catalog substrate built on top of ``EnvRegistry``."""

from __future__ import annotations

from typing import Any

import torch

from wm_infra.controlplane import EnvironmentSpec, TaskSpec, TemporalStore
from wm_infra.env_runtime.registry import EnvRegistry


class LearnedEnvCatalog:
    """Owns environment/task defaults plus model and reward resolution.

    This class now delegates to :class:`EnvRegistry` internally.
    """

    def __init__(
        self,
        temporal_store: TemporalStore,
        *,
        device: torch.device,
        dtype: torch.dtype,
        registry: EnvRegistry | None = None,
    ) -> None:
        self.temporal_store = temporal_store
        self.device = device
        self.dtype = dtype
        self._registry = registry or EnvRegistry()

    @property
    def registry(self) -> EnvRegistry:
        return self._registry

    def sync_to_store(self) -> None:
        self._registry.sync_to_store(self.temporal_store)

    def list_environment_specs(self) -> list[EnvironmentSpec]:
        return self.temporal_store.environment_specs.list()

    def list_task_specs(self, env_name: str | None = None) -> list[TaskSpec]:
        tasks = self.temporal_store.task_specs.list()
        if env_name is not None:
            tasks = [task for task in tasks if task.env_name == env_name]
        return tasks

    def resolve_env_spec(self, env_name: str) -> EnvironmentSpec:
        spec = self.temporal_store.environment_specs.get(env_name)
        if spec is None:
            raise KeyError(env_name)
        return spec

    def resolve_task(self, task_id: str, *, env_name: str) -> TaskSpec:
        return self._registry.resolve_task(task_id, env_name=env_name)

    def default_task_for_env(self, env_name: str) -> str:
        return self._registry.default_task_for_env(env_name)

    def action_dim_for_env(self, env_name: str) -> int:
        return self._registry.action_dim_for_env(env_name)

    def backend_for_env(self, env_name: str) -> str:
        return self._registry.backend_for_env(env_name)

    def world_model_for_env(self, env_name: str):
        return self._registry.world_model_for_env(env_name)

    def sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._registry.sample_initial_state(task, seed)

    def reward_fn_for_env(self, env_name: str):
        return self._registry.reward_fn_for_env(env_name)

    def session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        return self._registry.session_info_for_env(env_name, goal, step_idx)

    def transition_info_for_env(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        return self._registry.transition_info_for_env(
            env_name,
            goal=goal,
            info_tensors=info_tensors,
            index=index,
            step_idx=step_idx,
        )

__all__ = ["LearnedEnvCatalog"]
