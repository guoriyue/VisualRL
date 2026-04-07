"""Environment catalog and model/reward resolution for learned env runtime."""

from __future__ import annotations

from typing import Any

import torch

from wm_infra.controlplane import EnvironmentSpec, TaskSpec, TemporalStore
from wm_infra.runtime.env.genie import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter
from wm_infra.runtime.env.rewards import GoalReward
from wm_infra.runtime.env.toy import ToyLineWorldModel, ToyLineWorldSpec


def default_environment_specs() -> list[EnvironmentSpec]:
    return [
        EnvironmentSpec(
            env_name="toy-line-v0",
            backend="toy-line-world-model",
            observation_mode="latent_goal_concat",
            action_space={
                "type": "discrete_one_hot",
                "num_actions": 3,
                "labels": ["left", "stay", "right"],
            },
            reward_schema={
                "type": "dense_goal_mse",
                "success_threshold": 0.01,
                "reward_scale": 4.0,
            },
            default_horizon=12,
            supports_batch_step=True,
            supports_fork=True,
            metadata={
                "runtime_family": "rl_env_session",
                "world_model_contract": "predict_next",
            },
        ),
        EnvironmentSpec(
            env_name="genie-token-grid-v0",
            backend="genie-rollout",
            observation_mode="token_context_goal_concat",
            action_space={
                "type": "discrete_one_hot",
                "num_actions": 5,
                "labels": ["stay", "shift_left", "shift_right", "token_plus", "token_minus"],
            },
            reward_schema={
                "type": "token_l1_goal",
                "success_threshold": 0.01,
                "reward_scale": 4.0,
            },
            default_horizon=12,
            supports_batch_step=True,
            supports_fork=True,
            metadata={
                "runtime_family": "rl_env_session",
                "world_model_contract": "predict_next",
                "action_conditioning": "latest_prompt_token_control",
            },
        ),
    ]


def default_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id="toy-line-train",
            env_name="toy-line-v0",
            task_family="goal_reaching",
            goal_spec={"mode": "uniform", "low": -0.8, "high": 0.8},
            seed_policy="explicit",
            difficulty="default",
            split="train",
        ),
        TaskSpec(
            task_id="toy-line-eval",
            env_name="toy-line-v0",
            task_family="goal_reaching",
            goal_spec={"mode": "fixed", "target": 0.4},
            seed_policy="explicit",
            difficulty="default",
            split="eval",
        ),
        TaskSpec(
            task_id="genie-token-train",
            env_name="genie-token-grid-v0",
            task_family="token_goal_reaching",
            goal_spec={"mode": "seeded_random"},
            seed_policy="explicit",
            difficulty="default",
            split="train",
        ),
        TaskSpec(
            task_id="genie-token-eval",
            env_name="genie-token-grid-v0",
            task_family="token_goal_reaching",
            goal_spec={"mode": "fixed_seed", "seed": 404},
            seed_policy="explicit",
            difficulty="default",
            split="eval",
        ),
    ]


class LearnedEnvCatalog:
    """Owns environment/task defaults plus model and reward resolution."""

    def __init__(self, temporal_store: TemporalStore, *, device: torch.device, dtype: torch.dtype) -> None:
        self.temporal_store = temporal_store
        self.device = device
        self.dtype = dtype
        self.world_model = ToyLineWorldModel(ToyLineWorldSpec(), device=device, dtype=dtype)
        self.genie_world_model = GenieWorldModelAdapter(device=device, spec=GenieRLSpec())
        self._default_env_specs = {spec.env_name: spec for spec in default_environment_specs()}
        self._default_task_specs = {task.task_id: task for task in default_task_specs()}

    def register_defaults(self) -> None:
        for spec in self._default_env_specs.values():
            self.temporal_store.upsert_environment_spec(spec)
        for task in self._default_task_specs.values():
            self.temporal_store.upsert_task_spec(task)

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
        task = self.temporal_store.task_specs.get(task_id)
        if task is None or task.env_name != env_name:
            raise KeyError(task_id)
        return task

    def default_task_for_env(self, env_name: str) -> str:
        for task in self.temporal_store.task_specs.list():
            if task.env_name == env_name and task.split == "train":
                return task.task_id
        raise KeyError(env_name)

    def action_dim_for_env(self, env_name: str) -> int:
        if env_name == "genie-token-grid-v0":
            return self.genie_world_model.spec.action_dim
        return self.world_model.spec.action_dim

    def backend_for_env(self, env_name: str) -> str:
        return self.resolve_env_spec(env_name).backend

    def world_model_for_env(self, env_name: str):
        if env_name == "genie-token-grid-v0":
            return self.genie_world_model
        return self.world_model

    def sample_initial_state(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        if task.env_name == "genie-token-grid-v0":
            state = self.genie_world_model.sample_initial_state(seed=seed).to(self.device, self.dtype)
            goal_mode = task.goal_spec.get("mode", "seeded_random")
            goal_seed = seed
            if goal_mode == "fixed_seed":
                goal_seed = int(task.goal_spec.get("seed", 404))
            goal = self.genie_world_model.sample_goal_state(seed=goal_seed).to(self.device, self.dtype)
            return state, goal

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
        state = torch.zeros(1, 1, 1, device=self.device, dtype=self.dtype)
        goal_mode = task.goal_spec.get("mode", "fixed")
        if goal_mode == "uniform":
            low = float(task.goal_spec.get("low", -0.8))
            high = float(task.goal_spec.get("high", 0.8))
            goal = torch.empty(1, 1, 1, device=self.device, dtype=self.dtype)
            goal.uniform_(low, high, generator=generator)
        else:
            target = float(task.goal_spec.get("target", 0.4))
            goal = torch.full((1, 1, 1), target, device=self.device, dtype=self.dtype)
        return state, goal

    def reward_fn_for_env(self, env_name: str):
        reward_schema = self.resolve_env_spec(env_name).reward_schema
        if env_name == "genie-token-grid-v0":
            return GenieTokenReward(
                self.genie_world_model.spec,
                success_threshold=float(reward_schema.get("success_threshold", 0.01)),
                reward_scale=float(reward_schema.get("reward_scale", 4.0)),
            )
        return GoalReward(
            success_threshold=float(reward_schema.get("success_threshold", 0.01)),
            reward_scale=float(reward_schema.get("reward_scale", 4.0)),
        )

    def session_info_for_env(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        info: dict[str, Any] = {"step": step_idx}
        info["goal"] = goal.squeeze(0).detach().cpu().numpy().tolist()
        if env_name == "genie-token-grid-v0":
            info["goal_token_grid"] = goal[:, -self.genie_world_model.spec.frame_token_count :, :].reshape(
                1,
                self.genie_world_model.spec.spatial_h,
                self.genie_world_model.spec.spatial_w,
            ).squeeze(0).detach().cpu().numpy().tolist()
        return info

    def transition_info_for_env(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        info = self.session_info_for_env(env_name, goal, step_idx)
        if env_name == "genie-token-grid-v0":
            info["token_l1"] = float(info_tensors["token_l1"][index].item())
        else:
            info["goal_mse"] = float(info_tensors["goal_mse"][index].item())
        info["success"] = bool(info_tensors["success"][index].item() > 0)
        return info


__all__ = ["LearnedEnvCatalog", "default_environment_specs", "default_task_specs"]
