"""Default env registrations for the built-in RL environments."""

from __future__ import annotations

from typing import Any

import torch

from wm_infra.controlplane import EnvironmentSpec, TaskSpec, WorldModelKind
from wm_infra.workloads.reinforcement_learning.rewards import GoalReward
from wm_infra.workloads.reinforcement_learning.toy import ToyLineWorldModel, ToyLineWorldSpec
from wm_infra.env_runtime.registry import EnvRegistry


# ---------------------------------------------------------------------------
# Spec / task definitions (unchanged from old catalog.py)
# ---------------------------------------------------------------------------

def default_environment_specs() -> list[EnvironmentSpec]:
    return [
        EnvironmentSpec(
            env_name="toy-line-v0",
            backend="toy-line-world-model",
            world_model_kind=WorldModelKind.DYNAMICS,
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
                "runtime_family": "temporal_env_session",
                "world_model_contract": "predict_next",
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
    ]


# ---------------------------------------------------------------------------
# Initial-state samplers
# ---------------------------------------------------------------------------

class _ToyLineInitialStateSampler:
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def __call__(self, task: TaskSpec, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
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


class _ToyLineInfoProvider:
    def session_info(self, env_name: str, goal: torch.Tensor, step_idx: int) -> dict[str, Any]:
        return {
            "step": step_idx,
            "goal": goal.squeeze(0).detach().cpu().numpy().tolist(),
        }

    def transition_info(
        self,
        env_name: str,
        *,
        goal: torch.Tensor,
        info_tensors: dict[str, torch.Tensor],
        index: int,
        step_idx: int,
    ) -> dict[str, Any]:
        info = self.session_info(env_name, goal, step_idx)
        info["goal_mse"] = float(info_tensors["goal_mse"][index].item())
        info["success"] = bool(info_tensors["success"][index].item() > 0)
        return info


def register_defaults(
    registry: EnvRegistry,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Register the built-in toy-line environment."""

    _device = device or torch.device("cpu")
    _dtype = dtype or torch.float32

    specs = {s.env_name: s for s in default_environment_specs()}
    tasks = {t.task_id: t for t in default_task_specs()}

    # -- toy-line-v0 --------------------------------------------------------
    toy_spec = specs["toy-line-v0"]
    toy_model = ToyLineWorldModel(ToyLineWorldSpec(), device=_device, dtype=_dtype)
    toy_reward_schema = toy_spec.reward_schema
    toy_reward = GoalReward(
        success_threshold=float(toy_reward_schema.get("success_threshold", 0.01)),
        reward_scale=float(toy_reward_schema.get("reward_scale", 4.0)),
    )
    registry.register(
        "toy-line-v0",
        spec=toy_spec,
        tasks=[tasks["toy-line-train"], tasks["toy-line-eval"]],
        world_model=toy_model,
        reward_fn=toy_reward,
        action_dim=toy_model.spec.action_dim,
        initial_state_sampler=_ToyLineInitialStateSampler(_device, _dtype),
        info_provider=_ToyLineInfoProvider(),
    )

def build_default_registry(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> EnvRegistry:
    """Create an EnvRegistry populated with the built-in RL environments."""

    registry = EnvRegistry()
    register_defaults(registry, device=device, dtype=dtype)
    return registry


__all__ = [
    "build_default_registry",
    "default_environment_specs",
    "default_task_specs",
    "register_defaults",
]
