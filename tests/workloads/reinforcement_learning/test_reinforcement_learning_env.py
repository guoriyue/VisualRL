from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_infra.workloads.reinforcement_learning.env import GoalReward, WorldModelEnv, WorldModelVectorEnv
from wm_infra.workloads.reinforcement_learning.toy import ToyLineWorldModel


def _initial_sampler(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)


def _goal_sampler(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.full((batch_size, 1, 1), 0.4, device=device, dtype=dtype)


def _make_reward() -> GoalReward:
    return GoalReward(success_threshold=0.01, reward_scale=4.0)


def test_world_model_env_requires_reset() -> None:
    env = WorldModelEnv(
        ToyLineWorldModel(),
        initial_state_sampler=_initial_sampler,
        goal_state_sampler=_goal_sampler,
        reward_fn=_make_reward(),
        action_dim=3,
        max_episode_steps=5,
    )

    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_world_model_env_reset_and_step_contract() -> None:
    env = WorldModelEnv(
        ToyLineWorldModel(),
        initial_state_sampler=_initial_sampler,
        goal_state_sampler=_goal_sampler,
        reward_fn=_make_reward(),
        action_dim=3,
        max_episode_steps=5,
    )

    obs, info = env.reset(seed=3)
    assert obs.shape == (1, 2)
    assert info["goal"].shape == (1, 1)
    assert info["step"] == 0

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert obs.shape == (1, 2)
    assert reward == pytest.approx(-0.16)
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1
    assert info["goal_mse"] == pytest.approx(0.04)
    assert info["success"] is False


def test_world_model_vector_env_auto_reset_recycles_finished_slots() -> None:
    env = WorldModelVectorEnv(
        ToyLineWorldModel(),
        num_envs=4,
        initial_state_sampler=_initial_sampler,
        goal_state_sampler=_goal_sampler,
        reward_fn=_make_reward(),
        action_dim=3,
        max_episode_steps=2,
        auto_reset=True,
    )

    obs, info = env.reset(seed=11)
    assert obs.shape == (4, 1, 2)
    assert info["goal"].shape == (4, 1, 1)

    actions = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1))
    _, reward_1, terminated_1, truncated_1, info_1 = env.step(actions)
    assert reward_1.shape == (4,)
    assert not terminated_1.any()
    assert not truncated_1.any()
    np.testing.assert_array_equal(info_1["step"], np.ones(4, dtype=np.int64))

    obs_2, reward_2, terminated_2, truncated_2, info_2 = env.step(actions)
    assert terminated_2.all()
    assert truncated_2.all()
    assert "final_observation" in info_2
    np.testing.assert_array_equal(info_2["step"], np.zeros(4, dtype=np.int64))
    assert reward_2[0] == pytest.approx(0.0)
    assert obs_2.shape == (4, 1, 2)


# ---------------------------------------------------------------------------
# Genie adapter tests (merged from the old reinforcement-learning adapter test file)
# ---------------------------------------------------------------------------

from wm_infra.backends.genie_runner import GenieRunner
from wm_infra.workloads.reinforcement_learning.genie_adapter import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter


def _stub_adapter() -> GenieWorldModelAdapter:
    runner = GenieRunner(device="cpu")
    runner._mode = "stub"
    runner.load = lambda: "stub"  # type: ignore[method-assign]
    return GenieWorldModelAdapter(
        runner=runner,
        spec=GenieRLSpec(history_frames=4, spatial_h=16, spatial_w=16),
        device="cpu",
    )


def test_genie_adapter_predict_next_preserves_shape_and_changes_state() -> None:
    adapter = _stub_adapter()
    initial = adapter.sample_initial_state(seed=7)
    action = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    next_state = adapter.predict_next(initial, action)

    assert next_state.shape == initial.shape
    assert not torch.equal(next_state, initial)


def test_genie_adapter_distinguishes_different_actions() -> None:
    adapter = _stub_adapter()
    initial = adapter.sample_initial_state(seed=11)
    shift_left = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    token_plus = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    left_state = adapter.predict_next(initial, shift_left)
    plus_state = adapter.predict_next(initial, token_plus)

    assert not torch.equal(left_state, plus_state)


def test_genie_reward_reports_success_for_matching_goal() -> None:
    adapter = _stub_adapter()
    reward_fn = GenieTokenReward(adapter.spec, success_threshold=0.02, reward_scale=4.0)
    goal = adapter.sample_goal_state(seed=5)

    reward, terminated, info = reward_fn.evaluate(goal, goal.clone())

    assert reward.shape == (1,)
    assert bool(terminated.item()) is True
    assert float(info["token_l1"].item()) == 0.0
