"""Local RL workload entrypoint wrappers."""

from __future__ import annotations

from typing import Any

from wm_infra.workloads.reinforcement_learning.training import ExperimentSpec
from wm_infra.workloads.reinforcement_learning.training import run_local_experiment


DemoConfig = ExperimentSpec


def run_reinforce_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    """Run the default local RL experiment and return summarized metrics."""

    return run_local_experiment(config or DemoConfig())
