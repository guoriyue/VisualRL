"""RL workload accessors layered on top of the learned-env substrate."""

from __future__ import annotations

import torch

from wm_infra.controlplane import TemporalStore
from wm_infra.env_runtime.manager import TemporalEnvManager as RuntimeTemporalEnvManager
from wm_infra.workloads.reinforcement_learning.defaults import build_default_registry


class ReinforcementLearningEnvManager(RuntimeTemporalEnvManager):
    """RL workload manager with built-in environment registration."""

    def __init__(self, temporal_store: TemporalStore, *, max_chunk_size: int = 32) -> None:
        registry = build_default_registry(device=torch.device("cpu"), dtype=torch.float32)
        super().__init__(temporal_store, max_chunk_size=max_chunk_size, registry=registry)


# Compatibility alias for existing northbound code and tests.
TemporalEnvManager = ReinforcementLearningEnvManager


__all__ = ["ReinforcementLearningEnvManager", "TemporalEnvManager"]
