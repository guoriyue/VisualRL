"""Concrete learned-environment implementations.

Domain models live here, separate from the runtime substrate in
``wm_infra.runtime``.  Each env module implements the protocols defined
in ``wm_infra.runtime.env.registry`` so the runtime can dispatch to any
registered environment without hard-coded imports.
"""

from wm_infra.envs.genie import GenieRLSpec, GenieTokenReward, GenieWorldModelAdapter
from wm_infra.envs.rewards import GoalReward
from wm_infra.envs.toy import ToyContinuousWorldModel, ToyLineWorldModel, ToyLineWorldSpec, ToyWorldSpec

__all__ = [
    "GenieRLSpec",
    "GenieTokenReward",
    "GenieWorldModelAdapter",
    "GoalReward",
    "ToyContinuousWorldModel",
    "ToyLineWorldModel",
    "ToyLineWorldSpec",
    "ToyWorldSpec",
]
