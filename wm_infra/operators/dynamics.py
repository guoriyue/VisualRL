"""Dynamics-model invocation operators."""

from __future__ import annotations

from wm_infra.operators.base import ModelOperator, OperatorFamily
from wm_infra.rollout_engine import AsyncWorldModelEngine, RolloutJob, RolloutResult


class RolloutEngineDynamicsOperator(ModelOperator):
    """Operator wrapper around the existing rollout-engine async API."""

    operator_name = "rollout-engine"
    family = OperatorFamily.DYNAMICS

    def __init__(self, engine: AsyncWorldModelEngine) -> None:
        self.engine = engine

    async def rollout(self, job: RolloutJob) -> RolloutResult:
        return await self.engine.submit(job)
