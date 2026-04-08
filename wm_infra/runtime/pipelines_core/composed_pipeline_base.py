"""Stage-composition base class inspired by SGLang Diffusion."""

from __future__ import annotations

from typing import Any

from wm_infra.runtime.server_args import GenerationServerArgs

from .executors.sync_executor import SyncPipelineExecutor


class ComposedPipelineBase:
    """Own the ordered stage list and delegate execution to a pipeline executor."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        server_args: GenerationServerArgs | None = None,
        executor: Any | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.server_args = server_args or GenerationServerArgs()
        self.executor = executor or SyncPipelineExecutor()
        self._stages: list[Any] = []

    @property
    def stages(self) -> list[Any]:
        return list(self._stages)

    def add_stage(self, stage: Any) -> None:
        self._stages.append(stage)

    async def forward(self, context: Any, state: dict[str, Any]) -> list[tuple[Any, Any]]:
        return await self.executor.execute(self._stages, context, state)
