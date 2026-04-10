"""Cosmos video generation model adapter."""

from __future__ import annotations

from typing import Any

from wm_infra.models.base import VideoGenerationModel
from wm_infra.models.families.cosmos.variants import (
    CosmosLocalExecutor,
    CosmosVariant,
)
from wm_infra.schemas.video_generation import StageResult, VideoGenerationRequest


class CosmosGenerationModel(VideoGenerationModel):
    """Cosmos video generation model backed by an in-process local executor."""

    model_family = "cosmos"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        shell_runner: str | None = None,
        timeout_s: int = 600,
        executor: CosmosLocalExecutor | None = None,
        variant: str | None = None,
        model_size: str = "7B",
        model_id_or_path: str | None = None,
        device_id: int = 0,
        dtype: str = "bfloat16",
        enable_cpu_offload: bool = True,
    ) -> None:
        if base_url is not None or api_key is not None or shell_runner is not None:
            raise ValueError(
                "CosmosGenerationModel only supports in-process execution; "
                "base_url, api_key, and shell_runner are no longer supported"
            )
        self.model_name = model_name or "cosmos-predict1-7b-video2world"
        self.timeout_s = timeout_s

        if executor is not None:
            self._executor = executor
        elif variant is not None:
            cosmos_variant = CosmosVariant(variant)
            if cosmos_variant.value.startswith("predict2"):
                from wm_infra.models.families.cosmos.predict2 import (
                    DiffusersCosmosPredict2Executor,
                )

                self._executor = DiffusersCosmosPredict2Executor(
                    variant=cosmos_variant,
                    model_size=model_size,
                    model_id_or_path=model_id_or_path,
                    device_id=device_id,
                    dtype=dtype,
                    enable_cpu_offload=enable_cpu_offload,
                )
            else:
                from wm_infra.models.families.cosmos.predict1 import (
                    DiffusersCosmosPredict1Executor,
                )

                self._executor = DiffusersCosmosPredict1Executor(
                    variant=cosmos_variant,
                    model_size=model_size,
                    model_id_or_path=model_id_or_path,
                    device_id=device_id,
                    dtype=dtype,
                    enable_cpu_offload=enable_cpu_offload,
                )
        else:
            raise ValueError(
                "CosmosGenerationModel requires a real executor configuration. "
                "Pass executor=... or a supported variant=... "
                "(for example, predict1_text2world or predict1_video2world)."
            )

    @property
    def mode(self) -> str:
        return self._executor.execution_mode

    async def load(self) -> None:
        await self._executor.load()

    def describe(self) -> dict[str, Any]:
        return {
            "name": "cosmos-generation-model",
            "family": self.model_family,
            "model_name": self.model_name,
            "mode": self.mode,
            "executor": self._executor.describe(),
        }

    async def encode_text(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.encode_text(request, state)

    async def encode_conditioning(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.encode_conditioning(request, state)

    async def denoise(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.denoise(request, state)

    async def decode_vae(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.decode_vae(request, state)

    async def postprocess(
        self,
        request: VideoGenerationRequest,
        state: dict[str, Any],
    ) -> StageResult:
        return await self._executor.postprocess(request, state)
