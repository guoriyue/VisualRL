"""Gateway bootstrap helpers for assembling runtime dependencies."""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from wm_infra.backends import BackendRegistry, CosmosJobQueue, CosmosPredictBackend, WanJobQueue, WanVideoBackend
from wm_infra.config import EngineConfig
from wm_infra.controlplane import SampleManifestStore, TemporalStore
from wm_infra.gateway.state import GatewayRuntime

logger = logging.getLogger("wm_infra")


def build_default_store(config: EngineConfig) -> SampleManifestStore:
    """Create the sample manifest store used by Gateway-managed backends."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return SampleManifestStore(root)


def build_temporal_store(config: EngineConfig) -> TemporalStore:
    """Create the temporal entity store used by Gateway control-plane routes."""
    root = config.controlplane.manifest_store_root or str(Path(tempfile.gettempdir()) / "wm_infra")
    return TemporalStore(Path(root) / "temporal")


def create_gateway_runtime(
    config: EngineConfig,
    *,
    sample_store: SampleManifestStore | None = None,
    backend_registry: BackendRegistry | None = None,
    temporal_store: TemporalStore | None = None,
) -> GatewayRuntime:
    """Assemble static Gateway dependencies that do not require a running engine."""
    resolved_sample_store = sample_store or build_default_store(config)
    resolved_temporal_store = temporal_store or build_temporal_store(config)
    resolved_registry = backend_registry or BackendRegistry()
    return GatewayRuntime(
        config=config,
        sample_store=resolved_sample_store,
        temporal_store=resolved_temporal_store,
        backend_registry=resolved_registry,
    )


def build_gateway_lifespan(runtime: GatewayRuntime):
    """Create the FastAPI lifespan that starts/stops Gateway-managed resources."""

    @asynccontextmanager
    async def lifespan(_app):
        registry = runtime.backend_registry
        config = runtime.config

        if registry.get("wan-video") is None:
            wan_root = config.controlplane.wan_output_root or str(Path(tempfile.gettempdir()) / "wm_infra_wan")
            if config.controlplane.wan_shell_runner is not None:
                raise ValueError(
                    "Gateway only supports in-process Wan execution; remove controlplane.wan_shell_runner"
                )
            registry.register(
                WanVideoBackend(
                    wan_root,
                    wan_admission_max_units=config.controlplane.wan_admission_max_units,
                    wan_admission_max_vram_gb=config.controlplane.wan_admission_max_vram_gb,
                    max_batch_size=config.controlplane.wan_max_batch_size,
                    batch_wait_ms=config.controlplane.wan_batch_wait_ms,
                    warm_pool_size=config.controlplane.wan_warm_pool_size,
                    prewarm_common_signatures=config.controlplane.wan_prewarm_common_signatures,
                    wan_engine_adapter=config.controlplane.wan_engine_adapter,
                    wan_repo_dir=config.controlplane.wan_repo_dir,
                    wan_conda_env=config.controlplane.wan_conda_env,
                    wan_ckpt_dir=config.controlplane.wan_ckpt_dir,
                    wan_i2v_diffusers_dir=config.controlplane.wan_i2v_diffusers_dir,
                    conda_sh_path=config.controlplane.conda_sh_path,
                )
            )
        wan_backend = registry.get("wan-video")

        if registry.get("cosmos-predict") is None:
            cosmos_root = config.controlplane.cosmos_output_root or str(Path(tempfile.gettempdir()) / "wm_infra_cosmos")
            if (
                config.controlplane.cosmos_base_url is not None
                or config.controlplane.cosmos_api_key is not None
                or config.controlplane.cosmos_shell_runner is not None
            ):
                raise ValueError(
                    "Gateway only supports in-process Cosmos execution; remove cosmos_base_url, cosmos_api_key, and cosmos_shell_runner"
                )
            registry.register(
                CosmosPredictBackend(
                    cosmos_root,
                    model_name=config.controlplane.cosmos_model_name,
                    timeout_s=config.controlplane.cosmos_timeout_s,
                )
            )
        cosmos_backend = registry.get("cosmos-predict")

        if isinstance(wan_backend, WanVideoBackend):
            wan_queue_batch_size = wan_backend.queue_batch_size_limit(config.controlplane.wan_max_batch_size)
            runtime.wan_job_queue = WanJobQueue(
                execute_fn=wan_backend.execute_job,
                execute_many_fn=wan_backend.execute_job_batch,
                batch_key_fn=wan_backend.queue_batch_key,
                batch_select_fn=wan_backend.queue_batch_score,
                store=runtime.sample_store,
                queue_name="wan",
                max_queue_size=config.controlplane.wan_max_queue_size,
                max_concurrent=config.controlplane.wan_max_concurrent_jobs,
                max_batch_size=wan_queue_batch_size,
                batch_wait_ms=config.controlplane.wan_batch_wait_ms,
            )
            runtime.wan_job_queue.start()

        if isinstance(cosmos_backend, CosmosPredictBackend):
            runtime.cosmos_job_queue = CosmosJobQueue(
                execute_fn=cosmos_backend.execute_job,
                store=runtime.sample_store,
                queue_name="cosmos",
                max_queue_size=config.controlplane.cosmos_max_queue_size,
                max_concurrent=config.controlplane.cosmos_max_concurrent_jobs,
            )
            runtime.cosmos_job_queue.start()

        device_str = config.device.value if hasattr(config.device, "value") else str(config.device)
        logger.info("Gateway ready: device=%s", device_str)
        yield
        logger.info("Shutting down gateway")
        if runtime.wan_job_queue is not None:
            await runtime.wan_job_queue.stop()
            runtime.wan_job_queue = None
        if runtime.cosmos_job_queue is not None:
            await runtime.cosmos_job_queue.stop()
            runtime.cosmos_job_queue = None

    return lifespan
