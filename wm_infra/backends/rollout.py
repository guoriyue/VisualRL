"""Backend adapter that maps control-plane sample requests onto the rollout engine."""

from __future__ import annotations

import uuid

import torch

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.controlplane.schemas import ArtifactKind, ArtifactRecord, ProduceSampleRequest, ResourceEstimate, SampleRecord, SampleStatus, TaskType, WorldModelKind
from wm_infra.rollout_engine import AsyncWorldModelEngine, DEFAULT_RESOURCE_UNITS_PER_GB, RolloutJob, RolloutRequest
from wm_infra.operators import RolloutEngineDynamicsOperator


class RolloutBackend(ProduceSampleBackend):
    """Produce sample records using the existing world-model rollout engine."""

    world_model_kind = WorldModelKind.DYNAMICS
    capability_flags = frozenset({"actions", "rollout"})

    def __init__(self, engine: AsyncWorldModelEngine, backend_name: str = "rollout-engine") -> None:
        self.engine = engine
        self._operator = RolloutEngineDynamicsOperator(engine)
        self.backend_name = backend_name

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        self.validate_world_model_kind(request)
        if request.task_type != TaskType.TEMPORAL_ROLLOUT:
            raise ValueError(f"Backend {self.backend_name} only supports temporal_rollout")

        sample_id = str(uuid.uuid4())
        task_config = request.task_config
        num_steps = task_config.num_steps if task_config is not None else 1
        return_latents = ArtifactKind.LATENT in request.return_artifacts

        scheduling_request = RolloutRequest.from_task_config(sample_id, task_config, priority=request.priority)
        estimated_units = scheduling_request.estimate_resource_units()
        estimate = ResourceEstimate(
            estimated_units=estimated_units,
            estimated_vram_gb=round(estimated_units / DEFAULT_RESOURCE_UNITS_PER_GB, 2),
            bottleneck="frame_pressure",
            notes=[
                "Estimated from frame_count × num_steps × megapixels.",
                "Low-VRAM profiles reduce the relative score; high-quality profiles increase it.",
            ],
        )

        job = RolloutJob(job_id="", num_steps=num_steps, return_frames=False, return_latents=return_latents, stream=False)
        num_tokens = self.engine.engine.config.state_cache.num_latent_tokens
        latent_dim = self.engine.engine.config.dynamics.latent_token_dim
        job.initial_latent = torch.randn(num_tokens, latent_dim)

        result = await self._operator.rollout(job)
        record = SampleRecord(
            sample_id=sample_id,
            task_type=request.task_type,
            backend=request.backend,
            model=request.model,
            world_model_kind=self.world_model_kind,
            model_revision=request.model_revision,
            status=SampleStatus.SUCCEEDED,
            experiment=request.experiment,
            sample_spec=request.sample_spec,
            task_config=task_config,
            resource_estimate=estimate,
            runtime={
                "rollout_job_id": result.job_id,
                "steps_completed": result.steps_completed,
                "elapsed_ms": result.elapsed_ms,
                "operator": self._operator.describe(),
            },
            metadata={"evaluation_policy": request.evaluation_policy, "priority": request.priority, "labels": request.labels},
        )

        if result.predicted_latents is not None:
            record.artifacts.append(ArtifactRecord(
                artifact_id=f"{sample_id}:latents",
                kind=ArtifactKind.LATENT,
                uri=f"inline://samples/{sample_id}/latents",
                metadata={"latents": result.predicted_latents.cpu().tolist()},
            ))

        return record
