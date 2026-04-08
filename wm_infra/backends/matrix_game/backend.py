"""Matrix-Game dynamics backend built on top of the rollout-engine substrate.

This backend gives wm-infra an explicit dynamics-world-model path alongside
Cosmos-style generation backends. It intentionally reuses the rollout engine
for bring-up so the northbound API can stabilize before a dedicated Matrix
runtime adapter lands.
"""

from __future__ import annotations

import uuid
from typing import Any

import torch

from wm_infra.backends.base import ProduceSampleBackend
from wm_infra.controlplane import (
    ArtifactKind,
    ArtifactRecord,
    ProduceSampleRequest,
    ResourceEstimate,
    RolloutTaskConfig,
    SampleRecord,
    SampleStatus,
    TaskType,
    WorldModelKind,
)
from wm_infra.rollout_engine import (
    AsyncWorldModelEngine,
    DEFAULT_RESOURCE_UNITS_PER_GB,
    RolloutJob,
    RolloutRequest,
)
from wm_infra.operators import RolloutEngineDynamicsOperator


class MatrixGameBackend(ProduceSampleBackend):
    """Dynamics-world-model adapter for Matrix-style action-conditioned rollouts."""

    world_model_kind = WorldModelKind.DYNAMICS
    capability_flags = frozenset({"actions", "rollout", "checkpoint_ready"})

    def __init__(self, engine: AsyncWorldModelEngine, backend_name: str = "matrix-game") -> None:
        self.engine = engine
        self._operator = RolloutEngineDynamicsOperator(engine)
        self.backend_name = backend_name

    def _effective_task_config(self, request: ProduceSampleRequest) -> RolloutTaskConfig:
        return request.task_config.model_copy(deep=True) if request.task_config is not None else RolloutTaskConfig()

    def _resolve_action_tensor(
        self,
        request: ProduceSampleRequest,
        *,
        action_dim: int,
        default_num_steps: int,
    ) -> tuple[torch.Tensor, int]:
        controls = request.sample_spec.controls or {}
        raw_actions = controls.get("actions")
        if raw_actions is None:
            return torch.zeros((default_num_steps, action_dim), dtype=torch.float32), default_num_steps

        action_tensor = torch.tensor(raw_actions, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 2:
            raise ValueError("matrix-game controls.actions must be a 2D [T, A] float array")
        if int(action_tensor.shape[1]) != action_dim:
            raise ValueError(
                f"matrix-game controls.actions must use action_dim={action_dim}, got {int(action_tensor.shape[1])}"
            )

        resolved_steps = max(default_num_steps, int(action_tensor.shape[0]))
        if int(action_tensor.shape[0]) < resolved_steps:
            padding = torch.zeros((resolved_steps - int(action_tensor.shape[0]), action_dim), dtype=torch.float32)
            action_tensor = torch.cat([action_tensor, padding], dim=0)
        return action_tensor, resolved_steps

    def _resolve_initial_latent(
        self,
        request: ProduceSampleRequest,
        *,
        num_tokens: int,
        latent_dim: int,
    ) -> torch.Tensor:
        controls = request.sample_spec.controls or {}
        raw_latent = controls.get("initial_latent")
        if raw_latent is None:
            return torch.randn(num_tokens, latent_dim)

        latent_tensor = torch.tensor(raw_latent, dtype=torch.float32)
        if latent_tensor.ndim != 2:
            raise ValueError("matrix-game controls.initial_latent must be a 2D [N, D] float array")
        if int(latent_tensor.shape[1]) != latent_dim:
            raise ValueError(
                f"matrix-game controls.initial_latent must use latent_dim={latent_dim}, got {int(latent_tensor.shape[1])}"
            )
        if int(latent_tensor.shape[0]) != num_tokens:
            raise ValueError(
                f"matrix-game controls.initial_latent must use num_tokens={num_tokens}, got {int(latent_tensor.shape[0])}"
            )
        return latent_tensor

    async def produce_sample(self, request: ProduceSampleRequest) -> SampleRecord:
        self.validate_world_model_kind(request)
        if request.task_type != TaskType.TEMPORAL_ROLLOUT:
            raise ValueError(f"Backend {self.backend_name} only supports temporal_rollout")

        sample_id = str(uuid.uuid4())
        task_config = self._effective_task_config(request)
        num_tokens = self.engine.engine.config.state_cache.num_latent_tokens
        latent_dim = self.engine.engine.config.dynamics.latent_token_dim
        action_dim = self.engine.engine.config.dynamics.action_dim

        action_tensor, resolved_steps = self._resolve_action_tensor(
            request,
            action_dim=action_dim,
            default_num_steps=task_config.num_steps,
        )
        if resolved_steps != task_config.num_steps:
            task_config = task_config.model_copy(update={"num_steps": resolved_steps})

        scheduling_request = RolloutRequest.from_task_config(sample_id, task_config, priority=request.priority)
        estimated_units = scheduling_request.estimate_resource_units()
        estimate = ResourceEstimate(
            estimated_units=estimated_units,
            estimated_vram_gb=round(estimated_units / DEFAULT_RESOURCE_UNITS_PER_GB, 2),
            bottleneck="action_conditioned_rollout",
            notes=[
                "Estimated from rollout frame_count × num_steps × megapixels.",
                "Matrix dynamics requests treat action-conditioned state stepping as the dominant cost center.",
            ],
        )

        job = RolloutJob(
            job_id="",
            num_steps=resolved_steps,
            return_frames=False,
            return_latents=ArtifactKind.LATENT in request.return_artifacts,
            stream=False,
        )
        job.initial_latent = self._resolve_initial_latent(request, num_tokens=num_tokens, latent_dim=latent_dim)
        job.actions = action_tensor

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
            temporal=request.temporal,
            task_config=task_config,
            resource_estimate=estimate,
            runtime={
                "runtime_substrate": "rollout-engine",
                "operator": self._operator.describe(),
                "rollout_job_id": result.job_id,
                "steps_completed": result.steps_completed,
                "elapsed_ms": result.elapsed_ms,
                "action_count": resolved_steps,
                "action_dim": action_dim,
            },
            metadata={
                "evaluation_policy": request.evaluation_policy,
                "priority": request.priority,
                "labels": request.labels,
                "world_model_contract": "state_plus_action_to_next_state",
                "runtime_contract": "matrix_dynamics_bringup",
            },
        )

        if result.predicted_latents is not None:
            record.artifacts.append(
                ArtifactRecord(
                    artifact_id=f"{sample_id}:latents",
                    kind=ArtifactKind.LATENT,
                    uri=f"inline://samples/{sample_id}/latents",
                    metadata={"latents": result.predicted_latents.cpu().tolist()},
                )
            )

        return record
