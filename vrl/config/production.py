"""Production gates (a launch gate, validation tier 3).

``production.<reward>.enabled`` turns on the production contract for one
configured reward component. The gate itself is generic; the knowledge lives
with the owners:

- the **reward contract** is the reward class's ``validate_production_kwargs``
  (``DiskArtifactRewardFunction``): which media type, artifact format, task
  types and locked loader keys it is validated for;
- the **data provenance** is the data layer's ``validate_dataset_provenance``
  (``vrl/trainers/data/provenance.py``): manifests, per-row provenance
  metadata and the source report, keyed by ``data.task_type``, plus every
  artifact the configured rewards declare through ``required_prompt_artifacts``.

Adding a production gate: add its ``<reward>`` entry to ``ProductionSection``
and give the reward class its ``production_task_types``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig


def enabled_production_gates(root: RootConfig) -> tuple[str, ...]:
    """Reward component names whose production gate is switched on."""

    production = root.production
    if production is None:
        return ()
    return tuple(
        name
        for name in type(production).model_fields
        if bool(getattr(getattr(production, name), "enabled", False))
    )


def validate_production_gates(root: RootConfig) -> None:
    """The whole gate: every enabled reward's contract, then the data provenance."""

    if not enabled_production_gates(root):
        return
    validate_production_reward_contract(root)
    validate_production_data(root)


def validate_production_reward_contract(root: RootConfig) -> None:
    """Each enabled reward validates its own kwargs for the configured task type."""

    from vrl.rewards.functions.registry import get_reward

    reward = root.reward
    task_type = str((root.data.task_type if root.data is not None else None) or "")
    for name in enabled_production_gates(root):
        kwargs = (reward.kwargs.get(name) if reward is not None else None) or {}
        reward_cls = get_reward(name)
        validate = getattr(reward_cls, "validate_production_kwargs", None)
        if not callable(validate):
            raise ValueError(
                f"production.{name}: {reward_cls.__name__} has no production contract"
            )
        validate(name, kwargs, task_type=task_type)


def validate_production_data(root: RootConfig) -> None:
    """Manifests, per-row provenance and the source report, for the configured rewards."""

    from vrl.rewards.functions.registry import get_reward
    from vrl.trainers.data.provenance import validate_dataset_provenance

    if root.data is None:
        raise ValueError("config missing required field: data.manifest")
    components = root.reward.components if root.reward is not None else {}
    extra_artifacts = tuple(
        field
        for name in components
        for field in getattr(get_reward(name), "required_prompt_artifacts", ())
    )
    validate_dataset_provenance(root.data, extra_artifact_fields=extra_artifacts)


__all__ = [
    "enabled_production_gates",
    "validate_production_data",
    "validate_production_gates",
    "validate_production_reward_contract",
]
