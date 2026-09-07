"""Where each diffusion family resolves its shared base members.

Two of these members are silent-wrong-answer traps, which is why they are
pinned by MRO owner rather than by behavior:

- ``_set_transformer`` on ``DiffusersPipelineModelBase`` also writes
  ``pipeline.transformer``. A pipeline-less replay model sits BELOW that class
  in the MRO, so deleting its own override resolves back to the syncing version
  and every LoRA attach / torch.compile on the replay path raises "does not own
  a diffusers pipeline" — on the FSDP/DDP trainer path, far from the edit.
- ``from_build`` on ``DiffusersPipelineModelBase`` is driven by three class
  declarations. A family that declares none of them and also does not override
  ``from_build`` would only fail when a real checkpoint load is attempted.
"""

from __future__ import annotations

from typing import Any

import pytest

from vrl.models.families.registry import FAMILY_REGISTRY, DenoiseFamilyBuild
from vrl.models.steps.denoise import DiffusionModelBase
from vrl.models.steps.denoise.base import DiffusersPipelineModelBase
from vrl.utils.config import import_from_path

# Families whose registry entry declares a replay BUILDER instead of a replay
# class, so the class cannot be reached through ``DenoiseFamilyBuild``.
_UNREGISTERED_REPLAY_CLASSES = (
    "vrl.models.families.echo.model:EchoReplayModel",
    "vrl.models.families.causvid.model:CausVidReplayModel",
    "vrl.models.families.cosmos.cosmos3.model:Cosmos3ReplayModel",
    "vrl.models.families.cosmos.anima.model:AnimaReplayModel",
)

_LOADING_DECLARATIONS = (
    "_pipeline_classname",
    "_frozen_encoder_names",
    "_prompt_encoder_on_cpu",
)


def _owner(cls: type, name: str) -> type | None:
    """The class in ``cls``'s MRO that actually defines ``name``."""

    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    return None


def _denoise_model_classes() -> list[Any]:
    return [
        pytest.param(import_from_path(entry.family_build.model_cls), id=family)
        for family, entry in FAMILY_REGISTRY.items()
        if isinstance(entry.family_build, DenoiseFamilyBuild)
    ]


def _replay_model_classes() -> list[Any]:
    paths = [
        entry.family_build.replay_cls
        for entry in FAMILY_REGISTRY.values()
        if isinstance(entry.family_build, DenoiseFamilyBuild)
        and entry.family_build.replay_cls is not None
    ]
    paths.extend(_UNREGISTERED_REPLAY_CLASSES)
    classes = {}
    for path in paths:
        replay_cls = import_from_path(path)
        classes[replay_cls.__name__] = replay_cls
    return [pytest.param(cls, id=name) for name, cls in classes.items()]


@pytest.mark.parametrize("model_cls", _denoise_model_classes())
def test_family_either_declares_its_pipeline_load_or_owns_from_build(
    model_cls: type[DiffusionModelBase],
) -> None:
    """No family may fall through to the shared loader without declaring its inputs."""

    if not issubclass(model_cls, DiffusersPipelineModelBase):
        assert _owner(model_cls, "from_build") is model_cls
        return

    if not model_cls._pipeline_classname:
        assert _owner(model_cls, "from_build") is not DiffusersPipelineModelBase, (
            f"{model_cls.__name__} inherits the shared from_build but declares no "
            "_pipeline_classname; it would raise on the first real load"
        )
        return

    import diffusers

    declaring = _owner(model_cls, "_pipeline_classname")
    for name in _LOADING_DECLARATIONS:
        assert _owner(model_cls, name) is declaring, (
            f"{model_cls.__name__} must declare {name} on the same class as "
            "_pipeline_classname; the memory decision and the pipeline it applies "
            "to belong together"
        )
    assert hasattr(diffusers, model_cls._pipeline_classname)


def test_wan_i2v_replay_takes_forward_math_from_i2v_and_plumbing_from_t2v_replay() -> None:
    """The I2V replay's base order is load-bearing.

    ``WanT2VReplayModel`` must come first so the replay plumbing (constructor,
    dual-stage ``prepare_replay``, pipeline-less accessors) wins, while the I2V
    forward math must still resolve to ``WanI2VDiffusersModel``, not the T2V
    class further down the MRO. Reversing the bases keeps every test that only
    constructs the model green and silently swaps the forward.
    """
    from vrl.models.families.wan_2_1.model import (
        WanI2VDiffusersModel,
        WanI2VReplayModel,
        WanT2VReplayModel,
    )

    for name in ("forward_step", "restore_eval_state", "export_replay_tensors"):
        assert _owner(WanI2VReplayModel, name) is WanI2VDiffusersModel, name
    for name in ("__init__", "prepare_replay", "pipeline", "scheduler", "_set_transformer"):
        assert _owner(WanI2VReplayModel, name) is WanT2VReplayModel, name


@pytest.mark.parametrize("replay_cls", _replay_model_classes())
def test_pipeline_less_replay_models_never_inherit_the_pipeline_sync(
    replay_cls: type,
) -> None:
    """A replay model that has no pipeline must not resolve to the syncing setter."""

    if _owner(replay_cls, "pipeline") is DiffusersPipelineModelBase:
        # Cosmos3 replay wraps a pipeline SHELL, so the syncing setter is correct.
        return
    assert _owner(replay_cls, "_set_transformer") is not DiffusersPipelineModelBase, (
        f"{replay_cls.__name__} owns no diffusers pipeline; it must keep (or "
        "inherit from DiffusionModelBase/DiffusersReplayModelBase) a "
        "_set_transformer that does not touch self.pipeline"
    )
