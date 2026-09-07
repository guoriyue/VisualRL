"""Shared registry-derived fixtures for model interface contract tests."""

from __future__ import annotations

from vrl.models.families.registry import FAMILY_REGISTRY, DenoiseFamilyBuild, TokenFamilyBuild
from vrl.utils.config import import_from_path

# Custom replay construction is the only place the registry cannot derive the
# concrete replay class from the generic descriptor.
_CUSTOM_REPLAY_MODEL_CLASSES = {
    "causvid": "vrl.models.families.causvid.model:CausVidReplayModel",
    "cosmos-predict2-anima": "vrl.models.families.cosmos.anima.model:AnimaReplayModel",
    "cosmos3": "vrl.models.families.cosmos.cosmos3.model:Cosmos3ReplayModel",
    "echo": "vrl.models.families.echo.model:EchoReplayModel",
    "minimax_h3": "vrl.models.families.minimax_h3.model:MiniMaxH3ReplayModel",
}


def registered_runtime_model_classes() -> dict[str, type]:
    """Resolve every registered family's runtime model class."""

    return {
        family: import_from_path(entry.family_build.model_cls)
        for family, entry in FAMILY_REGISTRY.items()
    }


def registered_replay_model_classes() -> dict[str, type]:
    """Resolve replay model classes only for families with a replay recipe."""

    resolved: dict[str, type] = {}
    for family, entry in FAMILY_REGISTRY.items():
        if not entry.supports_policy_replay:
            continue
        build = entry.family_build
        if isinstance(build, TokenFamilyBuild):
            replay_path = build.replay_cls
        else:
            assert isinstance(build, DenoiseFamilyBuild)
            replay_path = build.replay_cls or _CUSTOM_REPLAY_MODEL_CLASSES.get(family)
            if replay_path is None:
                raise AssertionError(
                    f"custom replay family {family!r} lacks a contract-test model class",
                )
        resolved[family] = import_from_path(replay_path)
    return resolved


__all__ = [
    "registered_replay_model_classes",
    "registered_runtime_model_classes",
]
