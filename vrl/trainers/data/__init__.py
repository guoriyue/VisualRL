"""Trainer data utilities.

Lazy public boundary: the dataset/collate modules pull ``torch.utils.data``, but
``prompt_sampler.PromptSamplingStrategy`` is a plain Enum that
``vrl.config.schema`` reads while validating every recipe. An eager re-export
here charged all config parsing for the Pick-a-Pic and prompt datasets.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.trainers.data.artifacts import ArtifactManifestReport as ArtifactManifestReport
    from vrl.trainers.data.artifacts import ResolvedArtifact as ResolvedArtifact
    from vrl.trainers.data.artifacts import (
        resolve_prompt_example_artifacts as resolve_prompt_example_artifacts,
    )
    from vrl.trainers.data.artifacts import (
        resolve_prompt_example_references as resolve_prompt_example_references,
    )
    from vrl.trainers.data.preferences import (
        PickAPicPreferenceDataset as PickAPicPreferenceDataset,
    )
    from vrl.trainers.data.preferences import PreferenceBatch as PreferenceBatch
    from vrl.trainers.data.preferences import collate_preference as collate_preference
    from vrl.trainers.data.preferences import load_pickapic as load_pickapic
    from vrl.trainers.data.prompt_sampler import PromptBatchSampler as PromptBatchSampler
    from vrl.trainers.data.prompt_sampler import PromptSamplingStrategy as PromptSamplingStrategy
    from vrl.trainers.data.prompts import ImageCaptionPromptDataset as ImageCaptionPromptDataset
    from vrl.trainers.data.prompts import JsonlPromptDataset as JsonlPromptDataset
    from vrl.trainers.data.prompts import PromptExample as PromptExample
    from vrl.trainers.data.prompts import (
        load_prompt_examples_from_config as load_prompt_examples_from_config,
    )
    from vrl.trainers.data.prompts import load_prompt_image_manifest as load_prompt_image_manifest
    from vrl.trainers.data.prompts import load_prompt_manifest as load_prompt_manifest

_PUBLIC_EXPORTS = {
    "ArtifactManifestReport": ("vrl.trainers.data.artifacts", "ArtifactManifestReport"),
    "ImageCaptionPromptDataset": ("vrl.trainers.data.prompts", "ImageCaptionPromptDataset"),
    "JsonlPromptDataset": ("vrl.trainers.data.prompts", "JsonlPromptDataset"),
    "PickAPicPreferenceDataset": ("vrl.trainers.data.preferences", "PickAPicPreferenceDataset"),
    "PreferenceBatch": ("vrl.trainers.data.preferences", "PreferenceBatch"),
    "PromptBatchSampler": ("vrl.trainers.data.prompt_sampler", "PromptBatchSampler"),
    "PromptExample": ("vrl.trainers.data.prompts", "PromptExample"),
    "PromptSamplingStrategy": ("vrl.trainers.data.prompt_sampler", "PromptSamplingStrategy"),
    "ResolvedArtifact": ("vrl.trainers.data.artifacts", "ResolvedArtifact"),
    "collate_preference": ("vrl.trainers.data.preferences", "collate_preference"),
    "load_pickapic": ("vrl.trainers.data.preferences", "load_pickapic"),
    "load_prompt_examples_from_config": (
        "vrl.trainers.data.prompts",
        "load_prompt_examples_from_config",
    ),
    "load_prompt_image_manifest": ("vrl.trainers.data.prompts", "load_prompt_image_manifest"),
    "load_prompt_manifest": ("vrl.trainers.data.prompts", "load_prompt_manifest"),
    "DatasetProvenance": ("vrl.trainers.data.provenance", "DatasetProvenance"),
    "resolve_prompt_example_artifacts": (
        "vrl.trainers.data.artifacts",
        "resolve_prompt_example_artifacts",
    ),
    "resolve_prompt_example_references": (
        "vrl.trainers.data.artifacts",
        "resolve_prompt_example_references",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public trainer-data symbol only when it is requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
