"""Lightweight canonical names and aliases for model families.

This module is the naming boundary shared by config validation and the full
runtime registry. It deliberately contains no model, Ray, or generation
imports, so callers that only normalize a family name do not load the runtime
stack.
"""

from __future__ import annotations

from collections.abc import Iterable

# Deliberately isolated protocol taxonomy. Runtime wiring remains in
# registry.py; this table owns only stable external names.
_ALIASES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "flux": ("flux_1_dev",),
    "qwen_image": ("qwen-image",),
    "sana": ("sana_1600m",),
    "lumina2": ("lumina_image_2",),
    "hunyuan_video": ("hunyuanvideo",),
    "mochi": ("mochi_1",),
    "hunyuan_image": ("hunyuanimage_2_1",),
    "pixart_sigma": ("pixart",),
    "cogvideox": ("cogvideox_2b", "cogvideox_5b"),
    "wan_2_1": ("wan",),
    "wan_2_1_i2v": ("wan_i2v",),
    "cosmos-predict2": ("cosmos", "cosmos_predict2"),
    "cosmos-predict2.5": ("cosmos_predict2_5",),
    "cosmos3": ("cosmos3_omni", "cosmos_omni"),
    "cosmos-predict2-anima": ("anima", "cosmos_anima"),
    "minimax_h3": ("minimax-h3", "hailuo_3", "hailuo3"),
    "echo": ("joyai_echo",),
    "janus_pro": ("janus",),
    "janus_pro_r1": ("janus_r1",),
    "nextstep_1": ("nextstep",),
    "emu3": ("emu3_gen",),
    "glm_image": ("glm_image_t2i",),
    "llamagen": ("llamagen_t2i",),
    "causvid": ("caus_vid",),
    "magi_1": ("magi-1", "magi1"),
}


def _index_aliases() -> dict[str, str]:
    indexed: dict[str, str] = {}
    for family, aliases in _ALIASES_BY_FAMILY.items():
        for alias in aliases:
            if alias in indexed:
                raise ValueError(
                    f"duplicate model family alias {alias!r}: {indexed[alias]!r} and {family!r}",
                )
            indexed[alias] = family
    return indexed


_FAMILY_BY_ALIAS = _index_aliases()


def normalize_model_family(family: str) -> str:
    """Return the canonical family name for a known alias."""

    text = str(family)
    return _FAMILY_BY_ALIAS.get(text, text)


def validate_model_family_aliases(canonical_families: Iterable[str]) -> None:
    """Validate naming-table targets and collisions against a runtime registry."""

    canonical = frozenset(str(family) for family in canonical_families)
    missing = sorted(set(_ALIASES_BY_FAMILY) - canonical)
    if missing:
        raise ValueError(f"model family aliases target unregistered families: {missing}")
    conflicts = sorted(set(_FAMILY_BY_ALIAS) & canonical)
    if conflicts:
        raise ValueError(f"model family aliases collide with canonical names: {conflicts}")


__all__ = [
    "normalize_model_family",
    "validate_model_family_aliases",
]
