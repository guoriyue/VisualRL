"""Lightweight public schemas for the YAML ``sampling`` section.

The runtime registry refers to these classes by dotted path. Keeping them free
of family model imports lets config discovery reject family-inapplicable knobs
without importing torch, diffusers, or upstream model packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import StrictBool, StrictInt, field_validator

from vrl.config.base import ConfigBase


class SamplingSection(ConfigBase):
    """Base for family-selected public sampling configuration.

    The declared fields are also the per-prompt ``request_overrides`` vocabulary:
    the collector validates every override against the selected family class
    before it reaches a ``GenerationRequest`` (``SamplingSection.require_overrides``).
    """

    # Usually a per-prompt request override (paired rollouts/evals); a YAML value
    # seeds every request identically.
    seed: StrictInt | None = None

    @classmethod
    def require_overrides(cls, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Return per-prompt ``request_overrides`` validated against this family's fields.

        The override vocabulary is this class's fields, so a typo fails with the
        same ``unknown sampling.<key>`` message as a YAML typo instead of riding
        the wire to a runtime that ignores it. Returns the validated values.
        """

        section = cls.revalidate(dict(overrides), section="sampling")
        return section.model_dump(mode="python", exclude_unset=True)


class TeaCacheSection(ConfigBase):
    """``sampling.teacache`` mapping form; ``teacache: true`` selects the defaults.

    reader: vrl/generation/steps/denoise/teacache.py TeaCacheConfig.from_sampling
    (the request boundary), via the collector's request projection.
    """

    enabled: StrictBool = True
    threshold: float | None = None
    warmup_steps: StrictInt | None = None


class DenoiseImageSamplingSection(SamplingSection):
    """Sampling inputs shared by denoise image generators."""

    guidance_scale: Any = None
    height: Any = None
    num_steps: Any = None
    width: Any = None
    negative_prompt: str | None = None
    # Rollout-only forward approximation (skips denoise steps on a cached
    # noise_pred). A request-scoped drift source: config validation refuses it
    # unless a drift guard or importance-sampling correction is armed.
    teacache: StrictBool | TeaCacheSection | None = None


class TextEncodedImageSamplingSection(DenoiseImageSamplingSection):
    """Image sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class VideoSamplingSection(DenoiseImageSamplingSection):
    """Sampling inputs shared by denoise video generators."""

    fps: Any = None
    num_frames: Any = None


class TextEncodedVideoSamplingSection(VideoSamplingSection):
    """Video sampling whose prompt encoder exposes a sequence-length knob."""

    max_sequence_length: Any = None


class EchoSamplingSection(VideoSamplingSection):
    """Echo DMD sampling, whose checkpoint has guidance baked in."""

    guidance_scale: Literal[1.0] | None = None


class MiniMaxH3SamplingSection(TextEncodedVideoSamplingSection):
    """MiniMax-H3 sampling: guidance-distilled, ``max_sequence_length`` bounds the
    text rows of its packed sequence (``fps`` is fixed at 24 by the checkpoint)."""

    guidance_scale: Literal[1.0] | None = None


class MagiSamplingSection(SamplingSection):
    """Inputs mapped into MAGI-1's isolated official inference runtime."""

    fps: Any = None
    height: Any = None
    num_frames: Any = None
    num_steps: Any = None
    width: Any = None


class ARSamplingSection(SamplingSection):
    """Request-local scheduler controls shared by autoregressive generators."""

    ar_scheduler_batch_size: int | None = None

    @field_validator("ar_scheduler_batch_size", mode="before")
    @classmethod
    def _validate_ar_scheduler_batch_size(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "sampling.ar_scheduler_batch_size must be a positive integer or null",
            )
        return value


class TextEncodedARSamplingSection(ARSamplingSection):
    """AR sampling whose prompt encoder exposes a sequence-length knob."""

    max_text_length: Any = None


class SharedAttentionARSamplingSection(TextEncodedARSamplingSection):
    """AR sampling for families using the shared selectable attention adapter."""

    attention_backend: Literal["vllm_paged", "torch_native"] | None = None
    # vllm_paged knobs; readers: token_autoregressive/executor.py _ar_runner.
    ar_paged_block_size: StrictInt | None = None
    ar_paged_cache_dtype: str | None = None


class JanusProSamplingSection(SharedAttentionARSamplingSection):
    """Janus-Pro discrete image-token sampling."""

    guidance_scale: Any = None
    image_size: Any = None
    image_token_num: Any = None
    temperature: Any = None


class JanusProR1SamplingSection(JanusProSamplingSection):
    """Janus-Pro reflective sampling additions."""

    max_reflect_len: Any = None


class NextStepSamplingSection(SharedAttentionARSamplingSection):
    """NextStep continuous image-token sampling."""

    guidance_scale: Any = None
    image_size: Any = None
    image_token_num: Any = None
    num_steps: Any = None


class Emu3SamplingSection(SharedAttentionARSamplingSection):
    """Emu3 latent-grid-derived image sampling."""

    guidance_scale: Any = None
    image_area: Any = None
    ratio: Any = None
    temperature: Any = None


class GlmImageSamplingSection(TextEncodedARSamplingSection):
    """GLM-Image native-cache AR prior and frozen DiT decode controls."""

    decode_guidance_scale: Any = None
    decode_num_inference_steps: Any = None
    image_height: Any = None
    image_width: Any = None
    temperature: Any = None
    top_p: Any = None


class LlamaGenSamplingSection(ARSamplingSection):
    """LlamaGen native-cache discrete image-token sampling."""

    guidance_scale: Any = None
    temperature: Any = None
    top_k: Any = None
    top_p: Any = None


__all__ = [
    "ARSamplingSection",
    "DenoiseImageSamplingSection",
    "EchoSamplingSection",
    "Emu3SamplingSection",
    "GlmImageSamplingSection",
    "JanusProR1SamplingSection",
    "JanusProSamplingSection",
    "LlamaGenSamplingSection",
    "MagiSamplingSection",
    "MiniMaxH3SamplingSection",
    "NextStepSamplingSection",
    "SamplingSection",
    "SharedAttentionARSamplingSection",
    "TeaCacheSection",
    "TextEncodedARSamplingSection",
    "TextEncodedImageSamplingSection",
    "TextEncodedVideoSamplingSection",
    "VideoSamplingSection",
]
