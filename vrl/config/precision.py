"""Typed public ``precision`` section and its resolution into runtime policy.

``dtype`` is the ordinary parameter precision for a role; ``outer_autocast``
selects whether the shared diffusion boundary also applies that dtype through
AMP. Selective quantization is a separate kernel policy layered on the dtype:
FP8 replaces eligible attention/MLP Linears, while experimental NVFP4
conservatively replaces eligible MLP Linears only. Unswapped operations keep
the role dtype.

Rollout dtype and outer-autocast behavior default to the training role.
Prompt-encoder dtype defaults to the resolved rollout dtype; the canonical base
preset explicitly selects fp16 to preserve its established memory policy. VAE
precision is family-owned and is not represented by this prompt-encoder axis.
Diffusion math defaults to fp32. FP32 matmul precision is an explicit run-wide
policy because PyTorch process defaults are not a stable trainer/worker contract.

The pydantic sections below are the single declaration of the YAML shape:
``parse_config`` validates them (vocabulary, required keys, unknown keys), and
:func:`PrecisionPolicy` turns the validated section into the two role
policies. Training quantization is part of the structural schema so the
eventual training runtime will use the same role shape, but policy resolution
fails until a real autograd-capable consumer exists. This module remains
torch-free; runtime boundaries materialize dtype tokens through
``vrl.models.dtypes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from pydantic import StrictBool, field_validator, model_validator

from vrl.config.base import ConfigBase

Float32Precision = Literal["ieee", "tf32"]

# Protocol vocabularies are a real config boundary. Keep recipe compatibility in
# one deliberately isolated table so config resolution, defaults, and error text
# cannot drift when a new format lands.
_PLAIN_DTYPES = ("fp32", "bf16", "fp16")


@dataclass(frozen=True, slots=True)
class _QuantizationFormatRules:
    allowed_recipes: tuple[str, ...]
    default_recipe: str | None


_QUANTIZATION_FORMAT_RULES = {
    "fp8": _QuantizationFormatRules(
        allowed_recipes=("rowwise", "tensorwise", "blockwise"),
        default_recipe="rowwise",
    ),
    # NVFP4 is the complete two-level scaling scheme, not an FP8-style recipe.
    "nvfp4": _QuantizationFormatRules(allowed_recipes=(), default_recipe=None),
}
_PRECISION_TOKENS = (*_PLAIN_DTYPES, *_QUANTIZATION_FORMAT_RULES)


def normalize_precision(value: Any, *, default: str = "fp32") -> str:
    """Normalize a known precision token at a config or tool boundary."""

    if value is None:
        return default
    token = str(value).lower().strip()
    if not token:
        return default
    if token == "no":
        return "fp32"
    if token not in _PRECISION_TOKENS:
        raise ValueError(
            f"precision must be one of {_PRECISION_TOKENS} (or legacy 'no'); got {value!r}",
        )
    return token


def _normalize_plain_dtype(value: Any, *, path: str) -> str:
    token = normalize_precision(value)
    if token not in _PLAIN_DTYPES:
        raise ValueError(
            f"{path} must be one of {_PLAIN_DTYPES}; got {token!r}. FP8/NVFP4 "
            "belongs under a `quantization.format` key, not a `dtype` key.",
        )
    return token


def _normalize_float32_precision(value: Any) -> Float32Precision:
    mode = str(value).lower().strip()
    allowed = get_args(Float32Precision)
    if mode not in allowed:
        raise ValueError(
            f"precision.float32_precision must be one of {allowed}; got {value!r}",
        )
    return cast("Float32Precision", mode)


# ── Resolved runtime policy ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Selective low-precision kernel policy layered on a role's base dtype."""

    format: str
    recipe: str | None = None

    @classmethod
    def from_section(cls, section: QuantizationConfig | None) -> QuantizationPolicy | None:
        """Resolve an optional public quantization section."""
        if section is None:
            return None
        return cls(format=section.format, recipe=section.recipe)

    def __post_init__(self) -> None:
        format_name = str(self.format).lower().strip()
        recipe = str(self.recipe).lower().strip() if self.recipe is not None else None
        if format_name == "fp4":
            raise ValueError(
                "quantization.format='fp4' was replaced by the precise scheme name "
                "'nvfp4'; use `format: nvfp4` without a recipe.",
            )
        format_rules = _QUANTIZATION_FORMAT_RULES.get(format_name)
        if format_rules is None:
            raise ValueError(
                "quantization.format must be one of "
                f"{tuple(_QUANTIZATION_FORMAT_RULES)}; got {self.format!r}",
            )
        if recipe is None:
            recipe = format_rules.default_recipe
        elif recipe not in format_rules.allowed_recipes:
            if not format_rules.allowed_recipes:
                raise ValueError(
                    f"quantization.format={format_name!r} does not accept a recipe; "
                    "remove the `recipe` key.",
                )
            raise ValueError(
                f"quantization.recipe={recipe!r} is invalid for format "
                f"{format_name!r}; expected one of {format_rules.allowed_recipes}.",
            )
        object.__setattr__(self, "format", format_name)
        object.__setattr__(self, "recipe", recipe)


@dataclass(frozen=True, slots=True)
class RolePrecision:
    """Transformer precision selected for one trainer or rollout process."""

    dtype: str
    float32_precision: Float32Precision
    quantization: QuantizationPolicy | None = None
    outer_autocast: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dtype",
            _normalize_plain_dtype(self.dtype, path="precision role dtype"),
        )
        object.__setattr__(
            self,
            "float32_precision",
            _normalize_float32_precision(self.float32_precision),
        )

    @property
    def label(self) -> str:
        """Stable role label containing base dtype and execution policies."""

        parts = [self.dtype]
        if self.quantization is not None:
            parts.append(self.quantization.format)
        if not self.outer_autocast:
            parts.append("no-autocast")
        return "+".join(parts)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved precision for both roles, protected math, and prompt encoders."""

    training: RolePrecision
    rollout: RolePrecision
    diffusion_math: str
    prompt_encoder_dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diffusion_math",
            _normalize_plain_dtype(
                self.diffusion_math,
                path="precision.diffusion_math.dtype",
            ),
        )
        object.__setattr__(
            self,
            "prompt_encoder_dtype",
            _normalize_plain_dtype(
                self.prompt_encoder_dtype,
                path="precision.rollout.prompt_encoders.dtype",
            ),
        )
        if self.training.quantization is not None:
            raise ValueError(
                "precision.training.quantization is unavailable: the trainer has no "
                "quantized Linear forward/backward consumer. Remove the block until "
                "that runtime is implemented; VRL will not accept a silent no-op.",
            )

    @property
    def stages_match(self) -> bool:
        """Whether rollout and replay use the same complete execution policy."""

        return self.training == self.rollout

    @classmethod
    def from_section(cls, section: PrecisionConfig | None) -> PrecisionPolicy:
        """Build the precision policy from the parsed ``precision`` section."""

        if section is None:
            raise ValueError(
                "top-level `precision` is required. Configure "
                "`precision.training.dtype`; add a `precision.rollout` block only "
                "for a rollout-specific override.",
            )
        float32_precision = section.float32_precision
        training = RolePrecision(
            dtype=section.training.dtype,
            float32_precision=float32_precision,
            outer_autocast=section.training.outer_autocast,
            quantization=QuantizationPolicy.from_section(section.training.quantization),
        )
        rollout_section = section.rollout or RolloutPrecisionConfig()
        rollout = RolePrecision(
            dtype=rollout_section.dtype or training.dtype,
            float32_precision=float32_precision,
            outer_autocast=(
                training.outer_autocast
                if rollout_section.outer_autocast is None
                else rollout_section.outer_autocast
            ),
            quantization=QuantizationPolicy.from_section(rollout_section.quantization),
        )
        prompt_encoders = rollout_section.prompt_encoders
        return cls(
            training=training,
            rollout=rollout,
            diffusion_math=section.diffusion_math.dtype if section.diffusion_math else "fp32",
            prompt_encoder_dtype=prompt_encoders.dtype if prompt_encoders else rollout.dtype,
        )


# ── Public YAML sections ──────────────────────────────────────────────────────


class QuantizationConfig(ConfigBase):
    """``quantization`` block of one role: a format plus an optional recipe."""

    format: str
    recipe: str | None = None

    @model_validator(mode="after")
    def _validate_format_and_recipe(self) -> QuantizationConfig:
        # The runtime policy owns the format/recipe vocabulary; validating it at
        # parse time keeps a bad block from surviving past parse_config.
        QuantizationPolicy(format=self.format, recipe=self.recipe)
        return self


class TrainingPrecisionConfig(ConfigBase):
    dtype: str
    outer_autocast: StrictBool = True
    quantization: QuantizationConfig | None = None

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalize_dtype(cls, value: Any) -> str:
        return _normalize_plain_dtype(value, path="precision.training.dtype")


class PromptEncodersPrecisionConfig(ConfigBase):
    dtype: str

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalize_dtype(cls, value: Any) -> str:
        return _normalize_plain_dtype(value, path="precision.rollout.prompt_encoders.dtype")


class RolloutPrecisionConfig(ConfigBase):
    """Rollout-role overrides; every omitted key inherits the training role."""

    dtype: str | None = None
    outer_autocast: StrictBool | None = None
    quantization: QuantizationConfig | None = None
    prompt_encoders: PromptEncodersPrecisionConfig | None = None

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalize_dtype(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalize_plain_dtype(value, path="precision.rollout.dtype")


class DiffusionMathPrecisionConfig(ConfigBase):
    dtype: str

    @field_validator("dtype", mode="before")
    @classmethod
    def _normalize_dtype(cls, value: Any) -> str:
        return _normalize_plain_dtype(value, path="precision.diffusion_math.dtype")


class PrecisionConfig(ConfigBase):
    """Top-level ``precision`` section."""

    float32_precision: Float32Precision
    training: TrainingPrecisionConfig
    rollout: RolloutPrecisionConfig | None = None
    diffusion_math: DiffusionMathPrecisionConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_scalar_block(cls, value: Any) -> Any:
        if isinstance(value, (str, bool)):
            raise ValueError(
                "scalar `precision` is no longer supported because it hides the "
                "difference between base dtype and quantization. Use "
                "`precision: {training: {dtype: bf16}}`.",
            )
        return value

    @field_validator("float32_precision", mode="before")
    @classmethod
    def _normalize_float32(cls, value: Any) -> Float32Precision:
        return _normalize_float32_precision(value)


__all__ = [
    "DiffusionMathPrecisionConfig",
    "Float32Precision",
    "PrecisionConfig",
    "PrecisionPolicy",
    "PromptEncodersPrecisionConfig",
    "QuantizationConfig",
    "QuantizationPolicy",
    "RolePrecision",
    "RolloutPrecisionConfig",
    "TrainingPrecisionConfig",
    "normalize_precision",
]
