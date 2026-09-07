"""Canonical model-family registry.

YAML owns experiment values and defaults. This composition boundary owns the
single family table shared by model construction, generation, and collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from vrl.config.model_schema import MODEL_MEMORY_SECTIONS
from vrl.models.families.names import (
    normalize_model_family,
    validate_model_family_aliases,
)
from vrl.models.families.semantics import PolicySemantics, Task, TrajectoryLayout

if TYPE_CHECKING:
    from vrl.config.precision import PrecisionPolicy
    from vrl.config.schema import RootConfig

# Import-path protocol value shared by registry dispatch and generation workers.
# Keeping it here avoids making the neutral family table import a runtime module.
GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR = (
    "vrl.generation.bindings.full_sequence_denoise.executor:GenericDiffusionBatchExecutor"
)
SHARED_MODEL_SECTION_CLS = "vrl.config.model_schema:ModelSection"
# Lazy public-sampling schema protocol values. Families share a path only when
# they expose the same real request vocabulary; no per-family empty class is
# needed merely to make registry entries look unique.
DENOISE_IMAGE_SAMPLING_SECTION_CLS = "vrl.config.sampling_schema:DenoiseImageSamplingSection"
TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS = (
    "vrl.config.sampling_schema:TextEncodedImageSamplingSection"
)
VIDEO_SAMPLING_SECTION_CLS = "vrl.config.sampling_schema:VideoSamplingSection"
TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS = (
    "vrl.config.sampling_schema:TextEncodedVideoSamplingSection"
)
# This is a family capability selection, not the global schema namespace:
# adding a future memory section must not silently grant it to every VAE family.
_VAE_DECODE_MEMORY_SECTIONS = frozenset({"vae_decode"})


class GenerationParkingProfile(Enum):
    """Family-level declaration of the worker's single parking backend.

    Binding, not advisory: a ``CUMEM`` family whose allocator is unavailable
    fails the policy build instead of degrading to ``MODEL``. Only the resolved
    residency mode may override the choice (pipeline CPU offload already owns
    the model's residency), and that is a mechanism conflict, not a fallback.
    """

    MODEL = "model"
    CUMEM = "cumem"


@dataclass(frozen=True, slots=True)
class GenerationRuntimeCapabilities:
    """Concrete executor/runtime behaviors, separate from model semantics."""

    supports_torch_compile: bool = False
    # Dotted path of the family's sequence-parallel installer
    # (``module:function`` taking (transformer, process_group)). Its presence
    # IS the multi-GPU engine capability — one source, no capability bool to
    # drift. The launch preflight gate and the rank program both read it.
    sequence_parallel_installer: str | None = None
    memory_parking: GenerationParkingProfile = GenerationParkingProfile.MODEL
    supported_model_memory_sections: frozenset[str] = field(default_factory=frozenset)

    @property
    def supports_multi_gpu_engine(self) -> bool:
        """Derived: a family is multi-GPU capable iff it ships an installer."""

        return self.sequence_parallel_installer is not None

    def __post_init__(self) -> None:
        unsupported = sorted(
            self.supported_model_memory_sections - frozenset(MODEL_MEMORY_SECTIONS),
        )
        if unsupported:
            raise ValueError(
                "generation runtime capabilities declare unknown model.memory "
                f"section(s): {', '.join(unsupported)}",
            )


@dataclass(frozen=True, slots=True)
class DenoiseFamilyBuild:
    """Declarative build recipe for a descriptor-driven denoise policy.

    A family whose runtime construction is pure data records its build inputs
    here. ``ModelFamilyEntry`` derives the shared resolver and worker builder
    from whether this descriptor or a ``TokenFamilyBuild`` is present. Only real
    family-specific replay assembly remains as an explicit override.
    """

    model_cls: str
    # Replay recipe; None marks a family whose registry entry points directly to
    # its hand-written replay builder (echo/cosmos3/anima).
    replay_cls: str | None = None
    transformer_classname: str | None = None
    scheduler_classname: str | None = None
    # Only families whose replay construction cannot use the generic descriptor
    # path declare an override (echo/cosmos3/anima).
    replay_runtime_builder: str | None = None
    # Non-diffusers families may own rollout assembly as well as replay
    # assembly. The registry still carries ``model_cls`` as the importable
    # family surface, while this hook owns checkpoint/component construction.
    rollout_runtime_builder: str | None = None
    # Driver-side normalization for deployment inputs that must be resolved
    # before ModelBuild is serialized into a Ray launch contract.
    model_build_normalizer: str | None = None
    # LoRA-only family: the generic builders fail loud BEFORE paying the
    # transformer load. The per-family WHY belongs in a comment on the entry
    # (and in the model's own apply_full_finetune error), not in runtime data.
    requires_lora: bool = False
    # Generation-only denoise families retain their concrete reason so trainer
    # construction can fail before importing an upstream runtime or weights.
    replay_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        generic_replay = self.replay_cls is not None and self.transformer_classname is not None
        partial_generic_replay = (self.replay_cls is None) != (self.transformer_classname is None)
        custom_replay = self.replay_runtime_builder is not None
        unavailable = self.replay_unavailable_reason is not None
        if partial_generic_replay:
            raise ValueError(
                "denoise family generic replay requires both replay_cls and transformer_classname",
            )
        if unavailable and not self.replay_unavailable_reason.strip():
            raise ValueError("replay_unavailable_reason must be non-empty when set")
        if sum((generic_replay, custom_replay, unavailable)) != 1:
            raise ValueError(
                "denoise family build must declare exactly one replay mode: "
                "replay_cls plus transformer_classname, replay_runtime_builder, "
                "or replay_unavailable_reason",
            )
        if not generic_replay and self.scheduler_classname is not None:
            raise ValueError(
                "scheduler_classname belongs to the generic replay builder and "
                "cannot accompany custom or unavailable replay",
            )


@dataclass(frozen=True, slots=True)
class TokenFamilyBuild:
    """Declarative model construction for a descriptor-driven token policy."""

    model_cls: str
    replay_cls: str
    config_cls: str
    config_builder: str
    default_model_path: str
    # Positive, mandatory capability: new token families must explicitly state
    # whether replay with use_lora=false leaves a trainable policy.
    supports_full_parameter_training: bool
    # NextStep's upstream loader accepts checkpointing only during construction;
    # ordinary token families do not project the trainer knob into ModelBuild.
    gradient_checkpointing_at_load: bool = False


@dataclass(frozen=True, slots=True)
class ModelFamilyEntry:
    """Declarative runtime binding for one canonical model family."""

    family: str
    task: Task
    policy_semantics: PolicySemantics
    executor_cls: str
    gatherer_cls: str
    # Lazy public-YAML schema boundary. This is deliberately distinct from a
    # TokenFamilyBuild.config_cls, which constructs the resolved model wrapper.
    model_section_cls: str
    # Lazy request-vocabulary boundary selected by model.family. This remains
    # separate from mutable runtime sampling state and wire request protocols.
    sampling_section_cls: str
    family_build: DenoiseFamilyBuild | TokenFamilyBuild
    runtime_capabilities: GenerationRuntimeCapabilities = field(
        default_factory=GenerationRuntimeCapabilities,
    )

    def __post_init__(self) -> None:
        denoise_build = isinstance(self.family_build, DenoiseFamilyBuild)
        if denoise_build != (self.policy_semantics.step_kind == "denoise"):
            raise ValueError(
                f"model family {self.family!r} policy semantics "
                f"{self.policy_semantics!r} does not match its family build",
            )
        if not self.gatherer_cls:
            raise ValueError(f"model family {self.family!r} requires a gatherer binding")

    @property
    def supports_policy_replay(self) -> bool:
        """Whether this entry declares a concrete trainer replay recipe."""

        if isinstance(self.family_build, TokenFamilyBuild):
            return True
        return (
            self.family_build.replay_cls is not None
            or self.family_build.replay_runtime_builder is not None
        )

    def validate_gpus_per_engine(self, gpus_per_engine: int) -> None:
        """Gate multi-GPU engines on this family's sequence-parallel install.

        Without one, N ranks would each redundantly compute the full batch —
        N times the GPU cost for zero benefit. The knob therefore either works
        or fails loud here, before any actor launches.
        """

        if gpus_per_engine <= 1:
            return
        if not self.runtime_capabilities.supports_multi_gpu_engine:
            raise ValueError(
                f"model family {self.family!r} does not support multi-GPU "
                "engines, but distributed.resources.rollout.gpus_per_engine="
                f"{gpus_per_engine}. Remove the key (single-GPU engines) or "
                "use a family whose runtime capabilities declare a "
                "sequence_parallel_installer.",
            )

    def validate_model_runtime_sections(
        self,
        *,
        executor_config: Any | None,
        memory_config: Any | None,
    ) -> None:
        """Reject model runtime blocks that this family cannot consume."""

        from vrl.utils.config import plain_mapping

        executor = (
            {}
            if executor_config is None
            else plain_mapping(executor_config, field_name="model.executor")
        )
        memory = (
            {}
            if memory_config is None
            else plain_mapping(memory_config, field_name="model.memory")
        )

        if executor and self.executor_cls != GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
            raise ValueError(
                f"model family {self.family!r} does not support model.executor",
            )

        unsupported_memory = sorted(
            set(memory) - self.runtime_capabilities.supported_model_memory_sections,
        )
        if unsupported_memory:
            raise ValueError(
                f"model family {self.family!r} does not support model.memory "
                f"section(s): {', '.join(unsupported_memory)}",
            )

    def executor_kwargs(self, root: RootConfig) -> dict[str, Any]:
        """Return this family's executor arguments from a validated root."""

        from vrl.utils.config import plain_mapping

        if root.model is None:
            raise ValueError("validated root is missing model configuration")
        executor_config = root.model.executor

        kwargs: dict[str, Any] = {}
        if (
            self.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
            and executor_config is not None
        ):
            kwargs.update(
                plain_mapping(
                    executor_config,
                    field_name="model.executor",
                ),
            )
        return kwargs

    def resolve_model_build(
        self,
        root: RootConfig,
        device: Any,
        *,
        precision: PrecisionPolicy,
        for_rollout: bool = True,
        precision_role: Literal["training", "rollout"] | None = None,
        parameter_dtype_override: Any | None = None,
    ) -> Any:
        """Project validated config into the model inputs owned by this family.

        This is the only config-to-``ModelBuild`` boundary. Family identity was
        already selected by ``get_model_family_entry`` and must match the typed
        root. Execution precision is resolved once by the caller and cannot be
        reinterpreted at this boundary.
        """

        from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions

        if root.model is None:
            raise ValueError("validated root is missing model configuration")
        configured_family = normalize_model_family(str(root.model.family))
        if configured_family != self.family:
            raise ValueError(
                f"model config family {configured_family!r} does not match "
                f"registry entry {self.family!r}",
            )

        model_config = root.model.model_dump(
            mode="python",
            exclude_unset=True,
        )
        model_path = model_config.pop("path", None)
        model_revision = model_config.pop("revision", None)
        model_config.pop("family", None)
        model_config.pop("executor", None)
        model_config.pop("memory", None)
        if isinstance(self.family_build, TokenFamilyBuild) and model_path is None:
            model_path = self.family_build.default_model_path
        if model_path is None or not str(model_path).strip():
            raise ValueError("config missing required field: model.path")

        sampling_config = (
            None
            if root.sampling is None
            else root.sampling.model_dump(
                mode="python",
                exclude_unset=True,
            )
        )
        resolved_precision_role = precision_role or ("rollout" if for_rollout else "training")
        if resolved_precision_role not in ("training", "rollout"):
            raise ValueError(
                f"precision_role must be 'training' or 'rollout'; got {resolved_precision_role!r}",
            )
        role_precision = getattr(precision, resolved_precision_role)
        role_parameter_dtype = role_precision.dtype
        parameter_dtype = (
            parameter_dtype_override
            if parameter_dtype_override is not None
            else role_parameter_dtype
        )
        rollout = None
        generation_memory = None
        if for_rollout:
            rollout = RolloutBuildOptions(
                prompt_encoder_dtype=precision.prompt_encoder_dtype,
            )
            model_memory = root.model.memory
            if model_memory is not None and "vae_decode" in model_memory.model_fields_set:
                from vrl.models.interfaces.generation_memory import (
                    GenerationMemoryPolicy,
                    VaeDecodeMemory,
                )

                vae_decode = model_memory.vae_decode
                generation_memory = GenerationMemoryPolicy(
                    vae_decode=VaeDecodeMemory(
                        **(
                            {}
                            if vae_decode is None
                            else vae_decode.model_dump(
                                mode="python",
                                exclude_none=True,
                            )
                        ),
                    ),
                )
        build = ModelBuild(
            model_name_or_path=str(model_path),
            revision=None if model_revision is None else str(model_revision),
            device=device,
            parameter_dtype=parameter_dtype,
            family=self.family,
            precision=role_precision,
            model_config=model_config,
            sampling_config=sampling_config,
            generation_memory=generation_memory,
            rollout=rollout,
            defer_trainable_device_move=(
                isinstance(self.family_build, DenoiseFamilyBuild)
                and self.family_build.replay_cls is not None
                and not for_rollout
                and root.distributed is not None
                and root.distributed.training is not None
                and root.distributed.training.strategy == "fsdp"
            ),
        )

        if (
            isinstance(self.family_build, TokenFamilyBuild)
            and self.family_build.gradient_checkpointing_at_load
        ):
            # NextStep's upstream loader accepts this only during construction.
            # Rollout inference never checkpoints, and its bool-only API cannot
            # represent the trainer's selective checkpointing policy.
            from vrl.trainers.activation_checkpointing import (
                resolve_gradient_checkpointing_mode,
            )

            checkpointing = resolve_gradient_checkpointing_mode(root)
            if checkpointing == "selective" and not for_rollout:
                raise ValueError(
                    "nextstep_1 replay does not support selective gradient "
                    "checkpointing; use actor.gradient_checkpointing=full or off",
                )
            if build.model_config is not None:
                build.model_config["gradient_checkpointing"] = (
                    not for_rollout and checkpointing == "full"
                )
        if (
            isinstance(self.family_build, DenoiseFamilyBuild)
            and self.family_build.model_build_normalizer is not None
        ):
            from vrl.utils.config import import_from_path

            build = import_from_path(self.family_build.model_build_normalizer)(build)
        return build

    def build_replay(self, build: Any) -> Any:
        """Construct trainer replay through this entry's registered builder."""

        if build.family != self.family:
            raise ValueError(
                f"replay build family {build.family!r} does not match entry {self.family!r}",
            )
        if not self.supports_policy_replay:
            raise RuntimeError(
                f"{self.family} is generation-only: {self.family_build.replay_unavailable_reason}",
            )

        if isinstance(self.family_build, DenoiseFamilyBuild):
            if self.family_build.replay_runtime_builder is not None:
                from vrl.utils.config import import_from_path

                return import_from_path(self.family_build.replay_runtime_builder)(build)
            from vrl.models.steps.denoise.build import build_family_replay_runtime_bundle

            return build_family_replay_runtime_bundle(build, entry=self)
        from vrl.models.steps.token.build import build_token_family_bundle

        return build_token_family_bundle(build, entry=self, replay=True)

    def build_rollout(self, build: Any) -> Any:
        """Construct a rollout bundle from this entry's resolved model build."""

        if build.family != self.family:
            raise ValueError(
                f"rollout build family {build.family!r} does not match entry {self.family!r}",
            )
        if isinstance(self.family_build, DenoiseFamilyBuild):
            if self.family_build.rollout_runtime_builder is not None:
                from vrl.utils.config import import_from_path

                return import_from_path(self.family_build.rollout_runtime_builder)(build)
            from vrl.models.steps.denoise.build import build_family_runtime_bundle

            return build_family_runtime_bundle(build, entry=self)
        from vrl.models.steps.token.build import build_token_family_bundle

        return build_token_family_bundle(build, entry=self, replay=False)

    def new_gatherer(self) -> Any:
        """Construct the explicitly bound driver-side gatherer lazily."""

        from vrl.utils.config import import_from_path

        return import_from_path(self.gatherer_cls)()


FAMILY_REGISTRY: dict[str, ModelFamilyEntry] = {}


def _register_model_family(entry: ModelFamilyEntry) -> ModelFamilyEntry:
    """Register one canonical model family."""

    if entry.family in FAMILY_REGISTRY:
        raise ValueError(f"duplicate model family registration: {entry.family!r}")
    FAMILY_REGISTRY[entry.family] = entry
    return entry


def _full_sequence_denoise_entry(
    *,
    family: str,
    task: str,
    model_section_cls: str,
    sampling_section_cls: str,
    executor_cls: str | None = None,
    build: DenoiseFamilyBuild,
    runtime_capabilities: GenerationRuntimeCapabilities | None = None,
    supported_model_memory_sections: frozenset[str] | None = None,
) -> ModelFamilyEntry:
    # Default dispatch: the shared generic executor. Families with real
    # per-batch logic pass their own executor_cls. Per-family executor config
    # (num_frames / max_sequence_length / ...) lives in the model yaml's
    # ``executor`` block, read wholesale at launch — not here.
    if executor_cls is None:
        executor_cls = GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
    if runtime_capabilities is not None and supported_model_memory_sections is not None:
        raise ValueError(
            "full-sequence denoise entry must declare memory support either "
            "inside runtime_capabilities or through supported_model_memory_sections",
        )
    if runtime_capabilities is None:
        if supported_model_memory_sections is None:
            supported_model_memory_sections = (
                _VAE_DECODE_MEMORY_SECTIONS
                if executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR
                else frozenset()
            )
        runtime_capabilities = GenerationRuntimeCapabilities(
            supports_torch_compile=True,
            memory_parking=GenerationParkingProfile.CUMEM,
            supported_model_memory_sections=supported_model_memory_sections,
        )
    return ModelFamilyEntry(
        family=family,
        task=task,
        policy_semantics=PolicySemantics(
            generation_regime="full_sequence",
            step_kind="denoise",
            action_distribution="continuous",
            trajectory_layout="denoise",
        ),
        executor_cls=executor_cls,
        gatherer_cls="vrl.generation.bindings.full_sequence_denoise.gather:DiffusionBatchGatherer",
        model_section_cls=model_section_cls,
        sampling_section_cls=sampling_section_cls,
        family_build=build,
        runtime_capabilities=runtime_capabilities,
    )


def _token_autoregressive_entry(
    *,
    family: str,
    action_distribution: Literal["categorical", "continuous"],
    model_section_cls: str,
    sampling_section_cls: str,
    executor_cls: str,
    build: TokenFamilyBuild,
    task: str = "ar_t2i",
    trajectory_layout: TrajectoryLayout = "token",
    gatherer_cls: str = "vrl.generation.bindings.token_autoregressive.executor:ARDiscreteBatchGatherer",
) -> ModelFamilyEntry:
    """Construct common wiring for current token-autoregressive policy variants."""

    return ModelFamilyEntry(
        family=family,
        task=task,
        policy_semantics=PolicySemantics(
            generation_regime="token_autoregressive",
            step_kind="token",
            action_distribution=action_distribution,
            trajectory_layout=trajectory_layout,
        ),
        executor_cls=executor_cls,
        gatherer_cls=gatherer_cls,
        model_section_cls=model_section_cls,
        sampling_section_cls=sampling_section_cls,
        family_build=build,
    )


def _chunk_autoregressive_denoise_entry(
    *,
    family: str,
    model_section_cls: str,
    sampling_section_cls: str,
    executor_cls: str,
    build: DenoiseFamilyBuild,
    task: str = "t2v",
    runtime_capabilities: GenerationRuntimeCapabilities | None = None,
) -> ModelFamilyEntry:
    """Construct the shared typed binding for causal temporal-batch denoising."""

    return ModelFamilyEntry(
        family=family,
        task=task,
        policy_semantics=PolicySemantics(
            generation_regime="chunk_autoregressive",
            step_kind="denoise",
            action_distribution="continuous",
            trajectory_layout="denoise",
        ),
        executor_cls=executor_cls,
        gatherer_cls=(
            "vrl.generation.bindings.chunk_autoregressive_denoise.gather:"
            "ChunkAutoregressiveDenoiseGatherer"
        ),
        model_section_cls=model_section_cls,
        sampling_section_cls=sampling_section_cls,
        family_build=build,
        runtime_capabilities=(
            runtime_capabilities
            if runtime_capabilities is not None
            else GenerationRuntimeCapabilities()
        ),
    )


_register_model_family(
    _full_sequence_denoise_entry(
        family="sd3_5",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sd3_5.model:SD3_5Model",
            replay_cls="vrl.models.families.sd3_5.model:SD3_5ReplayModel",
            transformer_classname="SD3Transformer2DModel",
        ),
        # MMDiT joint attention has a verified Ulysses install (numeric
        # N=2-vs-reference equivalence in tests/models/test_sequence_parallel.py);
        # declaring it here is what opens gpus_per_engine > 1.
        runtime_capabilities=GenerationRuntimeCapabilities(
            supports_torch_compile=True,
            memory_parking=GenerationParkingProfile.CUMEM,
            supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
            sequence_parallel_installer=(
                "vrl.models.sequence_parallel:install_sd3_sequence_parallel"
            ),
        ),
    ),
)

_register_model_family(
    _chunk_autoregressive_denoise_entry(
        family="causvid",
        model_section_cls="vrl.models.families.causvid.config:CausVidModelSection",
        sampling_section_cls=VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.causvid.runtime:CausVidBatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.causvid.model:CausVidModel",
            replay_runtime_builder=(
                "vrl.models.families.causvid.runtime:build_causvid_replay_runtime_bundle"
            ),
        ),
        runtime_capabilities=GenerationRuntimeCapabilities(
            supports_torch_compile=True,
            memory_parking=GenerationParkingProfile.CUMEM,
        ),
    ),
)

_register_model_family(
    _chunk_autoregressive_denoise_entry(
        family="magi_1",
        model_section_cls="vrl.models.families.magi_1.config:Magi1ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:MagiSamplingSection",
        executor_cls="vrl.models.families.magi_1.runtime:Magi1BatchExecutor",
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.magi_1.model:Magi1Model",
            rollout_runtime_builder=(
                "vrl.models.families.magi_1.runtime:build_magi_1_runtime_bundle"
            ),
            model_build_normalizer=(
                "vrl.models.families.magi_1.model:normalize_magi_1_model_build"
            ),
            replay_unavailable_reason=(
                "the official 4.5B runtime exposes final-video inference only and no "
                "replayable transition likelihood or autograd model"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="flux",
        task="t2i",
        model_section_cls="vrl.models.families.flux.config:FluxModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.flux.model:FluxModel",
            replay_cls="vrl.models.families.flux.model:FluxReplayModel",
            transformer_classname="FluxTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="qwen_image",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        # Descriptor-driven family: the generic functions in
        # vrl.models.steps.denoise.build read the recipe below, so qwen_image ships
        # no per-family builder/resolver functions.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.qwen_image.model:QwenImageModel",
            replay_cls="vrl.models.families.qwen_image.model:QwenImageReplayModel",
            transformer_classname="QwenImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="sana",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.sana.model:SanaModel",
            replay_cls="vrl.models.families.sana.model:SanaReplayModel",
            transformer_classname="SanaTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="lumina2",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.lumina2.model:Lumina2Model",
            replay_cls="vrl.models.families.lumina2.model:Lumina2ReplayModel",
            transformer_classname="Lumina2Transformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="hunyuan_video",
        task="t2v",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.hunyuan_video.model:HunyuanVideoModel",
            replay_cls="vrl.models.families.hunyuan_video.model:HunyuanVideoReplayModel",
            transformer_classname="HunyuanVideoTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="mochi",
        task="t2v",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.mochi.model:MochiModel",
            replay_cls="vrl.models.families.mochi.model:MochiReplayModel",
            transformer_classname="MochiTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="hunyuan_image",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=DENOISE_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.hunyuan_image.model:HunyuanImageModel",
            replay_cls="vrl.models.families.hunyuan_image.model:HunyuanImageReplayModel",
            transformer_classname="HunyuanImageTransformer2DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="pixart_sigma",
        task="t2i",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.pixart_sigma.model:PixArtSigmaModel",
            replay_cls="vrl.models.families.pixart_sigma.model:PixArtSigmaReplayModel",
            transformer_classname="PixArtTransformer2DModel",
            # Epsilon DDPM family (sde_type=ddim): load a DDIMScheduler so the
            # shipped beta config survives into prepare_replay, which rebuilds
            # the rollout's DDIM ladder via pixart_ddim_scheduler.
            scheduler_classname="DDIMScheduler",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="cogvideox",
        task="t2v",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cogvideox.model:CogVideoXModel",
            replay_cls="vrl.models.families.cogvideox.model:CogVideoXReplayModel",
            transformer_classname="CogVideoXTransformer3DModel",
            # v-prediction DDPM family: replay recomputes log-probs on the
            # same ladder the rollout sampled (sde_type=ddim).
            scheduler_classname="CogVideoXDDIMScheduler",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="wan_2_1",
        task="t2v",
        model_section_cls="vrl.models.families.wan_2_1.config:WanModelSection",
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        # The two wan entries carry their own per-variant recipes, so the
        # t2v/i2v resolution is decided here, once, by family selection. The
        # dual-stage transformer_2 late-load lives in the replay model's
        # prepare_replay, so replay is generic too.
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanT2VDiffusersModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanT2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            # Replay recomputes log-probs on the schedule the rollout sampled.
            scheduler_classname="UniPCMultistepScheduler",
            model_build_normalizer=(
                "vrl.models.families.wan_2_1.config:normalize_wan_model_build"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="wan_2_1_i2v",
        task="i2v",
        model_section_cls="vrl.models.families.wan_2_1.config:WanModelSection",
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.wan_2_1.runtime:Wan_2_1I2VBatchExecutor",
        supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.wan_2_1.model:WanI2VDiffusersModel",
            replay_cls="vrl.models.families.wan_2_1.model:WanI2VReplayModel",
            transformer_classname="WanTransformer3DModel",
            scheduler_classname="UniPCMultistepScheduler",
            model_build_normalizer=(
                "vrl.models.families.wan_2_1.config:normalize_wan_model_build"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="cosmos-predict2",
        task="v2w",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.cosmos.predict2.runtime:CosmosBatchExecutor",
        supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2Model",
            replay_cls="vrl.models.families.cosmos.predict2.model:CosmosPredict2ReplayModel",
            transformer_classname="CosmosTransformer3DModel",
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="cosmos-predict2.5",
        task="t2w",
        model_section_cls=(
            "vrl.models.families.cosmos.predict2_5.config:CosmosPredict25ModelSection"
        ),
        sampling_section_cls=TEXT_ENCODED_VIDEO_SAMPLING_SECTION_CLS,
        executor_cls=(
            "vrl.models.families.cosmos.predict2_5.runtime:CosmosPredict25BatchExecutor"
        ),
        supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
        build=DenoiseFamilyBuild(
            model_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25Model"),
            replay_cls=("vrl.models.families.cosmos.predict2_5.model:CosmosPredict25ReplayModel"),
            transformer_classname="CosmosTransformer3DModel",
            # Upstream ships UniPC; replay must recompute log-probs under the
            # same schedule the rollout sampled with.
            scheduler_classname="UniPCMultistepScheduler",
            # DiffusionNFT needs the trainable default + frozen previous
            # adapters, which only exist on the LoRA path.
            requires_lora=True,
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="minimax_h3",
        task="t2v",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls="vrl.config.sampling_schema:MiniMaxH3SamplingSection",
        executor_cls="vrl.models.families.minimax_h3.runtime:MiniMaxH3BatchExecutor",
        supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.minimax_h3.model:MiniMaxH3Model",
            replay_runtime_builder=(
                "vrl.models.families.minimax_h3.runtime:build_minimax_h3_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="cosmos3",
        task="t2v",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls=VIDEO_SAMPLING_SECTION_CLS,
        executor_cls="vrl.models.families.cosmos.cosmos3.runtime:Cosmos3BatchExecutor",
        supported_model_memory_sections=_VAE_DECODE_MEMORY_SECTIONS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.cosmos3.model:Cosmos3Model",
            replay_runtime_builder=(
                "vrl.models.families.cosmos.cosmos3.runtime:build_cosmos3_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="cosmos-predict2-anima",
        task="t2i",
        model_section_cls="vrl.models.families.cosmos.anima.config:CosmosAnimaModelSection",
        sampling_section_cls=TEXT_ENCODED_IMAGE_SAMPLING_SECTION_CLS,
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.cosmos.anima.model:AnimaModel",
            replay_runtime_builder=(
                "vrl.models.families.cosmos.anima.runtime:build_anima_replay_runtime_bundle"
            ),
        ),
    ),
)

_register_model_family(
    _full_sequence_denoise_entry(
        family="echo",
        task="t2v",
        model_section_cls="vrl.models.families.echo.config:EchoModelSection",
        sampling_section_cls="vrl.config.sampling_schema:EchoSamplingSection",
        executor_cls="vrl.models.families.echo.runtime:EchoBatchExecutor",
        supported_model_memory_sections=frozenset(),
        build=DenoiseFamilyBuild(
            model_cls="vrl.models.families.echo.model:EchoModel",
            replay_runtime_builder=(
                "vrl.models.families.echo.runtime:build_echo_replay_runtime_bundle"
            ),
        ),
    ),
)

_JANUS_PRO_BUILD = TokenFamilyBuild(
    model_cls="vrl.models.families.janus_pro.model:JanusProModel",
    replay_cls="vrl.models.families.janus_pro.model:JanusProReplayModel",
    config_cls="vrl.models.families.janus_pro.config:JanusProConfig",
    config_builder="vrl.models.families.janus_pro.runtime:janus_config_from_build",
    default_model_path="deepseek-ai/Janus-Pro-1B",
    supports_full_parameter_training=False,
)

_register_model_family(
    _token_autoregressive_entry(
        family="janus_pro",
        action_distribution="categorical",
        model_section_cls="vrl.models.families.janus_pro.config:JanusProModelSection",
        sampling_section_cls="vrl.config.sampling_schema:JanusProSamplingSection",
        executor_cls="vrl.models.families.janus_pro.runtime:JanusProBatchExecutor",
        build=_JANUS_PRO_BUILD,
    ),
)

_register_model_family(
    _token_autoregressive_entry(
        family="janus_pro_r1",
        action_distribution="categorical",
        model_section_cls="vrl.models.families.janus_pro.config:JanusProModelSection",
        sampling_section_cls="vrl.config.sampling_schema:JanusProR1SamplingSection",
        task="ar_t2i_r1",
        executor_cls="vrl.models.families.janus_pro.runtime:JanusProR1BatchExecutor",
        gatherer_cls="vrl.models.families.janus_pro.runtime:JanusProR1GenerationBatchGatherer",
        build=_JANUS_PRO_BUILD,
        trajectory_layout="multisegment_token",
    ),
)

_register_model_family(
    _token_autoregressive_entry(
        family="nextstep_1",
        action_distribution="continuous",
        model_section_cls="vrl.models.families.nextstep_1.config:NextStep1ModelSection",
        sampling_section_cls="vrl.config.sampling_schema:NextStepSamplingSection",
        executor_cls="vrl.models.families.nextstep_1.runtime:NextStep1BatchExecutor",
        gatherer_cls="vrl.models.families.nextstep_1.runtime:NextStep1GenerationBatchGatherer",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.nextstep_1.model:NextStep1Model",
            replay_cls="vrl.models.families.nextstep_1.model:NextStep1ReplayModel",
            config_cls="vrl.models.families.nextstep_1.config:NextStep1Config",
            config_builder=("vrl.models.families.nextstep_1.runtime:nextstep_config_from_build"),
            default_model_path="stepfun-ai/NextStep-1.1",
            supports_full_parameter_training=True,
            gradient_checkpointing_at_load=True,
        ),
    ),
)

_register_model_family(
    _token_autoregressive_entry(
        family="emu3",
        action_distribution="categorical",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls="vrl.config.sampling_schema:Emu3SamplingSection",
        executor_cls="vrl.models.families.emu3.runtime:Emu3BatchExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.emu3.model:Emu3Model",
            replay_cls="vrl.models.families.emu3.model:Emu3ReplayModel",
            config_cls="vrl.models.families.emu3.config:Emu3Config",
            config_builder="vrl.models.families.emu3.runtime:emu3_config_from_build",
            default_model_path="BAAI/Emu3-Gen-hf",
            supports_full_parameter_training=False,
        ),
    ),
)

_register_model_family(
    _token_autoregressive_entry(
        family="glm_image",
        action_distribution="categorical",
        model_section_cls=SHARED_MODEL_SECTION_CLS,
        sampling_section_cls="vrl.config.sampling_schema:GlmImageSamplingSection",
        executor_cls="vrl.models.families.glm_image.runtime:GlmImageBatchExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.glm_image.model:GlmImageModel",
            replay_cls="vrl.models.families.glm_image.model:GlmImageReplayModel",
            config_cls="vrl.models.families.glm_image.config:GlmImageConfig",
            config_builder="vrl.models.families.glm_image.runtime:glm_image_config_from_build",
            default_model_path="zai-org/GLM-Image",
            supports_full_parameter_training=False,
        ),
    ),
)

_register_model_family(
    _token_autoregressive_entry(
        family="llamagen",
        action_distribution="categorical",
        model_section_cls="vrl.models.families.llamagen.config:LlamaGenModelSection",
        sampling_section_cls="vrl.config.sampling_schema:LlamaGenSamplingSection",
        executor_cls="vrl.models.families.llamagen.runtime:LlamaGenBatchExecutor",
        build=TokenFamilyBuild(
            model_cls="vrl.models.families.llamagen.model:LlamaGenModel",
            replay_cls="vrl.models.families.llamagen.model:LlamaGenReplayModel",
            config_cls="vrl.models.families.llamagen.config:LlamaGenConfig",
            config_builder="vrl.models.families.llamagen.runtime:llamagen_config_from_build",
            default_model_path="peizesun/llamagen_t2i",
            supports_full_parameter_training=False,
        ),
    ),
)


validate_model_family_aliases(FAMILY_REGISTRY)


def get_model_family_entry(family: str) -> ModelFamilyEntry:
    """Return the canonical model-family entry for ``family``."""

    normalized = normalize_model_family(family)
    try:
        return FAMILY_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model family: {family!r}; registered={sorted(FAMILY_REGISTRY)}",
        ) from exc


__all__ = [
    "FAMILY_REGISTRY",
    "GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR",
    "DenoiseFamilyBuild",
    "GenerationParkingProfile",
    "GenerationRuntimeCapabilities",
    "ModelFamilyEntry",
    "PolicySemantics",
    "TokenFamilyBuild",
    "get_model_family_entry",
]
