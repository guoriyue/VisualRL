"""Launch gates for a training config (validation tier 3).

``require_training_config`` is the one entry every training launch runs:
``parse_config`` (tier 1 shapes + tier 2 cross-section rules, see
``vrl/config/rules.py``), the precision policy, then ``TRAINING_GATES`` in
order. A gate is a check that tier 2 cannot afford: it needs the resolved
precision policy, a runtime module (the compile matrix reads the build-role
resolver and the checkpointing resolver) or the filesystem (the production
data gate reads manifests). Eval and perf tools call ``parse_config`` alone,
so a gate never taxes them.

Adding a gate: write ``def gate_<name>(root, precision) -> None`` that raises
``ValueError`` naming the offending keys, and append it to ``TRAINING_GATES``.
A check with no such dependency is a tier 2 rule instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig

from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig, parse_config

TrainingGate = Callable[[RootConfig, PrecisionPolicy], None]


def require_training_config(cfg: DictConfig) -> tuple[RootConfig, PrecisionPolicy]:
    """Validate config once and return the typed config plus its precision policy.

    Returns the pair rather than a wrapper struct: the precision policy is a pure
    derivation of ``root``, and it is returned only so the caller does not resolve
    it a second time (asserted by tests/config/test_builders.py).
    """

    root = parse_config(cfg)
    precision = PrecisionPolicy.from_section(root.precision)
    for gate in TRAINING_GATES:
        gate(root, precision)
    return root, precision


# ---- torch.compile compatibility matrix ---------------------------------------


@dataclass(frozen=True, slots=True)
class CompileConflict:
    """One feature that cannot coexist with the requested torch.compile."""

    # The config feature that collides, stable for programmatic filtering
    # (the runtime checkpointing pass re-checks only its own entry).
    feature: str
    message: str


def compile_conflicts(root: RootConfig) -> tuple[CompileConflict, ...]:
    """Every feature this config turns on that cannot coexist with torch.compile.

    ONE home for the compile compatibility matrix. Each of these was discovered
    separately and used to be enforced somewhere different — grad-checkpointing
    in the trainer, FSDP2 in the strategy builder, sequence parallelism nowhere
    at all — so adding the fifth meant first finding the other four. They are
    all decided by config keys, so config load is where they belong: a
    combination that can never run should fail before a GPU is touched.

    Returns one entry per conflict (empty when compatible). The blockwise-fp8
    conflict is deliberately NOT here: it is caught in ``vrl.models.loader``
    where the resolved quantization recipe lives, and this function is given
    only the parsed config.
    """

    compile_block = root.model.torch_compile if root.model is not None else None
    if compile_block is None or not bool(compile_block.enable):
        return ()

    # Each conflict below binds one build role, so the matrix honors
    # ``model.torch_compile.scope``: trainer constraints cannot veto a
    # rollout-only compile, and rollout constraints cannot veto a replay-only
    # one. The (block, role) decision is owned by the typed build contract.
    from vrl.models.interfaces.runtime import torch_compile_for_role

    compiles_replay = torch_compile_for_role(compile_block, "replay") is not None
    compiles_rollout = torch_compile_for_role(compile_block, "rollout") is not None

    conflicts: list[CompileConflict] = []

    # torch.compile traces torch.utils.checkpoint into an InternalTorchDynamoError
    # (measured for full and selective alike), and inductor's min-cut partitioner
    # already does automatic selective recompute.
    from vrl.trainers.activation_checkpointing import resolve_gradient_checkpointing_mode

    checkpointing = resolve_gradient_checkpointing_mode(root)
    if compiles_replay and checkpointing != "off":
        conflicts.append(
            CompileConflict(
                "gradient_checkpointing",
                f"actor.gradient_checkpointing={checkpointing!r}: torch.compile traces "
                "torch.utils.checkpoint into an InternalTorchDynamoError, and its "
                "min-cut partitioner already does automatic selective recompute. Pick "
                "one — compile alone (preferred when it fits memory), eager + "
                "checkpointing, or model.torch_compile.scope=rollout to keep the "
                "trainer eager while the rollout policy compiles.",
            )
        )

    distributed = root.distributed
    training = None if distributed is None else distributed.training
    strategy = "single_process" if training is None else str(training.strategy)
    # Inductor graph capture is unsound with FSDP2's reshard-after-forward
    # all-gathers. Previously only caught when the strategy was built, which is
    # after config load, so a bad recipe surfaced later than it needed to.
    if compiles_replay and strategy == "fsdp":
        conflicts.append(
            CompileConflict(
                "fsdp",
                "distributed.training.strategy=fsdp: torch.compile (inductor graph "
                "capture) is unsound with FSDP2 fully_shard's reshard-after-forward "
                "all-gathers. model.torch_compile.scope=rollout keeps the FSDP2 "
                "replay policy eager while the rollout policy compiles.",
            )
        )

    # Sequence parallelism is installed by the rollout WORKER, after the family
    # builder has already compiled the policy core: the installer swaps every
    # attention processor and registers forward hooks on the first/last block,
    # mutating a module inductor has already traced. sd3_5 declares BOTH
    # supports_torch_compile and a sequence_parallel_installer, so this is
    # reachable from config -- and it had no gate at all before.
    resources = None if distributed is None else distributed.resources
    gpus_per_engine = 1 if resources is None else int(resources.rollout.gpus_per_engine)
    if compiles_rollout and gpus_per_engine > 1:
        conflicts.append(
            CompileConflict(
                "sequence_parallel",
                f"distributed.resources.rollout.gpus_per_engine={gpus_per_engine}: "
                "sequence parallelism installs attention processors and forward hooks "
                "on the policy core AFTER the model is built and compiled, mutating "
                "the module torch.compile already traced.",
            )
        )

    return tuple(conflicts)


def validate_compile_compatible(root: RootConfig) -> None:
    """Refuse a config that enables torch.compile beside an incompatible feature."""

    conflicts = compile_conflicts(root)
    if not conflicts:
        return
    joined = "\n  - ".join(conflict.message for conflict in conflicts)
    raise ValueError(
        f"model.torch_compile.enable=true cannot combine with:\n  - {joined}",
    )


# ---- rollout drift ------------------------------------------------------------


def validate_guarded_rollout_drift(root: RootConfig, precision: PrecisionPolicy) -> None:
    """Refuse a rollout approximation that no drift correction will cover.

    Quantization needs no check here: it changes the rollout precision label, so
    ``stages_match`` goes False and the trainer already installs TIS correction
    plus a drift guard whose default ``mode="auto"`` resolves to ``"fail"``.

    A request-scoped approximation is the uncovered case. TeaCache reuses a
    cached ``noise_pred`` on skipped denoise steps, so the collection-time
    log-prob stops matching the trainer's exact replay forward -- while BOTH
    roles keep the same precision label, leaving every automatic correction off.
    Silently, the run would train on uncorrected off-policy gradients and still
    report convergence.
    """

    from vrl.nn.optimization import unguarded_drift_sources

    sampling = root.sampling
    sources = unguarded_drift_sources(
        sampling.model_dump(mode="python", exclude_none=True) if sampling is not None else None,
        precision,
    )
    if not sources:
        return
    # The same escape hatch the precision-split path honors: an explicit expert
    # block means the user has chosen the correction policy deliberately.
    explicit = set() if root.trainer is None else root.trainer.model_fields_set
    if "precision_drift_guard" in explicit or "precision_correction" in explicit:
        return
    raise ValueError(
        f"sampling enables {', '.join(sources)}, which makes the rollout log-probs "
        "diverge from the trainer's exact replay forward, but rollout and training "
        "precision are identical so no drift guard or importance-sampling "
        "correction is armed. Set an explicit trainer.precision_correction / "
        "trainer.precision_drift_guard for this run, or disable the optimization.",
    )


# ---- the gate registry --------------------------------------------------------


def gate_compile_compatible(root: RootConfig, precision: PrecisionPolicy) -> None:
    # Checked at config load — where the all-experiments test sees it — because
    # a model-layer torch_compile.enable=true default can silently flip compile
    # on underneath a recipe that needs checkpointing, FSDP, or a multi-rank engine.
    del precision
    validate_compile_compatible(root)


def gate_production(root: RootConfig, precision: PrecisionPolicy) -> None:
    # Reward-owned contracts plus the data layer's provenance check, for every
    # ``production.<reward>.enabled`` entry (vrl/config/production.py).
    del precision
    from vrl.config.production import validate_production_gates

    validate_production_gates(root)


TRAINING_GATES: tuple[TrainingGate, ...] = (
    gate_compile_compatible,
    validate_guarded_rollout_drift,
    gate_production,
)


__all__ = [
    "TRAINING_GATES",
    "CompileConflict",
    "TrainingGate",
    "compile_conflicts",
    "require_training_config",
    "validate_compile_compatible",
    "validate_guarded_rollout_drift",
]
