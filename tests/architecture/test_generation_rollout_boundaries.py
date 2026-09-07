"""Architecture checks for generation, rollout, and Ray package boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VRL_ROOT = ROOT / "vrl"
_GENERATION_MODEL_IMPORT_FLOOR = (
    "vrl.models.checkpoint_identity",
    "vrl.models.dtypes",
    "vrl.models.families.registry",
    "vrl.models.interfaces",
    "vrl.models.loader",
)


def test_generation_layer_does_not_import_rollout_or_training_layers() -> None:
    """``vrl.generation`` must not import the algorithm, reward, rollout, script or trainer
    layers: generation is the bottom of the stack.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "generation",
        forbidden=(
            "vrl.algorithms",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.scripts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_rewards_layer_does_not_import_generation_rollout_or_training_layers() -> None:
    """Rewards stay independently reusable below rollout orchestration."""
    violations = _forbidden_imports(
        VRL_ROOT / "rewards",
        forbidden=(
            "vrl.generation",
            "vrl.rollouts",
            "vrl.scripts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_generation_model_imports_stay_on_public_floor() -> None:
    """Generation may use model contracts and the registry, not family implementations."""
    violations: list[tuple[Path, str]] = []
    for path in _python_files(VRL_ROOT / "generation"):
        for target in _imports(path):
            if _is_generation_model_import_violation(target):
                violations.append((path.relative_to(ROOT), target))
    assert not violations, _format_violations(violations)


def test_import_scanner_preserves_from_import_targets(tmp_path: Path) -> None:
    """Imported aliases must remain visible to architecture boundary checks."""
    path = tmp_path / "vrl" / "generation" / "execution" / "probe.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
import vrl.models.loader as model_loader
from vrl import models
from vrl.models import checkpoint_identity, families
from vrl.models.families import registry as family_registry
from vrl.models.interfaces import RuntimeModel
from vrl.models.interfaces_bad import RuntimeModel as BadRuntimeModel
from ...models import steps

def lazy_import():
    from vrl.models.interfaces.runtime import ModelBuild
""",
        encoding="utf-8",
    )

    targets = set(_imports(path, root=tmp_path))
    assert targets == {
        "vrl.models",
        "vrl.models.checkpoint_identity",
        "vrl.models.families",
        "vrl.models.families.registry",
        "vrl.models.interfaces.RuntimeModel",
        "vrl.models.interfaces_bad.RuntimeModel",
        "vrl.models.interfaces.runtime.ModelBuild",
        "vrl.models.loader",
        "vrl.models.steps",
    }
    assert {target for target in targets if _is_generation_model_import_violation(target)} == {
        "vrl.models",
        "vrl.models.families",
        "vrl.models.interfaces_bad.RuntimeModel",
        "vrl.models.steps",
    }


def test_ray_working_dir_keeps_pinned_chunk_runtime_inputs() -> None:
    """Exercise Ray's real ignore traversal for required vendored runtime files."""

    import logging

    from ray._private import ray_constants
    from ray._private.runtime_env import packaging

    required = {
        "third_party/CausVid/causvid/models/wan/causal_model.py",
        "third_party/MAGI-1/example/4.5B/4.5B_base_config.json",
        "third_party/MAGI-1/example/assets/special_tokens.npz",
        "third_party/MAGI-1/inference/pipeline/entry.py",
    }
    excluded = {
        "third_party/CausVid/.git/HEAD",
        "third_party/MAGI-1/.git/HEAD",
        "third_party/DynamicEval/docs/static/videos/prompt_id_024_compressed.mp4",
        "third_party/VMBench/Grounded-SAM-2/assets/tracking_car.mp4",
    }
    targets = required | excluded
    visited: set[str] = set()

    def record(path: Path) -> None:
        relative = path.relative_to(ROOT).as_posix()
        if relative in targets:
            visited.add(relative)

    default_excludes = packaging._get_excludes(
        ROOT,
        ray_constants.get_runtime_env_default_excludes(),
    )
    packaging._dir_travel(
        ROOT,
        [default_excludes],
        record,
        include_gitignore=True,
        logger=logging.getLogger("test-ray-package-contents"),
    )

    assert required <= visited
    assert not excluded & visited


def test_ray_ignore_excludes_submodule_git_pointer_files(tmp_path: Path) -> None:
    """A normal submodule's .git is a file, unlike local standalone clones."""

    import logging

    from ray._private.runtime_env import packaging

    (tmp_path / ".rayignore").write_text(
        (ROOT / ".rayignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_root = tmp_path / "third_party" / "CausVid"
    source_root.mkdir(parents=True)
    (source_root / ".git").write_text(
        "gitdir: ../../../.git/modules/third_party/CausVid\n",
        encoding="utf-8",
    )
    runtime_file = source_root / "causvid" / "models" / "runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("# runtime\n", encoding="utf-8")
    visited: set[Path] = set()

    packaging._dir_travel(
        tmp_path,
        [],
        visited.add,
        include_gitignore=True,
        logger=logging.getLogger("test-ray-submodule-pointer"),
    )

    assert source_root / ".git" not in visited
    assert runtime_file in visited


def test_trajectory_layer_stays_family_neutral() -> None:
    """``vrl.trajectory`` stays family- and runtime-neutral: no imports from the generation
    bindings, the Ray runtime, rewards, rollouts, algorithms or trainers.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "trajectory",
        forbidden=(
            "vrl.algorithms",
            "vrl.generation.bindings.chunk_autoregressive_denoise",
            "vrl.generation.bindings.token_autoregressive",
            "vrl.generation.bindings.full_sequence_denoise",
            "vrl.generation.ray",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_model_family_registry_stays_import_light() -> None:
    """The model-family registry must stay importable during config parsing.

    Every module-level import must be stdlib, one of the three lightweight
    registry modules, or the torch-free ``vrl.config`` schema layer. Edges into
    concrete family implementations, trainers, generation, and utils remain
    function-level lazy so config parsing does not pay for the runtime stack.

    Walk ``tree.body`` only — NOT ``ast.walk`` — so the intentional function-level
    lazy imports (e.g. registry.py's gradient-checkpointing resolver) are not swept
    in and false-failed. This turns the lazy-import convention into a mechanical gate.
    """
    registry_modules = (
        VRL_ROOT / "models" / "families" / "names.py",
        VRL_ROOT / "models" / "families" / "registry.py",
        VRL_ROOT / "models" / "families" / "semantics.py",
    )
    allowed_registry_imports = frozenset(
        f"vrl.models.families.{path.stem}" for path in registry_modules
    )
    violations: list[tuple[Path, str]] = []
    for path in registry_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # module-level statements only
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if not module.startswith("vrl."):
                    continue  # stdlib / third-party import-light deps are unrestricted
                if module in allowed_registry_imports:
                    continue
                if module == "vrl.config" or module.startswith("vrl.config."):
                    continue  # torch-free config-schema layer (capability SoT)
                violations.append((path.relative_to(ROOT), module))
    assert not violations, _format_violations(violations)


def test_shared_ray_substrate_stays_domain_neutral() -> None:
    """``vrl.ray`` is shared substrate and must not know generation, rewards, rollouts or
    trainers.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "ray",
        forbidden=(
            "vrl.generation",
            "vrl.rewards",
            "vrl.rollouts",
            "vrl.trainers",
        ),
    )
    assert not violations, _format_violations(violations)


def test_reward_models_live_under_models() -> None:
    """Model-backed rewards own model modules; pure functions do not."""
    models_root = VRL_ROOT / "rewards" / "models"
    present = _module_filenames(models_root)
    model_modules = _registered_model_reward_modules()
    assert model_modules <= present
    # Only scaffolding may live alongside the per-reward modules.
    scaffolding = {"__init__.py", "base.py", "hub.py", "media.py", "qwen_vl_judge.py"}
    extras = present - model_modules - scaffolding
    assert not extras, f"unexpected modules under rewards/models/: {extras}"


def test_reward_function_implementations_live_under_functions() -> None:
    """Every registered reward function module lives under ``vrl/rewards/functions``, the package
    root keeps its fixed scaffolding files, and no unregistered module hides in ``functions/``.
    """
    rewards_root = VRL_ROOT / "rewards"
    required_root = {
        "__init__.py",
        "artifacts.py",
        "base.py",
        "inference.py",
        "protocols.py",
        "runtime.py",
        "types.py",
    }
    assert required_root <= _module_filenames(rewards_root)

    functions = _module_filenames(rewards_root / "functions")
    assert _registered_reward_modules() <= functions
    scaffolding = {"__init__.py", "base.py", "registry.py"}
    extras = functions - _registered_reward_modules() - scaffolding
    assert not extras, f"unexpected modules under rewards/functions/: {extras}"


def test_generation_execution_core_stays_ray_neutral() -> None:
    """``vrl.generation.execution`` is the Ray-free core: neither ``vrl.generation.ray`` nor
    ``vrl.ray`` may be imported there.
    """
    violations = _forbidden_imports(
        VRL_ROOT / "generation" / "execution",
        forbidden=("vrl.generation.ray", "vrl.ray"),
    )
    assert not violations, _format_violations(violations)


def test_chunk_executor_base_stays_family_registry_neutral() -> None:
    """The composition root injects gatherers; the shared base never re-resolves them."""
    path = VRL_ROOT / "generation" / "execution" / "executor_base.py"
    violations = [
        (path.relative_to(ROOT), target)
        for target in _imports(path)
        if _is_module_or_child(target, "vrl.models.families.registry")
    ]
    assert not violations, _format_violations(violations)


def _forbidden_imports(
    root: Path,
    *,
    forbidden: tuple[str, ...],
    allow_path_prefixes: tuple[Path, ...] = (),
) -> list[tuple[Path, str]]:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(root):
        rel = path.relative_to(ROOT)
        if any(_is_relative_to(rel, prefix) for prefix in allow_path_prefixes):
            continue
        for module in _imports(path):
            if any(_is_module_or_child(module, item) for item in forbidden):
                violations.append((rel, module))
    return violations


def _imports(path: Path, *, root: Path = ROOT) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package_parts = path.relative_to(root).with_suffix("").parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent_count = len(package_parts) - node.level + 1
                if parent_count < 0:
                    raise ValueError(f"{path}: relative import escapes its package")
                base_parts = package_parts[:parent_count]
                if node.module:
                    base_parts += tuple(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""

            # Preserve the imported name so ``from vrl import models`` cannot
            # bypass a boundary that watches ``vrl.models``.
            for alias in node.names:
                if alias.name == "*":
                    if base:
                        yield base
                elif base:
                    yield f"{base}.{alias.name}"
                else:
                    yield alias.name


def _python_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _module_filenames(root: Path) -> set[str]:
    return {path.name for path in root.glob("*.py") if "__pycache__" not in path.parts}


def _registered_reward_modules() -> set[str]:
    """Reward-impl filenames derived from the registry, the single source of truth.

    Each registered reward ``<name>`` owns a ``<name>.py`` module, so the
    expected module set is the registry keys — never a hand-typed ``ls``.
    Registration is lazy (``_register_builtins`` runs inside ``from_dict``),
    so trigger it once with an empty score dict before reading the keys.
    """
    from vrl.rewards.functions.registry import _REWARD_REGISTRY, MultiReward

    MultiReward.from_dict({}, device="cpu")  # populate _REWARD_REGISTRY
    return {f"{name}.py" for name in _REWARD_REGISTRY}


def _registered_model_reward_modules() -> set[str]:
    """Derive model-module owners from the registered reward class hierarchy."""
    from vrl.rewards.base import InferenceRewardFunction
    from vrl.rewards.functions.registry import _REWARD_REGISTRY, MultiReward

    MultiReward.from_dict({}, device="cpu")  # populate _REWARD_REGISTRY
    return {
        f"{name}.py"
        for name, reward_cls in _REWARD_REGISTRY.items()
        if issubclass(reward_cls, InferenceRewardFunction)
    }


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _is_module_or_child(module: str, parent: str) -> bool:
    return module == parent or module.startswith(f"{parent}.")


def _is_generation_model_import_violation(target: str) -> bool:
    return _is_module_or_child(target, "vrl.models") and not any(
        _is_module_or_child(target, allowed) for allowed in _GENERATION_MODEL_IMPORT_FLOOR
    )


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path}: imports {module}" for path, module in violations)
