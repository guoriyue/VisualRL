"""Opt-in RL integration tests against locally cached real model checkpoints."""

from __future__ import annotations

import asyncio
import gc
import inspect
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from tests import ci_envs
from vrl.config.builders import build_configs
from vrl.config.loading import load_config
from vrl.config.precision import PrecisionPolicy
from vrl.config.schema import RootConfig, parse_config
from vrl.generation import GenerationOutput, GenerationRequest
from vrl.generation.execution.planner import EnginePlan
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.registry import ModelFamilyEntry, get_model_family_entry
from vrl.ray.resources import ResolvedDistributedResources
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import build_rollout_collector
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.run import resolve_reward_inputs
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator,
    build_reward_function,
)
from vrl.trainers.data.prompts import PromptExample
from vrl.trainers.diagnostics import trainable_state_digest
from vrl.trainers.online import OnlineTrainer
from vrl.utils.config import import_from_path

RUN_REAL_ENV = "WM_RUN_REAL_MODEL_TESTS"
CASE_FILTER_ENV = "WM_REAL_MODEL_RL_CASES"


@dataclass(frozen=True, slots=True)
class CheckpointField:
    """A checkpoint the case needs on disk.

    ``cfg_path`` is the config key that receives the resolved path. ``None``
    means the runtime finds the checkpoint on its own (a base-model name baked
    into another checkpoint's config); the field then only gates the case on
    the cached snapshot so an offline run skips instead of erroring mid-load.
    """

    cfg_path: str | None
    repo_id: str
    required_files: tuple[str, ...] = ()
    allow_file: bool = False


@dataclass(frozen=True, slots=True)
class RealCheckpointCase:
    case_id: str
    config: str
    family: str
    prompt: str
    checkpoints: tuple[CheckpointField, ...]
    overrides: tuple[str, ...]
    min_cuda_memory_gib: float
    reference_image_cfg_path: str | None = None
    use_config_reward: bool = False
    # ``use_config_reward`` cases score through the config's reward stack. The
    # default swaps the reward MODEL for a tensor-mean factory so the case
    # proves reward transport (artifacts, manifest, debug rows) without the
    # real weights; ``None`` keeps the config's real model factory.
    reward_model_factory: str | None = "tests.e2e.test_real_checkpoint_rl:build_tensor_mean_model"
    synthetic_replay_rollout: bool = False
    # Sample->replay log-prob parity bound (GRPO ratio==1 invariant). With
    # ppo_epochs=1 the optimizer steps after the whole timestep loop, so every
    # replayed log-prob is computed against unchanged weights even at lr>0.
    # NFT cases report 0.0 (no evaluator log-probs) and pass trivially.
    logprob_parity_tol: float = 5e-3


_COSMOS_PREDICT2_KLING_TRANSPORT = RealCheckpointCase(
    case_id="cosmos_predict2",
    config="experiment/cosmos_predict2/online_grpo_kling_video_reward",
    family="cosmos-predict2",
    prompt="A quiet street with a clear RL sign",
    checkpoints=(
        CheckpointField(
            cfg_path="model.path",
            repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
            required_files=(
                "model_index.json",
                "transformer/config.json",
                "transformer/diffusion_pytorch_model.safetensors",
                "text_encoder/config.json",
                "text_encoder/model.safetensors.index.json",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "scheduler/scheduler_config.json",
                "tokenizer/tokenizer.json",
            ),
        ),
    ),
    overrides=(
        "model.torch_compile.enable=false",
        # The preset trains full-parameter (sized for multi-GPU, 2026-06-10);
        # on one card the 2B fp32 masters + Adam moments beside the 11B text
        # encoder do not fit 32 GiB. LoRA (the repo's own predict2 LoRA
        # experiments' targets) keeps this a single-card case while still
        # updating real trainable weights.
        "model.use_lora=true",
        "model.lora.rank=8",
        "model.lora.alpha=8",
        "model.lora.target_modules=[to_q,to_k,to_v,to_out.0]",
        "algorithm.kl_coef=0.0",
        "algorithm.kl_reward_coef=0.0",
        "actor.drop_zero_advantage=false",
        "rollout.n_samples_per_prompt=2",
        "rollout.prompts_per_batch=1",
        "rollout.samples_per_generation_batch=1",
        "rollout.noise_level=0.7",
        "rollout.sde.type=cps",
        "rollout.sde.window_size=0",
        "rollout.sde.window_range=[0,1]",
        "sampling.num_steps=2",
        "sampling.guidance_scale=1.0",
        "sampling.height=128",
        "sampling.width=128",
        "sampling.num_frames=5",
        "sampling.fps=16",
    ),
    # Measured 2026-09-06 on one 32 GiB card: 16.4 GiB peak (LoRA, 128x128x5f,
    # text encoder parked before the reward model loads).
    min_cuda_memory_gib=24.0,
    reference_image_cfg_path="data.preprocessing.reference_image",
    use_config_reward=True,
)
# Same run, real Kling VideoReward: the 2B Qwen2-VL judge scores the rollout
# clips instead of the tensor-mean stand-in. This is the only lane where the
# code after the factory string in vrl/rewards/models/kling_video_reward.py
# runs on real weights. ``Qwen/Qwen2-VL-2B-Instruct`` is the base model named
# inside the VideoReward snapshot's model_config.json; no config key can
# redirect it, so it is gated (cfg_path=None) rather than overridden.
_COSMOS_PREDICT2_KLING_REAL_REWARD = replace(
    _COSMOS_PREDICT2_KLING_TRANSPORT,
    case_id="cosmos_predict2_kling_real_reward",
    checkpoints=(
        *_COSMOS_PREDICT2_KLING_TRANSPORT.checkpoints,
        CheckpointField(
            cfg_path="reward.kwargs.kling_video_reward.worker_config.model_path",
            repo_id="KlingTeam/VideoReward",
            required_files=("model_config.json",),
        ),
        CheckpointField(
            cfg_path=None,
            repo_id="Qwen/Qwen2-VL-2B-Instruct",
            required_files=("config.json",),
        ),
    ),
    reward_model_factory=None,
)

CASES: tuple[RealCheckpointCase, ...] = (
    RealCheckpointCase(
        case_id="wan_2_1",
        config="experiment/wan_2_1/online_grpo_ocr",
        family="wan_2_1",
        prompt="A clear white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                required_files=("model_index.json",),
            ),
        ),
        overrides=(
            "model.family=wan_2_1",
            "model.torch_compile.enable=false",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.num_frames=1",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=24.0,
        # wan's bf16 replay carries ~2.6e-3 mean recompute noise (cross-model
        # smoke 2026-06-09) — a known, separately-tracked warn, not the EDM bug.
        logprob_parity_tol=1e-2,
    ),
    RealCheckpointCase(
        case_id="sd3_5",
        config="experiment/sd3_5/online_grpo_ocr",
        family="sd3_5",
        prompt="A square poster that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="stabilityai/stable-diffusion-3.5-medium",
                required_files=("model_index.json",),
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "precision.training.dtype=bf16",
            "precision.rollout.dtype=bf16",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=24.0,
    ),
    # New diffusion-RL algorithms (SPRINT_diffusion_algorithm_parity), each run
    # on SD3.5 by swapping only the /recipe/online group. flow_dppo / grpo_guard
    # exercise the §3 old_prev_sample_mean store->replay path on the real
    # executor; on the first step rollout==replay so the trust-region KL is ~0
    # (no masking / ratio_mean_bias), which keeps a live gradient. Their losses
    # report no sample->replay logprob diff, so parity is checked permissively.
    *(
        RealCheckpointCase(
            case_id=f"sd3_5_{_algo}",
            config="experiment/sd3_5/online_grpo_ocr",
            family="sd3_5",
            prompt="A square poster that says RL",
            checkpoints=(
                CheckpointField(
                    cfg_path="model.path",
                    repo_id="stabilityai/stable-diffusion-3.5-medium",
                    required_files=("model_index.json",),
                ),
            ),
            overrides=(
                f"/recipe/online=flow_matching_{_recipe}",
                "model.torch_compile.enable=false",
                "precision.training.dtype=bf16",
                "precision.rollout.dtype=bf16",
                "actor.gradient_accumulation_steps=0",
                # No ref model in this harness; only DanceGRPO exposes GRPO's
                # reference-KL coefficient and therefore needs it disabled.
                *(("algorithm.kl_coef=0.0",) if _algo == "dance_grpo" else ()),
                "algorithm.kl_reward_coef=0.0",
                "actor.drop_zero_advantage=false",
                "rollout.n_samples_per_prompt=2",
                "rollout.prompts_per_batch=1",
                "rollout.samples_per_generation_batch=1",
                "rollout.noise_level=0.7",
                "rollout.sde.window_size=0",
                "rollout.sde.window_range=[0,1]",
                "sampling.num_steps=1",
                "sampling.guidance_scale=1.0",
                "sampling.height=128",
                "sampling.width=128",
                "sampling.max_sequence_length=64",
            ),
            min_cuda_memory_gib=24.0,
            logprob_parity_tol=1.0,  # DPPO/Guard/Dance don't report a logprob diff
        )
        for _algo, _recipe in (
            ("dance_grpo", "dance_grpo"),
            ("flow_dppo", "dppo"),
            ("grpo_guard", "grpo_guard"),
        )
    ),
    RealCheckpointCase(
        case_id="janus_pro",
        config="experiment/janus_pro/online_grpo_ocr",
        family="janus_pro",
        prompt="Text RL on a small label",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="deepseek-ai/Janus-Pro-1B",
                required_files=(
                    "config.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "tokenizer.json",
                ),
            ),
        ),
        overrides=(
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "sampling.max_text_length=64",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.guidance_scale=1.0",
            "sampling.temperature=1.0",
            "sampling.attention_backend=torch_native",
        ),
        min_cuda_memory_gib=16.0,
    ),
    _COSMOS_PREDICT2_KLING_TRANSPORT,
    _COSMOS_PREDICT2_KLING_REAL_REWARD,
    RealCheckpointCase(
        case_id="cosmos_predict2_5",
        config="experiment/cosmos_predict2_5/online_nft_kling_video_reward",
        family="cosmos-predict2.5",
        prompt="A clear white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="nvidia/Cosmos-Predict2.5-2B",
                required_files=(
                    "model_index.json",
                    "transformer/config.json",
                    "transformer/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                    "scheduler/scheduler_config.json",
                ),
            ),
        ),
        overrides=(
            "model.torch_compile.enable=false",
            "model.skip_text_encoder=true",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "rollout.noise_level=0.7",
            "rollout.sde.type=cps",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=2",
            "sampling.guidance_scale=1.0",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.num_frames=1",
            "sampling.fps=16",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=28.0,
        use_config_reward=True,
    ),
    RealCheckpointCase(
        case_id="cosmos_anima",
        config="experiment/anima_preview3/online_grpo",
        family="cosmos-predict2-anima",
        prompt="anime portrait of a small white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="circlestone-labs/Anima",
                required_files=("split_files/diffusion_models/anima-preview3-base.safetensors",),
                allow_file=True,
            ),
        ),
        overrides=(
            "+reward=aesthetic",
            "+dataset=drawbench_train_192",
            "actor.optim.lr=1.0e-5",
            "model.torch_compile.enable=false",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=28.0,
        synthetic_replay_rollout=True,
    ),
    RealCheckpointCase(
        case_id="cosmos_anima_safe",
        config="experiment/anima_preview3/online_grpo",
        family="cosmos-predict2-anima",
        prompt="anime portrait of a small white sign that says RL",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="circlestone-labs/Anima",
                required_files=("split_files/diffusion_models/anima-preview3-base.safetensors",),
                allow_file=True,
            ),
        ),
        overrides=(
            "+reward=aesthetic",
            "+reward=nsfw_safety",
            "+dataset=anime_safety_stress",
            "reward.components.nsfw_safety=0.5",
            "actor.optim.lr=1.0e-5",
            "model.torch_compile.enable=false",
            "actor.gradient_accumulation_steps=0",
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=1",
            "rollout.noise_level=0.7",
            "rollout.sde.window_size=0",
            "rollout.sde.window_range=[0,1]",
            "sampling.num_steps=1",
            "sampling.guidance_scale=1.0",
            "sampling.height=128",
            "sampling.width=128",
            "sampling.max_sequence_length=64",
        ),
        min_cuda_memory_gib=28.0,
        synthetic_replay_rollout=True,
    ),
    RealCheckpointCase(
        case_id="nextstep_1",
        config="experiment/nextstep_1/online_grpo_ocr",
        family="nextstep_1",
        prompt="Text RL on a small label",
        checkpoints=(
            CheckpointField(
                cfg_path="model.path",
                repo_id="stepfun-ai/NextStep-1.1",
                required_files=("config.json",),
            ),
            CheckpointField(
                cfg_path="model.vae_path",
                repo_id="stepfun-ai/NextStep-1-f8ch16-Tokenizer",
                required_files=("config.json",),
            ),
        ),
        overrides=(
            "algorithm.kl_coef=0.0",
            "algorithm.kl_reward_coef=0.0",
            "actor.drop_zero_advantage=false",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "sampling.max_text_length=64",
            "rollout.noise_level=1.0",
            "sampling.image_token_num=4",
            "sampling.image_size=64",
            "sampling.num_steps=1",
            "rollout.noise_level=1.0",
            "sampling.guidance_scale=1.0",
            "sampling.attention_backend=torch_native",
        ),
        min_cuda_memory_gib=64.0,
    ),
)


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in CASES
        if case.case_id in {"sd3_5_dance_grpo", "sd3_5_flow_dppo", "sd3_5_grpo_guard"}
    ),
    ids=lambda case: case.case_id,
)
def test_new_diffusion_algorithm_case_overrides_build_without_gpu(
    case: RealCheckpointCase,
) -> None:
    """Real-checkpoint overrides expose only knobs consumed by each loss."""
    has_reference_kl = "algorithm.kl_coef=0.0" in case.overrides
    assert has_reference_kl is (case.case_id == "sd3_5_dance_grpo")

    built = build_configs(load_config(case.config, overrides=list(case.overrides)))
    assert hasattr(built.algorithm, "kl_coef") is has_reference_kl


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_every_case_config_parses_without_a_gpu(
    case: RealCheckpointCase,
    tmp_path: Path,
) -> None:
    """Each case's overrides still name knobs its family's schema accepts.

    The GPU lane only runs on a machine with the weights, so a stale override
    (``sampling.max_sequence_length`` on a family whose section dropped it)
    used to surface as an ERROR there and nowhere else. Checkpoint paths are
    placeholders: config validation does not open them.
    """
    overrides = [
        f"{field.cfg_path}={tmp_path / field.repo_id.replace('/', '--')}"
        for field in case.checkpoints
        if field.cfg_path is not None
    ]
    overrides += list(case.overrides)
    if case.reference_image_cfg_path is not None:
        overrides.append(f"{case.reference_image_cfg_path}={_write_reference_image(tmp_path)}")
    if case.use_config_reward:
        overrides.extend(_local_reward_overrides(tmp_path, case.reward_model_factory))
    overrides.extend(_common_training_overrides(tmp_path))

    parse_config(load_config(case.config, overrides=overrides))


class _IndexReward:
    async def score_batch(self, rollouts: list[Any]) -> list[float]:
        return [float(i) for i, _ in enumerate(rollouts)]


class _DirectExecutorGenerationRuntime:
    """Test-only runtime that drives the real family executor without Ray actors.

    Real-checkpoint e2e tests use this to validate replay/trainer integration
    without making Ray scheduling part of the assertion surface. The executor
    runs on the trainer's own modules, so this runtime plays the colocated
    rollout worker's wake/sleep on them: ``activate`` puts the model and its
    frozen pipeline components (text encoder / VAE) on the GPU after the
    trainer parked them for the phase, ``offload`` moves the frozen components
    (and, when the trainer shares the card and will restore its state after
    the phase, the model itself) back to the host before an in-process reward
    takes the card.
    """

    def __init__(
        self,
        executor: Any,
        *,
        model: Any,
        device: torch.device,
        park_model: bool = False,
    ) -> None:
        self.executor = executor
        self.model = model
        self.device = device
        self.park_model = park_model
        self.current_policy_version = 0

    @property
    def requires_driver_model_offload(self) -> bool:
        return False

    async def activate(self) -> None:
        self.model.to(self.device)
        self._move_frozen(self.device)

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        rows = request.sample_rows()
        with torch.no_grad():
            return self.executor.forward_plan(request, rows, EnginePlan.from_request(request))

    async def offload(self) -> None:
        self._move_frozen(torch.device("cpu"))
        if self.park_model:
            self.model.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def shutdown(self) -> None:
        return None

    def _move_frozen(self, device: torch.device) -> None:
        move_frozen = getattr(self.model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen(device)


class _SyntheticDiffusionReplayCollector:
    """Collector that exercises replay training without full generation assets."""

    requires_generation_offload_before_reward = False
    requires_driver_model_offload_for_reward = False
    # Synthetic replay has no reward runtime to overlap with or place beside
    # continuous work; both capabilities are honestly absent.
    supports_reward_generation_overlap = False
    supports_continuous_reward_execution = False

    def __init__(
        self,
        *,
        model: Any,
        case: RealCheckpointCase,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.case = case
        self.cfg = cfg
        self.device = device
        self.generation_runtime = _StaticPolicyRuntime()

    async def collect_unscored(self, prompts: list[str], **kwargs: Any) -> Any:
        return _synthetic_diffusion_replay_batch(
            model=self.model,
            case=self.case,
            cfg=self.cfg,
            prompts=prompts,
            group_size=int(kwargs["group_size"]),
            policy_version=kwargs.get("policy_version"),
            device=self.device,
        )

    async def score_rollouts(self, pendings: Any) -> Any:
        return list(pendings)

    async def activate_generation_runtime(self) -> None:
        return None

    async def offload_generation_runtime_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def shutdown(self) -> None:
        return None


class _StaticPolicyRuntime:
    current_policy_version = 0
    requires_driver_model_offload = False


@pytest.mark.e2e
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_real_checkpoint_online_rl_updates_trainable_weights(
    case: RealCheckpointCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One real online-RL step on a cached checkpoint: trainable weights change, reward std and
    advantage rate are live, loss and grad norm are finite, and rollout-vs-replay log-prob
    parity holds within the case's tolerance.
    """
    _skip_unless_case_enabled(case)
    _skip_unless_cuda_has_memory(case.min_cuda_memory_gib)

    checkpoint_overrides = []
    for field in case.checkpoints:
        path = _resolve_checkpoint_path(case, field)
        if field.cfg_path is not None:
            checkpoint_overrides.append(f"{field.cfg_path}={path}")
    case_overrides = list(case.overrides)
    step_inputs: list[Any] = [case.prompt]
    if case.reference_image_cfg_path is not None:
        reference_image = _write_reference_image(tmp_path)
        case_overrides.append(
            f"{case.reference_image_cfg_path}={reference_image.as_posix()}",
        )
        # The executor reads the reference image off the GenerationInput that
        # PromptExample.generation_input() builds; a bare string prompt carries
        # none (production fills the dataset default in via validate_reference_images).
        step_inputs = [PromptExample(prompt=case.prompt, reference_image=str(reference_image))]
    if case.use_config_reward:
        case_overrides.extend(_local_reward_overrides(tmp_path, case.reward_model_factory))

    cfg = load_config(
        case.config,
        overrides=[
            *checkpoint_overrides,
            *case_overrides,
            *_common_training_overrides(tmp_path),
        ],
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    entry = get_model_family_entry(case.family)
    bundle: Any | None = None
    collector: Any | None = None
    trainer: OnlineTrainer | None = None
    reward_fn: Any | None = None
    try:
        device = torch.device("cuda")
        built = build_configs(cfg)
        trainer_config = built.trainer
        dtype = resolve_torch_dtype(built.precision.training.dtype)
        bundle = _build_runtime_bundle(
            case,
            entry,
            built.root,
            built.precision,
            device,
            dtype,
        )
        collector_config = RolloutCollectorConfig.from_root(parse_config(cfg))
        resources = ResolvedDistributedResources.from_root(parse_config(cfg))
        # The real topology plan for this box. Generation runs on the trainer's
        # own modules through _DirectExecutorGenerationRuntime, whose
        # activate/offload move them the way a colocated rollout worker wakes
        # and sleeps, so the trainer parks and the reward releases exactly as
        # in production. Index-reward cases keep no plan: their CPU reward
        # runtime has nothing to park.
        lifecycle = resources.lifecycle if case.use_config_reward else None
        if case.synthetic_replay_rollout:
            collector = _SyntheticDiffusionReplayCollector(
                model=bundle.model,
                case=case,
                cfg=cfg,
                device=device,
            )
            reward_fn = None
        else:
            executor = _build_executor(entry, bundle.model, cfg)
            reward_fn = (
                build_reward_function(
                    resolve_reward_inputs(built, resources, trainer_device=device),
                )
                if case.use_config_reward
                else _IndexReward()
            )
            collector = build_rollout_collector(
                entry,
                reward_runtime=RewardFunctionRuntime(reward_fn),
                config=collector_config,
                generation_runtime=_DirectExecutorGenerationRuntime(
                    executor,
                    model=bundle.model,
                    device=device,
                    park_model=lifecycle is not None and lifecycle.trainer_and_rollout_share_gpu,
                ),
                lifecycle=lifecycle,
            )
        pair = build_algorithm_and_evaluator(
            family_entry=entry,
            built=built,
            collector_config=collector_config,
            scheduler=bundle.scheduler,
        )
        trainer = OnlineTrainer(
            algorithm=pair.algorithm,
            collector=collector,
            evaluator=pair.evaluator,
            model=bundle.model,
            ref_model=None,
            weight_syncer=None,
            sync_state_getter=None,
            config=trainer_config,
            device=device,
        )

        before = trainable_state_digest(bundle.model)
        metrics = asyncio.run(trainer.step(step_inputs))
        after = trainable_state_digest(bundle.model)

        assert trainer.state.step == 1
        assert trainer.state.global_step >= 1
        assert before["tensor_count"] > 0
        assert before["sha256"] != after["sha256"]
        assert metrics.reward_std > 0.0
        assert metrics.adv_zero_rate < 1.0
        _assert_finite("loss", metrics.loss)
        _assert_finite_positive("grad_norm", metrics.grad_norm)
        if not case.synthetic_replay_rollout:
            # Sample->replay log-prob parity (GRPO ratio==1). Regression guard
            # for family-specific replay wiring: predict2 sat at mean 13.9
            # before the EDM sigma-domain fix while every metric here still
            # looked alive. Synthetic-replay cases fabricate old log-probs and
            # are excluded.
            assert metrics.logprob_mismatch.logprob_abs_diff_mean < case.logprob_parity_tol, (
                f"{case.case_id}: replay log-prob diverged from collection "
                f"(mean {metrics.logprob_mismatch.logprob_abs_diff_mean:.6f} >= "
                f"{case.logprob_parity_tol}) — broken sample/replay parity"
            )
        if case.use_config_reward:
            _assert_local_reward_artifacts(
                tmp_path,
                metrics,
                samples=int(cfg.rollout.n_samples_per_prompt),
            )
    finally:
        if trainer is not None:
            # Production order: park the trainer's state, then release the
            # collector, so a shared in-process reward can prove its memory
            # went away against the pre-load baseline.
            asyncio.run(trainer.rollout_schedule.lifecycle.shutdown_collector_runtime())
        elif collector is not None:
            asyncio.run(collector.shutdown())
        elif reward_fn is not None:
            # Once constructed, the collector is the reward runtime's sole
            # terminal owner. The standalone fallback is only for failures
            # before that ownership transfer completes.
            asyncio.run(_shutdown_if_present(reward_fn))
        del trainer, collector, reward_fn, bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _common_training_overrides(tmp_path: Path) -> tuple[str, ...]:
    return (
        "trainer.total_epochs=1",
        "trainer.save_freq=0",
        f"trainer.output_dir={tmp_path.as_posix()}",
        "trainer.debug.first_step=false",
        "trainer.profile=false",
        "trainer.torch_profiler.enabled=false",
        "actor.ppo_epochs=1",
        "actor.gradient_checkpointing=false",
        "actor.timestep_fraction=1.0",
        "actor.ema.enable=false",
    )


def build_tensor_mean_model(worker_config):
    """Test RewardModel factory: score = mean pixel of the decoded artifact.

    Returns the Kling function's public ``overall_reward`` key (the preset's
    ``score_key``) so reward transport is exercised end to end without weights.
    """

    del worker_config
    from vrl.rewards.models.media import decode_artifact_frames

    def _model(artifact):
        frames = decode_artifact_frames(artifact)
        return {"overall_reward": float(frames.float().mean().item())}

    return _model


def _local_reward_overrides(tmp_path: Path, model_factory: str | None) -> tuple[str, ...]:
    artifact_dir = tmp_path / "reward_artifacts"
    debug_dir = tmp_path / "reward_debug"
    overrides = [
        f"reward.kwargs.kling_video_reward.artifact_dir={artifact_dir.as_posix()}",
        f"reward.kwargs.kling_video_reward.debug_dir={debug_dir.as_posix()}",
        # Scored artifacts are released after scoring by default; keep them so
        # the on-disk transport can be asserted.
        "reward.kwargs.kling_video_reward.retain_artifacts=true",
        "reward.kwargs.kling_video_reward.worker_config.local_files_only=true",
    ]
    if model_factory is not None:
        overrides += [
            f"reward.kwargs.kling_video_reward.worker_config.model_factory={model_factory}",
            "reward.kwargs.kling_video_reward.worker_config.reward_model_version=e2e-tensor-mean",
        ]
    return tuple(overrides)


def _assert_local_reward_artifacts(tmp_path: Path, metrics: Any, *, samples: int) -> None:
    components = metrics.reward_components
    assert "kling_video_reward" in components
    assert isinstance(components["kling_video_reward"], float)
    # One retained mp4 per scored sample, plus the per-request debug rows.
    retained = sorted((tmp_path / "reward_artifacts").glob("*.mp4"))
    assert len(retained) == samples, [path.name for path in retained]
    assert (tmp_path / "reward_debug" / "kling_video_reward_results.jsonl").exists()


async def _shutdown_if_present(value: Any) -> None:
    shutdown = getattr(value, "shutdown", None)
    if shutdown is not None:
        result = shutdown()
        if inspect.isawaitable(result):
            await result


def _build_runtime_bundle(
    case: RealCheckpointCase,
    entry: ModelFamilyEntry,
    root: RootConfig,
    precision: PrecisionPolicy,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    for_rollout = not case.synthetic_replay_rollout
    build = entry.resolve_model_build(
        root,
        device,
        precision=precision,
        for_rollout=for_rollout,
        parameter_dtype_override=dtype,
    )
    return entry.build_rollout(build) if for_rollout else entry.build_replay(build)


def _build_executor(
    entry: ModelFamilyEntry,
    model: Any,
    cfg: Any,
) -> Any:
    from vrl.models.families.registry import GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR

    executor_cls = import_from_path(entry.executor_cls)
    kwargs: dict[str, Any] = {"gatherer": entry.new_gatherer()}
    signature = inspect.signature(executor_cls)
    if "samples_per_generation_batch" in signature.parameters:
        kwargs["samples_per_generation_batch"] = int(cfg.rollout.samples_per_generation_batch)
    # Same construction as the rollout worker (vrl/generation/execution/worker.py):
    # the generic denoise executor takes its family/task from the registry entry.
    if entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
        kwargs.update(family=entry.family, task=entry.task)
    return executor_cls(model, **kwargs)


def _synthetic_diffusion_replay_batch(
    *,
    model: Any,
    case: RealCheckpointCase,
    cfg: Any,
    prompts: list[str],
    group_size: int,
    policy_version: int | None,
    device: torch.device,
) -> Any:
    from vrl.math.denoise.flow_matching import sde_step_with_logprob
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trajectory import build_diffusion_trajectory

    num_steps = max(1, int(cfg.sampling.num_steps))
    height = int(cfg.sampling.height)
    width = int(cfg.sampling.width)
    latent_scale = 8
    transformer_cfg = getattr(getattr(model, "transformer", None), "config", None)
    in_channels = int(getattr(transformer_cfg, "in_channels", 16))
    text_embed_dim = int(getattr(transformer_cfg, "text_embed_dim", 1024))
    dtype = _model_tensor_dtype(model)

    if hasattr(model, "set_num_steps"):
        model.set_num_steps(num_steps)
    else:
        model.scheduler.set_timesteps(num_steps, device=device)

    request = GenerationRequest(
        request_id=f"{case.case_id}:synthetic-replay",
        family=case.family,
        task="text_to_image",
        inputs=list(prompts),
        samples_per_prompt=group_size,
        sampling={
            "height": height,
            "width": width,
            "num_frames": 1,
            "num_steps": num_steps,
            "guidance_scale": float(cfg.sampling.guidance_scale),
        },
        policy_version=None if policy_version is None else int(policy_version),
    )
    sample_rows = request.sample_rows()
    batch_size = len(sample_rows)
    generator = torch.Generator(device=device)
    generator.manual_seed(1729)

    observations = torch.randn(
        (
            batch_size,
            num_steps,
            in_channels,
            1,
            max(1, height // latent_scale),
            max(1, width // latent_scale),
        ),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    prompt_embeds = torch.randn(
        (batch_size, 512, text_embed_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    padding_mask = torch.zeros(
        (batch_size, 1, height, width),
        device=device,
        dtype=dtype,
    )
    replay_tensors: dict[str, Any] = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": None,
        "padding_mask": padding_mask,
    }
    context = {
        "guidance_scale": float(cfg.sampling.guidance_scale),
        # Same derivation the families use at rollout (do_cfg = guidance > 1).
        "cfg": float(cfg.sampling.guidance_scale) > 1.0,
        "model_family": case.family,
    }
    timesteps = model.scheduler.timesteps[:num_steps].to(device)
    timesteps = timesteps.unsqueeze(0).expand(batch_size, -1).clone()
    noise_level = float(cfg.rollout.noise_level)
    sde_type = str(getattr(getattr(cfg.rollout, "sde", None), "type", "flow_grpo"))

    action_steps: list[torch.Tensor] = []
    old_log_prob_steps: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(num_steps):
            state = model.restore_eval_state(
                replay_tensors,
                context,
                observations[:, step_idx],
                step_idx,
            )
            noise_pred = model.forward_step(state, step_idx)["noise_pred"]
            sde = sde_step_with_logprob(
                model.scheduler,
                noise_pred,
                timesteps[:, step_idx],
                observations[:, step_idx],
                generator=generator,
                noise_level=noise_level,
                sde_type=sde_type,
            )
            action_steps.append(sde.prev_sample.detach())
            old_log_prob_steps.append(sde.log_prob.detach())

    actions = torch.stack(action_steps, dim=1)
    old_log_prob = torch.stack(old_log_prob_steps, dim=1)
    kl = torch.zeros_like(old_log_prob)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=observations.detach(),
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        kl=kl,
        replay_tensors=replay_tensors,
        context=context,
    )
    rewards = torch.arange(batch_size, device=device, dtype=torch.float32)
    return RolloutBatch(
        rewards=rewards,
        group_ids=torch.tensor(
            [row.prompt_index for row in sample_rows],
            dtype=torch.long,
            device=device,
        ),
        extras={},
        context=dict(trajectory.context),
        trajectory=trajectory,
    )


def _model_tensor_dtype(model: Any) -> torch.dtype:
    for parameter in model.parameters():
        return parameter.dtype
    return torch.bfloat16


def _skip_unless_case_enabled(case: RealCheckpointCase) -> None:
    if not ci_envs.WM_RUN_REAL_MODEL_TESTS:
        pytest.skip(f"set {RUN_REAL_ENV}=1 to run real checkpoint RL e2e tests")
    requested = _requested_case_ids()
    if requested != {"cached"} and case.case_id not in requested and "all" not in requested:
        pytest.skip(f"{case.case_id} not selected by {CASE_FILTER_ENV}")


def _requested_case_ids() -> set[str]:
    raw = ci_envs.WM_REAL_MODEL_RL_CASES
    return {item.strip() for item in raw.split(",") if item.strip()}


def _skip_unless_cuda_has_memory(min_gib: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("real checkpoint RL e2e tests require CUDA")
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gib < min_gib:
        pytest.skip(
            f"real checkpoint requires at least {min_gib:.1f} GiB CUDA memory; "
            f"device has {total_gib:.1f} GiB",
        )


def _resolve_checkpoint_path(case: RealCheckpointCase, field: CheckpointField) -> Path:
    env_name = _checkpoint_env_name(case, field)
    override = os.environ.get(env_name)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            pytest.skip(f"{env_name} points to a missing checkpoint: {path}")
        if path.is_file() and field.allow_file:
            return path
        missing = _missing_required_files(path, field.required_files)
        if missing:
            pytest.skip(
                f"{env_name} points to an incomplete checkpoint: {path}; missing={missing}",
            )
        return path

    snapshot = _cached_hf_snapshot(field.repo_id, required_files=field.required_files)
    if snapshot is None:
        selected = _requested_case_ids()
        skip_prefix = (
            "cached checkpoint is missing"
            if selected == {"cached"}
            else f"{case.case_id} selected but cached checkpoint is missing"
        )
        pytest.skip(
            f"{skip_prefix}: {field.repo_id}. "
            f"Set {env_name} to a filesystem checkpoint path to override.",
        )
    return snapshot


def _checkpoint_env_name(case: RealCheckpointCase, field: CheckpointField) -> str:
    target = field.cfg_path if field.cfg_path is not None else field.repo_id
    return f"WM_REAL_CHECKPOINT_{_env_token(case.case_id)}_{_env_token(target)}"


def _cached_hf_snapshot(
    repo_id: str,
    *,
    required_files: tuple[str, ...],
) -> Path | None:
    """Resolve a cached snapshot through the Hub's public cache API.

    This dependency seam is intentionally separate from checkpoint selection:
    Hugging Face owns cache layout and revision resolution; this test owns only
    offline behavior and the files its runtime needs.
    """
    try:
        snapshot = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    if not snapshot.is_dir() or _missing_required_files(snapshot, required_files):
        return None
    return snapshot


def _missing_required_files(path: Path, relative_files: tuple[str, ...]) -> list[str]:
    return [name for name in relative_files if not (path / name).exists()]


def _write_reference_image(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "reference.png"
    Image.new("RGB", (128, 128), color=(112, 118, 126)).save(path)
    return path


def _env_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _assert_finite_positive(name: str, value: float) -> None:
    _assert_finite(name, value)
    assert float(value) > 0.0, f"{name} must be positive, got {value}"


def _assert_finite(name: str, value: float) -> None:
    assert math.isfinite(float(value)), f"{name} must be finite, got {value}"
