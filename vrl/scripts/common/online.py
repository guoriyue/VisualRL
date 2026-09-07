"""Common runner skeleton for online training recipes."""

from __future__ import annotations

import gc
import inspect
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from vrl.config.builders import BuiltConfigs
from vrl.config.schema import RootConfig
from vrl.generation.ray.launcher import RayGenerationLauncher
from vrl.models.interfaces import require_runtime_model
from vrl.ray.dependencies import require_ray
from vrl.ray.placement import GlobalRayPlacementOwner, cross_node_preflight
from vrl.ray.resources import (
    ResolvedDistributedResources,
    format_distributed_resource_plan,
)
from vrl.rewards import RewardRuntime
from vrl.rollouts.collector import RolloutCollector, build_rollout_collector
from vrl.rollouts.orchestration import (
    RolloutSchedule,
    validate_rollout_schedule_topology,
)
from vrl.rollouts.stats import RolloutStats
from vrl.run import (
    resolve_model,
    resolve_online_run,
)
from vrl.scripts.common.factory import (
    build_algorithm_and_evaluator,
    build_reward_runtime,
    validate_reward_memory_parking,
)
from vrl.trainers.activation_checkpointing import (
    enable_transformer_gradient_checkpointing,
)
from vrl.trainers.checkpointing import (
    AdapterExport,
    build_adapter_exports,
    capture_rng_state,
    load_training_checkpoint_for_resume,
    restore_rng_state,
    restore_training_checkpoint,
    save_resolved_config,
    save_training_checkpoint,
    validate_checkpoint_compatibility,
)
from vrl.trainers.data import (
    PromptBatchSampler,
    PromptExample,
    load_prompt_examples_from_config,
    resolve_prompt_example_references,
)
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.metrics_io import (
    OnlineMetricRow,
    format_online_metric_row,
    online_metric_columns,
    prepare_metrics_csv,
)
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan
from vrl.trainers.strategy import Strategy, build_strategy
from vrl.trainers.weight_sync import RayRuntimeWeightSyncer
from vrl.utils.memory import capture_host_memory, format_host_memory, log_host_memory
from vrl.utils.profiling import profile_range

logger = logging.getLogger(__name__)

_RAY_ADDRESS_ENV = "RAY_ADDRESS"


@dataclass(slots=True)
class _RayClusterSession:
    """Connection owned by this recipe invocation.

    A pre-initialized Ray client belongs to the embedding caller and is left
    connected. Connections opened here are always closed: for a local cluster
    ``ray.shutdown`` terminates only the processes spawned by this driver; for an
    explicitly attached cluster it disconnects this driver without stopping the
    cluster.
    """

    ray: Any
    shutdown_on_exit: bool
    _closed: bool = False

    @classmethod
    def connect(
        cls,
        ray: Any,
        *,
        cross_node: bool,
        environ: Mapping[str, str] | None = None,
        local_num_cpus: int | None = None,
    ) -> _RayClusterSession:
        """Connect to exactly the Ray cluster selected by the resource topology.

        Single-node and per-rank-local runs always start a fresh local cluster,
        even when the host has a stale ``RAY_ADDRESS`` or another user's Ray
        instance. Cross-node runs require a concrete address; implicit
        ``address='auto'`` discovery is unsafe on a shared host.
        """

        if ray.is_initialized():
            logger.info(
                "Ray cluster session: ownership=preinitialized driver_pid=%d ray_version=%s",
                os.getpid(),
                getattr(ray, "__version__", "unknown"),
            )
            return cls(ray=ray, shutdown_on_exit=False)

        environment = os.environ if environ is None else environ
        if cross_node:
            address = str(environment.get(_RAY_ADDRESS_ENV, "")).strip()
            if not address or address in {"auto", "local"}:
                raise ValueError(
                    "distributed.resources.cross_node=true requires a concrete "
                    "RAY_ADDRESS (for example 10.0.0.1:6379); 'auto' and 'local' "
                    "do not identify the operator-owned multi-node cluster",
                )
            ownership = "attached"
        else:
            # ``local`` bypasses RAY_ADDRESS and Ray's latest-cluster discovery.
            address = "local"
            ownership = "owned_local"

        # Ray overwrites driver SIGTERM handling inside ray.init(). Signal
        # ownership belongs to the launcher, whose cooperative handlers must
        # survive cluster initialization so terminal cleanup still runs.
        import signal

        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        init_kwargs: dict[str, Any] = {"address": address}
        if ownership == "owned_local":
            if local_num_cpus is not None:
                if (
                    isinstance(local_num_cpus, bool)
                    or not isinstance(local_num_cpus, int)
                    or local_num_cpus <= 0
                ):
                    raise ValueError("local Ray num_cpus must be a positive integer")
                init_kwargs["num_cpus"] = local_num_cpus
            init_kwargs["include_dashboard"] = False
        try:
            context = ray.init(**init_kwargs)
        finally:
            for signum, handler in previous_handlers.items():
                if handler is not None:
                    signal.signal(signum, handler)
        address_info = getattr(context, "address_info", None)
        if not isinstance(address_info, Mapping):
            address_info = {}
        resolved_address = (
            address_info.get("gcs_address") or address_info.get("address") or address
        )
        logger.info(
            "Ray cluster session: ownership=%s driver_pid=%d address=%s "
            "session_dir=%s ray_version=%s",
            ownership,
            os.getpid(),
            resolved_address,
            address_info.get("session_dir", "unknown"),
            getattr(ray, "__version__", "unknown"),
        )
        return cls(ray=ray, shutdown_on_exit=True)

    def shutdown(self) -> None:
        if self._closed:
            return
        if self.shutdown_on_exit:
            self.ray.shutdown()
        self._closed = True


@dataclass(slots=True)
class _OnlineRecipeLifecycle:
    """Own partial-construction state and terminal teardown for one recipe."""

    placement_owner: GlobalRayPlacementOwner
    strategy: Strategy
    ray_session: _RayClusterSession | None = None
    reward_runtime: RewardRuntime | None = None
    collector: RolloutCollector | None = None
    rollout_schedule: RolloutSchedule | None = None
    _shutdown_errors: list[tuple[str, Exception]] = field(
        default_factory=list,
        init=False,
    )
    _closed: bool = field(default=False, init=False)

    async def shutdown(self, *, run_error: BaseException | None) -> None:
        """Release every acquired role in dependency order.

        Construction may fail after any field is assigned. The lifecycle keeps
        those partial acquisitions together so terminal cleanup does not need a
        parallel seven-argument state bundle.
        """

        if self._closed:
            return
        self._shutdown_errors.clear()

        # The schedule owns the complete rollout pipeline. Before trainer
        # construction, fall back to the collector; before collector
        # construction, only the standalone reward runtime exists.
        if self.rollout_schedule is not None:
            pipeline_released = await self._shutdown_one(
                "rollout_schedule",
                self.rollout_schedule,
            )
        elif self.collector is not None:
            pipeline_released = await self._shutdown_one(
                "collector",
                self.collector,
            )
        else:
            pipeline_released = await self._shutdown_one(
                "reward_runtime",
                self.reward_runtime,
            )

        await self._shutdown_one("placement_owner", self.placement_owner)

        # A failed role cleanup leaves shared-GPU ownership unknown. The
        # strategy must still destroy process groups, but it may not restore a
        # parked trainer into memory whose release was not proven.
        await self._shutdown_one(
            "strategy",
            self.strategy,
            call_kwargs={"restore_parked": pipeline_released},
        )
        # Ray is last: actor and placement cleanup above still need its client.
        await self._shutdown_one("ray_session", self.ray_session)
        self._closed = True

        if not self._shutdown_errors:
            return
        for name, exc in self._shutdown_errors:
            logger.error(
                "%s shutdown failed during online recipe cleanup",
                name,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if run_error is None:
            name, exc = self._shutdown_errors[0]
            raise RuntimeError(f"{name} shutdown failed during online recipe cleanup") from exc

    async def _shutdown_one(
        self,
        name: str,
        target: Any,
        *,
        call_kwargs: Mapping[str, Any] | None = None,
    ) -> bool:
        """Run one terminal cleanup, retrying one transient failure."""

        if target is None:
            return True
        shutdown = getattr(target, "shutdown", None)
        if not callable(shutdown):
            self._shutdown_errors.append(
                (name, TypeError(f"{name} does not expose callable shutdown()")),
            )
            return False
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                result = shutdown(**dict(call_kwargs or {}))
                if inspect.isawaitable(result):
                    await result
                return True
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        self._shutdown_errors.append((name, last_error))
        return False


def _require_supported_distributed_rollout_topology(
    context: DistributedTrainingContext,
    resources: ResolvedDistributedResources,
) -> None:
    """Reject multi-rank rollout ownership that the recipe cannot coordinate.

    The implemented DDP/FSDP orchestration is per-rank-local and colocated: each
    rank owns its own Ray runtime on its trainer GPU. With disjoint rollout GPUs,
    every rank would instead resolve the same global rollout plan, start or attach
    Ray independently, and launch duplicate workers onto those GPUs. Supporting
    that topology needs one explicit owner (for example rank-0 collection plus a
    broadcast), not a resource-only YAML change.
    """

    if not context.distributed:
        return
    if bool(resources.colocated) or int(resources.rollout_num_gpus) == 0:
        return
    raise NotImplementedError(
        "distributed training with disjoint rollout GPUs is not supported by "
        "run_online_recipe: every torchrun rank would independently initialize Ray "
        "and launch the same rollout device plan. Use single_process with dedicated "
        "rollout GPUs. Multi-rank DDP/FSDP needs rank-owned rollout placement or "
        "rank-0 collection/broadcast before this topology can run.",
    )


def _validate_reward_placement(
    placement: Any,
    *,
    resources: ResolvedDistributedResources,
    reward_device: str,
) -> None:
    """Link the reward GPU reservation to the device the runtime executes on.

    The placement group reserves the reward bundle and ``reward_torch_device``
    independently derives the execution device from the same resolution; this
    check fails fast if the two derivations ever drift apart.
    """

    if not resources.reward_devices:
        return
    if placement is None:
        raise RuntimeError(
            "reward GPUs are resolved "
            f"({list(resources.reward_devices)}) but the run placement group "
            "reserved no reward bundle; the reservation and execution device "
            "have diverged",
        )
    if reward_device.startswith("cuda"):
        # reward_device is a process-local torch ordinal while the placement
        # bundle reports Ray physical ids; on a rank-local torchrun launch the
        # two spaces differ (CUDA mask narrows torch to one logical device),
        # so translate back to plan space before comparing.
        ordinal = resources.plan_device_ordinal(int(reward_device.split(":", 1)[1]))
        if placement.expected_gpu_ids and ordinal not in placement.expected_gpu_ids:
            raise RuntimeError(
                f"in-process reward executes on {reward_device} (plan GPU {ordinal}) "
                f"but the placement group reserved GPUs "
                f"{list(placement.expected_gpu_ids)} for the reward role",
            )


def _log_rollout_memory_plan(
    batch_plan: OnlineBatchPlan,
    *,
    samples_per_generation_batch: int | str | None,
) -> None:
    """Log how many rollout tensors one optimizer update can hold at once."""

    prompts_per_batch = batch_plan.prompts_per_batch
    samples_per_prompt = batch_plan.n_samples_per_prompt
    target_samples = prompts_per_batch * samples_per_prompt
    replay_width = batch_plan.samples_per_replay_batch

    def describe_batch_width(value: Any) -> str:
        if value == "auto":
            return "auto(pending)"
        size = int(value or 0)
        return str(samples_per_prompt if size <= 0 else min(samples_per_prompt, size))

    generation_batch_text = describe_batch_width(samples_per_generation_batch)
    replay_batch_text = describe_batch_width(replay_width)
    gas = batch_plan.gradient_accumulation_steps
    if batch_plan.streaming:
        microbatch_prompts = batch_plan.microbatch_size
        microbatch_samples = microbatch_prompts * samples_per_prompt
        logger.info(
            "Rollout memory plan: streaming accumulation enabled "
            "(prompts_per_batch=%d, gradient_accumulation_steps=%d, "
            "microbatch_prompts=%d, microbatch_samples=%d, "
            "samples_per_generation_batch=%s, samples_per_replay_batch=%s, "
            "target_samples_per_update=%d)",
            prompts_per_batch,
            gas,
            microbatch_prompts,
            microbatch_samples,
            generation_batch_text,
            replay_batch_text,
            target_samples,
        )
        return

    logger.info(
        "Rollout memory plan: legacy full-batch accumulation "
        "(prompts_per_batch=%d, samples_per_generation_batch=%s, "
        "samples_per_replay_batch=%s, "
        "target_samples_per_update=%d)",
        prompts_per_batch,
        generation_batch_text,
        replay_batch_text,
        target_samples,
    )
    if prompts_per_batch > 1:
        logger.warning(
            "Legacy full-batch rollout accumulation is enabled; host RAM may hold "
            "up to %d prompt groups (%d samples) before backward. Set "
            "actor.gradient_accumulation_steps to a divisor of prompts_per_batch "
            "to stream rollout microbatches and fail earlier on memory issues.",
            prompts_per_batch,
            target_samples,
        )


def _warn_global_std_streaming_divergence(
    batch_plan: OnlineBatchPlan,
    *,
    global_std: bool,
) -> None:
    """Warn when global_std advantage normalization is silently per-microbatch.

    GRPO ``global_std=true`` normalizes advantages by the std across ALL prompt
    groups in the optimizer-target batch. Streaming accumulation computes
    advantages per microbatch (collect_training_batch runs once per slice), so
    with >1 group per microbatch the std is taken over the microbatch's groups
    only -- not the full batch -- and the gradient diverges from the full-batch
    global-std intent. ``microbatch_size=1`` is exempt: one group per microbatch
    makes per-group and "global" std identical. Surfaced, not blocked, because
    keeping global_std is an experiment-owner decision.

    Same signature shape as ``_log_rollout_memory_plan``: the batch plan the
    diagnostic reasons about, plus its one value from another owner as a keyword.
    ``global_std`` belongs to the algorithm config, so the caller passes the
    typed field rather than re-reading a YAML path whose default would silently
    win if the key ever moved.
    """
    gas = batch_plan.gradient_accumulation_steps
    if not batch_plan.streaming:
        return
    if not global_std:
        return
    rbs = batch_plan.prompts_per_batch
    groups_per_microbatch = batch_plan.microbatch_size
    if groups_per_microbatch <= 1:
        return
    logger.warning(
        "algorithm.global_std=true with streaming accumulation "
        "(gradient_accumulation_steps=%d, %d prompt groups per microbatch): the "
        "global-std advantage normalization is computed per microbatch, not over "
        "the full %d-group batch, so the gradient differs from the full-batch "
        "global-std intent. Set algorithm.global_std=false (per-group std, which "
        "is streaming-equivalent), actor.microbatch_size=1 (one group per "
        "microbatch), or drop streaming to keep the full-batch global std.",
        gas,
        groups_per_microbatch,
        rbs,
    )


def _default_reference_model(bundle: Any, built: BuiltConfigs) -> Any | None:
    """Reference model for KL: the (LoRA) policy itself when use_lora and kl_coef>0, else None."""

    # Read off the already-resolved typed bundle instead of re-walking raw cfg.
    # kl_coef is optional across algorithm-config families: only the
    # ClippedPolicy-derived configs (grpo/dance_grpo) define it, while
    # flow_dppo/grpo_guard extend GroupAdvantageConfig and legitimately omit it,
    # so read the typed field with a default rather than assume every algorithm
    # config carries it. model.use_lora is always a field on the typed
    # ModelSection (pydantic default None -> falsy), so read it directly.
    kl_coef = float(getattr(built.algorithm, "kl_coef", 0.0) or 0.0)
    if bool(built.root.model.use_lora) and kl_coef > 0:
        return bundle.model
    return None


def _load_sft_latents_from_config(built: BuiltConfigs, family: str) -> dict[str, Any] | None:
    """Load the clean-latents shard when the diffusion-loss regularizer is on.

    The schema cross-check already rejected sft_weight>0 without
    data.sft_latents, so this only turns a configured path into tensors (and
    fails loud on a family-mismatched or malformed shard).
    """

    # sft_weight is defined only on GRPOConfig (grpo/dance_grpo); every other
    # algorithm config legitimately omits it, so read the typed field off the
    # resolved bundle with a default instead of assuming presence.
    weight = float(getattr(built.algorithm, "sft_weight", 0.0) or 0.0)
    if weight <= 0:
        return None
    data = built.root.data
    path = data.sft_latents if data is not None else None
    if not path:
        raise ValueError("algorithm.sft_weight > 0 requires data.sft_latents")
    from vrl.trainers.data.sft_latents import load_sft_latents

    model = built.root.model
    return load_sft_latents(
        str(path),
        family=family,
        model_path=str(model.path or ""),
        model_revision=str(model.revision or ""),
    )


def _check_host_memory_budget(
    budget_fraction: float,
    *,
    microbatch_prompts: int,
    n_samples_per_prompt: int,
) -> None:
    """Fail fast if one streamed microbatch already pushes host RAM past budget.

    Streaming accumulation holds ~one microbatch of rollout/replay tensors at a
    time, so if system memory is already over budget right after collecting the
    first microbatch, a larger ``microbatch_size`` (or simply more
    epochs) would only OOM later in the run. Raising now — with the measured
    snapshot — turns a delayed mid-run OOM into an immediate, actionable error.
    Real RSS is measured (``capture_host_memory`` reads /proc), not estimated
    from tensor byte counts, because the Ray OOM monitor kills on RSS.
    """
    snapshot = capture_host_memory()
    used = snapshot.used_fraction
    if used is None or used <= budget_fraction:
        return
    raise MemoryError(
        f"Host RAM is at used={used:.1%} after collecting one streamed microbatch "
        f"({microbatch_prompts} prompt group(s) x {n_samples_per_prompt} samples), "
        f"above actor.host_memory_budget_fraction={budget_fraction:.1%} "
        f"({format_host_memory(snapshot)}). One microbatch already does not fit the "
        "host-RAM budget; reduce actor.microbatch_size to stream smaller "
        "slices, or lower rollout.n_samples_per_prompt / sample resolution if it is "
        "already 1.",
    )


async def _run_streaming_optimizer_update(
    trainer: OnlineTrainer,
    example_batch: list[Any],
    *,
    batch_plan: OnlineBatchPlan,
    next_example_batch: list[Any] | None = None,
) -> Any:
    """One optimizer update streamed over ``gradient_accumulation_steps`` microbatches.

    Splits the ``prompts_per_batch`` prompts into microbatches and runs
    collect -> backward -> RELEASE for each before the next, so host RAM holds
    ~one microbatch of rollout/replay tensors instead of the whole target batch
    (the memory fix that lets bigger models train on limited GPUs). One
    optimizer.step / EMA / weight-sync / metric row per update; gradients
    accumulate across microbatches with a global loss scale, so the update is
    gradient-equivalent to the legacy full-batch path.

    When ``host_memory_budget_fraction`` > 0, the first collected microbatch is
    checked against the host-RAM budget and the run fails fast if it is already
    over budget (SPRINT_memory_budgeted_microbatch T2).
    """
    if not batch_plan.streaming:
        raise ValueError("_run_streaming_optimizer_update requires a streaming batch plan")
    micro = batch_plan.microbatch_size
    microbatches = [example_batch[k : k + micro] for k in range(0, len(example_batch), micro)]
    total_groups = batch_plan.prompts_per_batch

    trainer.begin_optimizer_update()

    update_stats = RolloutStats()
    reward_mean_w = reward_std_w = adv_mean_w = adv_zero_w = adv_sat_w = 0.0
    weight_total = 0
    trained_prompt_num = 0
    group_size = float(batch_plan.n_samples_per_prompt)
    reward_component_values: dict[str, list[float]] = {}
    for mb_index, microbatch in enumerate(microbatches):
        if mb_index + 1 < len(microbatches):
            next_prompts = microbatches[mb_index + 1]
        elif next_example_batch:
            next_prompts = next_example_batch[:micro]
        else:
            next_prompts = None
        batch = await trainer.collect_training_batch(
            microbatch,
            next_prompts=next_prompts,
        )
        try:
            # Host-RAM fail-fast on the first microbatch: one slice is the host
            # peak under streaming, so if it is already over budget, stop now.
            if batch_plan.host_memory_budget_fraction > 0.0 and mb_index == 0:
                _check_host_memory_budget(
                    batch_plan.host_memory_budget_fraction,
                    microbatch_prompts=len(microbatch),
                    n_samples_per_prompt=batch_plan.n_samples_per_prompt,
                )
            trainer.backward_on_training_batch(batch, total_groups=total_groups)
            # Sample-count-weighted aggregation of this microbatch's pre-filter stats
            # so the one metric row reflects ALL samples, not the last microbatch.
            weight = max(1, len(microbatch) * batch_plan.n_samples_per_prompt)
            reward_mean_w += batch.pre_filter_reward_mean * weight
            reward_std_w += batch.pre_filter_reward_std * weight
            adv_mean_w += batch.pre_filter_adv_mean * weight
            adv_zero_w += batch.adv_zero_rate * weight
            adv_sat_w += batch.adv_saturation * weight
            weight_total += weight
            trained_prompt_num += int(batch.trained_prompt_num)
            if batch.group_size:
                group_size = float(batch.group_size)
            for name, values in getattr(batch, "reward_components", {}).items():
                reward_component_values.setdefault(name, []).extend(values)
            update_stats.merge(trainer._step_stats(batch.iteration, batch.timer))
        finally:
            # Release this microbatch's rollout/replay tensors before the next,
            # including exception paths where traceback locals can otherwise keep
            # large batches alive longer than needed.
            del batch

    weight_total = max(1, weight_total)
    return await trainer.finish_optimizer_update(
        stats=update_stats,
        reward_mean=reward_mean_w / weight_total,
        reward_std=reward_std_w / weight_total,
        adv_mean=adv_mean_w / weight_total,
        adv_zero_rate=adv_zero_w / weight_total,
        adv_saturation=adv_sat_w / weight_total,
        group_size=group_size,
        trained_prompt_num=trained_prompt_num,
        reward_components={
            name: sum(values) / len(values)
            for name, values in reward_component_values.items()
            if values
        },
    )


@dataclass(slots=True)
class OnlineRecipeRun:
    """Execution controller for one ``run_online_recipe`` invocation.

    Owns the wired training objects and the per-run CSV/checkpoint state. These
    fields used to sit in a one-owner nested wrapper; keeping them
    here makes the controller the single source of its own checkpoint inputs.
    """

    bundle: Any
    trainer: Any
    strategy: Any
    family: str
    component_names: tuple[str, ...]
    adapter_exports: dict[str, AdapterExport] | None
    csv_path: Path
    rng: Any
    resume_epoch: int | None
    model_identity: dict[str, Any]

    def prepare_metrics_csv(self) -> None:
        prepare_metrics_csv(
            self.csv_path,
            online_metric_columns(self.component_names),
            resume_at=("epoch", self.resume_epoch) if self.resume_epoch is not None else None,
        )

    def prepare_metrics_csv_rank_consistent(
        self,
        training_context: DistributedTrainingContext,
    ) -> None:
        """Run the rank-0 CSV preflight and propagate its verdict to every rank.

        Only rank 0 owns the output path in multi-node runs, so peers cannot safely
        inspect the header themselves. Broadcasting the small error description
        keeps every rank on the same side of the first training collective.
        """

        if not training_context.distributed:
            self.prepare_metrics_csv()
            return

        failure: str | None = None
        if training_context.is_primary:
            try:
                self.prepare_metrics_csv()
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"

        payload = [failure]
        torch.distributed.broadcast_object_list(
            payload,
            src=0,
            device=training_context.device,
        )
        if payload[0] is not None:
            raise RuntimeError(f"metrics CSV preflight failed on rank 0: {payload[0]}")

    def write_metric_row(self, epoch: int, metrics: Any) -> None:
        row = OnlineMetricRow.from_step_metrics(epoch, metrics, self.component_names)
        with self.csv_path.open("a", encoding="utf-8") as handle:
            handle.write(format_online_metric_row(row))

    def save_checkpoint(self, path: Path, *, epoch: int) -> None:
        # Called on EVERY rank: save_training_checkpoint runs the checkpoint-state
        # gather (a collective under FSDP2) on all ranks and writes files on the
        # primary only. Adapter artifacts use a separately gathered full state
        # under EMA and never read live DTensor shards during rank0 IO. The save
        # boundary also propagates publication success/failure to every rank.
        save_training_checkpoint(
            path,
            trainer=self.trainer,
            bundle=self.bundle,
            family=self.family,
            progress={
                "completed_epoch": epoch,
                "next_epoch": epoch,
                "global_step": self.trainer.state.global_step,
            },
            rng_state=capture_rng_state(prompt_generator=self.rng),
            adapter_exports=self.adapter_exports,
            export_ema=getattr(self.trainer, "_ema", None),
            model_identity=self.model_identity,
            strategy=self.strategy,
        )


async def run_online_recipe(
    cfg: DictConfig,
    *,
    prompt_examples: Sequence[PromptExample] | None = None,
) -> None:
    """Run a family online training job through shared recipe glue."""

    provided_examples = None if prompt_examples is None else list(prompt_examples)

    resolved = resolve_online_run(cfg)
    _preflight_production_video_reward(resolved.built.root)
    built = resolved.built
    run_config = resolved.run
    family_entry = resolved.family
    resources = resolved.resources
    generation_config = resolved.generation
    device = resolved.device
    trainer_config = built.trainer
    if trainer_config is None:
        raise ValueError("online recipe cannot use an offline-only trainer config")
    batch_plan = trainer_config.batch_plan
    reward_config = built.reward
    if reward_config is None:
        raise ValueError("online recipe requires a reward section")
    _log_rollout_memory_plan(
        batch_plan,
        samples_per_generation_batch=(
            built.root.rollout.samples_per_generation_batch
            if built.root.rollout is not None
            else None
        ),
    )
    _warn_global_std_streaming_divergence(
        batch_plan,
        global_std=built.algorithm.global_std,
    )
    if trainer_config.profile:
        os.environ["VRL_PROFILE"] = "1"

    # build_configs already normalized model.lora.path for resume and resolved the
    # resume policy, so the recipe only loads the raw checkpoint here. The
    # checkpoint-identity preflight below consumes it directly; the epoch/step/dir
    # fields are derived after the preflight, next to the trainer that reads them.
    resume_config = built.resume
    resume_checkpoint = load_training_checkpoint_for_resume(resume_config)
    validate_rollout_schedule_topology(trainer_config.rollout_orchestration, resources)
    validate_reward_memory_parking(resources=resources, built=built)
    family_entry.validate_gpus_per_engine(resources.rollout_gpus_per_engine)
    logger.info(format_distributed_resource_plan(resources))
    # Resolve the training process identity (rank/device) and fail-fast on
    # strategies the online recipe can't yet drive end-to-end, before building the
    # model / Ray runtime.
    training_context = DistributedTrainingContext.from_root(built.root, device=device)
    _require_supported_distributed_rollout_topology(training_context, resources)
    # Construct the strategy before any model or Ray actor. Shared-GPU on-demand
    # execution needs complete trainer-state parking; distributed strategies must
    # reject that topology here instead of failing after expensive launch work.
    strategy = build_strategy(built.root, training_context)
    if (
        resources.lifecycle.release_rollout_before_train
        or resources.lifecycle.release_trainer_before_reward
    ):
        strategy.validate_training_state_parking()
    # Under ddp every torchrun rank owns a distinct GPU: DistributedTrainingContext.from_root
    # returns cuda:<local_rank>, which overrides the resolver's (rank-agnostic)
    # trainer device so the trainer model, rollout, and weight sync all land on
    # this rank's card. single_process passes the resolver device straight through.
    device = training_context.device
    # Rewards execute in this driver process. A dedicated local reward reservation
    # must therefore select the reward model's actual CUDA device; cross-node reward
    # ordinals are remote budget tokens and fail here before any model is loaded.
    reward_inputs = resolved.reward_inputs(trainer_device=device)
    data_config = built.root.data
    if family_entry.task in {"i2v", "v2w"}:
        preprocessing = data_config.preprocessing if data_config is not None else None
        conditioning = preprocessing.conditioning if preprocessing is not None else None
        if conditioning != "reference_image":
            raise ValueError(
                f"{family_entry.family} requires data.preprocessing.conditioning=reference_image",
            )

    resolved_model = resolve_model(
        family_entry,
        built.root,
        device,
        precision=built.precision,
        for_rollout=False,
    )
    model_identity = resolved_model.identity
    validate_checkpoint_compatibility(
        resume_checkpoint,
        family=family_entry.family,
        expected_model_identity=model_identity,
        strict=resume_config.strict,
    )

    examples = (
        load_prompt_examples_from_config(data_config)
        if provided_examples is None
        else provided_examples
    )
    artifact_data_root = data_config.artifact_data_root if data_config is not None else None
    examples = [
        resolve_prompt_example_references(
            example,
            data_root=artifact_data_root,
            allow_absolute=True,
        )
        for example in examples
    ]
    if family_entry.task in {"i2v", "v2w"}:
        from vrl.trainers.data.artifacts import validate_reference_images

        validate_reference_images(
            examples,
            manifest_path=Path(str(data_config.manifest or "manifest")),
            default_reference_image=(
                data_config.preprocessing.reference_image if data_config.preprocessing else None
            ),
        )
    # Derive the per-rank resume verdict the trainer/weight-syncer read below. Kept
    # after the checkpoint-identity preflight so an incompatible checkpoint fails
    # fast before we start reading its epoch/step/dir fields.
    resumed = resume_checkpoint is not None
    resume_epoch = resume_checkpoint.next_epoch if resume_checkpoint is not None else None
    resume_step = resume_checkpoint.next_step if resume_checkpoint is not None else None
    resume_dir = resume_checkpoint.checkpoint_dir if resume_checkpoint is not None else None

    log_host_memory("before_trainer_bundle_build", log=logger)
    bundle = resolved_model.materialize(context="replay bundle construction")
    log_host_memory("after_trainer_bundle_build", log=logger)
    if family_entry.policy_semantics.step_kind == "denoise":
        enable_transformer_gradient_checkpointing(bundle, built.root)
    model = require_runtime_model(
        bundle.model,
        owner=f"{family_entry.family}.bundle.model",
    )
    # Scheduler feeds the flow-matching evaluator when the family bundle has one.
    scheduler = getattr(bundle, "scheduler", None)

    # One run-level Ray placement group owns the trainer/reward reservations and
    # rollout bundles for the whole run. It is created after the trainer model is
    # placed and before reward/rollout construction: the rollout actor receives
    # its role placement, while the local reward model uses the reserved physical
    # device selected above.
    placement_owner = GlobalRayPlacementOwner(
        resources,
        generation_config.worker,
    )
    ray = require_ray()
    lifecycle = _OnlineRecipeLifecycle(
        placement_owner=placement_owner,
        strategy=strategy,
    )
    run_error: BaseException | None = None
    try:
        lifecycle.ray_session = _RayClusterSession.connect(
            ray,
            cross_node=resources.cross_node,
            local_num_cpus=placement_owner.required_local_cluster_cpus(),
        )
        if resources.cross_node:
            cross_node_preflight(ray, resources)
        placement_owner.create()
        _validate_reward_placement(
            placement_owner.reward_placement,
            resources=resources,
            reward_device=reward_inputs.device,
        )
        collector_config = resolved.collector
        reward_runtime = lifecycle.reward_runtime = build_reward_runtime(reward_inputs)
        algorithm_and_evaluator = build_algorithm_and_evaluator(
            family_entry=family_entry,
            built=built,
            collector_config=collector_config,
            scheduler=scheduler,
        )
        # An unreachable or wrong-identity external reward service must fail
        # here, before the expensive rollout backend launch — not after the
        # first generation batch reaches scoring.
        await reward_runtime.preflight()
        collector = lifecycle.collector = build_rollout_collector(
            family_entry,
            reward_runtime=reward_runtime,
            config=collector_config,
            lifecycle=resources.lifecycle,
        )
        generation_launcher = RayGenerationLauncher()
        generation_config.validate_driver_state(driver_bundle=bundle)
        generation_launch_inputs = resolved.ray_launch_inputs(resolved_model)
        log_host_memory("before_rollout_backend_build", log=logger)
        generation_runtime = generation_launcher.create_runtime(
            generation_config,
            generation_launch_inputs,
            placement=placement_owner.rollout_placement,
        )
        # The generation twin of reward_runtime.preflight() above: a launched
        # fleet must answer one bounded health probe before the schedule starts.
        await generation_runtime.preflight()
        collector.set_generation_runtime(generation_runtime)
        log_host_memory("after_rollout_backend_build", log=logger)

        # Forward-process objectives (DiffusionNFT, V-GRPO) own their behaviour
        # policy through the previous adapter and run no evaluator; only the
        # evaluator-backed objectives read a reference model for KL.
        ref_model = (
            _default_reference_model(bundle, built)
            if family_entry.policy_semantics.step_kind == "denoise"
            and algorithm_and_evaluator.evaluator is not None
            else None
        )
        # The strategy built during preflight is the single owner of trainable-state
        # export for both rollout weight sync and checkpointing. prepare_model
        # (called once in the trainer) creates any process group and wraps the
        # trainable transformer only after topology/capability validation passed.
        trainer = OnlineTrainer(
            algorithm=algorithm_and_evaluator.algorithm,
            collector=collector,
            evaluator=algorithm_and_evaluator.evaluator,
            model=model,
            ref_model=ref_model,
            weight_syncer=RayRuntimeWeightSyncer.if_supported(
                collector.generation_runtime,
                initial_policy_version=resume_step,
            ),
            # Rollout weight sync re-reads live trainable state on every push, so
            # bind the strategy export lazily instead of snapshotting once.
            sync_state_getter=lambda: strategy.export_rollout_state(bundle),
            config=trainer_config,
            device=device,
            strategy=strategy,
            sft_latents=_load_sft_latents_from_config(
                built,
                family_entry.family,
            ),
        )
        lifecycle.rollout_schedule = trainer.rollout_schedule

        if resume_checkpoint is not None:
            restore_training_checkpoint(
                resume_checkpoint,
                trainer=trainer,
                bundle=bundle,
                family=family_entry.family,
                expected_model_identity=model_identity,
                strict=resume_config.strict,
            )
            logger.info(
                "Resuming from %s, start_epoch=%d",
                resume_dir,
                resume_epoch,
            )

        # Under any multi-rank strategy (ddp/fsdp) only rank0 owns run IO
        # (metrics/checkpoint/eval/resolved-config): the cross-rank collectives in
        # trainer.step (DDP grad all-reduce / FSDP all-gather+reduce-scatter) keep
        # ranks in lockstep, so rank0's gathered checkpoint is complete and a single
        # writer avoids N ranks racing the same files (on 2 servers, rank0's host).
        is_primary = training_context.is_primary
        output_dir = Path(trainer_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if is_primary:
            save_resolved_config(cfg, output_dir, resumed=resumed)

        component_names = tuple(reward_config.weights)

        rng = torch.Generator().manual_seed(run_config.seed)
        start_epoch = resume_epoch if resume_epoch is not None else 0
        if start_epoch > run_config.total_epochs:
            raise ValueError(
                "resume checkpoint starts after configured total_epochs: "
                f"start_epoch={start_epoch}, total_epochs={run_config.total_epochs}",
            )
        if resume_checkpoint is not None:
            restore_rng_state(resume_checkpoint.rng_state, prompt_generator=rng)
            # A full-param checkpoint is ~20 GB. Each torchrun rank loads its own
            # CPU payload, so retaining these dicts while three colocated rollout
            # models park on CPU exceeds this host's Ray memory threshold. All
            # consumers above have restored/captured their values; clear the
            # mutable payload before rollout actors load, then return pages now.
            resume_checkpoint.payload.clear()
            resume_checkpoint = None
            gc.collect()
            log_host_memory("after_resume_checkpoint_release", log=logger)

        adapter_exports = build_adapter_exports(
            bundle,
            use_lora=bool(built.root.model.use_lora),
        )
        run = OnlineRecipeRun(
            bundle=bundle,
            trainer=trainer,
            strategy=strategy,
            family=family_entry.family,
            component_names=component_names,
            adapter_exports=adapter_exports,
            csv_path=output_dir / "metrics.csv",
            rng=rng,
            resume_epoch=resume_epoch,
            model_identity=model_identity,
        )
        run.prepare_metrics_csv_rank_consistent(training_context)

        logger.info(
            "Starting %s online recipe: epochs=%d examples=%d n=%d",
            family_entry.family,
            run_config.total_epochs,
            len(examples),
            batch_plan.n_samples_per_prompt,
        )

        rank_batch = batch_plan.prompts_per_batch
        prompt_sampler = PromptBatchSampler(
            generator=rng,
            num_examples=len(examples),
            prompts_per_rank=rank_batch,
            num_replicas=training_context.world_size,
            rank=training_context.rank,
            strategy=str(data_config.sampler.type),
        )
        for epoch in range(start_epoch, run_config.total_epochs):
            indices = prompt_sampler.sample(epoch=epoch)
            example_batch = [examples[i] for i in indices]
            next_example_batch: list[Any] | None = None
            if epoch + 1 < run_config.total_epochs:
                next_indices = prompt_sampler.preview(epoch=epoch + 1)
                next_example_batch = [examples[i] for i in next_indices]
            # This wall range deliberately encloses collect, replay/backward,
            # optimizer step, and the post-step rollout weight sync. It is the
            # denominator for update-level barrier attribution in nsys traces.
            with profile_range("trainer.optimizer_update"):
                if batch_plan.streaming:
                    # Streaming accumulation: split the optimizer-target batch into
                    # microbatches collected/trained/released one at a time so host
                    # RAM does not have to hold the whole batch at once.
                    metrics = await _run_streaming_optimizer_update(
                        trainer,
                        example_batch,
                        batch_plan=batch_plan,
                        next_example_batch=next_example_batch,
                    )
                else:
                    metrics = await trainer.step(
                        example_batch,
                        next_prompts=next_example_batch,
                    )
            if is_primary:
                run.write_metric_row(epoch, metrics)

            # Checkpoint on EVERY rank (NOT gated by is_primary): the trainable-state
            # export inside is a collective under FSDP2 (all ranks all-gather), and
            # save_checkpoint writes files on the primary only. Gating the call to
            # rank0 deadlocks FSDP (rank0 waits at the gather for peers that skipped).
            if run_config.save_freq > 0 and (epoch + 1) % run_config.save_freq == 0:
                run.save_checkpoint(output_dir / f"checkpoint-{epoch + 1}", epoch=epoch + 1)

        # Final checkpoint on EVERY rank too (collective gather inside; rank0 writes).
        run.save_checkpoint(
            output_dir / "checkpoint-final",
            epoch=run_config.total_epochs,
        )
        if is_primary:
            logger.info("Training complete. Final checkpoint: %s", output_dir / "checkpoint-final")
    except BaseException as exc:
        run_error = exc
        # Log BEFORE lifecycle.shutdown: shutdown crosses collective barriers
        # (park_training_state -> all_ranks_succeeded), so on a multi-rank run
        # a peer still inside the update path deadlocks the barrier and the
        # re-raise below never prints. Without this line the failing rank's
        # traceback dies with the NCCL watchdog SIGABRT (observed on the hpsv3
        # fsdp 4-rank smoke, 2026-08-16).
        logger.exception("online recipe failed before shutdown: %r", exc)
        raise
    finally:
        await lifecycle.shutdown(run_error=run_error)


def _preflight_production_video_reward(root: RootConfig) -> None:
    """Fail fast on the driver if the production reward backend is unimportable."""

    production = root.production
    if production is None or not production.kling_video_reward.enabled:
        return
    from vrl.rewards.models.kling_video_reward import preflight_kling_video_reward_backend

    try:
        preflight_kling_video_reward_backend()
    except Exception as exc:
        raise RuntimeError(
            "production.kling_video_reward requires the repo-owned Kling VideoReward "
            "inference backend under vrl/rewards/models/kling_video_reward.py.",
        ) from exc


__all__ = [
    "run_online_recipe",
]
