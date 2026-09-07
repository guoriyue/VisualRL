"""Run the configured reward over the configured prompts before training does.

A reward that reads something off a prompt row (a target clip, a reference
image, a caption target) fails at scoring time, after the first rollout has
already been generated. This entry runs the same reward function, on the same
rows, with the same metadata projection the collector would hand it, against
synthetic media of the configured sampling geometry, so that failure surfaces
in seconds instead of after a rollout.

What it exercises: the reward factory (every configured component, device
placement, HTTP or in-process transport), ``preflight`` / ``activate``, one
``score`` call with per-component scores, and shutdown. What it does not
claim: anything about the scores' values, which come from random pixels.

Usage::

    python -m vrl.scripts.rewards.preflight \\
        --config experiment/wan_2_1/online_grpo_kling_video_reward \\
        --prompts 4 --device auto

Exit code 0 means every configured reward scored every sampled row; a
non-zero exit prints the failing component's exception.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.config.loading import load_config
from vrl.generation.types import GenerationInput
from vrl.models.families.semantics import task_modality
from vrl.rewards.types import REWARD_GROUP_ID_METADATA_KEY, RewardOutput, RewardSample
from vrl.run import ResolvedReward, _model_family
from vrl.scripts.eval._device import resolve_eval_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Per-row component scores from one preflight pass."""

    prompts: tuple[str, ...]
    output: RewardOutput

    def lines(self) -> list[str]:
        names = list(self.output.components)
        header = ["prompt", *names, "total"]
        rows = [header]
        for index, prompt in enumerate(self.prompts):
            rows.append(
                [
                    prompt[:48],
                    *(f"{self.output.components[name][index]:.4f}" for name in names),
                    f"{self.output.scores[index]:.4f}",
                ]
            )
        widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
        return ["  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="experiment config logical name")
    parser.add_argument("overrides", nargs="*", help="OmegaConf overrides, as for vrl-train")
    parser.add_argument("--prompts", type=int, default=4, help="rows to score from data.manifest")
    parser.add_argument("--eval", action="store_true", help="score data.eval_manifest instead")
    parser.add_argument("--device", default="auto", help="reward device: auto | cpu | cuda:N")
    parser.add_argument("--seed", type=int, default=0, help="seed for the synthetic media")
    return parser


def preflight_rewards(
    cfg: Any,
    *,
    prompts: int,
    use_eval_manifest: bool = False,
    device: torch.device,
    seed: int = 0,
) -> PreflightReport:
    """Score the first ``prompts`` rows of the configured manifest with the configured reward."""

    from vrl.config.builders import build_configs
    from vrl.rollouts.collector.config import RolloutCollectorConfig
    from vrl.rollouts.collector.requests import GenerationRequestBuilder
    from vrl.scripts.common.factory import build_reward_runtime
    from vrl.trainers.data.prompts import load_prompt_examples_from_config

    built = build_configs(cfg)
    if built.reward is None:
        raise ValueError("reward preflight needs a reward section")
    if built.root.data is None:
        raise ValueError("reward preflight needs a data section")
    entry = _model_family(built)
    data = built.root.data
    if use_eval_manifest:
        if not data.eval_manifest:
            raise ValueError("config missing required field: data.eval_manifest")
        data = data.model_copy(update={"manifest": str(data.eval_manifest)})
    examples = load_prompt_examples_from_config(data)[: max(1, int(prompts))]
    if not examples:
        raise ValueError(f"{data.manifest} holds no prompt rows")

    # The collector's own metadata projection: family task defaults, fps,
    # reference inputs, the example's reward metadata.
    request_builder = GenerationRequestBuilder(
        entry=entry, config=RolloutCollectorConfig.from_root(built.root)
    )
    samples = tuple(
        _sample_for(example, index, request_builder, built.root.sampling, entry.task, seed)
        for index, example in enumerate(examples)
    )
    runtime = build_reward_runtime(
        ResolvedReward(config=built.reward, device=str(device), memory_parking_required=False)
    )

    async def _run() -> RewardOutput:
        await runtime.preflight()
        await runtime.activate()
        try:
            return await runtime.score(samples)
        finally:
            await runtime.shutdown()

    output = asyncio.run(_run())
    return PreflightReport(prompts=tuple(sample.prompt for sample in samples), output=output)


def _sample_for(
    example: Any,
    index: int,
    request_builder: Any,
    sampling: Any,
    task: str,
    seed: int,
) -> RewardSample:
    generation_input = (
        example.generation_input()
        if hasattr(example, "generation_input")
        else GenerationInput(prompt=str(example))
    )
    collector_request = request_builder.build(
        [generation_input],
        1,
        metadata=example.reward_metadata() if hasattr(example, "reward_metadata") else None,
        request_overrides=dict(getattr(example, "request_overrides", None) or {}),
    )
    metadata = dict(collector_request.metadata)
    metadata[REWARD_GROUP_ID_METADATA_KEY] = f"{collector_request.request.request_id}:prompt:0"
    return RewardSample(
        prompt=generation_input.prompt,
        output=_synthetic_media(sampling, task, seed + index),
        sample_id=f"preflight-{index}",
        metadata=metadata,
    )


def _synthetic_media(sampling: Any, task: str, seed: int) -> torch.Tensor:
    """Random ``[0, 1]`` pixels in the collector's per-sample layout.

    Video tasks hand rewards ``[C, T, H, W]``, image tasks ``[C, H, W]``, the
    geometry the sampling section declares (a small default when it is unset,
    since only the pipeline is under test).
    """

    height = int(getattr(sampling, "height", None) or 64)
    width = int(getattr(sampling, "width", None) or 64)
    generator = torch.Generator().manual_seed(seed)
    if task_modality(task) == "video":
        frames = int(getattr(sampling, "num_frames", None) or 9)
        return torch.rand(3, frames, height, width, generator=generator)
    return torch.rand(3, height, width, generator=generator)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=list(args.overrides))
    try:
        report = preflight_rewards(
            cfg,
            prompts=args.prompts,
            use_eval_manifest=bool(args.eval),
            device=resolve_eval_device(args.device),
            seed=int(args.seed),
        )
    except Exception:
        logger.exception("reward preflight failed")
        return 1
    print("\n".join(report.lines()))
    print(
        f"✓ reward preflight: {len(report.prompts)} row(s) scored by {list(report.output.components)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
