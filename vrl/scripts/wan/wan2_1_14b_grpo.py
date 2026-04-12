"""Wan2.1-14B GRPO training (multi-GPU, official repo).

Uses OfficialWanModel (WanT2V/WanI2V dual-expert) with:
  WanCollector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Usage:
    python -m vrl.scripts.wan.wan2_1_14b_grpo --repo-dir /path/to/Wan2.1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WanGRPOConfig:
    """Top-level configuration for Wan GRPO training."""

    # Model
    repo_dir: str = ""
    checkpoint_dir: str = ""
    task_type: str = "text_to_video"
    model_size: str = "A14B"

    # Generation
    width: int = 832
    height: int = 480
    frame_count: int = 81
    num_steps: int = 50
    guidance_scale: float = 5.0
    shift: float = 5.0
    sample_solver: str = "dpmpp"

    # LoRA
    lora_rank: int = 128
    lora_alpha: int = 64

    # Training
    lr: float = 1e-5
    num_epochs: int = 1000
    group_size: int = 4
    num_inner_epochs: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    beta: float = 0.01
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    timestep_fraction: float = 1.0

    # EMA
    ema: bool = False
    ema_decay: float = 0.9999

    # Sampling — from flow_grpo config.sample.*
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    noise_level: float = 1.0
    sde_type: str = "sde"

    # Data
    prompt_file: str = ""
    prompts: list[str] = field(default_factory=list)

    # Reward
    reward_type: str = "aesthetic"

    # Logging
    log_interval: int = 1
    save_interval: int = 100
    output_dir: str = "outputs/wan_grpo"


async def train(config: WanGRPOConfig) -> None:
    """Main training loop."""
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rollouts.collectors.wan import WanCollector, WanCollectorConfig
    from vrl.models.families.wan.official import OfficialWanModel
    from vrl.rewards.multi import MultiReward, _register_builtins, get_reward
    from vrl.models.base import VideoGenerationRequest
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    # 1. Load Wan model
    logger.info("Loading Wan model from %s", config.repo_dir)
    wan_model = OfficialWanModel(
        repo_dir=config.repo_dir,
        default_checkpoint_dir=config.checkpoint_dir or None,
    )
    await wan_model.load()

    # 2. Build a template request for encode_text / denoise_init
    template_request = VideoGenerationRequest(
        prompt="placeholder",
        model_name=f"wan-{config.model_size}",
        task_type=config.task_type,
        model_size=config.model_size,
        width=config.width,
        height=config.height,
        frame_count=config.frame_count,
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        shift=config.shift,
        sample_solver=config.sample_solver,
    )

    # 3. Get pipeline and initialize scheduler for the evaluator
    # We do a dummy denoise_init to get the scheduler
    state: dict[str, Any] = {}
    encode_result = await wan_model.encode_text(template_request, state)
    state.update(encode_result.state_updates or {})
    denoise_loop = await wan_model.denoise_init(template_request, state)
    scheduler = denoise_loop.model_state.scheduler
    pipeline = denoise_loop.model_state.pipeline

    # 4. Apply LoRA to the transformer(s)
    # The active transformer is accessible via pipeline
    transformer = pipeline.high_noise_model  # or whichever is trainable
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        transformer = get_peft_model(transformer, lora_config)
        logger.info("Applied LoRA (rank=%d) to transformer", config.lora_rank)
    except ImportError:
        logger.warning("peft not installed — training full model (no LoRA)")

    # 5. Build reward function
    _register_builtins()
    reward_cls = get_reward(config.reward_type)
    reward_fn = reward_cls(device="cuda")

    # 6. Wire up CEA pipeline
    collector_config = WanCollectorConfig(
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        sample_solver=config.sample_solver,
        shift=config.shift,
        kl_reward=config.kl_reward,
        sde_window_size=config.sde_window_size,
        sde_window_range=config.sde_window_range,
        same_latent=config.same_latent,
    )
    collector = WanCollector(
        wan_model, reward_fn, collector_config,
        request_template=template_request,
    )

    evaluator = FlowMatchingEvaluator(
        scheduler,
        noise_level=config.noise_level,
        sde_type=config.sde_type,
    )

    grpo_config = GRPOConfig(
        clip_eps=config.clip_range,
        kl_coeff=config.beta,
        adv_clip_max=config.adv_clip_max,
    )
    algorithm = GRPO(grpo_config)

    trainer_config = TrainerConfig(
        lr=config.lr,
        max_grad_norm=config.max_grad_norm,
        num_inner_epochs=config.num_inner_epochs,
        group_size=config.group_size,
        clip_range=config.clip_range,
        adv_clip_max=config.adv_clip_max,
        beta=config.beta,
        mixed_precision=config.mixed_precision,
        ema=config.ema,
        ema_decay=config.ema_decay,
        timestep_fraction=config.timestep_fraction,
    )

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=transformer,
        config=trainer_config,
    )

    # 7. Load prompts
    prompts = config.prompts
    if config.prompt_file and Path(config.prompt_file).exists():
        prompts = Path(config.prompt_file).read_text().strip().splitlines()
    if not prompts:
        prompts = ["a beautiful sunset over the ocean"]

    # 8. Training loop
    logger.info("Starting Wan GRPO training for %d epochs", config.num_epochs)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        # Cycle through prompts
        prompt_batch = [prompts[epoch % len(prompts)]]

        # Update the request prompt for collection
        metrics = await trainer.step(prompt_batch)

        if epoch % config.log_interval == 0:
            logger.info(
                "Epoch %d | loss=%.4f policy_loss=%.4f kl=%.4f "
                "reward=%.4f+/-%.4f clip_frac=%.3f",
                epoch,
                metrics.loss,
                metrics.policy_loss,
                metrics.kl_penalty,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.clip_fraction,
            )

        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch+1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt_path / "lora_weights")
            logger.info("Saved checkpoint to %s", ckpt_path)

    logger.info("Training complete.")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Wan GRPO Training")
    parser.add_argument("--repo-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/wan_grpo")
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--mixed-precision", type=str, default="bf16")

    args = parser.parse_args()

    config = WanGRPOConfig(
        repo_dir=args.repo_dir,
        checkpoint_dir=args.checkpoint_dir,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        lora_rank=args.lora_rank,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        mixed_precision=args.mixed_precision,
    )

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
