"""Wan2.1-1.3B GRPO training (single-GPU, diffusers).

Uses diffusers.WanPipeline (Wan2.1-T2V-1.3B-Diffusers) with:
  WanDiffusersCollector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Usage:
    python -m vrl.scripts.wan.wan2_1_1_3b_grpo \
        --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --reward-type aesthetic \
        --prompt-file prompts.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Wan1_3BConfig:
    """Configuration matching flow_grpo general_ocr_wan2_1 defaults."""

    # Model — HuggingFace model path or local path
    model_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Generation (from flow_grpo general_ocr_wan2_1)
    width: int = 416
    height: int = 240
    num_frames: int = 33
    num_steps: int = 20
    guidance_scale: float = 4.5

    # LoRA (from flow_grpo: r=32, alpha=64, target attention layers)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_path: str = ""

    # Training (from flow_grpo general_ocr_wan2_1)
    lr: float = 1e-4
    num_epochs: int = 10000
    group_size: int = 4
    num_inner_epochs: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    beta: float = 0.004  # KL loss coefficient
    clip_range: float = 1e-3
    adv_clip_max: float = 5.0
    timestep_fraction: float = 0.99
    gradient_checkpointing: bool = True

    # EMA (from flow_grpo)
    ema: bool = True
    ema_decay: float = 0.9
    ema_update_interval: int = 8

    # Sampling
    kl_reward: float = 0.0
    sde_window_size: int = 0
    sde_window_range: tuple[int, int] = (0, 10)
    same_latent: bool = False
    global_std: bool = False
    cfg: bool = True

    # Data
    prompt_file: str = ""
    prompts: list[str] = field(default_factory=list)

    # Reward
    reward_type: str = "aesthetic"

    # Logging
    log_interval: int = 1
    save_interval: int = 100
    output_dir: str = "outputs/wan_1_3b_grpo"


async def train(config: Wan1_3BConfig) -> None:
    """Main training loop."""
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.rollouts.collectors.wan_diffusers import (
        WanDiffusersCollector,
        WanDiffusersCollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rewards.multi import MultiReward, _register_builtins, get_reward
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Diffusers WanPipeline
    logger.info("Loading WanPipeline from %s", config.model_path)

    from diffusers import WanPipeline

    pipeline = WanPipeline.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(
        device,
        dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )

    # 2. Apply LoRA to the transformer
    if config.use_lora:
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer.to(device)

        from peft import LoraConfig, PeftModel, get_peft_model

        if config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.lora_path,
            )
            pipeline.transformer.set_adapter("default")
        else:
            target_modules = [
                "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
                "to_k", "to_out.0", "to_q", "to_v",
            ]
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            pipeline.transformer = get_peft_model(
                pipeline.transformer, lora_config,
            )
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to transformer",
            config.lora_rank, config.lora_alpha,
        )
    else:
        pipeline.transformer.requires_grad_(True)
        pipeline.transformer.to(device)

    transformer = pipeline.transformer

    # Gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # 3. Get the scheduler for the evaluator
    pipeline.scheduler.set_timesteps(config.num_steps, device=device)

    # 4. Build reward function
    _register_builtins()
    reward_cls = get_reward(config.reward_type)
    reward_fn = reward_cls(device=str(device))

    # 5. Wire up 4-layer architecture
    collector_config = WanDiffusersCollectorConfig(
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        cfg=config.cfg,
        kl_reward=config.kl_reward,
        sde_window_size=config.sde_window_size,
        sde_window_range=config.sde_window_range,
        same_latent=config.same_latent,
    )
    from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel

    wan_model = DiffusersWanT2VModel(pipeline=pipeline, device=device)
    collector = WanDiffusersCollector(wan_model, reward_fn, collector_config)

    evaluator = FlowMatchingEvaluator(
        pipeline.scheduler,
        noise_level=1.0,  # WAN uses noise_level=1.0
        sde_type="sde",
    )

    grpo_config = GRPOConfig(
        clip_eps=config.clip_range,
        kl_coeff=config.beta,
        adv_clip_max=config.adv_clip_max,
        global_std=config.global_std,
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
        ema_update_interval=config.ema_update_interval,
        timestep_fraction=config.timestep_fraction,
        cfg=config.cfg,
    )

    # Use transformer itself as ref_model (LoRA disable_adapter for ref)
    ref_model = transformer if config.use_lora and config.beta > 0 else None

    trainer = OnlineTrainer(
        algorithm=algorithm,
        collector=collector,
        evaluator=evaluator,
        model=transformer,
        ref_model=ref_model,
        config=trainer_config,
        device=device,
    )

    # 6. Load prompts
    prompts = list(config.prompts)
    if config.prompt_file and Path(config.prompt_file).exists():
        prompts = Path(config.prompt_file).read_text().strip().splitlines()
    if not prompts:
        prompts = [
            "a beautiful sunset over the ocean",
            "a cat playing with a ball of yarn",
            "a robot walking through a forest",
            "fireworks exploding in the night sky",
        ]

    # 7. Training loop
    logger.info(
        "Starting Wan 1.3B GRPO training — %d epochs, %d prompts, group_size=%d",
        config.num_epochs, len(prompts), config.group_size,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        # Cycle through prompts
        prompt_batch = [prompts[epoch % len(prompts)]]

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

    parser = argparse.ArgumentParser(
        description="Wan 1.3B GRPO Training (single GPU)",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/wan_1_3b_grpo")
    parser.add_argument("--reward-type", type=str, default="aesthetic")
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.004)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--clip-range", type=float, default=1e-3)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()

    config = Wan1_3BConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        reward_type=args.reward_type,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        lora_rank=args.lora_rank,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        group_size=args.group_size,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        mixed_precision=args.mixed_precision,
        clip_range=args.clip_range,
        use_lora=not args.no_lora,
        ema=not args.no_ema,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
