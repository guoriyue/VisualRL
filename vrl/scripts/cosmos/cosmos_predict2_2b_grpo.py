"""Cosmos Predict2-2B GRPO training (single-GPU, diffusers).

Uses diffusers.Cosmos2VideoToWorldPipeline with:
  CosmosDiffusersCollector → FlowMatchingEvaluator → GRPO → OnlineTrainer

Usage:
    python -m vrl.scripts.cosmos.cosmos_predict2_2b_grpo \
        --model-path nvidia/Cosmos-Predict2-2B-Video2World \
        --reward-type aesthetic \
        --prompt-file prompts.txt \
        --reference-image ref.png
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CosmosPred2Config:
    """Configuration for Cosmos Predict2-2B GRPO training."""

    # Model — HuggingFace model path or local path
    model_path: str = "nvidia/Cosmos-Predict2-2B-Video2World"

    # Generation (Cosmos Predict2 defaults)
    width: int = 1280
    height: int = 704
    num_frames: int = 93
    num_steps: int = 35
    guidance_scale: float = 7.0
    fps: int = 16

    # Reference image for Video2World conditioning
    reference_image: str = ""

    # LoRA
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_path: str = ""

    # Training
    lr: float = 1e-5
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

    # EMA
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
    output_dir: str = "outputs/cosmos_pred2_2b_grpo"


async def train(config: CosmosPred2Config) -> None:
    """Main training loop."""
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.rollouts.collectors.cosmos import (
        CosmosDiffusersCollector,
        CosmosDiffusersCollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rewards.multi import _register_builtins, get_reward
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Cosmos2 Pipeline (bypass safety checker)
    logger.info("Loading Cosmos2VideoToWorldPipeline from %s", config.model_path)

    import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _v2w_mod
    from diffusers import Cosmos2VideoToWorldPipeline

    # Bypass safety checker
    _orig_safety = _v2w_mod.CosmosSafetyChecker

    class _PassthroughSafetyChecker:
        def to(self, device):
            return self

        def check_text_safety(self, prompt):
            return True

        def check_video_safety(self, video):
            return video

    _v2w_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
    try:
        pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        )
    finally:
        _v2w_mod.CosmosSafetyChecker = _orig_safety

    pipeline.set_progress_bar_config(disable=True)
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
                "to_k", "to_q", "to_v", "to_out.0",
                "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
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

    # 5. Load reference image (for Video2World conditioning)
    reference_image = None
    if config.reference_image:
        from PIL import Image

        reference_image = Image.open(config.reference_image).convert("RGB")
        logger.info("Loaded reference image from %s", config.reference_image)

    # 6. Wire up 4-layer architecture
    collector_config = CosmosDiffusersCollectorConfig(
        num_steps=config.num_steps,
        guidance_scale=config.guidance_scale,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        fps=config.fps,
        cfg=config.cfg,
        kl_reward=config.kl_reward,
        sde_window_size=config.sde_window_size,
        sde_window_range=config.sde_window_range,
        same_latent=config.same_latent,
    )
    from vrl.models.families.cosmos.predict2 import DiffusersCosmosPredict2Executor
    from vrl.models.families.cosmos.variants import CosmosVariant

    cosmos_executor = DiffusersCosmosPredict2Executor(
        variant=CosmosVariant.PREDICT2_VIDEO2WORLD,
        model_size="2B",
    )
    cosmos_executor._pipeline = pipeline
    cosmos_executor._modules_loaded = True

    collector = CosmosDiffusersCollector(
        cosmos_executor, reward_fn, collector_config,
        reference_image=reference_image,
    )

    evaluator = FlowMatchingEvaluator(
        pipeline.scheduler,
        noise_level=1.0,
        sde_type="sde",  # Cosmos2 uses FlowMatchEulerDiscreteScheduler — same as Wan
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

    # 7. Load prompts
    prompts = list(config.prompts)
    if config.prompt_file and Path(config.prompt_file).exists():
        prompts = Path(config.prompt_file).read_text().strip().splitlines()
    if not prompts:
        prompts = [
            "a car driving through a cityscape at sunset",
            "waves crashing on a rocky shoreline",
            "a drone flying over mountain terrain",
            "a person walking through a park in autumn",
        ]

    # 8. Training loop
    logger.info(
        "Starting Cosmos Predict2-2B GRPO training — %d epochs, %d prompts, group_size=%d",
        config.num_epochs, len(prompts), config.group_size,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
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
        description="Cosmos Predict2-2B GRPO Training (single GPU)",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="nvidia/Cosmos-Predict2-2B-Video2World",
        help="HuggingFace model path or local directory",
    )
    parser.add_argument("--reference-image", type=str, default="")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/cosmos_pred2_2b_grpo")
    parser.add_argument("--reward-type", type=str, default="aesthetic")
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.004)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--mixed-precision", type=str, default="bf16")
    parser.add_argument("--clip-range", type=float, default=1e-3)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=1)

    args = parser.parse_args()

    config = CosmosPred2Config(
        model_path=args.model_path,
        reference_image=args.reference_image,
        lora_path=args.lora_path,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        reward_type=args.reward_type,
        num_epochs=args.num_epochs,
        lr=args.lr,
        beta=args.beta,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        group_size=args.group_size,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
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
