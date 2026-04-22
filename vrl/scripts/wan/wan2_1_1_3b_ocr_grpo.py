"""Wan2.1-1.3B OCR GRPO training (single-GPU, diffusers).

Trains the model to generate videos containing readable text matching a target
string. Uses PromptExample manifests (JSONL) with per-sample target_text.

Usage:
    python -m vrl.scripts.wan.wan2_1_1_3b_ocr_grpo \
        --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers

    # Override with your own manifest
    python -m vrl.scripts.wan.wan2_1_1_3b_ocr_grpo --manifest path/to/prompts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "datasets" / "ocr" / "train.txt"


@dataclass
class WanOCRConfig:
    """Configuration for OCR GRPO training."""

    # Model
    model_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    # Generation
    width: int = 416
    height: int = 240
    num_frames: int = 33
    num_steps: int = 20
    guidance_scale: float = 4.5

    # LoRA
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_path: str = ""

    # torch.compile on the transformer (L3 perf)
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"  # "default" | "reduce-overhead" | "max-autotune"

    # Profile each training phase (collect / advantage / evaluate / backward /
    # optim_step) with CUDA sync. Adds sync overhead — use only for A/B tuning.
    profile: bool = False

    # Training — matching flow_grpo general_ocr_wan2_1
    lr: float = 1e-4
    num_epochs: int = 10000
    group_size: int = 4
    num_inner_epochs: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    beta: float = 0.004
    clip_range: float = 1e-4
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
    global_std: bool = False  # flow_grpo general_ocr_wan2_1: False — per-prompt std preserves local signal when prompt difficulty varies
    cfg: bool = True

    # Data — JSONL manifest with PromptExample fields
    manifest: str = str(DEFAULT_MANIFEST)
    eval_manifest: str = ""
    prompts_per_step: int = 1
    # Rollout batches per outer epoch (flow_grpo `num_batches_per_epoch`).
    # Each epoch collects `num_batches_per_epoch * prompts_per_step` prompts and
    # trains on the full stack — a larger effective batch without more optimizer
    # steps. Orthogonal to gradient accumulation.
    num_batches_per_epoch: int = 1

    # Reward mix — weights set to 0 disable the component.
    # OCR alone is prone to reward hacking (PPT-style text, static frames).
    # Stack aesthetic + clipscore to regularise visual quality + prompt fidelity.
    # Note scale: aesthetic raw output ~[3, 9], clipscore ~[0, 1.5], ocr [0, 1].
    # Weights here are applied AFTER raw reward values, so tune accordingly.
    ocr_weight: float = 1.0
    aesthetic_weight: float = 0.0
    clip_weight: float = 0.0

    # Logging
    log_interval: int = 1
    save_interval: int = 50
    output_dir: str = "outputs/wan_1_3b_ocr_grpo"
    ocr_debug_dir: str = ""  # if set, saves OCR debug frames
    seed: int = 0


async def train(config: WanOCRConfig) -> None:
    """Main training loop."""
    import torch

    from vrl.algorithms.grpo import GRPO, GRPOConfig
    from vrl.rollouts.collectors.wan_diffusers import (
        WanDiffusersCollector,
        WanDiffusersCollectorConfig,
    )
    from vrl.rollouts.evaluators.diffusion.flow_matching import FlowMatchingEvaluator
    from vrl.rewards.multi import MultiReward
    from vrl.trainers.data import PromptExample, load_prompt_manifest
    from vrl.trainers.online import OnlineTrainer
    from vrl.trainers.types import TrainerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load pipeline
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

    # 2. Apply LoRA
    if config.use_lora:
        pipeline.transformer.requires_grad_(False)
        pipeline.transformer.to(device)
        from peft import LoraConfig, PeftModel, get_peft_model

        if config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, config.lora_path,
                is_trainable=True,
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
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
        logger.info("Applied LoRA (rank=%d, alpha=%d)", config.lora_rank, config.lora_alpha)
    else:
        pipeline.transformer.requires_grad_(True)
        pipeline.transformer.to(device)

    transformer = pipeline.transformer

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # torch.compile the transformer — traces and fuses kernels for the hot path.
    # Works with LoRA (compile wraps the PeftModel); first call pays a ~1-2 min
    # trace cost but subsequent denoise steps run significantly faster.
    # Incompatible with gradient_checkpointing in some cases — if compile breaks,
    # disable gradient_checkpointing first.
    if config.torch_compile:
        logger.info("Compiling transformer with mode=%s", config.compile_mode)
        pipeline.transformer = torch.compile(
            pipeline.transformer,
            mode=config.compile_mode,
            fullgraph=False,  # Wan DiT has conditional paths; avoid fullgraph
        )
        transformer = pipeline.transformer

    # 3. Scheduler
    pipeline.scheduler.set_timesteps(config.num_steps, device=device)

    # 4. Build reward function — three-layer defence against reward hacking.
    # Layer 1: OCR (primary signal, easy to game).
    # Layer 2: aesthetic (keeps visual quality from collapsing).
    # Layer 3: clipscore (keeps prompt semantics alive).
    # KL penalty (beta) is applied inside GRPO, not here.
    reward_weights: dict[str, float] = {}
    if config.ocr_weight > 0:
        reward_weights["ocr"] = config.ocr_weight
    if config.aesthetic_weight > 0:
        reward_weights["aesthetic"] = config.aesthetic_weight
    if config.clip_weight > 0:
        reward_weights["clipscore"] = config.clip_weight

    if not reward_weights:
        raise ValueError(
            "At least one of --ocr-weight / --aesthetic-weight / --clip-weight must be > 0"
        )

    # Per-reward init kwargs (currently only OCR takes debug_dir).
    reward_kwargs: dict[str, dict] = {}
    if "ocr" in reward_weights and config.ocr_debug_dir:
        reward_kwargs["ocr"] = {"debug_dir": config.ocr_debug_dir}

    reward_fn = MultiReward.from_dict(
        reward_weights,
        device=str(device),
        reward_kwargs=reward_kwargs,
    )
    logger.info("Reward mix: %s", reward_weights)
    if config.ocr_debug_dir:
        logger.info("OCR debug frames → %s", config.ocr_debug_dir)

    # 5. Wire up architecture
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
        noise_level=1.0,
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
        profile=config.profile,
        debug_first_step=True,
    )

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

    # 6. Load prompts from manifest
    manifest_path = Path(config.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            f"Default demo manifest is {DEFAULT_MANIFEST}."
        )
    examples: list[PromptExample] = load_prompt_manifest(manifest_path)

    logger.info(
        "Starting OCR GRPO training — %d epochs, %d examples, group_size=%d",
        config.num_epochs, len(examples), config.group_size,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV log — include per-component reward columns (empty when unused).
    csv_path = output_dir / "metrics.csv"
    csv_component_names = list(reward_weights.keys())
    component_cols = ",".join(f"r_{n}" for n in csv_component_names)
    csv_path.write_text(
        "epoch,loss,policy_loss,kl_penalty,reward_mean,reward_std,"
        "clip_fraction,approx_kl,advantage_mean,"
        "grad_norm,adv_saturation,adv_zero_rate," + component_cols + "\n"
    )

    # 7. Training loop
    if config.global_std and config.prompts_per_step == 1:
        logger.warning(
            "global_std collapses to per-group std with a single prompt per step; "
            "consider --prompts-per-step >= 2"
        )

    # i.i.d. sampling — GRPO per-prompt stat tracking assumes prompts are drawn
    # independently; deterministic cycling biases the advantage estimate.
    rng = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.num_epochs):
        n = config.num_batches_per_epoch * config.prompts_per_step
        idx = torch.randperm(len(examples), generator=rng)[:n].tolist()
        example_batch = [examples[i] for i in idx]

        # OnlineTrainer.step() accepts list[str | PromptExample]
        metrics = await trainer.step(example_batch)

        if epoch % config.log_interval == 0:
            # Pull per-component raw reward means (set on last score call).
            component_means: dict[str, float] = {}
            last = getattr(reward_fn, "last_components", {}) or {}
            for name in csv_component_names:
                values = last.get(name, [])
                component_means[name] = (
                    sum(values) / len(values) if values else float("nan")
                )
            component_str = " ".join(
                f"{n}={component_means[n]:.3f}" for n in csv_component_names
            )
            logger.info(
                "Epoch %d | loss=%.4f kl=%.4f reward=%.4f+/-%.4f "
                "grad_norm=%.4f adv_sat=%.3f adv_zero=%.3f | %s",
                epoch,
                metrics.loss,
                metrics.kl_penalty,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.grad_norm,
                metrics.adv_saturation,
                metrics.adv_zero_rate,
                component_str,
            )
            with open(csv_path, "a") as f:
                component_vals = ",".join(
                    f"{component_means[n]:.4f}" for n in csv_component_names
                )
                f.write(
                    f"{epoch},{metrics.loss:.6f},{metrics.policy_loss:.6f},"
                    f"{metrics.kl_penalty:.6f},"
                    f"{metrics.reward_mean:.4f},{metrics.reward_std:.4f},"
                    f"{metrics.clip_fraction:.4f},{metrics.approx_kl:.6f},"
                    f"{metrics.advantage_mean:.6f},"
                    f"{metrics.grad_norm:.6f},{metrics.adv_saturation:.4f},"
                    f"{metrics.adv_zero_rate:.4f},"
                    + component_vals + "\n"
                )

        if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint-{epoch+1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.state_dict(), ckpt_path / "trainer_state.pt")
            if hasattr(transformer, "save_pretrained"):
                transformer.save_pretrained(ckpt_path / "lora_weights")
            logger.info("Saved checkpoint to %s", ckpt_path)

    # Always save a final LoRA snapshot so it can be loaded for inference,
    # regardless of save_interval. Otherwise a full training run leaves nothing
    # behind on disk for downstream use.
    final_path = output_dir / "checkpoint-final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), final_path / "trainer_state.pt")
    if hasattr(transformer, "save_pretrained"):
        transformer.save_pretrained(final_path / "lora_weights")
    logger.info("Saved final checkpoint to %s", final_path)

    logger.info("Training complete.")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Wan 1.3B OCR GRPO Training")
    parser.add_argument("--model-path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--eval-manifest", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/wan_1_3b_ocr_grpo")
    parser.add_argument("--ocr-debug-dir", type=str, default="")
    parser.add_argument(
        "--ocr-weight", type=float, default=1.0,
        help="OCR reward weight (primary signal). Set 0 to disable."
    )
    parser.add_argument(
        "--aesthetic-weight", type=float, default=0.0,
        help="Aesthetic reward weight. Raw score ~[3,9], so small values (0.05-0.3) matter."
    )
    parser.add_argument(
        "--clip-weight", type=float, default=0.0,
        help="CLIP text-image similarity weight for prompt-fidelity regularisation."
    )
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
    parser.add_argument("--clip-range", type=float, default=1e-4)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--prompts-per-step", type=int, default=1)
    parser.add_argument(
        "--num-batches-per-epoch", type=int, default=1,
        help="Rollout batches per epoch (flow_grpo sample.num_batches_per_epoch). "
             "Effective prompts/epoch = this × --prompts-per-step.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--global-std", action="store_true",
        help="Use global-batch std for advantage normalization (default: per-prompt std, flow_grpo wan2_1).",
    )
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="torch.compile the transformer (L3 perf; ~1-2 min trace on first step).",
    )
    parser.add_argument(
        "--compile-mode", type=str, default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (ignored unless --torch-compile is set).",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Time each training phase with CUDA sync and log per-step results.",
    )

    args = parser.parse_args()

    config = WanOCRConfig(
        model_path=args.model_path,
        lora_path=args.lora_path,
        manifest=args.manifest,
        output_dir=args.output_dir,
        ocr_debug_dir=args.ocr_debug_dir,
        seed=args.seed,
        ocr_weight=args.ocr_weight,
        aesthetic_weight=args.aesthetic_weight,
        clip_weight=args.clip_weight,
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
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        prompts_per_step=args.prompts_per_step,
        num_batches_per_epoch=args.num_batches_per_epoch,
        global_std=args.global_std,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        profile=args.profile,
    )

    asyncio.run(train(config))


if __name__ == "__main__":
    main()
