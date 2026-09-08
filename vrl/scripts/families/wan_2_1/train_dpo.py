"""Wan 2.1 Diffusion-DPO training recipe (offline, Pick-a-Pic v2).

The unified ``vrl.scripts.train`` entry point dispatches Wan-DPO configs here.
Pipeline construction, LoRA, encoders, training loop, and checkpointing live in
this module.

DPO is offline preference learning — no rollout collection, no algorithm
ABC, just a pure functional loss. We therefore drive the synchronous
``OfflineDPOTrainer`` directly rather than going through ``OnlineTrainer``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

if TYPE_CHECKING:
    # Config dispatch imports this module to find ``train_wan_2_1_dpo``; keep
    # torch out of that import path (the helpers below import it themselves).
    import torch

logger = logging.getLogger(__name__)


def _build_encoders(pipeline, num_frames: int, device, dtype):
    """Returns ``(encode_pixels, encode_text)`` closures bound to a WanPipeline.

    ``encode_pixels`` replicates each image to ``num_frames`` along the
    temporal dim before VAE encoding — this lets image-only datasets
    (Pick-a-Pic) train a video model with ``num_frames=1`` (image-style)
    or ``num_frames>1`` (video-style).
    """
    import torch

    vae = pipeline.vae
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )

    def encode_pixels(pixels_2bchw: torch.Tensor) -> torch.Tensor:
        # Replicate to T frames along a new temporal dim.
        x = pixels_2bchw.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        x = x.to(device=device, dtype=vae.dtype)
        latents = vae.encode(x).latent_dist.sample()
        # Inverse of the Wan decode denormalization (raw = z * std + mean, see
        # decode_latents in vrl/models/families/wan_2_1/model.py): the transformer
        # consumes z = (raw - mean) / std.
        latents = (latents.float() - latents_mean) / latents_std
        return latents.to(dtype)

    @torch.no_grad()
    def encode_text(captions: list[str]) -> torch.Tensor:
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=captions,
            negative_prompt=[""] * len(captions),
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        return prompt_embeds.to(dtype)

    return encode_pixels, encode_text


def wan_forward(
    model: torch.nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    extra: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Wan transformer forward; unwrap policy modules before raw backbone call.

    The ``ForwardFn`` this recipe hands to ``OfflineDPOTrainer``. The trainer
    itself stays family-neutral, so the Wan-shaped call — ``hidden_states`` /
    ``timestep`` / ``encoder_hidden_states`` and the tuple return — belongs
    here with the recipe that selects it.
    """
    del extra
    forward_model = getattr(model, "transformer", None)
    if forward_model is None:
        forward_model = model
    out = forward_model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
    return out


def _required(value: Any, path: str) -> Any:
    if value is None:
        raise ValueError(f"config missing required field: {path}")
    return value


def train_wan_2_1_dpo(cfg: DictConfig) -> None:
    """Run Wan-family Diffusion-DPO training driven by a merged YAML config."""

    from vrl.run import resolve_model, resolve_run
    from vrl.trainers.offline import OfflineDPOTrainerConfig

    resolved = resolve_run(cfg)
    built = resolved.built
    family_entry = resolved.family
    resources = resolved.resources
    device = resolved.device
    configured_family = None if built.root.model is None else built.root.model.family
    family = family_entry.family
    if family != "wan_2_1":
        raise ValueError(
            "Wan Diffusion-DPO requires model.family='wan_2_1' "
            f"(alias 'wan' is accepted); got {configured_family!r}",
        )

    from torch.utils.data import DataLoader

    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.models.dtypes import resolve_torch_dtype
    from vrl.ray.resources import format_distributed_resource_plan
    from vrl.trainers.activation_checkpointing import enable_transformer_gradient_checkpointing
    from vrl.trainers.checkpointing import (
        build_adapter_exports,
        capture_rng_state,
        load_training_checkpoint_for_resume,
        restore_rng_state,
        restore_training_checkpoint,
        save_resolved_config,
        save_training_checkpoint,
        validate_checkpoint_compatibility,
    )
    from vrl.trainers.data import collate_preference, load_pickapic
    from vrl.trainers.metrics_io import prepare_metrics_csv
    from vrl.trainers.offline import OfflineDPOTrainer

    trainer_section = built.root.trainer
    if trainer_section is None:
        raise ValueError("config missing required field: trainer")
    sampling = _required(built.root.sampling, "sampling")
    data_cfg = _required(built.root.data, "data")
    preprocessing = _required(data_cfg.preprocessing, "data.preprocessing")
    sampler = _required(data_cfg.sampler, "data.sampler")

    dpo_config = built.algorithm
    if not isinstance(dpo_config, DiffusionDPOConfig):
        raise TypeError(
            f"Wan-DPO expects algorithm.kind=diffusion_dpo, got {type(dpo_config).__name__}",
        )

    precision = built.precision
    trainer_cfg = OfflineDPOTrainerConfig.from_root(built.root, dpo_config)
    assert built.root.actor is not None  # the builder required actor.train_batch_size
    train_batch_size = int(built.root.actor.train_batch_size)
    resume_config = built.resume
    resume_checkpoint = load_training_checkpoint_for_resume(resume_config)
    logger.info(format_distributed_resource_plan(resources))
    weight_dtype = resolve_torch_dtype(precision.training.dtype)

    # DPO needs the full family bundle because its VAE and text encoder prepare
    # preference pairs. Registry selection and model projection stay identical
    # to generation; only the downstream optimizer makes this a training path.
    resolved_model = resolve_model(
        family_entry,
        built.root,
        device,
        precision=precision,
        for_rollout=True,
        precision_role="training",
    )
    model_identity = resolved_model.identity
    validate_checkpoint_compatibility(
        resume_checkpoint,
        family=family,
        expected_model_identity=model_identity,
        strict=resume_config.strict,
    )
    bundle = resolved_model.materialize(context="Wan DPO bundle construction")
    wan_model = bundle.model
    pipeline = bundle.raw_handle

    enable_transformer_gradient_checkpointing(bundle, built.root)

    # 2. Encoders bound to the loaded pipeline
    num_frames = int(sampling.num_frames)
    encode_pixels, encode_text = _build_encoders(
        pipeline,
        num_frames=num_frames,
        device=device,
        dtype=weight_dtype,
    )

    # 3. Data — Pick-a-Pic v2 preference pairs
    # `data.preprocessing.resolution: 0` means "fall back to sampling.height";
    # the schema requires the key for this loader, the `or` keeps that semantic.
    resolution = int(preprocessing.resolution) or int(sampling.height)
    logger.info(
        "Loading Pick-a-Pic from %s split=%s",
        data_cfg.dataset_name,
        data_cfg.split,
    )
    ds = load_pickapic(
        split=str(data_cfg.split),
        cache_dir=str(data_cfg.cache_dir) or None,
        max_samples=data_cfg.max_train_samples,
        resolution=resolution,
        random_crop=bool(preprocessing.random_crop),
        no_hflip=not bool(preprocessing.horizontal_flip),
        dataset_name=str(data_cfg.dataset_name),
    )
    dataloader = DataLoader(
        ds,
        batch_size=train_batch_size,
        shuffle=bool(sampler.shuffle),
        num_workers=int(sampler.dataloader_num_workers),
        collate_fn=collate_preference,
        drop_last=bool(sampler.drop_last),
    )
    logger.info("Loaded %d preference pairs", len(ds))

    # 4. Trainer — its config was resolved before runtime/data construction so
    # unsupported optimizer modes fail without paying model or dataset startup.
    pipeline.scheduler.set_timesteps(
        pipeline.scheduler.config.num_train_timesteps,
        device=device,
    )
    trainer = OfflineDPOTrainer(
        model=wan_model,
        ref_model=None,  # use LoRA disable_adapter for ref
        forward_fn=wan_forward,
        noise_scheduler=pipeline.scheduler,
        encode_pixels=encode_pixels,
        encode_text=encode_text,
        config=trainer_cfg,
        device=device,
    )
    if resume_checkpoint is not None:
        restore_training_checkpoint(
            resume_checkpoint,
            trainer=trainer,
            bundle=bundle,
            family="wan_2_1",
            expected_model_identity=model_identity,
            strict=resume_config.strict,
        )
        logger.info(
            "Resuming from %s, start_step=%d",
            resume_checkpoint.checkpoint_dir,
            resume_checkpoint.next_step,
        )
        restore_rng_state(resume_checkpoint.rng_state)

    # 5. Output dir + CSV log + resolved config snapshot
    out_dir = Path(str(trainer_section.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out_dir, resumed=resume_checkpoint is not None)

    csv_path = out_dir / "metrics.csv"
    # Derive the metrics columns from DPOStepMetrics so adding a metric updates
    # both the header and each row instead of silently mis-columning the CSV.
    from dataclasses import fields as _dc_fields

    from vrl.trainers.offline.dpo import DPOStepMetrics

    metric_fields = tuple(f.name for f in _dc_fields(DPOStepMetrics))
    prepare_metrics_csv(
        csv_path,
        ("step", *metric_fields),
        resume_at=(
            ("step", resume_checkpoint.next_step) if resume_checkpoint is not None else None
        ),
    )

    # 6. Training loop
    max_train_steps = int(_required(trainer_section.max_train_steps, "trainer.max_train_steps"))
    checkpointing_steps = int(
        _required(trainer_section.checkpointing_steps, "trainer.checkpointing_steps"),
    )
    log_interval = int(_required(trainer_section.log_interval, "trainer.log_interval"))

    logger.info(
        "Starting Wan-1.3B DPO — %d steps, beta=%g, lr=%g, num_frames=%d",
        max_train_steps,
        trainer_cfg.beta,
        trainer_cfg.lr,
        num_frames,
    )

    # LoRA-only checkpoints export the adapter weights; full fine-tunes export
    # nothing extra. The decision is fixed for the run, so resolve it once.
    adapter_exports = build_adapter_exports(
        bundle,
        use_lora=bool(built.root.model.use_lora),
    )

    step = resume_checkpoint.next_step if resume_checkpoint is not None else 0
    if step > max_train_steps:
        raise ValueError(
            "resume checkpoint starts after configured max_train_steps: "
            f"start_step={step}, max_train_steps={max_train_steps}",
        )
    data_iter = iter(dataloader)
    while step < max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        m = trainer.step(batch)

        if step % log_interval == 0:
            logger.info(
                "step %d | loss=%.4f acc=%.3f model_diff=%.4f ref_diff=%.4f "
                "model_mse=%.4f ref_mse=%.4f gn=%.3f",
                step,
                m.loss,
                m.implicit_acc,
                m.model_diff,
                m.ref_diff,
                m.raw_model_loss,
                m.raw_ref_loss,
                m.grad_norm,
            )
            with open(csv_path, "a") as f:
                row = ",".join(f"{getattr(m, name):.6f}" for name in metric_fields)
                f.write(f"{step},{row}\n")

        if checkpointing_steps > 0 and (step + 1) % checkpointing_steps == 0:
            ckpt = out_dir / f"checkpoint-{step + 1}"
            save_training_checkpoint(
                ckpt,
                trainer=trainer,
                bundle=bundle,
                family="wan_2_1",
                progress={
                    "completed_step": step + 1,
                    "next_step": step + 1,
                    "global_step": trainer.global_step,
                },
                rng_state=capture_rng_state(),
                adapter_exports=adapter_exports,
                model_identity=model_identity,
            )
            logger.info("Saved checkpoint -> %s", ckpt)

        step += 1

    final_path = out_dir / "checkpoint-final"
    save_training_checkpoint(
        final_path,
        trainer=trainer,
        bundle=bundle,
        family="wan_2_1",
        progress={
            "completed_step": max_train_steps,
            "next_step": max_train_steps,
            "global_step": trainer.global_step,
        },
        rng_state=capture_rng_state(),
        adapter_exports=adapter_exports,
        model_identity=model_identity,
    )
    logger.info("Final checkpoint -> %s", final_path)
    logger.info("DPO training complete.")
