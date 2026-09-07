"""FLUX.1 t2i diffusers-backed model.

Diffusion implementation for Black Forest Labs FLUX.1 image generation. The
generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does one
guidance-distilled transformer forward and returns the noise prediction.

FLUX specifics vs SD3 (the reference family):
- Latents are PACKED: prepare_latents returns ``[B, (h/2)*(w/2), C*4]`` 2x2
  patch-packed tokens (not ``[B, C, H, W]``). They stay packed through the whole
  denoise loop; only ``decode_latents`` unpacks before the VAE.
- The transformer needs rotary position ids: ``img_ids`` (per latent patch) and
  ``txt_ids`` (per text token, all zeros for FLUX). Both are batch-independent
  and deterministic from the spatial shape, so the replay path rebuilds them from
  ``height``/``width`` instead of storing per-sample copies.
- The transformer time embedding multiplies its input by 1000 internally, so
  ``forward_step`` feeds ``t / 1000`` (matching diffusers' FluxPipeline loop).
- FLUX.1-dev is guidance-distilled (``config.guidance_embeds``): a ``guidance``
  scalar is embedded, there is no classifier-free unconditional branch. So
  ``do_cfg`` is always False and the single-branch runner runs one forward.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDiffusionSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
    set_mu_shifted_timesteps,
)
from vrl.models.steps.denoise.common.lora import (
    LoraModelMixin,
    require_lora_for_previous_policy_adapter,
)


@dataclass
class FluxSamplingState(GuidedDiffusionSamplingStateBase):
    """Private FLUX sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    latent_image_ids: torch.Tensor
    guidance_embeds: bool
    height: int
    width: int


class FluxModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed FLUX.1 t2i model.

    The frozen ``previous`` LoRA mirror DiffusionNFT and V-GRPO evaluate the
    behaviour policy through comes from ``LoraModelMixin``
    (``model.nft_previous_adapter: true``); plain GRPO runs never attach it.

    Implements the backbone-runner protocol itself. FLUX.1-dev is
    guidance-distilled: a single transformer forward conditioned on a
    ``guidance`` embedding, NOT classifier-free guidance with an unconditional
    branch — hence ``single_branch`` and ``forward_step`` always sets
    ``do_cfg=False``; the ``guidance`` scalar rides ``request.extra`` as an
    ordinary conditioning input.
    """

    cfg_mode = "single_branch"
    cfg_base = "cond"

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "FluxPipeline"
    _frozen_encoder_names = ("text_encoder", "text_encoder_2")
    # FLUX.1-dev's T5-XXL encoder is ~9.4 GB and the transformer is ~24 GB
    # (bf16); together they exceed a single 32 GB card. The encoders feed only
    # the one-shot encode_prompt, so park them on CPU (the enable_model_cpu_offload
    # discipline) and leave the whole card for the denoiser. encode_prompt runs
    # them on their own device and moves the embeds to the GPU.
    _prompt_encoder_on_cpu = True

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map FLUX transformer kwargs (packed latents + rotary ids + pooled)."""
        if branch != "cond":
            raise ValueError("FLUX is guidance-distilled and has no uncond branch")
        extra_kwargs = {
            "pooled_projections": request.extra["pooled_projections"],
            "img_ids": request.extra["img_ids"],
            "txt_ids": request.extra["txt_ids"],
        }
        # ``guidance`` is None for non-distilled checkpoints (config.guidance_embeds
        # off); pass it through either way so the transformer's own None-check fires.
        extra_kwargs["guidance"] = request.extra.get("guidance")
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=request.prompt_embeds,
            extra_kwargs=extra_kwargs,
        )

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        # decode_latents only receives the packed latent tensor; the executor runs
        # prepare -> denoise -> decode sequentially on this one model instance per
        # batch (no concurrency), so prepare_sampling records the spatial shape it
        # must unpack to here. Defaults are overwritten on the first prepare call.
        self._decode_height = 1024
        self._decode_width = 1024

    @classmethod
    def from_build(cls, build: ModelBuild) -> FluxModel:
        """Reject the previous-adapter config before paying the pipeline load."""
        require_lora_for_previous_policy_adapter(build)
        return super().from_build(build)

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set FLUX timesteps with the resolution-derived ``mu`` (diffusers parity)."""
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        return set_mu_shifted_timesteps(
            self.scheduler,
            num_steps=num_steps,
            image_seq_len=image_seq_len,
            device=device,
            calculate_shift=calculate_shift,
        )

    @staticmethod
    def _build_latent_image_ids(
        height: int,
        width: int,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        """Rebuild FLUX latent position ids (diffusers ``_prepare_latent_image_ids``).

        Inlined as a pure function so the pipeline-less replay model can rebuild
        the batch-shared position grid without owning a FluxPipeline.
        """
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1]
            + torch.arange(
                height,
                device=device,
                dtype=dtype,
            )[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2]
            + torch.arange(
                width,
                device=device,
                dtype=dtype,
            )[None, :]
        )
        return latent_image_ids.reshape(height * width, 3)

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via FLUX's CLIP (pooled) + T5 (sequence) encoders.

        FLUX.1-dev is guidance-distilled, so there is no unconditional branch and
        negative prompts are unsupported. Returns the T5 sequence embeds, the
        CLIP pooled vector, and the batch-shared ``text_ids`` position grid
        (float32 — rotary positions are computed in float).
        """
        self._reject_unsupported_negative_prompt(negative_prompt)
        max_seq = kwargs.get("max_sequence_length", 512)
        # The frozen encoders live on CPU (see from_build); run encode there, then
        # move the embeds onto the model/transformer device for the denoise forward.
        enc_device = self._encoder_device()
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=enc_device,
        )
        td = self.transformer.dtype
        return {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.device, dtype=td),
            "text_ids": text_ids.to(self.device, dtype=torch.float32),
        }

    def _encoder_device(self) -> Any:
        """Device the frozen prompt encoders live on (CPU when offloaded)."""
        for name in ("text_encoder_2", "text_encoder"):
            enc = getattr(self.pipeline, name, None)
            if enc is not None:
                try:
                    return next(enc.parameters()).device
                except StopIteration:
                    continue
        return self.device

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> FluxSamplingState:
        """Build the per-request packed-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        text_ids = encoded["text_ids"]

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        # FLUX packs 2x2 patches into channels, so the transformer's in_channels
        # is 4x the unpacked latent channel count.
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        # Timesteps depend on the packed image sequence length (dynamic shifting),
        # so set them only now that the latents (hence seq len) exist.
        timesteps = self._set_dynamic_timesteps(
            request.num_steps,
            latents.shape[1],
            device,
        )

        # Record the spatial shape decode_latents must unpack to (safe: same model
        # instance runs prepare -> denoise -> decode sequentially per batch).
        self._decode_height = int(request.height)
        self._decode_width = int(request.width)

        return FluxSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=request.guidance_scale,
            guidance_embeds=self._guidance_embeds,
            height=int(request.height),
            width=int(request.width),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: FluxSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """FLUX transformer forward (single guidance-distilled branch).

        Returns noise_pred (packed) plus the cond/uncond placeholder branches; the
        caller owns scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # The transformer multiplies its timestep input by 1000, so feed t / 1000
        # (matches diffusers FluxPipeline). If t is already [B] (eval path packs
        # [1, B]) expand is a no-op.
        timestep_batch = (
            expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td) / 1000.0
        )
        guidance = None
        if state.guidance_embeds:
            guidance = torch.full(
                (bsz,),
                float(state.guidance_scale),
                device=latent_input.device,
                dtype=td,
            )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=None,
                guidance_scale=state.guidance_scale,
                do_cfg=False,
                output_dtype=td,
                extra={
                    "pooled_projections": state.pooled_prompt_embeds.to(td),
                    # Rotary position ids stay float32 (computed in float).
                    "img_ids": state.latent_image_ids.to(
                        device=latent_input.device,
                        dtype=torch.float32,
                    ),
                    "txt_ids": state.text_ids.to(
                        device=latent_input.device,
                        dtype=torch.float32,
                    ),
                    "guidance": guidance,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: FluxSamplingState) -> dict[str, Any]:
        """Project FLUX sampling state into trajectory context.

        ``text_ids`` / ``latent_image_ids`` are batch-shared and deterministic
        from the spatial shape, so only scalars travel here; the replay path
        rebuilds the position grids from height/width + ``vae_scale_factor``.
        """
        return {
            "guidance_scale": state.guidance_scale,
            "guidance_embeds": state.guidance_embeds,
            "height": state.height,
            "width": state.width,
            "vae_scale_factor": int(self.pipeline.vae_scale_factor),
        }

    def export_replay_tensors(self, state: FluxSamplingState) -> dict[str, Any]:
        """Project FLUX sampling state into per-sample trajectory replay tensors.

        ``latents_clean`` is the final (fully denoised) packed latent, captured at
        decode time — it is the x0 the DiffusionNFT loss regresses toward. GRPO
        ignores it (it reads observations/actions/log_probs instead); the small
        extra per-sample tensor is the price of one family-neutral export path.
        """
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "latents_clean": state.latents.detach(),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> FluxSamplingState:
        """Rebuild FluxSamplingState from a batch slice for the eval forward path.

        Rebuilds the batch-shared position grids deterministically: ``text_ids``
        is zeros sized to the stored prompt sequence length, ``latent_image_ids``
        from the stored spatial shape — pipeline-free, so the replay model never
        touches a FluxPipeline.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        prompt_embeds = replay_tensors["prompt_embeds"]
        height = batch_context["height"]
        width = batch_context["width"]
        vae_scale_factor = int(batch_context["vae_scale_factor"])
        device = prompt_embeds.device

        text_seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32)
        latent_h = 2 * (int(height) // (vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (vae_scale_factor * 2))
        latent_image_ids = self._build_latent_image_ids(
            latent_h // 2,
            latent_w // 2,
            device,
            torch.float32,
        )
        return FluxSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=replay_tensors["pooled_prompt_embeds"],
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=batch_context["guidance_scale"],
            guidance_embeds=batch_context["guidance_embeds"],
            height=height,
            width=width,
        )

    # -- DiffusionNFT forward input ------------------------------------

    def diffusion_nft_prepare_transformer_input(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
        pooled_prompt_embeds: torch.Tensor | None,
        timestep: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build raw FLUX transformer forward kwargs for the DiffusionNFT loss.

        NFT calls ``transformer(**inputs)[0]`` directly on the unwrapped
        transformer (no DiffusionBackboneCaller), so this returns exactly what
        ``FluxTransformer2DModel.forward`` consumes: packed ``hidden_states``, the
        rotary ``img_ids`` / ``txt_ids`` position grids, pooled CLIP projections,
        and the guidance scalar (FLUX.1-dev is guidance-distilled). FLUX latents
        stay PACKED ``[B, seq, C*4]`` through the NFT xt interpolation, so they
        feed the transformer packed exactly as in ``forward_step``.
        """
        del prompt_attention_mask, num_frames, kwargs
        device = latents.device
        td = self._transformer_dtype()

        # Recover the packed position grid (grid_h * grid_w == seq_len, with
        # grid_h:grid_w == height:width) from the packed sequence length + the
        # rollout aspect ratio. The replay/training model that runs this loss owns
        # no FluxPipeline, so it cannot read vae_scale_factor; deriving the grid
        # from seq_len + aspect is pipeline-free and exact for the 16-divisible
        # FLUX resolutions. img_ids/txt_ids match what restore_eval_state rebuilds.
        seq_len = int(latents.shape[1])
        grid_w = round(math.sqrt(seq_len * float(width) / float(height)))
        grid_h = seq_len // grid_w if grid_w else 0
        if grid_w <= 0 or grid_h * grid_w != seq_len:
            raise RuntimeError(
                "FLUX DiffusionNFT could not recover a packed latent grid: "
                f"seq_len={seq_len}, height={height}, width={width}",
            )
        img_ids = self._build_latent_image_ids(grid_h, grid_w, device, torch.float32)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=torch.float32)

        # FLUX's transformer multiplies its timestep input by 1000 internally; the
        # rollout feeds t/1000 (forward_step) and the NFT loss passes the raw
        # buffered timestep (the [0, 1000] flow-match grid), so apply the same
        # /1000 here to land on the identical conditioning the rollout used.
        ts = timestep.to(device=device, dtype=torch.float32)
        if bool((ts > 1.0).any()):
            ts = ts / 1000.0

        guidance = None
        if self._guidance_embeds:
            # NFT's three forwards (previous / trainable / disable-adapter ref)
            # share these kwargs, so any consistent guidance keeps the
            # positive/negative decomposition sound; matching the rollout guidance
            # keeps train/sample conditioning aligned. batch.context carries it
            # (export_batch_context); fall back to the FLUX.1-dev default.
            g = 3.5 if guidance_scale is None else float(guidance_scale)
            guidance = torch.full((latents.shape[0],), g, device=device, dtype=td)

        if pooled_prompt_embeds is None:
            raise RuntimeError(
                "FLUX DiffusionNFT requires pooled_prompt_embeds in the replay tensors",
            )

        return {
            "hidden_states": latents.to(td),
            "timestep": ts.to(td),
            "encoder_hidden_states": prompt_embeds.to(td),
            "pooled_projections": pooled_prompt_embeds.to(td),
            "img_ids": img_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
            "return_dict": False,
        }

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode packed latents → image via FLUX VAE (unpack, then 4D decode)."""
        pipe = self.pipeline
        vae_scale_factor = pipe.vae_scale_factor
        scaling_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        height = self._decode_height
        width = self._decode_width

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            unpacked = pipe._unpack_latents(batch, height, width, vae_scale_factor)
            return unpacked.to(pipe.vae.dtype) / scaling_factor + shift_factor

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda batch: pipe.vae.decode(batch, return_dict=False)[0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class FluxReplayModel(DiffusersReplayModelBase, FluxModel):
    """Replay-only FLUX model that owns no prompt encoders, VAE, or pipeline."""

    def prepare_replay(self, build: ModelBuild) -> None:
        """Set the mu-shifted replay timesteps FLUX's dynamic scheduler needs.

        The replay scheduler was loaded WITHOUT timesteps (mu unknown in the
        generic loader). The replay SDE log-prob math reads scheduler.sigmas +
        index_for_timestep, so the replay scheduler must carry the SAME
        mu-shifted schedule the rollout used. Resolution is fixed per run, so
        derive the packed image_seq_len from it and set the dynamic timesteps
        now — identical to the rollout's prepare_sampling. (debug.first_step
        asserts old==new log-prob, so any drift here surfaces immediately.)
        FLUX packs an 8x VAE + 2x2 patch grid: seq_len = (H // 16) * (W // 16).
        """
        require_lora_for_previous_policy_adapter(build)
        sampling = build.sampling_config or {}
        num_steps = build.num_steps
        height, width = sampling.get("height"), sampling.get("width")
        if num_steps is not None and height and width:
            image_seq_len = (int(height) // 16) * (int(width) // 16)
            self._set_dynamic_timesteps(int(num_steps), image_seq_len, build.device)


__all__ = ["FluxModel", "FluxReplayModel", "FluxSamplingState"]
