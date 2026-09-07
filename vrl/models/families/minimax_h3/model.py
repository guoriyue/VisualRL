"""MiniMax-H3 (Hailuo 3.0) joint video+audio generator wrapped for flow-matching GRPO RL.

Wraps the ``MiniMaxH3ModularPipeline`` components shipped in diffusers 0.40
(transformer, video VAE, audio VAE, Qwen3-VL conditioner, two schedulers). Four
facts about the released model drive the shape of this file:

1. **One packed sequence, strictly batch=1 per prompt.** The transformer
   attends over ``[text | audio rows | video rows]`` at once; the row layout
   (``position_ids`` / ``token_tags`` / index tensors) is a function of the
   prompt length and the canvas, so it is built once in ``prepare_sampling``
   and again in ``restore_eval_state`` from the recorded geometry. The batch
   executor pins one sample per generation batch, like Cosmos3.

2. **Video is the RL action; audio is a deterministic side stream.** The engine's
   denoise loop owns exactly one latent (``state.latents``) and one scheduler.
   The video latent is that action: it takes the shared SDE step and its
   log-prob. The audio rows are stepped by this model with the checkpoint's own
   Euler (``eta = 0``) update on the audio schedule (``shift = 3``), one step
   behind the loop: ``forward_step(i)`` consumes the audio rows at step ``i``
   and prepares the rows for ``i + 1``. Every step's audio input is recorded
   (``audio_rows_by_step``) so replay feeds the transformer the exact rows
   rollout used; audio therefore never enters the policy ratio, and any
   loop mode that skips a ``forward_step`` (TeaCache) desynchronises the side
   stream and is refused loudly.

3. **Velocity sign.** MiniMax-H3 predicts a *data-ward* velocity
   (``x0 = x_t + sigma * v``) on a ``t = 1 - sigma`` clock, the opposite of the
   ``x0 = x_t - sigma * v`` convention ``sde_step_with_logprob`` derives from.
   ``forward_step`` returns ``-v`` as ``noise_pred``; ``MiniMaxH3FlowScheduler``
   flips it back inside ``step`` so ``denoise_mode: native`` reproduces the
   reference sampler. The sigma table lives in ``[0, 1]`` so the shared SDE
   auto-detects the rectified-flow domain.

4. **Guidance-distilled.** There is no unconditional branch; ``guidance_scale``
   must be ``1.0`` and every step runs one forward.

The 5-15 s duration envelope and the 16:9 768p canvas of the released weights
are properties of the checkpoint, not of this wrapper: the wrapper only refuses
geometry the VAE cannot decode (frame counts off the ``17n + 5`` grid, canvases
off the ``32`` multiple).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.families.cosmos import CosmosReplayForward
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusionModelBase,
    DiffusionSamplingStateBase,
    ReplayRolloutStubs,
)
from vrl.models.steps.denoise.base import diffusers_pipeline_dtypes
from vrl.models.steps.denoise.common import ChunkedLatentDecoder, LatentDecodePlan
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

MINIMAX_H3_FPS = 24
# The clock the reference pins its keyframe anchors at; only reached by the
# row-timestep plan (no anchors in t2v), kept for parity with the pipeline.
_KEYFRAME_NOISE_AUG = 0.999
# ``hidden_states[50]`` of the 64-layer Qwen3-VL-32B conditioner; the last
# layer is post-norm and is not what the released weights were trained on.
_TEXT_ENCODER_LAYER = 50
_AUDIO_CHANNELS = 2
_VIDEO_TAG, _TEXT_TAG, _AUDIO_TAG = 0, 1, 2
_WORKFLOW = "t2va"
_DEFAULT_MAX_TEXT_TOKENS = 512


def _import_h3_layout():
    """The diffusers 0.40 layout builders, imported lazily (optional dependency)."""

    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames,
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    return (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        align_num_frames,
        audio_latent_num_frames,
        video_latent_num_frames,
    )


def build_flow_scheduler_class() -> type:
    """``MiniMaxH3Scheduler`` whose ``step`` takes the flow-matching velocity.

    Defined inside a factory so importing this module never imports diffusers.
    The engine's SDE step only reads ``sigmas`` / ``timesteps`` /
    ``index_for_timestep`` (inherited unchanged); ``step`` is reached by
    ``denoise_mode: native`` and receives the ``-v`` that ``forward_step``
    reports, so it negates once more before the checkpoint's data-ward update.
    """

    from diffusers import MiniMaxH3Scheduler

    class MiniMaxH3FlowScheduler(MiniMaxH3Scheduler):
        """MiniMax-H3 Euler scheduler on the ``x0 = x_t - sigma * v`` convention."""

        def step(self, model_output, timestep, sample, return_dict=True):
            return super().step(-model_output, timestep, sample, return_dict=return_dict)

    return MiniMaxH3FlowScheduler


def patchify_video_latents(
    latents: torch.Tensor, patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """``[B, C, T, H, W]`` -> ``[B, rows, C * prod(patch)]``, frame-major then row-major.

    The diffusers helper of the same name folds the batch into the row axis
    (its pipeline is batch=1); this one keeps the batch axis so replay can feed
    a micro-batch of samples that share one layout.
    """

    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}"
        )
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()


def unpatchify_video_rows(
    rows: torch.Tensor,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of :func:`patchify_video_latents` (mirrors ``MiniMaxH3AfterDenoiseStep``)."""

    patch_t, patch_h, patch_w = patch_size
    batch = rows.shape[0]
    grid = rows.reshape(
        batch,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    grid = grid.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return grid.reshape(batch, channels, frames, height, width).contiguous()


def audio_euler_step(
    audio_rows: torch.Tensor,
    velocity: torch.Tensor,
    *,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """One ``MiniMaxH3Scheduler.step`` on the audio rows, stateless.

    Byte-for-byte the checkpoint's update: ``x0 = x_t + (1 - t) * v`` in the
    sample dtype, then the blend ``r * x_t + (1 - r) * x0`` with
    ``r = sigma_next / sigma`` in fp32. Written out here rather than calling
    the scheduler because that call mutates ``_step_index``, which replay must
    not depend on.
    """

    sigma_from_timestep = (1.0 - timestep).to(device=audio_rows.device, dtype=audio_rows.dtype)
    denoised = audio_rows + sigma_from_timestep * velocity.to(audio_rows.dtype)
    compute_dtype = (
        torch.float32 if audio_rows.dtype in (torch.float16, torch.bfloat16) else audio_rows.dtype
    )
    ratio = sigma_next.to(compute_dtype) / sigma.to(compute_dtype)
    prev = ratio * audio_rows.to(compute_dtype) + (1.0 - ratio) * denoised.to(compute_dtype)
    return prev.to(audio_rows.dtype)


@dataclass
class MiniMaxH3Components:
    """The MiniMax-H3 modules the family reads, detached from the modular pipeline.

    ``DiffusersPipelineModelBase`` addresses its backend as ``pipeline`` and
    reads ``.transformer`` / ``.vae`` / ``.components`` off it; the modular
    pipeline's component registry is validated on assignment, which the LoRA
    and compile passes (which swap ``transformer`` for a wrapper) would trip.
    This shell is that attribute surface and nothing else.
    """

    transformer: Any
    vae: Any
    audio_vae: Any
    text_encoder: Any
    tokenizer: Any
    processor: Any
    scheduler: Any
    audio_scheduler: Any
    text_encoder_layer: int = _TEXT_ENCODER_LAYER

    @property
    def components(self) -> dict[str, Any]:
        """Every module, for the frozen-component offload discipline."""

        return {
            "transformer": self.transformer,
            "vae": self.vae,
            "audio_vae": self.audio_vae,
            "text_encoder": self.text_encoder,
        }

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return tuple(int(p) for p in self.transformer.config.patch_size)

    @property
    def canvas_multiple(self) -> int:
        return int(self.vae.spatial_compression_ratio) * self.patch_size[2]


@dataclass
class MiniMaxH3Layout:
    """Step-invariant packed-sequence description for one canvas and prompt length."""

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    num_text_tokens: int
    # One ``(distinct timesteps, per-row index)`` pair per denoise step.
    row_timestep_plan: list[tuple[torch.Tensor, torch.Tensor]]

    def transformer_kwargs(self) -> dict[str, Any]:
        return {
            "token_tags": self.token_tags,
            "position_ids": self.position_ids,
            "video_indices": self.video_indices,
            "audio_indices": self.audio_indices,
            "text_indices": self.text_indices,
        }


@dataclass
class MiniMaxH3SamplingState(DiffusionSamplingStateBase):
    """Per-rollout state: the video latent is the engine-facing action, the rest is private.

    ``latents`` is the live video ``x_t`` ``[B, C, T, H, W]`` fp32; ``timesteps``
    is the video scheduler's ``t = 1 - sigma`` ladder (``len == num_steps``);
    ``scheduler`` is the flow-convention video scheduler. ``audio_rows`` are the
    audio ``x_t`` ``[B, rows, C_audio]`` that belong to step ``audio_step``;
    ``audio_rows_next`` is the Euler successor prepared by the last forward.
    """

    audio_scheduler: Any
    prompt_embeds: torch.Tensor
    layout: MiniMaxH3Layout
    audio_rows: torch.Tensor
    audio_step: int
    height: int
    width: int
    num_frames: int
    fps: int = MINIMAX_H3_FPS
    max_text_tokens: int = _DEFAULT_MAX_TEXT_TOKENS
    audio_rows_next: torch.Tensor | None = None
    # Rollout only: the audio input of every step, exported for replay.
    audio_rows_by_step: list[torch.Tensor] = field(default_factory=list)


class MiniMaxH3Model(CosmosReplayForward, LoraModelMixin, DiffusersPipelineModelBase):
    """MiniMax-H3 text-to-video(+audio) generator wrapped for the vrl diffusion RL seam."""

    # ``from_build`` is family-owned (modular pipeline, not ``DiffusionPipeline``);
    # the two declarations below are what it applies to the frozen conditioner.
    # Qwen3-VL-32B co-resides with the transformer: a deployment that can hold
    # the 33B policy holds the encoder too, and parking it would put a 32B
    # forward on the CPU per prompt.
    _frozen_encoder_names: tuple[str, ...] = ("text_encoder",)
    _prompt_encoder_on_cpu: bool = False

    @classmethod
    def from_build(cls, build: ModelBuild) -> MiniMaxH3Model:
        from diffusers import ModularPipeline

        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, build.parameter_dtype)
        pipeline = ModularPipeline.from_pretrained(
            build.model_name_or_path,
            workflow=_WORKFLOW,
            **build.pretrained_kwargs,
        )
        # ``workflow`` keeps the 33B ``transformer_ref`` partition (ref2va) off
        # the load; the dtype mapping follows the shared pipeline convention
        # (per-component keys plus ``default``).
        pipeline.load_components(workflow=_WORKFLOW, **load_kwargs)
        flow_scheduler_cls = build_flow_scheduler_class()
        components = MiniMaxH3Components(
            transformer=pipeline.transformer,
            vae=pipeline.vae,
            audio_vae=pipeline.audio_vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            processor=pipeline.processor,
            scheduler=flow_scheduler_cls.from_config(dict(pipeline.scheduler.config)),
            audio_scheduler=pipeline.audio_scheduler,
        )
        encoder_device = "cpu" if cls._prompt_encoder_on_cpu else build.device
        for name in cls._frozen_encoder_names:
            encoder = getattr(components, name)
            encoder.requires_grad_(False)
            encoder.to(encoder_device, dtype=prompt_encoder_dtype)
        for vae in (components.vae, components.audio_vae):
            vae.requires_grad_(False)
            vae.to(build.device, dtype=torch.float32)
        logger.info(
            "loaded MiniMax-H3 generator %s",
            kv(path=build.model_name_or_path, device=build.device, dtype=build.parameter_dtype),
        )
        return cls(pipeline=components, device=build.device)

    def _lora_dtype(self, build: ModelBuild) -> Any | None:
        # The checkpoint is mixed-precision (fp32 patch projections, timestep
        # MLP and output heads inside a bf16 stack; ``_keep_in_fp32_modules``).
        # A dtype cast at LoRA attach would flatten that, so the pre-wrap move
        # is device-only, as for cosmos.
        del build
        return None

    # ---- schedule ----
    def set_num_steps(self, n: int) -> None:
        """``n`` model evaluations: the H3 grid holds ``n + 1`` sigmas (terminal 0 included)."""

        self.scheduler.set_timesteps(int(n) + 1, device=self.device)
        self.audio_scheduler.set_timesteps(int(n) + 1, device=self.device)

    @property
    def audio_scheduler(self) -> Any:
        return self.pipeline.audio_scheduler

    # ---- encode ----
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize the bare prompt (no chat template, no special tokens) and read
        ``hidden_states[text_encoder_layer]`` of the Qwen3-VL conditioner.

        The negative prompt is dropped: the checkpoint is guidance-distilled.
        ``max_sequence_length`` bounds the text rows of the packed sequence and
        is the padded width the replay tensor is stored at.
        """

        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

        del negative_prompt
        text = prompt if isinstance(prompt, str) else prompt[0]
        max_text_tokens = int(kwargs.get("max_sequence_length", _DEFAULT_MAX_TEXT_TOKENS))
        components = self.pipeline
        token_ids = components.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_text_tokens:
            logger.warning(
                "MiniMax-H3 prompt truncated %s",
                kv(tokens=len(token_ids), max_sequence_length=max_text_tokens),
            )
            token_ids = token_ids[:max_text_tokens]
        if not token_ids:
            raise ValueError("MiniMax-H3 needs at least one prompt token")
        embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            components.processor,
            list(token_ids),
            {},
            text_encoder_layer=components.text_encoder_layer,
            device=self._encoder_device(),
            dtype=components.text_encoder.dtype,
        )
        return {
            "prompt_embeds": embeds.to(self.device),
            "max_text_tokens": max_text_tokens,
        }

    # ---- sampling-state assembly ----
    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> MiniMaxH3SamplingState:
        del kwargs
        if float(request.guidance_scale) != 1.0:
            raise ValueError(
                "MiniMax-H3 is guidance-distilled; sampling.guidance_scale must be 1.0, "
                f"got {request.guidance_scale}"
            )
        fps = int(request.fps or MINIMAX_H3_FPS)
        if fps != MINIMAX_H3_FPS:
            raise ValueError(f"MiniMax-H3 generates at {MINIMAX_H3_FPS} fps only, got fps={fps}")
        components = self.pipeline
        device = self.device
        geometry = self._latent_geometry(
            request.height, request.width, request.frame_count, vae_geometry=self._vae_geometry()
        )
        num_latent_frames, latent_height, latent_width, num_audio_latents = geometry
        self.set_num_steps(request.num_steps)

        prompt_embeds = encoded["prompt_embeds"]
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError(
                "MiniMax-H3 runs one prompt per generation batch; prompt_embeds must be "
                f"[1, tokens, dim], got {tuple(prompt_embeds.shape)}"
            )
        layout = self._build_layout(
            num_text_tokens=int(prompt_embeds.shape[1]),
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
        )

        # Same draw order as the reference pipeline: video noise first, then
        # the audio rows, from one CPU generator seeded by the request.
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (
                1,
                components.vae.config.latent_channels,
                num_latent_frames,
                latent_height,
                latent_width,
            ),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        audio_rows = torch.randn(
            (1, num_audio_latents * _AUDIO_CHANNELS, components.audio_vae.config.latent_channels),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        return MiniMaxH3SamplingState(
            latents=latents,
            timesteps=self.scheduler.timesteps,
            scheduler=self.scheduler,
            audio_scheduler=self.audio_scheduler,
            prompt_embeds=prompt_embeds.to(device),
            layout=layout,
            audio_rows=audio_rows,
            audio_step=0,
            height=int(request.height),
            width=int(request.width),
            num_frames=int(request.frame_count),
            fps=fps,
            max_text_tokens=int(encoded.get("max_text_tokens", _DEFAULT_MAX_TEXT_TOKENS)),
        )

    def _vae_geometry(self) -> tuple[int, int, int]:
        """``(clip_length, tokens_chunk_size, spatial_compression_ratio)`` of the video VAE."""

        vae = self.pipeline.vae
        return (
            int(vae.config.clip_length),
            int(vae.tokens_chunk_size),
            int(vae.spatial_compression_ratio),
        )

    def _latent_geometry(
        self,
        height: int,
        width: int,
        num_frames: int,
        *,
        vae_geometry: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        """Validate a canvas against the VAE geometry and return the latent/audio row counts.

        Refuses what the VAE cannot decode instead of rounding: a frame count
        off the ``17n + 5`` grid would silently change the duration the reward
        and the prompt manifest describe. ``vae_geometry`` is read off the VAE
        in rollout and off the batch context in replay (which owns no VAE).
        """

        _, _, align_num_frames, audio_latent_num_frames, video_latent_num_frames = (
            _import_h3_layout()
        )
        frames_per_chunk, latents_per_chunk, ratio = (int(v) for v in vae_geometry)
        multiple = ratio * self.pipeline.patch_size[2]
        if height % multiple or width % multiple:
            raise ValueError(
                f"MiniMax-H3 canvases must be multiples of {multiple}, got {height}x{width}"
            )
        aligned = align_num_frames(int(num_frames), frames_per_chunk, latents_per_chunk)
        if aligned != int(num_frames):
            raise ValueError(
                f"MiniMax-H3 frame counts must be {frames_per_chunk} * n + {latents_per_chunk}; "
                f"{num_frames} is not, the next valid count is {aligned}"
            )
        return (
            int(video_latent_num_frames(aligned, frames_per_chunk, latents_per_chunk)),
            height // ratio,
            width // ratio,
            int(audio_latent_num_frames(aligned, MINIMAX_H3_FPS)),
        )

    def _build_layout(
        self,
        *,
        num_text_tokens: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
    ) -> MiniMaxH3Layout:
        """The t2va ``[text | audio | video]`` layout plus the per-step row-timestep plan.

        Both come from the diffusers block statics so the rows, rotary grid and
        per-row timestep table are the reference's, not a re-derivation. The
        plan is built for the schedules the two schedulers currently hold.
        """

        layout_step, timesteps_step, *_ = _import_h3_layout()
        components = self.pipeline
        device = self.device
        text_token_tags = torch.full((num_text_tokens,), _TEXT_TAG, dtype=torch.long)
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            cond_video,
            cond_audio,
        ) = layout_step.build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            components.patch_size,
            _AUDIO_CHANNELS,
            _AUDIO_TAG,
            _VIDEO_TAG,
            (),
        )
        video_timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        if video_timesteps.numel() != audio_timesteps.numel():
            raise RuntimeError(
                "MiniMax-H3 video and audio schedules must hold the same step count"
            )
        plan = [
            tuple(
                tensor.to(device)
                for tensor in timesteps_step.build_row_timesteps(
                    video_indices,
                    audio_indices,
                    cond_video,
                    cond_audio,
                    num_text_tokens,
                    float(video_t),
                    float(audio_t),
                    max(float(video_t), _KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for video_t, audio_t in zip(
                video_timesteps.tolist(), audio_timesteps.tolist(), strict=True
            )
        ]
        return MiniMaxH3Layout(
            position_ids=position_ids.to(device),
            token_tags=token_tags.to(device),
            video_indices=video_indices.to(device),
            audio_indices=audio_indices.to(device),
            text_indices=text_indices.to(device),
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            num_text_tokens=num_text_tokens,
            row_timestep_plan=plan,
        )

    # ---- the per-step transformer call ----
    def forward_step(self, state: MiniMaxH3SamplingState, step_idx: int) -> dict[str, Any]:
        """One joint forward: video velocity out as ``noise_pred`` (flow convention),
        audio rows advanced to the next step as a side effect.

        Re-entrant per step: a second call at the same ``step_idx`` (the frozen
        reference forward under ``disable_adapter``) sees the same audio input.
        A call that skips a step means the loop did not run the forward in
        between, so the audio trajectory is stale; that is refused rather than
        silently continued.
        """

        self._advance_audio(state, step_idx)
        layout = state.layout
        components = self.pipeline
        patch = components.patch_size
        batch = state.latents.shape[0]
        if state.prompt_embeds.shape[0] not in (1, batch):
            raise ValueError(
                f"prompt_embeds batch {state.prompt_embeds.shape[0]} does not match latents batch {batch}"
            )
        video_rows = patchify_video_latents(state.latents.to(torch.float32), patch)
        unique_timesteps, timestep_indices = layout.row_timestep_plan[step_idx]
        prompt_embeds = state.prompt_embeds
        if prompt_embeds.shape[0] == 1 and batch > 1:
            prompt_embeds = prompt_embeds.expand(batch, -1, -1)
        audio_rows = state.audio_rows
        if audio_rows.shape[0] == 1 and batch > 1:
            audio_rows = audio_rows.expand(batch, -1, -1)
        video_out, audio_out = self.transformer(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows.to(torch.float32),
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            return_dict=False,
            **layout.transformer_kwargs(),
        )
        h3_velocity = unpatchify_video_rows(
            video_out.to(torch.float32),
            channels=state.latents.shape[1],
            frames=layout.num_latent_frames,
            height=layout.latent_height,
            width=layout.latent_width,
            patch_size=patch,
        )
        audio_velocity = audio_out.to(torch.float32)
        state.audio_rows_next = audio_euler_step(
            state.audio_rows,
            audio_velocity[:1] if state.audio_rows.shape[0] == 1 else audio_velocity,
            timestep=state.audio_scheduler.timesteps[step_idx],
            sigma=state.audio_scheduler.sigmas[step_idx],
            sigma_next=state.audio_scheduler.sigmas[step_idx + 1],
        )
        # Data-ward H3 velocity -> the ``noise - x0`` velocity the shared SDE expects.
        return {
            "noise_pred": -h3_velocity,
            "audio_velocity": audio_velocity,
        }

    @staticmethod
    def _advance_audio(state: MiniMaxH3SamplingState, step_idx: int) -> None:
        if step_idx == state.audio_step:
            pass
        elif step_idx == state.audio_step + 1 and state.audio_rows_next is not None:
            state.audio_rows = state.audio_rows_next
            state.audio_rows_next = None
            state.audio_step = step_idx
        else:
            raise RuntimeError(
                "MiniMax-H3 audio side stream is out of step with the denoise loop "
                f"(audio at step {state.audio_step}, forward requested at {step_idx}); "
                "every step must run forward_step exactly in order (TeaCache is unsupported)"
            )
        if len(state.audio_rows_by_step) == step_idx:
            state.audio_rows_by_step.append(state.audio_rows.detach())

    def final_audio_rows(self, state: MiniMaxH3SamplingState) -> torch.Tensor:
        """The fully denoised audio rows after the last loop step (``sigma = 0``)."""

        if state.audio_rows_next is None or state.audio_step != len(state.timesteps) - 1:
            raise RuntimeError("the audio side stream has not reached the last denoise step")
        return state.audio_rows_next

    # ---- decode ----
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Video latents -> ``[B, C, T, H, W]`` in ``[0, 1]``.

        Mirrors ``MiniMaxH3VideoDecodeStep``: undo the per-channel latent
        normalization, decode (fp16 autocast on CUDA, as the reference), undo
        the ImageNet pixel normalization, clamp. The soundtrack is not decoded
        here; see ``decode_audio``.
        """

        components = self.pipeline
        vae = components.vae
        device = latents.device
        latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, -1, 1, 1, 1)

        def _vae_decode(batch: torch.Tensor) -> torch.Tensor:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                return vae.decode(batch, return_dict=False)[0]

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda batch: batch.to(torch.float32) * latents_std + latents_mean,
                vae_decode=_vae_decode,
                postprocess=lambda video: self._video_processor().postprocess_video(
                    (video.float() * pixel_std + pixel_mean).clamp(0, 1),
                    output_type="pt",
                ),
                output_layout="video_btchw",
            ),
        )
        return decoder(latents)

    def decode_audio(self, audio_rows: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Final audio rows ``[1, rows, C]`` -> ``(waveform [channels, samples], sample_rate)``.

        Mirrors ``MiniMaxH3AfterDenoiseStep`` + ``MiniMaxH3AudioDecodeStep``.
        Not on the engine's decode path (the RL artifact is the video); offered
        for evaluation scripts that want the soundtrack of a rollout.
        """

        components = self.pipeline
        audio_vae = components.audio_vae
        if audio_rows.ndim != 3 or audio_rows.shape[0] != 1:
            raise ValueError(
                f"audio rows must be [1, rows, channels], got {tuple(audio_rows.shape)}"
            )
        rows = audio_rows[0]
        num_latents = rows.shape[0] // _AUDIO_CHANNELS
        latents = rows.reshape(_AUDIO_CHANNELS, num_latents, rows.shape[-1]).permute(0, 2, 1)
        device = latents.device
        mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
        audio = audio_vae.decode(latents.float() * std + mean, return_dict=False)[0]
        return audio.float().permute(1, 0, 2)[0], int(audio_vae.config.sampling_rate)

    def _video_processor(self) -> Any:
        from diffusers.video_processor import VideoProcessor

        return VideoProcessor(
            vae_scale_factor=int(self.pipeline.vae.spatial_compression_ratio), do_normalize=False
        )

    # ---- replay export/restore ----
    def export_batch_context(self, state: MiniMaxH3SamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": 1.0,
            "height": state.height,
            "width": state.width,
            "num_frames": state.num_frames,
            "fps": state.fps,
            "num_steps": int(state.timesteps.numel()),
            # Replay owns no VAE: the geometry the layout was built for rides along.
            "vae_geometry": [int(v) for v in self._vae_geometry()],
        }

    def export_replay_tensors(self, state: MiniMaxH3SamplingState) -> dict[str, Any]:
        """Prompt embeds padded to a fixed width plus the audio input of every step.

        Prompts differ in length across requests, and the gatherer concatenates
        replay tensors along the sample axis, so the embeds are stored at the
        request's ``max_sequence_length`` width with the true length beside
        them; ``restore_eval_state`` slices the padding off before the forward,
        so the transformer never sees a pad row.
        """

        if len(state.audio_rows_by_step) != state.timesteps.numel():
            raise RuntimeError(
                "MiniMax-H3 audio trajectory is incomplete: "
                f"{len(state.audio_rows_by_step)} of {state.timesteps.numel()} steps recorded"
            )
        batch = state.latents.shape[0]
        embeds = state.prompt_embeds.detach()
        num_text_tokens = int(embeds.shape[1])
        width = max(num_text_tokens, int(state.max_text_tokens))
        padded = embeds.new_zeros((embeds.shape[0], width, embeds.shape[-1]))
        padded[:, :num_text_tokens] = embeds
        audio_by_step = torch.stack(state.audio_rows_by_step, dim=1)  # [B, steps, rows, C]
        return {
            "prompt_embeds": _expand_batch(padded, batch),
            "num_text_tokens": torch.full((batch,), num_text_tokens, dtype=torch.int64),
            "latents_clean": state.latents.detach(),
            "audio_rows_by_step": _expand_batch(audio_by_step, batch),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> MiniMaxH3SamplingState:
        """Rebuild the packed layout from the recorded geometry and put the audio
        side stream at ``step_idx``; the forward then indexes the full schedule
        (``CosmosReplayForward``)."""

        num_steps = int(batch_context["num_steps"])
        self.set_num_steps(num_steps)
        lengths = replay_tensors["num_text_tokens"].reshape(-1).tolist()
        if any(int(n) != int(lengths[0]) for n in lengths):
            raise ValueError(
                "MiniMax-H3 replay micro-batches must share one prompt length "
                f"(one packed layout per forward); got {lengths}"
            )
        num_text_tokens = int(lengths[0])
        prompt_embeds = replay_tensors["prompt_embeds"][:, :num_text_tokens]
        audio_rows = replay_tensors["audio_rows_by_step"][:, step_idx]
        geometry = self._latent_geometry(
            int(batch_context["height"]),
            int(batch_context["width"]),
            int(batch_context["num_frames"]),
            vae_geometry=tuple(batch_context["vae_geometry"]),
        )
        num_latent_frames, latent_height, latent_width, num_audio_latents = geometry
        layout = self._build_layout(
            num_text_tokens=num_text_tokens,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
        )
        return MiniMaxH3SamplingState(
            latents=latents,
            timesteps=self.scheduler.timesteps,
            scheduler=self.scheduler,
            audio_scheduler=self.audio_scheduler,
            prompt_embeds=prompt_embeds,
            layout=layout,
            audio_rows=audio_rows,
            audio_step=int(step_idx),
            height=int(batch_context["height"]),
            width=int(batch_context["width"]),
            num_frames=int(batch_context["num_frames"]),
            fps=int(batch_context.get("fps", MINIMAX_H3_FPS)),
        )


def _expand_batch(value: torch.Tensor, batch: int) -> torch.Tensor:
    if value.shape[0] == batch:
        return value
    if value.shape[0] != 1:
        raise ValueError(f"cannot align a batch-{value.shape[0]} replay tensor to batch {batch}")
    return value.expand(batch, *value.shape[1:]).contiguous()


class MiniMaxH3ReplayModel(ReplayRolloutStubs, MiniMaxH3Model):
    """Trainer-side replay model: the transformer and the two schedulers, no pipeline.

    Holds the same ``MiniMaxH3Components`` shell as the rollout model with the
    generation-only slots left empty, so the layout/plan builders and the
    ``scheduler`` / ``audio_scheduler`` accessors are shared verbatim while
    ``decode_latents`` / ``encode_prompt`` stay stubbed.
    """

    def __init__(
        self, *, transformer: Any, scheduler: Any, audio_scheduler: Any, device: Any = None
    ) -> None:
        DiffusionModelBase.__init__(self)
        object.__setattr__(
            self,
            "_pipeline",
            MiniMaxH3Components(
                transformer=transformer,
                vae=None,
                audio_vae=None,
                text_encoder=None,
                tokenizer=None,
                processor=None,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            ),
        )
        self.transformer = transformer
        self._device = device


__all__ = [
    "MINIMAX_H3_FPS",
    "MiniMaxH3Components",
    "MiniMaxH3Layout",
    "MiniMaxH3Model",
    "MiniMaxH3ReplayModel",
    "MiniMaxH3SamplingState",
    "audio_euler_step",
    "build_flow_scheduler_class",
    "patchify_video_latents",
    "unpatchify_video_rows",
]
