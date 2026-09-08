"""Offline Diffusion-DPO trainer.

Generic over diffusion model family: the caller supplies the forward adapter,
the encoders, and the scheduler, so nothing here names a family. What IS built
in is the noise/target convention, one branch of ``_inject_noise`` per
``prediction_type``:

  * ``epsilon`` / ``v_prediction`` (``DDPMScheduler.add_noise``)
  * ``flow_matching`` velocity (``scheduler.scale_noise``, or a sigma lookup)

For Wan video models, image-only datasets (Pick-a-Pic) are handled by the
caller's ``encode_pixels`` replicating each image along the temporal dim
before VAE encoding (see ``vrl/scripts/families/wan_2_1/train_dpo.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vrl.algorithms.dpo import diffusion_dpo_loss, diffusion_sft_loss
from vrl.models.precision import (
    apply_float32_precision,
    model_autocast,
    model_precision,
)
from vrl.trainers.data.preferences import PreferenceBatch

if TYPE_CHECKING:
    from vrl.algorithms.dpo import DiffusionDPOConfig
    from vrl.config.schema import RootConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OfflineDPOTrainerConfig:
    """Configuration for the offline DPO training loop."""

    # --- DPO ---
    beta: float = field(default=5000.0)
    sft_weight: float = field(default=0.0)

    # --- optimizer ---
    # The caller resolves lr scaling (e.g. by effective batch size) before
    # constructing this config; the trainer applies ``lr`` as-is.
    lr: float = field(default=1e-8)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_weight_decay: float = field(default=1e-2)
    adam_epsilon: float = field(default=1e-8)
    use_adafactor: bool = field(default=False)
    max_grad_norm: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)

    # --- noise / schedule ---
    prediction_type: str = field(
        default="flow_matching"
    )  # "epsilon" | "v_prediction" | "flow_matching"

    @classmethod
    def from_root(
        cls,
        root: RootConfig,
        dpo_config: DiffusionDPOConfig,
    ) -> OfflineDPOTrainerConfig:
        """Project the public actor section into the offline trainer config."""
        from vrl.trainers.core.types import OptimConfig

        actor = root.actor

        def required(name: str) -> Any:
            value = None if actor is None else getattr(actor, name)
            if value is None:
                raise ValueError(f"config missing required field: actor.{name}")
            return value

        train_batch_size = int(required("train_batch_size"))
        gradient_accumulation_steps = int(required("gradient_accumulation_steps"))
        optim: OptimConfig = required("optim")
        if optim.optim_8bit:
            raise ValueError(
                "actor.optim.optim_8bit=true is not supported by OfflineDPOTrainer; "
                "use AdamW/Adafactor without 8-bit optimizer state",
            )
        use_adafactor = bool(required("use_adafactor"))
        if use_adafactor:
            # An AdamW-only knob moved off its default would be silently ignored
            # under Adafactor; refuse rather than train with a no-op setting.
            defaults = OptimConfig(lr=optim.lr)
            adam_only_keys = sorted(
                key
                for key in ("adam_beta1", "adam_beta2", "eps")
                if getattr(optim, key) != getattr(defaults, key)
            )
            if adam_only_keys:
                paths = ", ".join(f"actor.optim.{key}" for key in adam_only_keys)
                raise ValueError(
                    f"actor.use_adafactor=true does not consume AdamW-only key(s): {paths}",
                )

        scale_lr = bool(required("scale_lr"))
        effective_batch_size = train_batch_size * gradient_accumulation_steps
        lr = float(optim.lr) * effective_batch_size if scale_lr else float(optim.lr)
        max_grad_norm = actor.max_norm if actor is not None else None
        if max_grad_norm is None:
            max_grad_norm = cls().max_grad_norm
        return cls(
            beta=float(dpo_config.beta),
            sft_weight=float(dpo_config.sft_weight),
            lr=lr,
            adam_beta1=float(optim.adam_beta1),
            adam_beta2=float(optim.adam_beta2),
            adam_weight_decay=float(optim.weight_decay),
            adam_epsilon=float(optim.eps),
            max_grad_norm=float(max_grad_norm),
            gradient_accumulation_steps=gradient_accumulation_steps,
            prediction_type=str(required("prediction_type")),
            use_adafactor=use_adafactor,
        )


@dataclass(slots=True)
class DPOStepMetrics:
    """One training-step's metrics."""

    loss: float = 0.0
    raw_model_loss: float = 0.0
    raw_ref_loss: float = 0.0
    model_diff: float = 0.0
    ref_diff: float = 0.0
    implicit_acc: float = 0.0
    sft_loss: float = 0.0
    grad_norm: float = 0.0


def _build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: OfflineDPOTrainerConfig,
) -> torch.optim.Optimizer:
    if cfg.use_adafactor:
        try:
            from transformers.optimization import Adafactor
        except ImportError as e:
            raise ImportError(
                "Install transformers for Adafactor: pip install transformers"
            ) from e
        return Adafactor(
            list(parameters),
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.adam_weight_decay,
        )
    return torch.optim.AdamW(
        list(parameters),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )


# ---------------------------------------------------------------------------
# Forward adapters — caller plugs in a model-specific forward function.
# ---------------------------------------------------------------------------

# A ForwardFn takes ``(model, noisy_latents, timesteps, encoder_hidden_states,
# extra_kwargs)`` and returns the prediction tensor. Concrete adapters live with
# the family recipe that selects them (e.g. ``wan_forward`` in
# ``vrl/scripts/families/wan_2_1/train_dpo.py``), never here: this module must
# stay family-neutral.
ForwardFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# OfflineDPOTrainer
# ---------------------------------------------------------------------------


class OfflineDPOTrainer:
    """Offline DPO trainer over preference pairs.

    The trainer is **synchronous** — DPO has no rollout collection so we
    avoid the OnlineTrainer's async machinery.

    Caller responsibilities:
      * Build the policy ``model`` (typically a LoRA-wrapped backbone).
      * Build the frozen ``ref_model`` (or pass ``None`` and rely on
        LoRA adapter-disable inside ``forward_fn``).
      * Provide ``encode_pixels`` — turns ``[2B, 3, H, W]`` pixels into
        latents in the shape the model expects (handles VAE + temporal
        replication for video models).
      * Provide ``encode_text`` — turns a list of captions into the
        ``encoder_hidden_states`` tensor shaped ``[2B, ..., D]``
        (winner-then-loser convention).
      * Provide ``forward_fn`` — the model-family-specific forward; it lives
        with the family recipe, not in this module.
      * Provide ``noise_scheduler`` for sampling timesteps + injecting
        noise. Flow-matching schedulers use ``scale_noise``; epsilon
        schedulers use ``add_noise``.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        forward_fn: ForwardFn,
        noise_scheduler: Any,
        encode_pixels: Callable[[torch.Tensor], torch.Tensor],
        encode_text: Callable[[list[str]], torch.Tensor],
        config: OfflineDPOTrainerConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.forward_fn = forward_fn
        self.noise_scheduler = noise_scheduler
        self.encode_pixels = encode_pixels
        self.encode_text = encode_text
        self.precision = model_precision(model)
        self.config = config or OfflineDPOTrainerConfig()
        self.device = torch.device(device) if isinstance(device, str) else device
        apply_float32_precision(self.precision.float32_precision)
        self.global_step = 0
        self._gradient_accumulation_micro_step = 0

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("model has no trainable parameters — wire up LoRA / unfreeze first")
        self._optimizer = _build_optimizer(trainable, self.config)

    # ------------------------------------------------------------------
    # Noise injection — branch on prediction_type
    # ------------------------------------------------------------------

    def _sample_timesteps(self, bsz: int) -> torch.Tensor:
        # Resolve the timestep range explicitly. Silently using
        # ``num_train_timesteps`` when ``scheduler.timesteps`` is empty
        # would mask the common bug of forgetting to call
        # ``scheduler.set_timesteps(...)`` before training, which can
        # silently change the sampling distribution.
        ts = getattr(self.noise_scheduler, "timesteps", None)
        if ts is None or len(ts) == 0:
            raise RuntimeError(
                "noise_scheduler.timesteps is empty/missing. "
                "Call scheduler.set_timesteps(num_inference_steps) "
                "before training."
            )
        lo, hi = 0, len(ts)
        return torch.randint(lo, hi, (bsz,), device=self.device).long()

    def _inject_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (noisy_latents, target).

        For epsilon prediction:    target = noise
        For flow-matching:         target = noise - latents (velocity)
        """
        pt = self.config.prediction_type
        if pt == "epsilon":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            return noisy, noise
        if pt == "v_prediction":
            noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            return noisy, target
        if pt == "flow_matching":
            # Flow-matching schedulers expose ``scale_noise(sample, timestep, noise)``.
            # Forward process: x_t = (1 - sigma) * x_0 + sigma * noise.
            if hasattr(self.noise_scheduler, "scale_noise"):
                noisy = self.noise_scheduler.scale_noise(latents, timesteps, noise)
            else:
                # Fallback: derive from sigmas tensor directly.
                sigmas = self.noise_scheduler.sigmas.to(latents.device)
                # timesteps here index into scheduler.timesteps; map to sigma.
                # Use nearest timestep match.
                ts_idx = torch.searchsorted(
                    self.noise_scheduler.timesteps.to(self.device).flip(0),
                    timesteps.to(self.device),
                ).clamp(0, len(sigmas) - 1)
                view = (-1,) + (1,) * (latents.ndim - 1)
                sigma = sigmas[ts_idx].view(*view)
                noisy = (1.0 - sigma) * latents + sigma * noise
            target = noise - latents
            return noisy, target
        raise ValueError(f"unknown prediction_type: {pt}")

    # ------------------------------------------------------------------
    # One training step
    # ------------------------------------------------------------------

    def step(self, batch: PreferenceBatch) -> DPOStepMetrics:
        """Single optimizer step over one preference batch."""
        cfg = self.config
        self.model.train()

        # 1. Stack winner-then-loser → [2B, 3, H, W]
        pixels = batch.stacked_winner_then_loser().to(self.device)

        # 2. Pixels → latents (caller handles VAE + temporal replication)
        with torch.no_grad():
            latents = self.encode_pixels(pixels)
            # 3. Text encoding — duplicated to match 2B layout
            encoder_hidden_states = self.encode_text(batch.captions)
            if encoder_hidden_states.shape[0] != latents.shape[0]:
                # caller may return [B, ...] — repeat to [2B, ...]
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                    latents.shape[0] // encoder_hidden_states.shape[0],
                    dim=0,
                )

        # 4. Sample shared noise + timestep across each pair
        bsz_pair = latents.shape[0] // 2
        noise = torch.randn(
            (bsz_pair, *tuple(latents.shape[1:])),
            device=latents.device,
            dtype=latents.dtype,
        ).repeat(2, *([1] * (latents.ndim - 1)))
        ts_pair = self._sample_timesteps(bsz_pair)
        timesteps = ts_pair.repeat(2)

        noisy_latents, target = self._inject_noise(latents, noise, timesteps)

        # 5. Forward — policy + frozen reference
        with model_autocast(self.model, self.device):
            model_pred = self.forward_fn(
                self.model,
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
        with torch.no_grad(), model_autocast(self.model, self.device):
            ref_pred = self._reference_forward(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).detach()

        # DPO/SFT reductions are protected FP32 math, not transformer forward.
        stats = diffusion_dpo_loss(
            model_pred=model_pred.float(),
            ref_pred=ref_pred.float(),
            target=target.float(),
            beta=cfg.beta,
        )
        loss = stats["loss"]

        sft_loss_val = torch.tensor(0.0, device=self.device)
        if cfg.sft_weight > 0:
            # Compute MSE on winner only
            bsz = model_pred.shape[0] // 2
            sft_loss_val = diffusion_sft_loss(
                model_pred[:bsz].float(),
                target[:bsz].float(),
            )
            loss = loss + cfg.sft_weight * sft_loss_val

        # 6. Backward + step (with optional accumulation)
        loss_scaled = loss / max(1, cfg.gradient_accumulation_steps)
        loss_scaled.backward()

        grad_norm = 0.0
        if self._mark_gradient_accumulation_step():
            if cfg.max_grad_norm > 0:
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                grad_norm = float(gn) if isinstance(gn, torch.Tensor) else gn
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)

        self.global_step += 1

        return DPOStepMetrics(
            loss=float(loss.detach()),
            raw_model_loss=float(stats["raw_model_loss"].detach()),
            raw_ref_loss=float(stats["raw_ref_loss"].detach()),
            model_diff=float(stats["model_diff"].detach()),
            ref_diff=float(stats["ref_diff"].detach()),
            implicit_acc=float(stats["implicit_acc"].detach()),
            sft_loss=float(sft_loss_val.detach()),
            grad_norm=grad_norm,
        )

    def _reference_forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the reference policy.

        If a separate ``ref_model`` was provided, use it directly. Otherwise
        assume the policy is a LoRA-wrapped backbone and disable adapters
        for the reference pass (PEFT convention).
        """
        if self.ref_model is not None:
            return self.forward_fn(
                self.ref_model,
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return self.forward_fn(
                    self.model,
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                )
        raise RuntimeError(
            "no ref_model and policy has no ``disable_adapter`` — "
            "cannot compute reference prediction"
        )

    def _mark_gradient_accumulation_step(self) -> bool:
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        self._gradient_accumulation_micro_step += 1
        if self._gradient_accumulation_micro_step < accumulation_steps:
            return False
        self._gradient_accumulation_micro_step = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return resumable trainer state."""

        return {
            "global_step": self.global_step,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        """Restore resumable trainer state."""

        if not isinstance(state, dict):
            raise TypeError("OfflineDPOTrainer.load_state_dict expects a dict")
        self.global_step = int(state.get("global_step", 0))
        # Parameter .grad buffers are not checkpointed, so resume must start a
        # fresh accumulation window instead of deriving the boundary from
        # global_step.
        self._gradient_accumulation_micro_step = 0
        if "optimizer" in state:
            try:
                self._optimizer.load_state_dict(state["optimizer"])
            except Exception:
                if strict:
                    raise
                logger.warning("Skipping incompatible optimizer state during non-strict load")
        elif strict:
            raise ValueError("checkpoint missing optimizer state")
