"""V-GRPO: GRPO on an ELBO-based likelihood surrogate (arXiv:2604.23380).

Variational GRPO replaces the per-step SDE log-prob of the induced MDP with the
flow-matching training loss of the *final* sample as a likelihood surrogate:

    log pi(o | c)  <-  -L(theta | o, c) = -E_{t, eps}[ w_t ||NN_theta(z_t) - r_t||^2 ]

so the importance ratio between the current and the behaviour policy is
``rho = exp(-L(theta) + L(theta_old))`` and the GRPO objective is the usual
clipped ``min(rho A, clip(rho) A)``. Rollout is plain generation (no per-step
log-probs, any ODE sampler); training needs the clean sample, the prompt
conditioning and a few ``(t, eps)`` pairs.

What this module implements, and how it maps onto the trainer:

- **One ``(t, eps)`` pair per trainer replay index.** The trainer's per-step
  loss loop hands the objective one ``timestep_index`` at a time; that index
  picks ``t`` off the rollout grid and the objective draws ``eps`` for it. With
  ``actor.timestep_selection: stratified`` and
  ``timestep_fraction = N_MC / num_steps`` the indices are the paper's
  stratified draw (one per equal-length interval of the schedule), resampled
  every update.
- **Group-shared noise.** ``eps`` is generated from a seed derived from the
  prompt group id, the replay index and the update counter, so every sample of
  a group is scored on the same pairs (the paper's within-group variance fix)
  while pairs still change from update to update.
- **Adaptive loss weighting.** Both losses are the x-prediction MSE normalized
  by its own detached mean absolute error (``normalized_mse``, Eq. 14).
- **Behaviour policy = the frozen ``previous`` LoRA adapter**, refreshed after
  every optimizer step (``after_optimizer_step``) like DiffusionNFT. With
  ``ppo_epochs: 1`` that is exactly the paper's ``theta_old``; with more
  gradient steps per rollout it is the previous *step's* policy.
- **Per-pair ratio.** The paper's ratio uses the mean over all ``N_MC`` pairs
  inside one ``exp``; here each pair carries its own ratio and clip, because the
  trainer backpropagates one index at a time. For a fully on-policy update
  (``rho == 1``) the two are identical; with clipping active the per-pair form
  clips more conservatively.
- **Gradient step control.** Optional ratio clipping ``clip_ratio`` (epsilon),
  optional simple KL to the behaviour policy
  ``||x_theta(z_t) - x_theta_old(z_t)||^2`` (``kl_coef``, Eq. 16) and optional
  advantage soft clipping ``eta * tanh(A / eta)`` (``adv_soft_clip``, Eq. 17).

The forward-process model surface is the one DiffusionNFT introduced:
``diffusion_nft_prepare_transformer_input`` (raw transformer kwargs for a
noised clean latent) and the ``previous`` adapter with
``sync_previous_policy_adapter``. A family that has both runs either objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from vrl.algorithms.advantages import group_relative_advantages
from vrl.algorithms.config_contract import AlgorithmConfigContract
from vrl.algorithms.diffusion_nft import normalized_mse
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.algorithms.types import PolicyUpdateStats, TrainStepMetrics
from vrl.models.precision import model_autocast

_SEED_UPDATE = 1_000_003
_SEED_GROUP = 7_919
_SEED_INDEX = 104_729


@dataclass(slots=True)
class VGRPOConfig:
    """Hyper-parameters for the V-GRPO objective.

    ``clip_ratio`` / ``kl_coef`` / ``adv_soft_clip`` are the paper's three
    gradient-step controls; each is off when ``None`` / ``0``. The paper's
    SD 3.5 M Stage-1 recipe (fully on-policy) uses only ``adv_soft_clip=3``;
    its multi-epoch stages add ``clip_ratio`` or ``kl_coef=0.3``.
    """

    config_contract: ClassVar[AlgorithmConfigContract] = AlgorithmConfigContract(
        needs_sde_rollout=False,
        supports_step_kl_reward=True,
        sft_source="unsupported",
    )

    eps: float = 1e-8
    adv_clip_max: float = 5.0
    global_std: bool = False
    clip_ratio: float | None = None
    kl_coef: float = 0.0
    adv_soft_clip: float | None = 3.0
    weight_copy_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.clip_ratio is not None and not 0.0 < float(self.clip_ratio) < 1.0:
            raise ValueError(
                f"VGRPOConfig.clip_ratio must be in (0, 1) or null, got {self.clip_ratio}"
            )
        if float(self.kl_coef) < 0.0:
            raise ValueError(f"VGRPOConfig.kl_coef must be >= 0, got {self.kl_coef}")
        if self.adv_soft_clip is not None and float(self.adv_soft_clip) <= 0.0:
            raise ValueError(
                f"VGRPOConfig.adv_soft_clip must be > 0 or null, got {self.adv_soft_clip}"
            )
        if not 0.0 <= float(self.weight_copy_decay) <= 1.0:
            raise ValueError(
                f"VGRPOConfig.weight_copy_decay must be in [0, 1], got {self.weight_copy_decay}"
            )


class VGRPO:
    """Variational GRPO objective on the forward-process replay branch."""

    uses_evaluator = False
    required_data_keys = ("latents_clean", "prompt_embeds", "timesteps")
    required_signal_keys: tuple[str, ...] = ()
    needs_kl_intermediates = False
    # The ratio is a real trust region only when a second gradient step runs on
    # the same rollouts; at ppo_epochs=1 it is identically 1 and the objective
    # is REINFORCE with a group baseline, which is the paper's Stage-1 recipe.
    requires_active_trust_region = False
    # The behaviour policy is the previous adapter refreshed every optimizer
    # step, not the policy that generated a stale rollout: training on rollouts
    # from a superseded policy would score them against the wrong theta_old.
    tolerates_off_policy_staleness = False

    def __init__(self, config: VGRPOConfig | None = None) -> None:
        self.config = config or VGRPOConfig()
        # Advances with the optimizer so the group-shared noise changes across
        # updates while staying fixed within one.
        self._update_counter = 0

    def compute_advantages_from_tensors(self, rewards: Any, group_ids: Any) -> Any:
        cfg = self.config
        return group_relative_advantages(
            rewards,
            group_ids,
            eps=cfg.eps,
            adv_clip_max=cfg.adv_clip_max,
            global_std=cfg.global_std,
        )

    def compute_loss(self, inputs: AlgorithmInput) -> tuple[Any, TrainStepMetrics]:
        return self.compute_batch_timestep_loss(
            inputs.model,
            inputs.rollout_batch,
            inputs.timestep_index,
            inputs.advantages,
        )

    # -- the objective ------------------------------------------------------

    def compute_batch_timestep_loss(
        self,
        model: Any,
        batch: Any,
        timestep_index: int,
        advantages: Any,
    ) -> tuple[Any, TrainStepMetrics]:
        """One ``(t, eps)`` pair of the surrogate for a rollout batch.

        ``t`` is the rollout grid's ``timesteps[:, timestep_index]``; ``eps`` is
        the group-shared draw for this index and update.
        """

        import torch

        from vrl.trajectory import TrajectoryResolver

        cfg = self.config
        replay_tensors = TrajectoryResolver.from_batch(batch).replay_tensor_dict("denoise")
        x0 = replay_tensors["latents_clean"]
        prompt_embeds = replay_tensors["prompt_embeds"]
        timesteps = replay_tensors["timesteps"]
        timestep_width = 1 if timesteps.ndim == 1 else int(timesteps.shape[1])
        if not 0 <= timestep_index < timestep_width:
            raise RuntimeError(
                "V-GRPO timestep_index out of range: "
                f"timestep_index={timestep_index}, width={timestep_width}, "
                f"timesteps.shape={tuple(timesteps.shape)}",
            )
        t_raw = timesteps if timesteps.ndim == 1 else timesteps[:, timestep_index]
        batch_size = int(x0.shape[0])
        if prompt_embeds.shape[0] != batch_size or advantages.shape[0] != batch_size:
            raise RuntimeError(
                "V-GRPO batch mismatch: latents_clean, prompt_embeds and advantages "
                f"have leading dims {batch_size}, {prompt_embeds.shape[0]}, {advantages.shape[0]}",
            )
        transformer = getattr(model, "transformer", None)
        if transformer is None:
            raise RuntimeError("V-GRPO model must expose a transformer module")
        prepare = getattr(model, "diffusion_nft_prepare_transformer_input", None)
        if not callable(prepare):
            raise RuntimeError(
                "V-GRPO model must expose diffusion_nft_prepare_transformer_input(...) "
                "(the forward-process transformer input hook)",
            )

        t = _flow_time(t_raw, x0)
        t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
        noise = self._group_shared_noise(
            x0,
            group_ids=getattr(batch, "group_ids", None),
            timestep_index=int(timestep_index),
        )
        xt = (1 - t_expanded) * x0.float() + t_expanded * noise
        transformer_inputs = prepare(
            latents=xt.to(x0.dtype),
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            pooled_prompt_embeds=replay_tensors.get("pooled_prompt_embeds"),
            timestep=t_raw,
            num_frames=int(
                batch.context.get(
                    "num_frames",
                    int(x0.shape[2]) if getattr(x0, "ndim", 0) >= 3 else 1,
                )
            ),
            height=int(batch.context.get("height", 0)),
            width=int(batch.context.get("width", 0)),
            guidance_scale=batch.context.get("guidance_scale"),
        )
        with (
            model.activate_adapter("previous"),
            torch.no_grad(),
            model_autocast(model, x0.device),
        ):
            old_prediction = transformer(**transformer_inputs)[0].detach()
        with model_autocast(model, x0.device):
            prediction = transformer(**transformer_inputs)[0]

        # x-prediction reparameterization of the rectified-flow velocity.
        x0_float = x0.float()
        x_pred = xt - t_expanded * prediction.float()
        x_old = xt - t_expanded * old_prediction.float()
        surrogate = normalized_mse(x_pred, x0_float)  # [B], adaptive weighting
        with torch.no_grad():
            surrogate_old = normalized_mse(x_old, x0_float)
        log_ratio = surrogate_old - surrogate
        ratio = torch.exp(log_ratio)

        adv = advantages.to(device=x0.device, dtype=ratio.dtype)
        if cfg.adv_soft_clip is not None:
            eta = float(cfg.adv_soft_clip)
            adv = eta * torch.tanh(adv / eta)
        unclipped = ratio * adv
        if cfg.clip_ratio is not None:
            eps = float(cfg.clip_ratio)
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
            objective = torch.minimum(unclipped, clipped)
            clip_fraction = float(((ratio - 1.0).abs() > eps).float().mean().item())
            active_clip_fraction = float((clipped < unclipped).float().mean().item())
        else:
            objective = unclipped
            clip_fraction = 0.0
            active_clip_fraction = 0.0
        policy_loss = -objective.mean()

        kl_loss = ((x_pred - x_old) ** 2).mean()
        kl_term = float(cfg.kl_coef) * kl_loss
        loss = policy_loss + kl_term
        kl_value = float(kl_loss.detach().item())
        return loss, TrainStepMetrics(
            loss=float(loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            kl_penalty=kl_value,
            weighted_kl_loss=float(kl_term.detach().item()),
            update=PolicyUpdateStats(
                clip_fraction=clip_fraction,
                active_clip_fraction=active_clip_fraction,
                # k2 estimator of KL(theta_old || theta) on the surrogate.
                approx_kl=float((0.5 * log_ratio.detach() ** 2).mean().item()),
            ),
        )

    def _group_shared_noise(
        self,
        x0: Any,
        *,
        group_ids: Any,
        timestep_index: int,
    ) -> Any:
        """One ``eps`` per prompt group, identical across the group's samples.

        Seeded from ``(update counter, group id, replay index)`` so the pairs a
        group is scored on are shared within an update and fresh across them.
        The draw runs on the CPU generator so it is device-independent.
        """

        import torch

        batch = int(x0.shape[0])
        if group_ids is None:
            ids = [0] * batch
        else:
            ids = [int(value) for value in torch.as_tensor(group_ids).reshape(-1).tolist()]
            if len(ids) != batch:
                raise RuntimeError(
                    f"V-GRPO group_ids has {len(ids)} rows for a batch of {batch} samples",
                )
        shape = tuple(x0.shape[1:])
        draws: dict[int, Any] = {}
        rows = []
        for group in ids:
            if group not in draws:
                seed = (
                    self._update_counter * _SEED_UPDATE
                    + group * _SEED_GROUP
                    + timestep_index * _SEED_INDEX
                ) & 0x7FFFFFFF
                generator = torch.Generator(device="cpu").manual_seed(seed)
                draws[group] = torch.randn(shape, generator=generator, dtype=torch.float32)
            rows.append(draws[group])
        return torch.stack(rows, dim=0).to(device=x0.device)

    # -- lifecycle ------------------------------------------------------------

    def first_step_invariant_check(
        self,
        *,
        model: Any,
        batch: Any,
        advantages: Any,
        timestep_index: int = 0,
        threshold: float = 1.0e-6,
    ) -> dict[str, Any]:
        """The lr=0 invariant: with ``previous == default`` the ratio is 1, so
        the objective is linear in the advantage and the loss is antisymmetric
        under flipping it (``loss(A) + loss(-A) == 2 * kl_term == 0``).

        Called by the trainer's debug.first_step branch through this optional
        protocol method.
        """

        import torch

        def _loss(adv: Any) -> float:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                loss, _ = self.compute_batch_timestep_loss(model, batch, timestep_index, adv)
            return float(loss.detach().float().item())

        loss = _loss(advantages)
        flipped_loss = _loss(-advantages)
        abs_sum = abs(loss + flipped_loss)
        return {
            "event": "first_step_v_grpo_invariant",
            "invariant": "advantage_antisymmetry",
            "loss": loss,
            "flipped_loss": flipped_loss,
            "abs_diff": abs_sum,
            "threshold": threshold,
            "passed": abs_sum <= threshold,
        }

    def after_optimizer_step(self, model: Any, global_step: int) -> None:
        """Refresh the behaviour policy and advance the group-noise counter."""

        sync = getattr(model, "sync_previous_policy_adapter", None)
        if not callable(sync):
            raise RuntimeError(
                "V-GRPO model must expose sync_previous_policy_adapter(decay=...) "
                "for behaviour-policy refresh",
            )
        sync(decay=float(self.config.weight_copy_decay))
        self._update_counter = int(global_step) + 1


def _flow_time(t_raw: Any, x0: Any) -> Any:
    """Normalize a rollout timestep grid into flow time ``t`` in ``[0, 1]``.

    The same ``/1000`` heuristic and EDM guard as DiffusionNFT: a ``[0, 1000]``
    grid (SD3, FLUX, Wan) divides down; an EDM-scale grid would leave ``[0, 1]``
    and push ``x_t`` off the data manifold, so it fails loud.
    """

    import torch

    t = t_raw.to(device=x0.device, dtype=torch.float32)
    if bool((t > 1.0).any()):
        t = t / 1000.0
    if bool((t > 1.0).any()) or bool((t < 0.0).any()):
        raise RuntimeError(
            "V-GRPO timestep grid must normalize into [0, 1]; got "
            f"min={float(t.min()):.4g}, max={float(t.max()):.4g} after the /1000 heuristic",
        )
    return t


__all__ = ["VGRPO", "VGRPOConfig"]
