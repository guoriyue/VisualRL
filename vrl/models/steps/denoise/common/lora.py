"""Shared PEFT LoRA attach logic for diffusion family models.

Five families carried near-identical copies of ``apply_lora``; the PEFT call
convention (validated saved adapter for warm start vs
LoraConfig+get_peft_model for fresh adapters) is family-agnostic, so it lives
here once. Families only override the small hooks that actually differ.

The previous-policy adapter primitives (``build_lora_config`` /
``copy_adapter_weights`` / ``freeze_checkpoint_owned_adapter_params``) also
live here: they are pure PEFT operations with no family specifics. The frozen
``previous`` LoRA mirror they build is what DiffusionNFT's negative branch and
V-GRPO's importance ratio evaluate the behaviour policy through, so the mixin
carries the attach / sync pair once (``model.nft_previous_adapter: true`` opts
a LoRA family in) instead of every family copying it; cosmos/predict2.5 keeps
its own ``apply_lora`` because it always builds the mirror.
"""

from __future__ import annotations

from typing import Any

from vrl.models.interfaces.runtime import ModelBuild, register_checkpoint_owned_state
from vrl.models.peft_adapter import load_trainable_lora_adapter


class LoraModelMixin:
    """Attach a PEFT LoRA adapter to the family transformer per runtime build."""

    # Fresh-adapter weight init when the config does not set init_lora_weights.
    # Wan overrides to True: empty training adapters must initially preserve
    # base Wan output.
    _lora_default_init_weights: Any = "gaussian"

    def _lora_transformer(self) -> Any:
        """The trainable transformer to wrap.

        Reads ``self.transformer`` — kept in sync with ``pipeline.transformer`` by
        every family's ``_set_transformer`` — instead of ``pipeline.transformer``,
        so this one attach path also serves the pipeline-less replay models (which
        set ``self.transformer`` directly and raise on ``pipeline``).
        """
        return self.transformer

    def _lora_dtype(self, build: ModelBuild) -> Any | None:
        """Dtype for the pre-wrap device move; ``None`` skips the cast.

        Default: the build's parameter dtype — the near-universal diffusers-family
        behavior. Overrides: cosmos predict2/cosmos3 return ``None`` (their
        transformer is already cast at load); anima/echo return their stored
        ``self._dtype`` (single-file checkpoints with their own parameter storage).
        """

        return build.parameter_dtype

    def apply_lora(self, build: ModelBuild) -> None:
        """Wrap the family transformer with PEFT LoRA per ``build.lora_*``."""
        from peft import LoraConfig, get_peft_model

        transformer = self._lora_transformer()
        transformer.requires_grad_(False)
        dtype = self._lora_dtype(build)
        # Quantized rollouts keep the checkpoint on CPU until base-weight
        # compaction. FSDP replay keeps it there until fully_shard can move and
        # shard one transformer block at a time. Either path would otherwise
        # materialize the full base model on one GPU before its memory-saving
        # transform owns the parameters.
        rollout = getattr(build, "rollout", None)
        defer_device_move = bool(
            getattr(build, "defer_trainable_device_move", False)
            or (
                rollout is not None
                and getattr(getattr(build, "precision", None), "quantization", None)
            ),
        )
        if not defer_device_move:
            if dtype is None:
                transformer.to(self.device)
            else:
                transformer.to(self.device, dtype=dtype)

        lora_config = build.lora
        if lora_config is None:
            raise ValueError("LoRA runtime build requires model.lora configuration")

        lora_path = build.lora_path
        if lora_path:
            wrapped = load_trainable_lora_adapter(
                transformer,
                lora_path,
                expected_rank=lora_config["rank"],
                expected_alpha=lora_config["alpha"],
                expected_dropout=lora_config.get("dropout", 0.0),
                expected_target_modules=lora_config["target_modules"],
            )
            wrapped.set_adapter("default")
            self._set_transformer(wrapped)
            return

        cfg = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config.get("dropout", 0.0),
            init_lora_weights=lora_config.get(
                "init_lora_weights",
                self._lora_default_init_weights,
            ),
            target_modules=lora_config["target_modules"],
        )
        self._set_transformer(get_peft_model(transformer, cfg))
        if previous_policy_adapter_requested(build):
            self.attach_previous_policy_adapter(build)

    # -- previous-policy adapter (DiffusionNFT / V-GRPO) ------------------
    # Both objectives evaluate the behaviour policy through a frozen ``previous``
    # copy of the trainable adapter: forward-only under no_grad, refreshed by
    # weight copy after each optimizer step, never optimized. Attach runs right
    # after the normal LoRA attach (``self.transformer`` must carry ``default``).

    def attach_previous_policy_adapter(self, build: ModelBuild) -> None:
        """Build the frozen ``previous`` adapter, seeded from ``default``.

        Idempotent on the adapter slot: only adds it once, then (re)seeds it from
        the current ``default`` so ``previous == default`` at attach time (the
        lr=0 invariants of NFT and V-GRPO). Leaves ``default`` active.
        """

        transformer = self.transformer
        lora_config = getattr(build, "lora", None)
        if lora_config is None:
            raise ValueError(
                "attach_previous_policy_adapter requires build.lora (LoRA only)",
            )
        if "previous" not in getattr(transformer, "peft_config", {}):
            transformer.add_adapter("previous", build_lora_config(lora_config))
        copy_adapter_weights(transformer, src="default", dst="previous")
        freeze_checkpoint_owned_adapter_params(transformer, "previous")
        transformer.set_adapter("default")

    def sync_previous_policy_adapter(self, *, decay: float = 0.0) -> None:
        """Refresh the ``previous`` adapter from the trainable ``default`` adapter.

        Reached via getattr dispatch from the objectives' ``after_optimizer_step``
        (vrl/algorithms/diffusion_nft.py, vrl/algorithms/v_grpo.py), not a
        direct call — keep even though textual call-site searches miss it.
        """

        copy_adapter_weights(self.transformer, src="default", dst="previous", decay=decay)


def previous_policy_adapter_requested(build: ModelBuild) -> bool:
    """Whether ``model.nft_previous_adapter`` asks for the frozen mirror."""

    # Bare test builds are namespaces without model_config; treat as "no".
    model_config = getattr(build, "model_config", None) or {}
    return bool(model_config.get("nft_previous_adapter", False))


def require_lora_for_previous_policy_adapter(build: ModelBuild) -> None:
    """Reject the previous-adapter switch without LoRA before paying a model load."""

    if previous_policy_adapter_requested(build) and not build.use_lora:
        raise RuntimeError(
            "model.nft_previous_adapter requires LoRA (the frozen previous "
            "adapter is a PEFT adapter); set model.use_lora=true.",
        )


def build_lora_config(lora_config: Any) -> Any:
    """Build the LoRA config for a family's adapters from one ``model.lora`` block.

    The ``default`` and the frozen ``previous`` mirror use identical settings, so
    this names that single shape instead of repeating the literal. Init does not
    matter for ``previous`` (it is overwritten by ``copy_adapter_weights`` right
    after creation), so ``gaussian`` matches the fresh-default init.
    """

    from peft import LoraConfig

    return LoraConfig(
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config.get("dropout", 0.0),
        init_lora_weights="gaussian",
        target_modules=lora_config["target_modules"],
    )


def copy_adapter_weights(
    module: Any,
    *,
    src: str,
    dst: str,
    decay: float = 0.0,
) -> None:
    """Copy (or EMA-blend) one PEFT adapter's params into another in place.

    ``decay=0`` is an exact copy; ``decay`` in (0, 1] is a soft update
    ``dst <- decay*dst + (1-decay)*src`` (NFT ``weight_copy_decay``). Matches
    params by the ``.{src}.`` / ``.{dst}.`` marker PEFT puts in every adapter path.
    """

    named = dict(module.named_parameters())
    copied = 0
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"adapter weight copy decay must be in [0, 1], got {decay}")
    for name, param in named.items():
        src_marker = f".{src}."
        if src_marker not in name:
            continue
        dst_name = name.replace(src_marker, f".{dst}.")
        dst_param = named.get(dst_name)
        if dst_param is None:
            continue
        if decay == 0.0:
            dst_param.data.copy_(param.data)
        else:
            dst_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        copied += 1
    if copied == 0:
        raise RuntimeError(
            f"failed to copy adapter weights from {src!r} to {dst!r}; "
            "no matching adapter parameters were found",
        )


def freeze_checkpoint_owned_adapter_params(module: Any, adapter: str) -> None:
    """Freeze a mutable PEFT adapter and register it for exact checkpoint resume.

    Used for NFT's ``previous`` adapter: it is only forward-evaluated under
    no_grad and refreshed by weight copy (``sync_previous_policy_adapter``),
    never optimized. PEFT creates adapter params with ``requires_grad=True``, so
    without this DDP's reducer expects a gradient for them that the no_grad
    replay never produces — failing the first backward unless the more expensive
    ``find_unused_parameters=true`` is forced. Freezing it keeps the cheaper
    ``find_unused_parameters=false`` correct.
    """

    marker = f".{adapter}."
    matched = [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if marker in name
    ]
    if not matched:
        raise RuntimeError(
            f"no parameters found for adapter {adapter!r} to freeze",
        )
    prior_requires_grad = [parameter.requires_grad for _, parameter in matched]
    try:
        for _, parameter in matched:
            parameter.requires_grad_(False)
        register_checkpoint_owned_state(module, (name for name, _ in matched))
    except BaseException:
        for (_, parameter), requires_grad in zip(
            matched,
            prior_requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(requires_grad)
        raise


__all__ = [
    "LoraModelMixin",
    "build_lora_config",
    "copy_adapter_weights",
    "freeze_checkpoint_owned_adapter_params",
    "previous_policy_adapter_requested",
    "require_lora_for_previous_policy_adapter",
]
