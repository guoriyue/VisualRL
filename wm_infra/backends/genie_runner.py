"""Genie model runner with real and stub execution modes.

This module adapts the 1x-technologies/GENIE STMaskGIT model into a runner
that can be called by the GenieRolloutBackend.

Modes:
- **real**: Loads the STMaskGIT model via ``from_pretrained`` and runs actual
  token-level autoregressive generation (maskgit_generate per frame).
- **stub**: No model loaded — returns synthetic token tensors seeded from the
  prompt hash. Used in tests or environments without Genie dependencies.

The runner is deliberately separated from the backend so the backend stays
focused on control-plane / artifact plumbing while the runner owns model I/O.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("wm_infra.genie_runner")

# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

_GENIE_AVAILABLE: bool | None = None


def genie_available() -> bool:
    """Return True if the genie + torch stack can be imported."""
    global _GENIE_AVAILABLE
    if _GENIE_AVAILABLE is not None:
        return _GENIE_AVAILABLE
    try:
        import torch  # noqa: F401
        from genie.st_mask_git import STMaskGIT  # noqa: F401
        _GENIE_AVAILABLE = True
    except Exception:
        _GENIE_AVAILABLE = False
    return _GENIE_AVAILABLE


# ---------------------------------------------------------------------------
# Runner result
# ---------------------------------------------------------------------------


@dataclass
class GenieRunResult:
    """Output of a single Genie runner invocation."""

    mode: str  # "real" or "stub"
    tokens_generated: int = 0
    frames_generated: int = 0
    prompt_frames: int = 0
    total_frames: int = 0
    spatial_h: int = 16
    spatial_w: int = 16
    vocab_size: int = 262144
    elapsed_s: float = 0.0
    model_name: str = ""
    device: str = "cpu"

    # Persisted file paths (filled by runner)
    tokens_path: Optional[str] = None
    logits_path: Optional[str] = None
    state_path: Optional[str] = None

    # Error info for degraded runs
    error: Optional[str] = None

    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class GenieRunner:
    """Thin adapter that owns model loading and token generation.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id or local path passed to ``STMaskGIT.from_pretrained``.
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    num_prompt_frames : int
        Number of leading frames treated as context (not generated).
    maskgit_steps : int
        Refinement iterations per generated frame.
    temperature : float
        Sampling temperature (0 = argmax).
    """

    def __init__(
        self,
        model_name_or_path: str = "1x-technologies/GENIE_210M_v0",
        device: str = "cuda",
        num_prompt_frames: int = 8,
        maskgit_steps: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.num_prompt_frames = num_prompt_frames
        self.maskgit_steps = maskgit_steps
        self.temperature = temperature

        self._model = None
        self._mode: str = "stub"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> str:
        """Attempt to load the real Genie model.  Returns the mode string."""
        if self._model is not None:
            return self._mode

        if not genie_available():
            logger.warning("Genie dependencies not available — staying in stub mode")
            self._mode = "stub"
            return self._mode

        try:
            import torch
            from genie.st_mask_git import STMaskGIT

            logger.info("Loading Genie model from %s …", self.model_name_or_path)
            t0 = time.monotonic()
            model = STMaskGIT.from_pretrained(self.model_name_or_path)
            model = model.to(self.device)
            model.eval()
            elapsed = time.monotonic() - t0
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(
                "Genie model loaded: %.1fM params on %s in %.1fs",
                param_count / 1e6,
                self.device,
                elapsed,
            )
            self._model = model
            self._mode = "real"
        except Exception as exc:
            logger.warning("Failed to load Genie model (%s) — falling back to stub: %s", self.model_name_or_path, exc)
            self._mode = "stub"

        return self._mode

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        output_dir: Path,
        prompt: str = "",
        seed: int = 42,
        num_frames: int = 16,
        input_tokens: Optional[np.ndarray] = None,
    ) -> GenieRunResult:
        """Run generation and persist artifacts to *output_dir*.

        Parameters
        ----------
        output_dir
            Directory to write tokens.npy, state.json, and optionally logits.npy.
        prompt
            Text prompt (used for seed derivation in stub mode, or metadata).
        seed
            Random seed.
        num_frames
            Total frames in the output window (including prompt frames).
        input_tokens
            Optional pre-existing prompt tokens as ``(T, H, W)`` uint32 array.
            If None, synthetic tokens are created from the seed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "real" and self._model is not None:
            result = self._run_real(output_dir, prompt, seed, num_frames, input_tokens)
            if result.error:
                logger.warning(
                    "Genie real-mode execution failed; falling back to stub mode: %s",
                    result.error,
                )
                self._mode = "stub"
                fallback = self._run_stub(output_dir, prompt, seed, num_frames, input_tokens)
                fallback.extra.update(
                    {
                        "fallback_from": "real",
                        "fallback_error": result.error,
                    }
                )
                return fallback
            return result
        return self._run_stub(output_dir, prompt, seed, num_frames, input_tokens)

    # ------------------------------------------------------------------
    # Real execution
    # ------------------------------------------------------------------

    def _run_real(
        self,
        output_dir: Path,
        prompt: str,
        seed: int,
        num_frames: int,
        input_tokens: Optional[np.ndarray],
    ) -> GenieRunResult:
        import torch

        model = self._model
        assert model is not None

        T = min(num_frames, model.config.T)
        h, w = model.h, model.w
        mask_id = model.mask_token_id

        torch.manual_seed(seed)

        # Build prompt tensor -------------------------------------------
        if input_tokens is not None:
            prompt_np = np.asarray(input_tokens, dtype=np.int64)
            if prompt_np.ndim == 1:
                # Flat (T*H*W,) → reshape
                prompt_np = prompt_np[: T * h * w].reshape(T, h, w)
            elif prompt_np.ndim == 3:
                prompt_np = prompt_np[:T]
            prompt_THW = torch.from_numpy(prompt_np).unsqueeze(0).to(self.device)
        else:
            # Derive synthetic prompt tokens from seed
            rng = np.random.RandomState(seed)
            prompt_np = rng.randint(0, model.config.image_vocab_size, size=(T, h, w)).astype(np.int64)
            prompt_THW = torch.from_numpy(prompt_np).unsqueeze(0).to(self.device)

        # Mask future frames
        num_prompt = min(self.num_prompt_frames, T - 1)
        prompt_THW[:, num_prompt:] = mask_id

        t0 = time.monotonic()
        error: str | None = None
        frames_generated = 0

        try:
            with torch.no_grad():
                for t_idx in range(num_prompt, T):
                    samples_HW, _logits = model.maskgit_generate(
                        prompt_THW,
                        out_t=t_idx,
                        maskgit_steps=self.maskgit_steps,
                        temperature=self.temperature,
                    )
                    prompt_THW[:, t_idx] = samples_HW
                    frames_generated += 1
        except Exception as exc:
            error = str(exc)
            logger.error("Genie generation error at frame %d: %s", num_prompt + frames_generated, exc)

        elapsed = time.monotonic() - t0

        # Persist tokens ------------------------------------------------
        out_tokens = prompt_THW[0].cpu().numpy().astype(np.uint32)
        tokens_path = output_dir / "tokens.npy"
        np.save(str(tokens_path), out_tokens)

        state_payload = {
            "mode": "real",
            "model": self.model_name_or_path,
            "device": self.device,
            "seed": seed,
            "prompt": prompt,
            "num_prompt_frames": num_prompt,
            "total_frames": T,
            "frames_generated": frames_generated,
            "spatial": [h, w],
            "vocab_size": model.config.image_vocab_size,
            "maskgit_steps": self.maskgit_steps,
            "temperature": self.temperature,
            "elapsed_s": round(elapsed, 4),
            "error": error,
        }
        state_path = output_dir / "state.json"
        state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True))

        return GenieRunResult(
            mode="real",
            tokens_generated=frames_generated * h * w,
            frames_generated=frames_generated,
            prompt_frames=num_prompt,
            total_frames=T,
            spatial_h=h,
            spatial_w=w,
            vocab_size=model.config.image_vocab_size,
            elapsed_s=round(elapsed, 4),
            model_name=self.model_name_or_path,
            device=self.device,
            tokens_path=str(tokens_path),
            state_path=str(state_path),
            error=error,
        )

    # ------------------------------------------------------------------
    # Stub execution
    # ------------------------------------------------------------------

    def _run_stub(
        self,
        output_dir: Path,
        prompt: str,
        seed: int,
        num_frames: int,
        input_tokens: Optional[np.ndarray],
    ) -> GenieRunResult:
        T, h, w = min(num_frames, 16), 16, 16
        vocab_size = 262144

        # Deterministic synthetic tokens from prompt + seed
        combined_seed = int(hashlib.sha256(f"{prompt}:{seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(combined_seed % (2**31))
        tokens = rng.randint(0, vocab_size, size=(T, h, w)).astype(np.uint32)

        num_prompt = min(self.num_prompt_frames, T - 1)
        frames_generated = T - num_prompt

        t0 = time.monotonic()

        tokens_path = output_dir / "tokens.npy"
        np.save(str(tokens_path), tokens)

        state_payload = {
            "mode": "stub",
            "model": self.model_name_or_path,
            "device": "cpu",
            "seed": seed,
            "prompt": prompt,
            "num_prompt_frames": num_prompt,
            "total_frames": T,
            "frames_generated": frames_generated,
            "spatial": [h, w],
            "vocab_size": vocab_size,
            "maskgit_steps": self.maskgit_steps,
            "temperature": self.temperature,
            "elapsed_s": 0.0,
            "error": None,
            "note": "Stub mode — synthetic tokens generated from prompt hash, no real model execution.",
        }
        state_path = output_dir / "state.json"
        state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True))

        elapsed = time.monotonic() - t0

        return GenieRunResult(
            mode="stub",
            tokens_generated=frames_generated * h * w,
            frames_generated=frames_generated,
            prompt_frames=num_prompt,
            total_frames=T,
            spatial_h=h,
            spatial_w=w,
            vocab_size=vocab_size,
            elapsed_s=round(elapsed, 4),
            model_name=self.model_name_or_path,
            device="cpu",
            tokens_path=str(tokens_path),
            state_path=str(state_path),
        )
