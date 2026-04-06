"""Genie model runner with explicit stage-level execution helpers.

This module adapts the 1x-technologies/GENIE STMaskGIT model into a runner
that can be called by the GenieRolloutBackend.

Modes:
- **real**: Loads the STMaskGIT model via ``from_pretrained`` and runs actual
  token-level autoregressive generation (maskgit_generate per frame).
- **stub**: No model loaded — returns synthetic token tensors seeded from the
  prompt hash. Used in tests or environments without Genie dependencies.

The runner stays focused on model-facing execution. Control-plane persistence
and temporal lineage remain in the backend.
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
import torch

logger = logging.getLogger("wm_infra.genie_runner")

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
    tokens_path: Optional[str] = None
    logits_path: Optional[str] = None
    state_path: Optional[str] = None
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeniePreparedRun:
    """Prepared mutable runtime state for stage-oriented Genie execution."""

    mode: str
    prompt: str
    seed: int
    total_frames: int
    prompt_frames: int
    spatial_h: int
    spatial_w: int
    vocab_size: int
    maskgit_steps: int
    temperature: float
    model_name: str
    device: str
    tokens_buffer: object
    generated_until: int
    dtype: str = "uint32"
    error: Optional[str] = None
    elapsed_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def frames_generated(self) -> int:
        return max(self.generated_until - self.prompt_frames, 0)

    def current_tokens_numpy(self) -> np.ndarray:
        if self.mode == "real":
            tokens = self.tokens_buffer
            assert isinstance(tokens, torch.Tensor)
            return tokens[0, :self.total_frames].detach().cpu().numpy().astype(np.uint32)
        return np.asarray(self.tokens_buffer, dtype=np.uint32)[: self.total_frames]


@dataclass
class GenieWindowResult:
    """Result of advancing one bounded frame window."""

    frame_start: int
    frame_end: int
    frames_generated: int
    elapsed_s: float
    error: Optional[str] = None
    batch_size: int = 1
    batched: bool = False


class GenieRunner:
    """Thin adapter that owns model loading and token generation."""

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

    def load(self) -> str:
        """Attempt to load the real Genie model. Returns the mode string."""
        if self._model is not None:
            return self._mode

        if not genie_available():
            logger.warning("Genie dependencies not available — staying in stub mode")
            self._mode = "stub"
            return self._mode

        try:
            from genie.st_mask_git import STMaskGIT

            logger.info("Loading Genie model from %s …", self.model_name_or_path)
            t0 = time.monotonic()
            model = STMaskGIT.from_pretrained(self.model_name_or_path)
            if str(self.device).startswith("cuda"):
                model = model.half()
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
            logger.warning(
                "Failed to load Genie model (%s) — falling back to stub: %s",
                self.model_name_or_path,
                exc,
            )
            self._mode = "stub"

        return self._mode

    def _resolve_overrides(
        self,
        *,
        num_prompt_frames: Optional[int],
        maskgit_steps: Optional[int],
        temperature: Optional[float],
    ) -> tuple[int, int, float]:
        prompt_frames = self.num_prompt_frames if num_prompt_frames is None else num_prompt_frames
        steps = self.maskgit_steps if maskgit_steps is None else maskgit_steps
        temp = self.temperature if temperature is None else temperature
        return prompt_frames, steps, temp

    def prepare_inputs(
        self,
        *,
        prompt: str = "",
        seed: int = 42,
        num_frames: int = 16,
        input_tokens: Optional[np.ndarray] = None,
        num_prompt_frames: Optional[int] = None,
        maskgit_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GeniePreparedRun:
        """Materialize prompt state and build mutable runtime buffers."""

        prompt_frames, steps, temp = self._resolve_overrides(
            num_prompt_frames=num_prompt_frames,
            maskgit_steps=maskgit_steps,
            temperature=temperature,
        )

        if self._mode == "real" and self._model is not None:
            model = self._model
            requested_T = max(num_frames, 1)
            T = int(model.config.T)
            h, w = int(model.h), int(model.w)
            mask_id = int(model.mask_token_id)

            if requested_T > T:
                return GeniePreparedRun(
                    mode="real",
                    prompt=prompt,
                    seed=seed,
                    total_frames=requested_T,
                    prompt_frames=min(max(prompt_frames, 0), requested_T - 1),
                    spatial_h=h,
                    spatial_w=w,
                    vocab_size=int(model.config.image_vocab_size),
                    maskgit_steps=steps,
                    temperature=temp,
                    model_name=self.model_name_or_path,
                    device=self.device,
                    tokens_buffer=torch.empty((1, T, h, w), dtype=torch.long, device=self.device),
                    generated_until=0,
                    dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
                    error=f"Requested num_frames={requested_T} exceeds Genie model context T={T}",
                )

            torch.manual_seed(seed)
            prompt_THW = torch.full((1, T, h, w), mask_id, dtype=torch.long, device=self.device)
            if input_tokens is not None:
                prompt_np = np.asarray(input_tokens, dtype=np.int64)
                if prompt_np.ndim == 1:
                    prompt_np = prompt_np[: requested_T * h * w].reshape(requested_T, h, w)
                if prompt_np.ndim != 3:
                    raise ValueError("Genie real-mode input_tokens must resolve to a 3D [T,H,W] token tensor")
                if prompt_np.shape[1:] != (h, w):
                    raise ValueError(
                        f"Genie real-mode input_tokens must have spatial shape {(h, w)}, got {tuple(prompt_np.shape[1:])}"
                    )
                available_prompt_frames = min(prompt_np.shape[0], requested_T)
                prompt_THW[:, :available_prompt_frames] = (
                    torch.from_numpy(prompt_np[:available_prompt_frames]).unsqueeze(0).to(self.device)
                )
            else:
                rng = np.random.RandomState(seed)
                prompt_np = rng.randint(0, model.config.image_vocab_size, size=(T, h, w)).astype(np.int64)
                prompt_THW = torch.from_numpy(prompt_np).unsqueeze(0).to(self.device)

            num_prompt = min(max(prompt_frames, 0), requested_T - 1)
            if input_tokens is not None:
                num_prompt = min(num_prompt, prompt_np.shape[0])
            prompt_THW[:, num_prompt:] = mask_id

            return GeniePreparedRun(
                mode="real",
                prompt=prompt,
                seed=seed,
                total_frames=requested_T,
                prompt_frames=num_prompt,
                spatial_h=h,
                spatial_w=w,
                vocab_size=int(model.config.image_vocab_size),
                maskgit_steps=steps,
                temperature=temp,
                model_name=self.model_name_or_path,
                device=self.device,
                tokens_buffer=prompt_THW,
                generated_until=num_prompt,
                dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
            )

        T, h, w = min(max(num_frames, 1), 16), 16, 16
        vocab_size = 262144
        combined_seed = int(hashlib.sha256(f"{prompt}:{seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(combined_seed % (2**31))
        tokens = rng.randint(0, vocab_size, size=(T, h, w)).astype(np.uint32)

        num_prompt = min(max(prompt_frames, 0), T - 1)
        if input_tokens is not None:
            prompt_np = np.asarray(input_tokens, dtype=np.uint32)
            if prompt_np.ndim == 1:
                prompt_np = prompt_np[: T * h * w].reshape(T, h, w)
            elif prompt_np.ndim == 3:
                prompt_np = prompt_np[:T]
            prompt_frames_available = min(prompt_np.shape[0], num_prompt)
            if prompt_frames_available > 0:
                tokens[:prompt_frames_available] = prompt_np[:prompt_frames_available]

        return GeniePreparedRun(
            mode="stub",
            prompt=prompt,
            seed=seed,
            total_frames=T,
            prompt_frames=num_prompt,
            spatial_h=h,
            spatial_w=w,
            vocab_size=vocab_size,
            maskgit_steps=steps,
            temperature=temp,
            model_name=self.model_name_or_path,
            device="cpu",
            tokens_buffer=tokens,
            generated_until=num_prompt,
        )

    def run_window(self, prepared: GeniePreparedRun, *, frame_start: int, frame_end: int) -> GenieWindowResult:
        """Advance a bounded frame window within a prepared run."""

        frame_start = max(frame_start, prepared.prompt_frames)
        frame_end = min(frame_end, prepared.total_frames)
        if frame_end <= frame_start or prepared.error is not None:
            return GenieWindowResult(
                frame_start=frame_start,
                frame_end=frame_end,
                frames_generated=0,
                elapsed_s=0.0,
                error=prepared.error,
                batch_size=1,
                batched=False,
            )

        t0 = time.monotonic()
        error: str | None = None

        if prepared.mode == "real":
            model = self._model
            assert model is not None
            prompt_THW = prepared.tokens_buffer
            assert isinstance(prompt_THW, torch.Tensor)
            try:
                with torch.no_grad():
                    for t_idx in range(frame_start, frame_end):
                        samples_HW, _logits = model.maskgit_generate(
                            prompt_THW,
                            out_t=t_idx,
                            maskgit_steps=prepared.maskgit_steps,
                            temperature=prepared.temperature,
                        )
                        prompt_THW[:, t_idx] = samples_HW
            except Exception as exc:
                error = str(exc)
                logger.error("Genie generation error at frame %d: %s", frame_start, exc)

        elapsed = time.monotonic() - t0
        if error is None:
            prepared.generated_until = max(prepared.generated_until, frame_end)
        prepared.elapsed_s += elapsed
        prepared.error = prepared.error or error
        return GenieWindowResult(
            frame_start=frame_start,
            frame_end=frame_end,
            frames_generated=max(frame_end - frame_start, 0) if error is None else 0,
            elapsed_s=elapsed,
            error=error,
            batch_size=1,
            batched=False,
        )

    def run_chunk(
        self,
        prepared_runs: list[GeniePreparedRun],
        *,
        frame_start: int,
        frame_end: int,
    ) -> list[GenieWindowResult]:
        """Advance one shared frame window across multiple prepared Genie runs."""

        if not prepared_runs:
            return []
        if len(prepared_runs) == 1:
            result = self.run_window(prepared_runs[0], frame_start=frame_start, frame_end=frame_end)
            result.batch_size = 1
            result.batched = False
            return [result]

        batch_size = len(prepared_runs)
        first = prepared_runs[0]
        modes = {prepared.mode for prepared in prepared_runs}
        if len(modes) != 1:
            return [
                GenieWindowResult(
                    frame_start=frame_start,
                    frame_end=frame_end,
                    frames_generated=0,
                    elapsed_s=0.0,
                    error="Mixed Genie runner modes in one transition chunk",
                    batch_size=batch_size,
                    batched=False,
                )
                for _ in prepared_runs
            ]

        if first.mode != "real":
            results = [
                self.run_window(prepared, frame_start=frame_start, frame_end=frame_end)
                for prepared in prepared_runs
            ]
            for result in results:
                result.batch_size = batch_size
                result.batched = True
            return results

        model = self._model
        assert model is not None
        prompt_tensors = [prepared.tokens_buffer for prepared in prepared_runs]
        if not all(isinstance(tensor, torch.Tensor) for tensor in prompt_tensors):
            raise TypeError("Genie real-mode batch execution requires torch.Tensor token buffers")

        batch_prompt = torch.cat([tensor for tensor in prompt_tensors if isinstance(tensor, torch.Tensor)], dim=0)
        safe_start = max(frame_start, max(prepared.prompt_frames for prepared in prepared_runs))
        safe_end = min(frame_end, min(prepared.total_frames for prepared in prepared_runs))
        if safe_end <= safe_start:
            return [
                GenieWindowResult(
                    frame_start=safe_start,
                    frame_end=safe_end,
                    frames_generated=0,
                    elapsed_s=0.0,
                    error=prepared.error,
                    batch_size=batch_size,
                    batched=True,
                )
                for prepared in prepared_runs
            ]

        error: str | None = None
        t0 = time.monotonic()
        try:
            with torch.no_grad():
                for t_idx in range(safe_start, safe_end):
                    samples_hw, _logits = model.maskgit_generate(
                        batch_prompt,
                        out_t=t_idx,
                        maskgit_steps=first.maskgit_steps,
                        temperature=first.temperature,
                    )
                    batch_prompt[:, t_idx] = samples_hw
        except Exception as exc:
            error = str(exc)
            logger.error("Genie batched generation error at frame %d: %s", safe_start, exc)

        elapsed = time.monotonic() - t0
        results: list[GenieWindowResult] = []
        for index, prepared in enumerate(prepared_runs):
            prepared.tokens_buffer = batch_prompt[index:index + 1]
            if error is None:
                prepared.generated_until = max(prepared.generated_until, safe_end)
            prepared.elapsed_s += elapsed
            prepared.error = prepared.error or error
            results.append(
                GenieWindowResult(
                    frame_start=safe_start,
                    frame_end=safe_end,
                    frames_generated=max(safe_end - safe_start, 0) if error is None else 0,
                    elapsed_s=elapsed,
                    error=error,
                    batch_size=batch_size,
                    batched=True,
                )
            )
        return results

    def run_window_batch(
        self,
        prepared_runs: list[GeniePreparedRun],
        *,
        frame_start: int,
        frame_end: int,
    ) -> list[GenieWindowResult]:
        """Advance one bounded frame window across multiple compatible Genie runs."""

        if not prepared_runs:
            return []
        if len(prepared_runs) == 1:
            return [self.run_window(prepared_runs[0], frame_start=frame_start, frame_end=frame_end)]

        first = prepared_runs[0]
        if any(prepared.mode != first.mode for prepared in prepared_runs):
            return [self.run_window(prepared, frame_start=frame_start, frame_end=frame_end) for prepared in prepared_runs]

        normalized_start = max(frame_start, max(prepared.prompt_frames for prepared in prepared_runs))
        normalized_end = min(frame_end, min(prepared.total_frames for prepared in prepared_runs))
        if normalized_end <= normalized_start:
            return [
                GenieWindowResult(
                    frame_start=normalized_start,
                    frame_end=normalized_end,
                    frames_generated=0,
                    elapsed_s=0.0,
                    error=prepared.error,
                )
                for prepared in prepared_runs
            ]

        if first.mode == "stub":
            t0 = time.monotonic()
            elapsed = time.monotonic() - t0
            results: list[GenieWindowResult] = []
            for prepared in prepared_runs:
                if prepared.error is None:
                    prepared.generated_until = max(prepared.generated_until, normalized_end)
                prepared.elapsed_s += elapsed
                results.append(
                    GenieWindowResult(
                        frame_start=normalized_start,
                        frame_end=normalized_end,
                        frames_generated=max(normalized_end - normalized_start, 0) if prepared.error is None else 0,
                        elapsed_s=elapsed,
                        error=prepared.error,
                    )
                )
            return results

        if any(
            prepared.maskgit_steps != first.maskgit_steps
            or prepared.temperature != first.temperature
            or prepared.spatial_h != first.spatial_h
            or prepared.spatial_w != first.spatial_w
            or prepared.total_frames != first.total_frames
            for prepared in prepared_runs
        ):
            return [self.run_window(prepared, frame_start=frame_start, frame_end=frame_end) for prepared in prepared_runs]

        model = self._model
        if model is None:
            return [self.run_window(prepared, frame_start=frame_start, frame_end=frame_end) for prepared in prepared_runs]

        prompt_batches = []
        for prepared in prepared_runs:
            assert isinstance(prepared.tokens_buffer, torch.Tensor)
            prompt_batches.append(prepared.tokens_buffer)

        stacked = torch.cat(prompt_batches, dim=0)
        t0 = time.monotonic()
        error: str | None = None
        try:
            with torch.no_grad():
                for t_idx in range(normalized_start, normalized_end):
                    samples_HW, _logits = model.maskgit_generate(
                        stacked,
                        out_t=t_idx,
                        maskgit_steps=first.maskgit_steps,
                        temperature=first.temperature,
                    )
                    stacked[:, t_idx] = samples_HW
        except Exception as exc:
            error = str(exc)
            logger.warning("Genie batched window failed, falling back to per-request execution: %s", exc)

        if error is not None:
            return [self.run_window(prepared, frame_start=frame_start, frame_end=frame_end) for prepared in prepared_runs]

        elapsed = time.monotonic() - t0
        results = []
        for index, prepared in enumerate(prepared_runs):
            prepared.tokens_buffer = stacked[index:index + 1]
            prepared.generated_until = max(prepared.generated_until, normalized_end)
            prepared.elapsed_s += elapsed
            results.append(
                GenieWindowResult(
                    frame_start=normalized_start,
                    frame_end=normalized_end,
                    frames_generated=max(normalized_end - normalized_start, 0),
                    elapsed_s=elapsed,
                    error=None,
                )
            )
        return results

    def build_checkpoint_delta(self, prepared: GeniePreparedRun, *, frame_start: int, frame_end: int) -> np.ndarray:
        """Return the current token slice for a bounded frame window."""

        tokens = prepared.current_tokens_numpy()
        return tokens[frame_start:frame_end]

    def persist_outputs(self, prepared: GeniePreparedRun, *, output_dir: Path) -> GenieRunResult:
        """Persist tokens and state for a prepared run."""

        output_dir.mkdir(parents=True, exist_ok=True)
        out_tokens = prepared.current_tokens_numpy()

        tokens_path = output_dir / "tokens.npy"
        np.save(str(tokens_path), out_tokens.astype(np.uint32))

        state_payload = {
            "mode": prepared.mode,
            "model": prepared.model_name,
            "device": prepared.device,
            "seed": prepared.seed,
            "prompt": prepared.prompt,
            "num_prompt_frames": prepared.prompt_frames,
            "total_frames": prepared.total_frames,
            "frames_generated": prepared.frames_generated,
            "spatial": [prepared.spatial_h, prepared.spatial_w],
            "vocab_size": prepared.vocab_size,
            "maskgit_steps": prepared.maskgit_steps,
            "temperature": prepared.temperature,
            "dtype": prepared.dtype,
            "elapsed_s": round(prepared.elapsed_s, 4),
            "error": prepared.error,
        }
        if prepared.mode == "stub":
            state_payload["note"] = "Stub mode — synthetic tokens generated from prompt hash, no real model execution."

        state_path = output_dir / "state.json"
        state_path.write_text(json.dumps(state_payload, indent=2, sort_keys=True))

        return GenieRunResult(
            mode=prepared.mode,
            tokens_generated=prepared.frames_generated * prepared.spatial_h * prepared.spatial_w,
            frames_generated=prepared.frames_generated,
            prompt_frames=prepared.prompt_frames,
            total_frames=prepared.total_frames,
            spatial_h=prepared.spatial_h,
            spatial_w=prepared.spatial_w,
            vocab_size=prepared.vocab_size,
            elapsed_s=round(prepared.elapsed_s, 4),
            model_name=prepared.model_name,
            device=prepared.device,
            tokens_path=str(tokens_path),
            state_path=str(state_path),
            error=prepared.error,
            extra=dict(prepared.extra),
        )

    def run(
        self,
        *,
        output_dir: Path,
        prompt: str = "",
        seed: int = 42,
        num_frames: int = 16,
        input_tokens: Optional[np.ndarray] = None,
        num_prompt_frames: Optional[int] = None,
        maskgit_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenieRunResult:
        """Backward-compatible single-call wrapper around the stage APIs."""

        output_dir.mkdir(parents=True, exist_ok=True)

        if self._mode == "real" and self._model is not None:
            return self._run_real(
                output_dir=output_dir,
                prompt=prompt,
                seed=seed,
                num_frames=num_frames,
                input_tokens=input_tokens,
                num_prompt_frames=self.num_prompt_frames if num_prompt_frames is None else num_prompt_frames,
                maskgit_steps=self.maskgit_steps if maskgit_steps is None else maskgit_steps,
                temperature=self.temperature if temperature is None else temperature,
            )
        return self._run_stub(
            output_dir=output_dir,
            prompt=prompt,
            seed=seed,
            num_frames=num_frames,
            input_tokens=input_tokens,
            num_prompt_frames=self.num_prompt_frames if num_prompt_frames is None else num_prompt_frames,
            maskgit_steps=self.maskgit_steps if maskgit_steps is None else maskgit_steps,
            temperature=self.temperature if temperature is None else temperature,
        )

    def _run_real(
        self,
        output_dir: Path,
        prompt: str,
        seed: int,
        num_frames: int,
        input_tokens: Optional[np.ndarray],
        num_prompt_frames: int,
        maskgit_steps: int,
        temperature: float,
    ) -> GenieRunResult:
        prepared = self.prepare_inputs(
            prompt=prompt,
            seed=seed,
            num_frames=num_frames,
            input_tokens=input_tokens,
            num_prompt_frames=num_prompt_frames,
            maskgit_steps=maskgit_steps,
            temperature=temperature,
        )
        if prepared.error is None:
            self.run_window(
                prepared,
                frame_start=prepared.prompt_frames,
                frame_end=prepared.total_frames,
            )
        return self.persist_outputs(prepared, output_dir=output_dir)

    def _run_stub(
        self,
        output_dir: Path,
        prompt: str,
        seed: int,
        num_frames: int,
        input_tokens: Optional[np.ndarray],
        num_prompt_frames: int,
        maskgit_steps: int,
        temperature: float,
    ) -> GenieRunResult:
        prepared = self.prepare_inputs(
            prompt=prompt,
            seed=seed,
            num_frames=num_frames,
            input_tokens=input_tokens,
            num_prompt_frames=num_prompt_frames,
            maskgit_steps=maskgit_steps,
            temperature=temperature,
        )
        self.run_window(
            prepared,
            frame_start=prepared.prompt_frames,
            frame_end=prepared.total_frames,
        )
        return self.persist_outputs(prepared, output_dir=output_dir)
