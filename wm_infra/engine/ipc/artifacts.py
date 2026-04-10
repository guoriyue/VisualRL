"""Artifact store for writing rollout results to tmpfs.

The ONE place where tensors cross the process boundary — as numpy file
writes to ``/dev/shm``, not as pickled objects.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from wm_infra.engine.ipc.protocol import ArtifactRef
from wm_infra.engine.types import StepResult

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Manages rollout artifacts on tmpfs (``/dev/shm`` by default)."""

    def __init__(self, root: str = "/dev/shm/wm-engine") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_results(self, request_id: str, results: list[StepResult]) -> ArtifactRef:
        """Write step results to tmpfs.

        Creates:
        - ``meta.json``: list of per-step dicts (no ``output_latent``).
        - ``latent.npy``: final ``output_latent`` as numpy array (if present).

        Returns an ``ArtifactRef`` pointing to the artifact directory.
        """
        artifact_dir = self.root / request_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Write meta.json (lightweight per-step metadata, no tensors)
        meta = []
        for r in results:
            meta.append({
                "request_id": r.request_id,
                "step_index": r.step_index,
                "done": r.done,
                "metadata": r.metadata,
            })
        meta_path = artifact_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, separators=(",", ":")))

        # Write latent.npy from the final step's output_latent (if any)
        latent_path = artifact_dir / "latent.npy"
        shape: tuple[int, ...] | None = None
        dtype_str: str | None = None
        size_bytes = 0

        final_latent = None
        for r in reversed(results):
            if r.output_latent is not None:
                final_latent = r.output_latent
                break

        if final_latent is not None:
            arr = _to_numpy(final_latent)
            np.save(str(latent_path), arr)
            shape = arr.shape
            dtype_str = str(arr.dtype)
            size_bytes = latent_path.stat().st_size
        else:
            size_bytes = meta_path.stat().st_size

        return ArtifactRef(
            path=str(artifact_dir),
            content_type="application/x-npy",
            size_bytes=size_bytes,
            shape=shape,
            dtype=dtype_str,
        )

    def read_meta(self, request_id: str) -> list[dict] | None:
        """Read ``meta.json`` for a completed request."""
        meta_path = self.root / request_id / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())

    def read_latent_path(self, request_id: str) -> Path | None:
        """Return path to ``latent.npy`` if it exists."""
        latent_path = self.root / request_id / "latent.npy"
        return latent_path if latent_path.exists() else None

    def cleanup(self, request_id: str) -> None:
        """Remove artifact directory for a request."""
        artifact_dir = self.root / request_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    def cleanup_older_than(self, seconds: float) -> int:
        """Remove artifacts older than TTL. Returns count removed."""
        cutoff = time.time() - seconds
        removed = 0
        if not self.root.exists():
            return 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(child)
                    removed += 1
            except OSError:
                pass
        return removed


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch Tensor (or numpy array) to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    # torch.Tensor
    return tensor.detach().cpu().numpy()
