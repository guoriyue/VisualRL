"""Artifact store for writing engine results to tmpfs.

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
from wm_infra.engine.types import RequestOutput

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Manages request artifacts on tmpfs (``/dev/shm`` by default)."""

    def __init__(self, root: str = "/dev/shm/wm-engine") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_result(self, request_id: str, result: RequestOutput) -> ArtifactRef:
        """Write one completed request output to tmpfs.

        Creates:
        - ``meta.json``: JSON-safe metadata about the completed request.
        - ``result.npy``: tensor/ndarray-like payload when the output carries one.

        Returns an ``ArtifactRef`` pointing to the artifact directory.
        """
        artifact_dir = self.root / request_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        payload = result.data
        array_payload = _extract_array_payload(payload)
        meta = {
            "request_id": result.request_id,
            "finished": result.finished,
            "finish_reason": result.finish_reason,
            "payload": _json_safe(payload),
        }
        meta_path = artifact_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, separators=(",", ":")))

        result_path = artifact_dir / "result.npy"
        shape: tuple[int, ...] | None = None
        dtype_str: str | None = None
        size_bytes = 0

        if array_payload is not None:
            arr = _to_numpy(array_payload)
            np.save(str(result_path), arr)
            shape = arr.shape
            dtype_str = str(arr.dtype)
            size_bytes = result_path.stat().st_size
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

    def read_result_path(self, request_id: str) -> Path | None:
        """Return path to ``result.npy`` if it exists."""
        result_path = self.root / request_id / "result.npy"
        return result_path if result_path.exists() else None

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


def _extract_array_payload(payload):
    """Extract the main ndarray/tensor payload from a request result."""
    if payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return payload
    if hasattr(payload, "detach") and hasattr(payload, "cpu") and hasattr(payload, "numpy"):
        return payload
    if isinstance(payload, dict):
        for key in ("_pipeline_output", "output", "video_frames", "frames"):
            value = payload.get(key)
            if value is None:
                continue
            extracted = _extract_array_payload(value)
            if extracted is not None:
                return extracted
        return None
    if isinstance(payload, list):
        for item in reversed(payload):
            extracted = _extract_array_payload(item)
            if extracted is not None:
                return extracted
        return None
    if hasattr(payload, "output"):
        return _extract_array_payload(payload.output)
    if hasattr(payload, "state_updates"):
        return _extract_array_payload(payload.state_updates)
    return None


def _json_safe(value):
    """Best-effort JSON-safe projection for metadata persistence."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "detach"):
        shape = list(value.shape) if value.shape is not None else None
        return {"type": "tensor", "shape": shape, "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)
