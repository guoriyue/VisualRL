from __future__ import annotations

import numpy as np

from vrl.engine.ipc.artifacts import ArtifactStore
from vrl.engine.types import RequestOutput
from vrl.models.base import ModelResult


def test_write_and_read_video_tensor(tmp_path):
    store = ArtifactStore(root=str(tmp_path))
    video_tensor = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    result = RequestOutput(
        request_id="req-1",
        finished=True,
        data=[ModelResult(state_updates={"video_tensor": video_tensor})],
    )

    artifact = store.write_result("req-1", result)

    assert artifact.shape == video_tensor.shape
    assert store.read_result_path("req-1") is not None
