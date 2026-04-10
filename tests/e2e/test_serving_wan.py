"""Real Wan checkpoint end-to-end test through the engine stack."""

from __future__ import annotations

import numpy as np
import pytest

from tests.e2e.helpers import build_engine, require_cuda, require_real_model_opt_in, resolve_hf_snapshot
from wm_infra.models.families.wan.diffusers_i2v import DiffusersWanI2VModel
from wm_infra.schemas.video_generation import VideoGenerationRequest


@pytest.mark.asyncio
async def test_wan_diffusers_i2v_real_engine_e2e():
    """Run a real Wan diffusers checkpoint through engine -> pipeline -> result."""
    require_real_model_opt_in()
    require_cuda()

    snapshot = resolve_hf_snapshot("models--Wan-AI--Wan2.2-I2V-A14B-Diffusers")
    if snapshot is None or not (snapshot / "model_index.json").exists():
        pytest.skip("Local Wan diffusers snapshot is missing or incomplete.")

    reference_image = snapshot / "examples" / "i2v_input.JPG"
    if not reference_image.exists():
        pytest.skip("Local Wan diffusers example input image is missing.")

    model = DiffusersWanI2VModel(
        default_model_dir=snapshot,
        device_id=0,
        default_dtype="bfloat16",
    )
    engine = build_engine(model)
    await engine.start()
    try:
        request = VideoGenerationRequest(
            prompt="A calm white cat surfing on a beach at sunset",
            task_type="image_to_video",
            references=[str(reference_image)],
            width=544,
            height=720,
            frame_count=9,
            num_steps=1,
            seed=0,
            guidance_scale=3.5,
            high_noise_guidance_scale=3.5,
            fps=16,
        )
        await engine.add_request("wan-real-e2e", request)
        result = await engine.get_result("wan-real-e2e")
    finally:
        await engine.stop()

    assert isinstance(result, list)
    assert len(result) == 5
    assert all(stage.status == "succeeded" for stage in result)

    frames = result[-1].state_updates.get("video_frames")
    if frames is None:
        frames = result[-1].state_updates.get("_pipeline_output")
    assert frames is not None

    arr = np.asarray(frames)
    assert arr.dtype == np.uint8
    assert arr.ndim == 4
    assert arr.shape[0] == request.frame_count
    assert arr.shape[1] == request.height
    assert arr.shape[2] == request.width
    assert arr.shape[3] == 3
