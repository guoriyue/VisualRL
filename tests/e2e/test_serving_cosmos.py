"""Real Cosmos checkpoint end-to-end test through the engine stack."""

from __future__ import annotations

import pytest

from tests.e2e.helpers import (
    build_engine,
    require_cuda,
    require_real_model_opt_in,
    resolve_hf_snapshot,
)
from vrl.models.families.cosmos.model import CosmosGenerationModel
from vrl.models.base import VideoGenerationRequest


def _find_reference_image():
    """Locate a reference image from any locally-cached model snapshot."""
    wan_snapshot = resolve_hf_snapshot("models--Wan-AI--Wan2.2-I2V-A14B-Diffusers")
    if wan_snapshot is not None:
        image_path = wan_snapshot / "examples" / "i2v_input.JPG"
        if image_path.exists():
            return image_path
    return None


@pytest.mark.asyncio
async def test_cosmos_real_engine_e2e():
    """Run a real Cosmos checkpoint through engine -> pipeline -> result."""
    require_real_model_opt_in()
    require_cuda()

    # Try Predict2 first (diffusers-compatible download)
    predict2_snapshot = resolve_hf_snapshot(
        "models--nvidia--Cosmos-Predict2-2B-Video2World"
    )
    predict1_snapshot = resolve_hf_snapshot(
        "models--nvidia--Cosmos-1.0-Diffusion-7B-Video2World"
    )

    if predict2_snapshot is not None:
        variant = "predict2_video2world"
        model_size = "2B"
        snapshot_path = predict2_snapshot
    elif predict1_snapshot is not None:
        variant = "predict1_video2world"
        model_size = "7B"
        snapshot_path = predict1_snapshot
    else:
        pytest.skip(
            "No local Cosmos snapshot found. Need either "
            "nvidia/Cosmos-Predict2-2B-Video2World or "
            "nvidia/Cosmos-1.0-Diffusion-7B-Video2World in HF cache."
        )

    image_path = _find_reference_image()
    if image_path is None:
        pytest.skip("A local reference image is required for Cosmos Video2World E2E.")

    model = CosmosGenerationModel(
        variant=variant,
        model_size=model_size,
        model_id_or_path=str(snapshot_path),
        device_id=0,
        dtype="bfloat16",
        enable_cpu_offload=True,
    )
    engine = build_engine(model)
    await engine.start()
    try:
        request = VideoGenerationRequest(
            prompt="A cinematic ocean scene with gentle camera motion",
            task_type="video_to_world",
            references=[str(image_path)],
            width=704,
            height=480,
            num_steps=1,
            guidance_scale=5.0,
            seed=0,
        )
        await engine.add_request("cosmos-real-e2e", request)
        result = await engine.get_result("cosmos-real-e2e")
    finally:
        await engine.stop()

    assert isinstance(result, list)
    assert len(result) == 5
    assert all(stage.status == "succeeded" for stage in result)
