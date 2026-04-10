"""Real Cosmos checkpoint end-to-end test through the engine stack."""

from __future__ import annotations

import pytest

from tests.real_model_helpers import build_engine, require_cuda, require_real_model_opt_in, resolve_hf_snapshot
from wm_infra.models.families.cosmos.model import CosmosGenerationModel
from wm_infra.schemas.video_generation import VideoGenerationRequest


@pytest.mark.asyncio
async def test_cosmos_real_engine_e2e():
    """Run a real Cosmos checkpoint through engine -> pipeline -> result."""
    require_real_model_opt_in()
    require_cuda()

    predict1_snapshot = resolve_hf_snapshot(
        "models--nvidia--Cosmos-1.0-Diffusion-7B-Video2World"
    )
    if predict1_snapshot is None:
        predict25_ref = resolve_hf_snapshot("models--nvidia--Cosmos-Predict2.5-2B")
        if predict25_ref is not None:
            pytest.skip(
                "Local cache contains Cosmos Predict2.5, but current wm_infra code "
                "only wires Predict1 executors."
            )
        pytest.skip("Local Cosmos Predict1 snapshot is missing.")

    wan_snapshot = resolve_hf_snapshot("models--Wan-AI--Wan2.2-I2V-A14B-Diffusers")
    if wan_snapshot is None:
        pytest.skip("A local reference image is required for Cosmos Video2World E2E.")
    image_path = wan_snapshot / "examples" / "i2v_input.JPG"
    if not image_path.exists():
        pytest.skip("Local reference image is missing.")

    model = CosmosGenerationModel(
        variant="predict1_video2world",
        model_size="7B",
        model_id_or_path=str(predict1_snapshot),
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
