"""Tests for vrl.models.families.wan.diffusers_t2v (DiffusersWanT2VModel)."""

from __future__ import annotations

import asyncio

import pytest


class TestDiffusersWanT2VModelDenoiseInit:
    """denoise_init returns a DenoiseLoopState with DiffusersDenoiseState."""

    def test_denoise_init_returns_denoise_loop_state(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.base import VideoGenerationRequest
        from vrl.models.families.diffusers_state import DiffusersDenoiseState
        from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel

        B, C, D, H, W = 2, 16, 5, 8, 12
        num_steps = 3

        # Mock pipeline
        class MockTransformerConfig:
            in_channels = C

        class MockTransformer:
            dtype = torch.bfloat16
            config = MockTransformerConfig()

        class MockScheduler:
            timesteps = None

            def set_timesteps(self, n, device=None):
                self.timesteps = torch.linspace(1.0, 0.0, n, device=device)

        class MockPipeline:
            transformer = MockTransformer()
            scheduler = MockScheduler()
            device = torch.device("cpu")

            def encode_prompt(self, **kwargs):
                bs = 1  # single prompt
                return (
                    torch.randn(bs, 10, 64, dtype=torch.bfloat16),
                    torch.randn(bs, 10, 64, dtype=torch.bfloat16),
                )

            def prepare_latents(self, batch_size, channels, h, w, frames, dtype, device, gen, lat):
                return torch.randn(batch_size, channels, frames, h // 8, w // 8, dtype=dtype, device=device)

        pipe = MockPipeline()
        model = DiffusersWanT2VModel(pipeline=pipe, device=torch.device("cpu"))

        request = VideoGenerationRequest(
            prompt="test prompt",
            num_steps=num_steps,
            guidance_scale=4.5,
            height=H * 8,
            width=W * 8,
            frame_count=D,
        )

        # encode_text first
        state: dict = {}
        result = asyncio.run(model.encode_text(request, state))
        state.update(result.state_updates)

        # denoise_init
        loop = asyncio.run(model.denoise_init(request, state))

        assert isinstance(loop, DenoiseLoopState)
        assert loop.total_steps == num_steps
        assert loop.current_step == 0
        assert isinstance(loop.model_state, DiffusersDenoiseState)
        assert loop.model_state.model_family == "wan-diffusers-t2v"
        assert loop.model_state.latents is not None
        assert loop.model_state.do_cfg is True


class TestDiffusersWanT2VModelPredictNoise:
    """predict_noise returns dict with noise_pred key."""

    def test_predict_noise_returns_dict(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState
        from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel

        B, C, D, H, W = 2, 4, 3, 8, 8
        prompt_embeds = torch.randn(B, 10, 64)
        neg_embeds = torch.randn(B, 10, 64)

        # Mock transformer that returns input
        class MockTransformer:
            def __call__(self, **kwargs):
                return (kwargs["hidden_states"],)

        class MockPipeline:
            transformer = MockTransformer()

        ms = DiffusersDenoiseState(
            latents=torch.randn(B, C, D, H, W),
            timesteps=torch.tensor([1.0, 0.5, 0.0]),
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            guidance_scale=4.5,
            do_cfg=False,  # no CFG for simpler test
            model_family="wan-diffusers-t2v",
        )
        ds = DenoiseLoopState(current_step=0, total_steps=3, model_state=ms)

        model = DiffusersWanT2VModel(pipeline=MockPipeline())

        result = asyncio.run(model.predict_noise(ds, step_idx=0))

        assert "noise_pred" in result
        assert "noise_pred_cond" in result
        assert "noise_pred_uncond" in result
        assert result["noise_pred"].shape == (B, C, D, H, W)

    def test_predict_noise_with_cfg(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState
        from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel

        B, C, D, H, W = 1, 4, 3, 8, 8

        # Mock transformer: return constant so we can verify CFG math
        call_count = [0]

        class MockTransformer:
            def __call__(self, **kwargs):
                call_count[0] += 1
                return (torch.ones_like(kwargs["hidden_states"]) * call_count[0],)

        class MockPipeline:
            transformer = MockTransformer()

        ms = DiffusersDenoiseState(
            latents=torch.randn(B, C, D, H, W),
            timesteps=torch.tensor([1.0, 0.5, 0.0]),
            prompt_embeds=torch.randn(B, 10, 64),
            negative_prompt_embeds=torch.randn(B, 10, 64),
            guidance_scale=2.0,
            do_cfg=True,
            model_family="wan-diffusers-t2v",
        )
        ds = DenoiseLoopState(current_step=0, total_steps=3, model_state=ms)

        model = DiffusersWanT2VModel(pipeline=MockPipeline())
        result = asyncio.run(model.predict_noise(ds, step_idx=0))

        # With CFG: two forward calls
        assert call_count[0] == 2
        assert "noise_pred" in result


class TestDiffusersWanT2VModelPredictNoiseWithModel:
    """_predict_noise_with_model works with external model."""

    def test_uses_external_model(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.diffusers_state import DiffusersDenoiseState
        from vrl.models.families.wan.diffusers_t2v import DiffusersWanT2VModel

        B, C, D, H, W = 1, 4, 3, 8, 8

        external_called = [False]

        class ExternalModel:
            def __call__(self, **kwargs):
                external_called[0] = True
                return (kwargs["hidden_states"],)

        class MockPipeline:
            transformer = None  # not used

        ms = DiffusersDenoiseState(
            latents=torch.randn(B, C, D, H, W),
            timesteps=torch.tensor([1.0]),
            prompt_embeds=torch.randn(B, 10, 64),
            negative_prompt_embeds=torch.randn(B, 10, 64),
            guidance_scale=1.0,
            do_cfg=False,
            model_family="wan-diffusers-t2v",
        )
        ds = DenoiseLoopState(current_step=0, total_steps=1, model_state=ms)

        model = DiffusersWanT2VModel(pipeline=MockPipeline())
        result = model._predict_noise_with_model(ExternalModel(), ds, step_idx=0)

        assert external_called[0]
        assert result["noise_pred"].shape == (B, C, D, H, W)
