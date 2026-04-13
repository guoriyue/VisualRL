"""Tests for Cosmos Predict2 step-level runtime."""

from __future__ import annotations

import asyncio

import pytest


class TestCosmosPredict2DenoiseInit:
    """DiffusersCosmosPredict2Executor.denoise_init returns DenoiseLoopState."""

    def test_denoise_init_returns_loop_state(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.base import VideoGenerationRequest
        from vrl.models.families.cosmos.predict2 import DiffusersCosmosPredict2Executor
        from vrl.models.families.cosmos.variants import CosmosVariant
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        B, C, D, H, W = 1, 16, 4, 22, 40
        num_steps = 3

        # Mock pipeline
        class MockTransformerConfig:
            in_channels = C

        class MockTransformer:
            dtype = torch.bfloat16
            config = MockTransformerConfig()

        class MockSchedulerConfig:
            sigma_data = 1.0
            num_train_timesteps = 1000

        class MockScheduler:
            timesteps = None
            config = MockSchedulerConfig()

            def set_timesteps(self, n, device=None):
                self.timesteps = torch.linspace(1.0, 0.0, n, device=device)

        class MockVAEConfig:
            latents_mean = [0.0] * C
            latents_std = [1.0] * C
            z_dim = C

        class MockVAE:
            dtype = torch.float32
            config = MockVAEConfig()

        class MockVideoProcessor:
            def preprocess_video(self, img, height=None, width=None):
                return torch.zeros(B, 3, 1, height, width)

        class MockPipeline:
            transformer = MockTransformer()
            scheduler = MockScheduler()
            vae = MockVAE()
            video_processor = MockVideoProcessor()

            def prepare_latents(self, **kwargs):
                bs = kwargs["batch_size"]
                ch = kwargs["num_channels_latents"]
                h = kwargs["height"] // 32
                w = kwargs["width"] // 32
                d = 4  # temporal
                latents = torch.randn(bs, ch, d, h, w, dtype=kwargs["dtype"])
                init_latents = torch.randn(bs, ch, d, h, w, dtype=kwargs["dtype"])
                cond_indicator = torch.zeros(1, ch, d, 1, 1)
                uncond_indicator = torch.zeros(1, ch, d, 1, 1)
                cond_mask = torch.zeros(1, 1, d, 1, 1)
                uncond_mask = torch.zeros(1, 1, d, 1, 1)
                return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask

        executor = DiffusersCosmosPredict2Executor(
            variant=CosmosVariant.PREDICT2_VIDEO2WORLD,
            model_size="2B",
        )
        # Inject mock pipeline
        executor._pipeline = MockPipeline()
        executor._modules_loaded = True
        executor._torch = torch

        state = {
            "prompt_embeds": torch.randn(B, 10, 64),
            "negative_prompt_embeds": torch.randn(B, 10, 64),
        }

        request = VideoGenerationRequest(
            prompt="test",
            num_steps=num_steps,
            guidance_scale=7.0,
            height=704,
            width=1280,
            frame_count=93,
            fps=16,
        )

        loop = asyncio.run(executor.denoise_init(request, state))

        assert isinstance(loop, DenoiseLoopState)
        assert loop.total_steps == num_steps
        assert loop.current_step == 0
        ms = loop.model_state
        assert isinstance(ms, DiffusersDenoiseState)
        assert ms.model_family == "cosmos-predict2"
        assert ms.init_latents is not None
        assert ms.cond_indicator is not None
        assert ms.cond_mask is not None
        assert ms.do_cfg is True
        assert ms.fps == 16


class TestCosmosPredict2PredictNoise:
    """predict_noise returns dict with noise_pred key."""

    def test_predict_noise_returns_dict(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.cosmos.predict2 import DiffusersCosmosPredict2Executor
        from vrl.models.families.cosmos.variants import CosmosVariant
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        B, C, D, H, W = 2, 16, 4, 22, 40

        class MockTransformer:
            def __call__(self, **kwargs):
                return (kwargs["hidden_states"],)

        class MockPipeline:
            transformer = MockTransformer()

        executor = DiffusersCosmosPredict2Executor(
            variant=CosmosVariant.PREDICT2_VIDEO2WORLD,
            model_size="2B",
        )
        executor._pipeline = MockPipeline()
        executor._modules_loaded = True

        ms = DiffusersDenoiseState(
            latents=torch.randn(B, C, D, H, W),
            timesteps=torch.tensor([1.0, 0.5, 0.0]),
            prompt_embeds=torch.randn(B, 10, 64),
            negative_prompt_embeds=torch.randn(B, 10, 64),
            guidance_scale=7.0,
            do_cfg=False,
            init_latents=torch.randn(B, C, D, H, W),
            cond_indicator=torch.zeros(1, C, D, 1, 1),
            uncond_indicator=torch.zeros(1, C, D, 1, 1),
            cond_mask=torch.zeros(1, 1, D, 1, 1),
            uncond_mask=torch.zeros(1, 1, D, 1, 1),
            fps=16,
            sigma_conditioning=0.0001,
            model_family="cosmos-predict2",
        )
        ds = DenoiseLoopState(current_step=0, total_steps=3, model_state=ms)

        result = asyncio.run(executor.predict_noise(ds, step_idx=0))

        assert "noise_pred" in result
        assert "noise_pred_cond" in result
        assert "noise_pred_uncond" in result
        assert result["noise_pred"].shape == (B, C, D, H, W)


class TestCosmosPredict2PredictNoiseWithModel:
    """_predict_noise_with_model uses external model."""

    def test_uses_external_model(self) -> None:
        import torch

        from vrl.engine.model_executor.execution_state import DenoiseLoopState
        from vrl.models.families.cosmos.predict2 import DiffusersCosmosPredict2Executor
        from vrl.models.families.cosmos.variants import CosmosVariant
        from vrl.models.families.diffusers_state import DiffusersDenoiseState

        B, C, D, H, W = 1, 16, 4, 22, 40

        external_called = [False]

        class ExternalModel:
            def __call__(self, **kwargs):
                external_called[0] = True
                return (kwargs["hidden_states"],)

        executor = DiffusersCosmosPredict2Executor(
            variant=CosmosVariant.PREDICT2_VIDEO2WORLD,
            model_size="2B",
        )
        executor._modules_loaded = True

        ms = DiffusersDenoiseState(
            latents=torch.randn(B, C, D, H, W),
            timesteps=torch.tensor([1.0]),
            prompt_embeds=torch.randn(B, 10, 64),
            negative_prompt_embeds=torch.randn(B, 10, 64),
            guidance_scale=1.0,
            do_cfg=False,
            init_latents=torch.randn(B, C, D, H, W),
            cond_indicator=torch.zeros(1, C, D, 1, 1),
            uncond_indicator=torch.zeros(1, C, D, 1, 1),
            cond_mask=torch.zeros(1, 1, D, 1, 1),
            uncond_mask=torch.zeros(1, 1, D, 1, 1),
            fps=16,
            model_family="cosmos-predict2",
        )
        ds = DenoiseLoopState(current_step=0, total_steps=1, model_state=ms)

        result = executor._predict_noise_with_model(ExternalModel(), ds, step_idx=0)

        assert external_called[0]
        assert result["noise_pred"].shape == (B, C, D, H, W)
