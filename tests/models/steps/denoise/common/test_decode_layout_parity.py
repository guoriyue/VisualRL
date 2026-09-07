"""Decode-wrapper denormalization + layout tests.

``decode_latents`` is pure wrapper math: per-model latent denormalization (Wan
per-channel mean/std, SD3 scaling/shift, Cosmos sigma_data), chunked VAE decode,
then layout normalization (permute to ``B,C,T,H,W``, frame matching). It never
calls the transformer.

The VAE here is an IDENTITY decode probe, NOT a fake model: a real VAE spatially
up-samples, so its output shape would no longer equal the latent shape and the
exact denorm/layout assertions this test exists to pin would become unverifiable.
The transparent probe is what makes the wrapper's arithmetic observable. The
transformer is a real (tiny, cache-free) one purely to construct the model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import (
    build_tiny_cosmos_transformer,
    build_tiny_sd3_transformer,
    build_tiny_wan_transformer,
)
from vrl.models.families.cosmos.predict2.model import CosmosPredict2Model
from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model
from vrl.models.families.sd3_5.model import SD3_5Model
from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel


class _IdentityDecodeVAE:
    """A real-signature VAE whose decode is the identity, so the wrapper's
    denormalization and layout transforms are exactly observable."""

    def __init__(self, config: SimpleNamespace) -> None:
        self.config = config
        self.dtype = torch.float32
        self.calls: list[torch.Tensor] = []

    def decode(self, latents: torch.Tensor, *, return_dict: bool) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.calls.append(latents)
        return (latents,)


class _ImageProcessor:
    def postprocess(self, image: torch.Tensor, *, output_type: str) -> torch.Tensor:
        assert output_type == "pt"
        return image


class _VideoProcessor:
    def postprocess_video(self, video: torch.Tensor, *, output_type: str) -> torch.Tensor:
        assert output_type == "pt"
        return video.permute(0, 2, 1, 3, 4)


def test_sd3_decode_latents_uses_shared_chunked_decoder_and_keeps_layout() -> None:
    """SD3 decode runs the shared chunked decoder (one VAE call per ``decode_batch_size`` chunk)
    and applies ``latents / scaling_factor + shift_factor`` before decoding.
    """
    vae = _IdentityDecodeVAE(SimpleNamespace(scaling_factor=2.0, shift_factor=0.5))
    pipeline = SimpleNamespace(
        transformer=build_tiny_sd3_transformer(),
        vae=vae,
        image_processor=_ImageProcessor(),
        device=torch.device("cpu"),
        decode_batch_size=1,
    )
    model = SD3_5Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(8.0).view(2, 1, 2, 2)

    image = model.decode_latents(latents)

    assert len(vae.calls) == 2
    torch.testing.assert_close(image, latents / 2.0 + 0.5)


def test_wan_decode_latents_preserves_bcthw_layout() -> None:
    """Wan decode denormalizes with ``latents * latents_std + latents_mean`` and returns
    [B,C,T,H,W]: the video processor's ``permute(0,2,1,3,4)`` and ``ChunkedLatentDecoder``'s
    permute cancel, so equal shapes here mean a round-trip identity, not the absence of a
    layout change.
    """
    vae = _IdentityDecodeVAE(
        SimpleNamespace(latents_mean=[1.0], latents_std=[2.0], z_dim=1),
    )
    pipeline = SimpleNamespace(
        transformer=build_tiny_wan_transformer(),
        vae=vae,
        video_processor=_VideoProcessor(),
        device=torch.device("cpu"),
    )
    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 2.0 + 1.0)


def test_cosmos_predict2_decode_latents_applies_sigma_data_and_layout() -> None:
    """predict2 decode maps latents back with ``latents * (latents_std / sigma_data) +
    latents_mean`` and keeps the [B,C,T,H,W] layout through the video processor.
    """
    vae = _IdentityDecodeVAE(
        SimpleNamespace(latents_mean=[1.0], latents_std=[4.0], z_dim=1),
    )
    pipeline = SimpleNamespace(
        transformer=build_tiny_cosmos_transformer(),
        scheduler=SimpleNamespace(config=SimpleNamespace(sigma_data=2.0)),
        vae=vae,
        video_processor=_VideoProcessor(),
        device=torch.device("cpu"),
    )
    model = CosmosPredict2Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 2.0 + 1.0)


def test_cosmos_predict25_decode_latents_matches_frames_and_layout() -> None:
    """predict2.5 decode denormalizes with ``latents * latents_std + latents_mean``, asks the
    pipeline to match ``(latent_t - 1) * vae_scale_factor_temporal + 1`` frames, and keeps the
    [B,C,T,H,W] layout.
    """
    vae = _IdentityDecodeVAE(SimpleNamespace())
    matched: list[int] = []

    def match_num_frames(video: torch.Tensor, frames: int) -> torch.Tensor:
        matched.append(frames)
        return video

    pipeline = SimpleNamespace(
        transformer=build_tiny_cosmos_transformer(),
        vae=vae,
        latents_mean=torch.ones(1, 1, 1, 1, 1),
        latents_std=torch.full((1, 1, 1, 1, 1), 3.0),
        vae_scale_factor_temporal=4,
        video_processor=_VideoProcessor(),
        _match_num_frames=match_num_frames,
        device=torch.device("cpu"),
    )
    model = CosmosPredict25Model(pipeline=pipeline, device=torch.device("cpu"))
    latents = torch.arange(16.0).view(2, 1, 2, 2, 2)

    video = model.decode_latents(latents)

    assert matched == [5]
    assert video.shape == latents.shape
    torch.testing.assert_close(video, latents * 3.0 + 1.0)
