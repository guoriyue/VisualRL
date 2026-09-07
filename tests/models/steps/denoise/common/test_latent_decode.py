from __future__ import annotations

import torch

from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    LatentDecodePlan,
)


def test_chunked_latent_decoder_decodes_in_batch_chunks() -> None:
    """With ``decode_batch_size=1`` every latent row is its own VAE call, and prepare -> decode ->
    postprocess compose in that order (``x*2``, ``+10``, ``-1``).
    """
    calls: list[torch.Tensor] = []

    def decode(batch: torch.Tensor) -> torch.Tensor:
        calls.append(batch)
        return batch + 10

    decoder = ChunkedLatentDecoder(
        LatentDecodePlan(
            prepare_latents=lambda x: x * 2,
            vae_decode=decode,
            postprocess=lambda x: x - 1,
            output_layout="image_bchw",
            decode_batch_size=1,
        )
    )

    latents = torch.arange(3.0).view(3, 1, 1, 1)
    out = decoder(latents)

    assert len(calls) == 3
    torch.testing.assert_close(out, latents * 2 + 9)


def test_chunked_latent_decoder_standardizes_video_layout() -> None:
    """A ``video_btchw`` plan's postprocess permute is undone by the decoder, so callers always
    get [B,C,T,H,W] back whatever layout the VAE postprocess emits.
    """
    decoder = ChunkedLatentDecoder(
        LatentDecodePlan(
            prepare_latents=lambda x: x,
            vae_decode=lambda x: x,
            postprocess=lambda x: x.permute(0, 2, 1, 3, 4),
            output_layout="video_btchw",
        )
    )

    latents = torch.zeros(2, 3, 4, 1, 1)
    out = decoder(latents)

    assert out.shape == (2, 3, 4, 1, 1)
