"""Tiny real (cache-free) diffusion model fixtures for CPU tests.

Each ``build_tiny_*`` returns a genuine diffusers transformer constructed straight
from config — no ``from_pretrained`` / no download / no cached weights — so the
source fully defines it, forward outputs are reproducible, and tests run real
inference on CPU. ``add_lora_adapters`` attaches real diffusers-native LoRA.
"""

from __future__ import annotations

from typing import Any

import torch

from vrl.config.precision import RolePrecision

# Tiny real Wan DiT geometry (CPU, ~6.7K params): latent video [B, C, T, H, W]
# with patch size (1, 2, 2); text embeds are [B, TEXT_LEN, TEXT_DIM].
TINY_WAN_LATENT_SHAPE = (1, 4, 1, 4, 4)
TINY_WAN_TEXT_LEN = 3
TINY_WAN_TEXT_DIM = 16
_TINY_WAN_LORA_TARGETS = ["to_q", "to_v"]


def build_tiny_wan_transformer(*, seed: int = 0) -> Any:
    """A real ~6.7K-param ``WanTransformer3DModel`` on CPU, random-init from a seed.

    Built straight from config — no ``from_pretrained`` / no download / no cached
    weights — so the source fully defines it and forward outputs are reproducible.
    Use this instead of a hand-written fake when a test needs the genuine
    transformer's adapter API or real gradient flow (e.g. DiffusionNFT branches).
    """

    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=TINY_WAN_LATENT_SHAPE[1],
        out_channels=TINY_WAN_LATENT_SHAPE[1],
        text_dim=TINY_WAN_TEXT_DIM,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=64,
    )


TINY_COSMOS_LATENT_SHAPE = (2, 4, 1, 4, 4)
TINY_COSMOS_TEXT_DIM = 16


def build_tiny_cosmos_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``CosmosTransformer3DModel`` on CPU, cache-free (config-init).

    Cosmos predict2/2.5 concatenate a 1-channel condition mask into the latent
    channel axis, so ``in_channels`` is the latent channels (4) + 1; ``out_channels``
    stays at the latent channels. ``attention_head_dim`` is 16 (8 divides the 3D
    rope unevenly and trips a div-by-zero at construction).
    """

    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(seed)
    return CosmosTransformer3DModel(
        in_channels=TINY_COSMOS_LATENT_SHAPE[1] + 1,
        out_channels=TINY_COSMOS_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        mlp_ratio=2.0,
        text_embed_dim=TINY_COSMOS_TEXT_DIM,
        adaln_lora_dim=8,
        max_size=(4, 16, 16),
        patch_size=(1, 2, 2),
        concat_padding_mask=True,
    )


def build_tiny_anima_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``CosmosTransformer3DModel`` with Anima's channel geometry.

    Anima reuses the Cosmos Predict2 Text2Image backbone but, unlike predict2/2.5,
    feeds the latents to the transformer DIRECTLY (no runner-side condition-mask
    channel expansion — see ``AnimaModel.forward_step``), so ``in_channels`` equals
    ``out_channels`` (the bare latent channels) rather than latent + 1. It still
    sets ``concat_padding_mask=True`` (the real Anima config does), so the model
    appends a resized padding-mask channel internally; the wrapper therefore must
    pass a ``[1, 1, H, W]`` ``padding_mask``. This asymmetry — in==out at the
    wrapper boundary, with the mask channel added inside the model — is exactly
    what a hand-written ``torch.ones_like(hidden_states)`` fake cannot exercise.
    """

    from diffusers import CosmosTransformer3DModel

    torch.manual_seed(seed)
    return CosmosTransformer3DModel(
        in_channels=TINY_COSMOS_LATENT_SHAPE[1],
        out_channels=TINY_COSMOS_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        mlp_ratio=2.0,
        text_embed_dim=TINY_COSMOS_TEXT_DIM,
        adaln_lora_dim=8,
        max_size=(4, 16, 16),
        patch_size=(1, 2, 2),
        concat_padding_mask=True,
    )


TINY_SD3_LATENT_SHAPE = (2, 4, 8, 8)
TINY_SD3_JOINT_DIM = 16
TINY_SD3_POOLED_DIM = 16


def build_tiny_sd3_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``SD3Transformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import SD3Transformer2DModel

    torch.manual_seed(seed)
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=TINY_SD3_LATENT_SHAPE[1],
        out_channels=TINY_SD3_LATENT_SHAPE[1],
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=TINY_SD3_JOINT_DIM,
        caption_projection_dim=16,
        pooled_projection_dim=TINY_SD3_POOLED_DIM,
        pos_embed_max_size=8,
    )


# Tiny real FLUX geometry (CPU): PACKED latents [B, seq, C*4] with C=4 -> 16
# in_channels (patch_size=1 in packed token space). axes_dims_rope must sum to
# attention_head_dim (8). guidance_embeds=True mirrors FLUX.1-dev.
TINY_FLUX_IN_CHANNELS = 16
TINY_FLUX_JOINT_DIM = 16
TINY_FLUX_POOLED_DIM = 16


def build_tiny_flux_transformer(*, seed: int = 0, guidance_embeds: bool = True) -> Any:
    """Tiny real ``FluxTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import FluxTransformer2DModel

    torch.manual_seed(seed)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=TINY_FLUX_IN_CHANNELS,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=TINY_FLUX_JOINT_DIM,
        pooled_projection_dim=TINY_FLUX_POOLED_DIM,
        guidance_embeds=guidance_embeds,
        axes_dims_rope=(2, 2, 4),
    )


# Tiny real Qwen-Image geometry (CPU): PACKED latents [B, seq, C*4] with C=4 ->
# 16 in_channels. out_channels(4) * patch_size**2(4) == in_channels(16) so the
# noise_pred matches the packed latent for the SDE step. axes_dims_rope sums to
# attention_head_dim (16).
TINY_SANA_LATENT_SHAPE = (2, 4, 8, 8)
TINY_SANA_CAPTION_DIM = 16


def build_tiny_sana_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``SanaTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import SanaTransformer2DModel

    torch.manual_seed(seed)
    return SanaTransformer2DModel(
        in_channels=TINY_SANA_LATENT_SHAPE[1],
        out_channels=TINY_SANA_LATENT_SHAPE[1],
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        num_cross_attention_heads=2,
        cross_attention_head_dim=8,
        cross_attention_dim=16,
        caption_channels=TINY_SANA_CAPTION_DIM,
        sample_size=TINY_SANA_LATENT_SHAPE[2],
        patch_size=1,
    )


TINY_LUMINA2_LATENT_SHAPE = (2, 4, 8, 8)
TINY_LUMINA2_CAP_DIM = 16


def build_tiny_lumina2_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``Lumina2Transformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import Lumina2Transformer2DModel

    torch.manual_seed(seed)
    return Lumina2Transformer2DModel(
        sample_size=TINY_LUMINA2_LATENT_SHAPE[2],
        patch_size=2,
        in_channels=TINY_LUMINA2_LATENT_SHAPE[1],
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        num_attention_heads=2,
        num_kv_heads=2,
        multiple_of=16,
        axes_dim_rope=(4, 2, 2),
        cap_feat_dim=TINY_LUMINA2_CAP_DIM,
    )


TINY_HUNYUAN_VIDEO_LATENT_SHAPE = (2, 4, 3, 8, 8)
TINY_HUNYUAN_VIDEO_TEXT_DIM = 16
TINY_HUNYUAN_VIDEO_POOLED_DIM = 8


def build_tiny_hunyuan_video_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``HunyuanVideoTransformer3DModel`` on CPU, cache-free."""

    from diffusers import HunyuanVideoTransformer3DModel

    torch.manual_seed(seed)
    return HunyuanVideoTransformer3DModel(
        in_channels=TINY_HUNYUAN_VIDEO_LATENT_SHAPE[1],
        out_channels=TINY_HUNYUAN_VIDEO_LATENT_SHAPE[1],
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        patch_size=2,
        patch_size_t=1,
        guidance_embeds=True,
        text_embed_dim=TINY_HUNYUAN_VIDEO_TEXT_DIM,
        pooled_projection_dim=TINY_HUNYUAN_VIDEO_POOLED_DIM,
        rope_axes_dim=(2, 4, 2),
    )


TINY_MOCHI_LATENT_SHAPE = (2, 4, 3, 8, 8)
TINY_MOCHI_TEXT_DIM = 16


def build_tiny_mochi_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``MochiTransformer3DModel`` on CPU, cache-free (config-init)."""

    from diffusers import MochiTransformer3DModel

    torch.manual_seed(seed)
    return MochiTransformer3DModel(
        patch_size=2,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        pooled_projection_dim=16,
        in_channels=TINY_MOCHI_LATENT_SHAPE[1],
        text_embed_dim=TINY_MOCHI_TEXT_DIM,
        time_embed_dim=8,
        max_sequence_length=16,
    )


TINY_COGVIDEOX_LATENT_SHAPE = (2, 3, 4, 8, 8)  # [B, F, C, H, W]
TINY_COGVIDEOX_TEXT_DIM = 16
TINY_COGVIDEOX_TEXT_LEN = 8


def build_tiny_cogvideox_transformer(*, seed: int = 0, rope: bool = False) -> Any:
    """Tiny real ``CogVideoXTransformer3DModel`` on CPU, cache-free.

    ``rope=True`` mirrors the 5b config (external rotary embeddings passed
    into forward); False mirrors 2b (learned positional embeddings).
    """

    from diffusers import CogVideoXTransformer3DModel

    torch.manual_seed(seed)
    # 3D RoPE splits head_dim across (t, h, w); 16 keeps every split even.
    return CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16 if rope else 8,
        in_channels=TINY_COGVIDEOX_LATENT_SHAPE[2],
        out_channels=TINY_COGVIDEOX_LATENT_SHAPE[2],
        time_embed_dim=8,
        text_embed_dim=TINY_COGVIDEOX_TEXT_DIM,
        num_layers=1,
        sample_width=16,
        sample_height=16,
        sample_frames=9,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=TINY_COGVIDEOX_TEXT_LEN,
        use_rotary_positional_embeddings=rope,
    )


TINY_QWEN_IN_CHANNELS = 16
TINY_QWEN_JOINT_DIM = 16


def build_tiny_qwen_image_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``QwenImageTransformer2DModel`` on CPU, cache-free (config-init)."""

    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(seed)
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=TINY_QWEN_IN_CHANNELS,
        out_channels=TINY_QWEN_IN_CHANNELS // 4,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=TINY_QWEN_JOINT_DIM,
        guidance_embeds=False,
        axes_dims_rope=(8, 4, 4),
    )


def build_tiny_wan_i2v_transformer(*, seed: int = 0) -> Any:
    """Tiny real Wan I2V ``WanTransformer3DModel`` on CPU, cache-free.

    I2V cats the conditioning latent into the channel axis, so ``in_channels`` is
    doubled (4 latent + 4 condition); ``image_dim`` enables the CLIP image-embed
    cross-attention branch. Same config-init/no-download contract as
    :func:`build_tiny_wan_transformer`.
    """

    from diffusers import WanTransformer3DModel

    torch.manual_seed(seed)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=2 * TINY_WAN_LATENT_SHAPE[1],
        out_channels=TINY_WAN_LATENT_SHAPE[1],
        text_dim=TINY_WAN_TEXT_DIM,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=64,
        image_dim=TINY_WAN_TEXT_DIM,
    )


def add_lora_adapters(
    transformer: Any,
    *,
    names: tuple[str, ...] = ("default", "previous"),
    rank: int = 4,
    seed: int = 0,
) -> Any:
    """Attach independently gaussian-init LoRA adapters via diffusers' native API.

    Uses ``PeftAdapterMixin.add_adapter`` (not ``get_peft_model``) so the model
    exposes the ``disable_adapters`` / ``enable_adapters`` / ``set_adapter`` surface
    DiffusionNFT drives. One shared RNG stream seeds all adapters, so they end up
    with distinct weights; the first name is left active.
    """

    from peft import LoraConfig

    torch.manual_seed(seed)
    transformer.requires_grad_(False)
    for name in names:
        transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                init_lora_weights="gaussian",
                target_modules=_TINY_WAN_LORA_TARGETS,
            ),
            adapter_name=name,
        )
    if names:
        transformer.set_adapter(names[0])
    return transformer


def record_forward_calls(module: torch.nn.Module) -> list[dict[str, Any]]:
    """Capture the kwargs of every forward call on ``module``.

    Registers a forward pre-hook and returns the list it appends to, so a test can
    assert exactly how a wrapper invoked a real transformer (call count + kwargs)
    against the genuine signature — instead of a hand-written fake that re-declares
    that signature and silently rots when the real model changes it.
    """

    calls: list[dict[str, Any]] = []
    module.register_forward_pre_hook(
        lambda _m, _args, kwargs: calls.append(dict(kwargs)),
        with_kwargs=True,
    )
    return calls


class RecordingModule:
    """Records the freeze / placement calls the shared loaders make on a component.

    Deliberately a plain object, not an ``nn.Module``: the loader tests assert on
    ``"cuda:0"`` / ``torch.device("cuda:1")`` placements, and a real ``Module.to``
    would fail on the CUDA-less default lane. ``to_calls`` appends whatever it is
    handed -- strings, ``torch.device`` objects, ``None`` -- with no normalization,
    so every existing assertion reads the exact argument the loader passed.
    """

    def __init__(self) -> None:
        self.dtype: torch.dtype | None = None
        self.requires_grad_enabled: bool | None = None
        self.to_calls: list[tuple[Any, torch.dtype | None]] = []

    def requires_grad_(self, enabled: bool) -> None:
        self.requires_grad_enabled = enabled

    def to(self, device: Any = None, dtype: torch.dtype | None = None) -> RecordingModule:
        self.to_calls.append((device, dtype))
        if dtype is not None:
            self.dtype = dtype
        return self


def stamp_model_precision(model: Any) -> None:
    """Mirror RuntimeBundle precision assembly for direct-model forward tests."""

    model.precision = RolePrecision(
        dtype="fp32",
        float32_precision="ieee",
        outer_autocast=False,
    )


def build_tiny_autoencoder_kl(
    *,
    seed: int = 0,
    downsamples: int = 1,
    latent_channels: int = 4,
) -> Any:
    """A real tiny ``AutoencoderKL`` on CPU, random-init from a seed.

    ``downsamples`` is the number of 2x spatial steps, so the decode geometry a
    test asserts on is COMPUTED by diffusers (latent HxW * 2**downsamples) rather
    than declared by the fixture — a hand-written decoder can agree with a wrong
    expectation forever. ``downsamples=3`` is the NextStep tokenizer's f8
    geometry; the f2 default is enough for flag and state tests.
    """

    from diffusers import AutoencoderKL

    torch.manual_seed(seed)
    blocks = downsamples + 1
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * blocks,
        up_block_types=("UpDecoderBlock2D",) * blocks,
        block_out_channels=(4,) * blocks,
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=2,
        sample_size=32,
    )


def build_tiny_pipeline_shell(
    *,
    transformer: Any,
    vae: Any,
    scheduler: Any,
    text_encoder: Any = None,
) -> Any:
    """A real ``DiffusionPipeline`` whose ``.components`` diffusers itself derives.

    Every slot is a REQUIRED ``__init__`` parameter, so diffusers'
    ``_get_signature_keys`` keeps all of them in ``.components`` -- including
    the ``None`` slot (``text_encoder`` by default) and the non-module
    scheduler. That is the shape ``move_frozen_components`` has to survive,
    and a hand-written dict cannot promise it.
    """

    from diffusers import DiffusionPipeline

    class _TinyPipelineShell(DiffusionPipeline):
        def __init__(self, transformer: Any, vae: Any, scheduler: Any, text_encoder: Any) -> None:
            super().__init__()
            self.register_modules(
                transformer=transformer,
                vae=vae,
                scheduler=scheduler,
                text_encoder=text_encoder,
            )

    return _TinyPipelineShell(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
    )


def build_tiny_wan_vae(
    *,
    seed: int = 0,
    z_dim: int = 4,
    latents_mean: float = 0.0,
    latents_std: float = 1.0,
) -> Any:
    """A real tiny ``AutoencoderKLWan`` on CPU, random-init from a seed.

    Wan's VAE carries ``z_dim`` / ``latents_mean`` / ``latents_std`` as genuine
    diffusers config fields, which is the point: production reads them off
    ``vae.config``, so a double that re-declares them cannot catch a rename.
    Non-trivial ``latents_mean`` / ``latents_std`` make a denormalizing decode
    observable (the identity stats hide a dropped mean or a swapped std).
    """

    from diffusers import AutoencoderKLWan

    torch.manual_seed(seed)
    return AutoencoderKLWan(
        base_dim=4,
        z_dim=z_dim,
        dim_mult=[1, 1],
        num_res_blocks=1,
        latents_mean=[latents_mean] * z_dim,
        latents_std=[latents_std] * z_dim,
    )


# Tiny real Cosmos3 Omni geometry: the transformer's latent channel count is
# the VAE's ``z_dim``; ``patch_latent_dim = latent_channel * latent_patch_size**2``
# and ``sum(mrope_section) == head_dim // 2`` are construction constraints, so
# the numbers below are honest, not coincidental.
TINY_COSMOS3_LATENT_CHANNELS = 4
TINY_COSMOS3_HEAD_DIM = 16
TINY_COSMOS3_LATENT_PATCH_SIZE = 2


def build_tiny_cosmos3_tokenizer() -> Any:
    """A real ``PreTrainedTokenizerFast`` carrying what ``Cosmos3OmniPipeline`` reads.

    The pipeline's ``__init__`` calls ``convert_tokens_to_ids("<|vision_start|>")``
    and reads ``eos_token_id``; ``tokenize_prompt`` renders a Qwen-style chat
    template. A word-level vocabulary with those special tokens and a minimal
    template satisfies all three through the library's own code paths.
    """

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>"]
    words = ["system", "user", "assistant", "a", "cat", "video", "the", "is", "of", "long"]
    vocab = {token: index for index, token in enumerate([*specials, *words])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<unk>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<|vision_start|>"],
        chat_template=template,
    )


def build_tiny_cosmos3_transformer(*, vocab_size: int, seed: int = 0) -> Any:
    """Tiny real ``Cosmos3OmniTransformer`` (~30K params) on CPU, cache-free."""

    from diffusers import Cosmos3OmniTransformer

    torch.manual_seed(seed)
    return Cosmos3OmniTransformer(
        head_dim=TINY_COSMOS3_HEAD_DIM,
        hidden_size=32,
        intermediate_size=64,
        latent_channel=TINY_COSMOS3_LATENT_CHANNELS,
        latent_patch_size=TINY_COSMOS3_LATENT_PATCH_SIZE,
        patch_latent_dim=TINY_COSMOS3_LATENT_CHANNELS * TINY_COSMOS3_LATENT_PATCH_SIZE**2,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_key_value_heads=1,
        vocab_size=vocab_size,
        rope_scaling={"mrope_section": [TINY_COSMOS3_HEAD_DIM // 4, 2, 2]},
        dtype="float32",
    )


def build_tiny_cosmos3_pipeline(
    *,
    seed: int = 0,
    latents_mean: float = 0.0,
    latents_std: float = 1.0,
) -> Any:
    """Tiny real ``Cosmos3OmniPipeline``: real transformer + VAE + scheduler + tokenizer.

    ``enable_safety_checker=False`` mirrors ``Cosmos3Model.from_build`` and keeps
    the cosmos_guardrail dependency out; the pipeline still registers the two
    ``None`` slots (``sound_tokenizer`` / ``safety_checker``) in ``.components``.
    """

    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler

    tokenizer = build_tiny_cosmos3_tokenizer()
    return Cosmos3OmniPipeline(
        transformer=build_tiny_cosmos3_transformer(vocab_size=len(tokenizer), seed=seed),
        text_tokenizer=tokenizer,
        vae=build_tiny_wan_vae(
            seed=seed,
            z_dim=TINY_COSMOS3_LATENT_CHANNELS,
            latents_mean=latents_mean,
            latents_std=latents_std,
        ),
        scheduler=UniPCMultistepScheduler(),
        enable_safety_checker=False,
    )


# ---- MiniMax-H3 (diffusers 0.40) ------------------------------------------------
# Tiny real geometry: video latents [1, 4, T_lat, 4, 4] from a 16x16 canvas
# (spatial ratio 4, patch (1, 2, 2) -> canvas multiple 8), a 5-frame VAE clip
# keeping 3 latents with 1 dropped (the released 17/5/3 shape scaled down), so
# 8 pixel frames -> 5 latent frames. Audio: 6 latent channels, 2 channels packed
# channel-major. The Qwen3-VL conditioner is a real 2-layer model read at
# ``hidden_states[1]``.
TINY_MINIMAX_H3_LATENT_CHANNELS = 4
TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS = 6
TINY_MINIMAX_H3_TEXT_DIM = 16
TINY_MINIMAX_H3_PATCH_SIZE = (1, 2, 2)
TINY_MINIMAX_H3_TEXT_ENCODER_LAYER = 1
TINY_MINIMAX_H3_VAE_CLIP_LENGTH = 5
TINY_MINIMAX_H3_VAE_TOKEN_DROP = 1


def build_tiny_minimax_h3_tokenizer() -> Any:
    """A real ``Qwen2TokenizerFast`` over a word-level vocabulary carrying the
    four vision-pad specials ``Qwen3VLProcessor`` derives its token types from."""

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast

    specials = [
        "<unk>",
        "<|endoftext|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    words = ["a", "cat", "video", "the", "is", "of", "long", "dog", "runs", "on", "grass"]
    vocab = {token: index for index, token in enumerate([*specials, *words])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    return Qwen2TokenizerFast(
        tokenizer_object=core,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        unk_token="<unk>",
        additional_special_tokens=specials[2:],
    )


def build_tiny_minimax_h3_text_encoder(tokenizer: Any, *, seed: int = 0) -> Any:
    """Tiny real ``Qwen3VLForConditionalGeneration`` (2 decoder layers, ~30K params)."""

    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    token_id = tokenizer.convert_tokens_to_ids
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": TINY_MINIMAX_H3_TEXT_DIM,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": len(tokenizer),
            "head_dim": 8,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "mrope_section": [1, 1, 2],
                "rope_type": "default",
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": TINY_MINIMAX_H3_TEXT_DIM,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
        },
        image_token_id=token_id("<|image_pad|>"),
        video_token_id=token_id("<|video_pad|>"),
        vision_start_token_id=token_id("<|vision_start|>"),
        vision_end_token_id=token_id("<|vision_end|>"),
    )
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(config)


def build_tiny_minimax_h3_processor(tokenizer: Any) -> Any:
    """A real ``Qwen3VLProcessor`` around the tiny tokenizer (image/video processors at defaults)."""

    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    return Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )


def build_tiny_minimax_h3_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``MiniMaxH3Transformer3DModel`` (~9K params): one block, one refiner."""

    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(seed)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=TINY_MINIMAX_H3_LATENT_CHANNELS,
        audio_in_channels=TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        patch_size=TINY_MINIMAX_H3_PATCH_SIZE,
        text_dim=TINY_MINIMAX_H3_TEXT_DIM,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        # 2 * 3 * rope_freq_dim channels of the head are rotated; 6 of 8 here.
        rope_freq_dim=1,
    )


def build_tiny_minimax_h3_video_vae(
    *,
    seed: int = 0,
    latents_mean: float = 0.0,
    latents_std: float = 1.0,
) -> Any:
    """Tiny real ``AutoencoderKLMiniMaxH3`` (~23K params) with the released chunking shape scaled down."""

    from diffusers import AutoencoderKLMiniMaxH3

    torch.manual_seed(seed)
    channels = TINY_MINIMAX_H3_LATENT_CHANNELS
    return AutoencoderKLMiniMaxH3(
        latent_channels=channels,
        block_out_channels=(8, 8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 1),
        temporal_downsample_factors=(1, 2, 1),
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        clip_length=TINY_MINIMAX_H3_VAE_CLIP_LENGTH,
        token_drop=TINY_MINIMAX_H3_VAE_TOKEN_DROP,
        latents_mean=(latents_mean,) * channels,
        latents_std=(latents_std,) * channels,
    )


def build_tiny_minimax_h3_audio_vae(*, seed: int = 0) -> Any:
    """Tiny real ``AutoencoderKLMiniMaxH3Audio`` (hop 4 at a 100 Hz sample rate)."""

    from diffusers import AutoencoderKLMiniMaxH3Audio

    torch.manual_seed(seed)
    return AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4,
        encoder_rates=(2, 2),
        latent_dim=12,
        latent_channels=TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        num_attention_heads=2,
        decoder_dim=8,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=100,
        latents_mean=[0.0] * TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        latents_std=[1.0] * TINY_MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    )


def build_tiny_minimax_h3_components(
    *,
    seed: int = 0,
    latents_mean: float = 0.0,
    latents_std: float = 1.0,
) -> Any:
    """The full tiny real component set the family reads, on the family's own shell.

    Video scheduler is the flow-convention subclass the rollout model installs
    (``shift=12``); the audio scheduler is the plain ``MiniMaxH3Scheduler``
    (``shift=3``), exactly the released pairing.
    """

    from diffusers import MiniMaxH3Scheduler

    from vrl.models.families.minimax_h3.model import (
        MiniMaxH3Components,
        build_flow_scheduler_class,
    )

    tokenizer = build_tiny_minimax_h3_tokenizer()
    return MiniMaxH3Components(
        transformer=build_tiny_minimax_h3_transformer(seed=seed),
        vae=build_tiny_minimax_h3_video_vae(
            seed=seed, latents_mean=latents_mean, latents_std=latents_std
        ),
        audio_vae=build_tiny_minimax_h3_audio_vae(seed=seed),
        text_encoder=build_tiny_minimax_h3_text_encoder(tokenizer, seed=seed),
        tokenizer=tokenizer,
        processor=build_tiny_minimax_h3_processor(tokenizer),
        scheduler=build_flow_scheduler_class()(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        text_encoder_layer=TINY_MINIMAX_H3_TEXT_ENCODER_LAYER,
    )
