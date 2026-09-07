"""Tests for KlingVideoReward model loading: repo-root resolution, checkpoint
paths, materialized-artifact validation, repo-owned model build, and Qwen2VL
checkpoint key remapping.

The loader, the checkpoint remap, and ``_create_model_and_processor`` run on a
tiny real Qwen2-VL reward model (``fixtures.py``); the two recorder tests that
remain observe hub arguments a local directory cannot express (see their labels).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.rewards.kling_video_reward.fixtures import (
    build_tiny_kling_reward_model,
    build_tiny_qwen2vl_repo,
)
from vrl.rewards.inference import RewardInferenceArtifact

_LORA_EXCLUDE = ["lm_head", "rm_head", "embed_tokens", "visual"]


def _video_reward_root(tmp_path: Path) -> Path:
    root = tmp_path / "VideoReward"
    checkpoint = root / "checkpoint-11352"
    checkpoint.mkdir(parents=True)
    (root / "model_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.pth").write_bytes(b"")
    return root


def test_kling_video_reward_snapshot_root_keeps_model_config_at_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hub ``reward_model_name`` resolves through ``snapshot_download`` to the snapshot root,
    where ``model_config.json`` lives.
    """
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(root))

    resolved = _resolve_model_root({"reward_model_name": "KlingTeam/VideoReward@main"})

    assert resolved == root


def test_kling_video_reward_checkpoint_path_resolves_to_repo_root(tmp_path: Path) -> None:
    """A ``model_path`` pointing at the ``checkpoint-NNNN`` subdirectory resolves up to the repo
    root that holds ``model_config.json``.
    """
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)

    resolved = _resolve_model_root({"model_path": str(root / "checkpoint-11352")})

    assert resolved == root


def test_kling_video_reward_requires_materialized_artifact_path() -> None:
    """An artifact with only in-memory media is refused before any scoring: Kling reads the video
    from a materialized file path.
    """
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    def _reward(*args, **kwargs):
        raise AssertionError("file-path validation should run before model scoring")

    model = KlingVideoRewardModel.__new__(KlingVideoRewardModel)
    model._reward = _reward
    model.use_norm = True
    artifact = RewardInferenceArtifact(
        artifact_id="a0",
        sample_id="sample-0",
        path="",
        media=object(),
        prompt="prompt",
    )

    with pytest.raises(ValueError, match="no materialized path"):
        model(artifact)


@pytest.mark.real_cover(
    "tests/rewards/kling_video_reward/test_model_loading.py"
    "::test_create_model_and_processor_runs_offline_on_a_tiny_repo",
    why=(
        "this pins the constructor's wiring (dtype, attention flag, offline flag, "
        "checkpoint dir) with recorders; the counterpart drives the same "
        "_create_model_and_processor and checkpoint loader on a real tiny model"
    ),
)
def test_kling_video_reward_builds_repo_owned_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction wires ``_create_model_and_processor`` with the worker's dtype / flash-attn /
    offline flags, loads the checkpoint from the resolved root, moves the model to the device
    and puts it in eval mode.
    """
    from vrl.rewards.models import kling_video_reward as kling_reward
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    captured = {}

    class _FakeModel:
        def eval(self):
            captured["model_eval"] = True

        def to(self, device):
            captured["model_device"] = device
            return self

    def _fake_create_model_and_processor(
        model_config,
        peft_config,
        *,
        dtype,
        disable_flash_attn2,
        local_files_only,
    ):
        del model_config, peft_config
        captured.update(
            {
                "dtype": dtype,
                "disable_flash_attn2": disable_flash_attn2,
                "local_files_only": local_files_only,
            },
        )
        return _FakeModel(), object()

    def _fake_load_checkpoint(model, checkpoint_dir):
        captured["checkpoint_dir"] = checkpoint_dir
        return model, "final"

    root = _video_reward_root(tmp_path)
    monkeypatch.setattr(
        kling_reward, "_create_model_and_processor", _fake_create_model_and_processor
    )
    monkeypatch.setattr(kling_reward, "load_kling_video_reward_checkpoint", _fake_load_checkpoint)

    KlingVideoRewardModel(
        {
            "model_path": str(root),
            "device": "cpu",
            "dtype": "bfloat16",
            "disable_flash_attn2": True,
            "local_files_only": True,
        },
    )

    assert captured == {
        "dtype": kling_reward.torch.bfloat16,
        "disable_flash_attn2": True,
        "local_files_only": True,
        "checkpoint_dir": root,
        "model_eval": True,
        "model_device": "cpu",
    }


def test_kling_video_reward_parses_frame_pixel_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks min/max frame pixel bounds parse from worker_config."""
    from vrl.rewards.models import kling_video_reward as kling_reward
    from vrl.rewards.models.kling_video_reward import KlingVideoRewardModel

    class _FakeModel:
        def eval(self):
            return None

        def to(self, device):
            del device
            return self

    monkeypatch.setattr(
        kling_reward,
        "_create_model_and_processor",
        lambda *args, **kwargs: (_FakeModel(), object()),
    )
    monkeypatch.setattr(
        kling_reward,
        "load_kling_video_reward_checkpoint",
        lambda model, checkpoint_dir: (model, "final"),
    )

    root = _video_reward_root(tmp_path)
    model = KlingVideoRewardModel(
        {
            "model_path": str(root),
            "device": "cpu",
            "disable_flash_attn2": True,
            "min_frame_pixels": 200704,
        },
    )

    assert model.min_frame_pixels == 200704
    assert model.max_frame_pixels is None  # checkpoint budget stays in charge


def test_kling_video_reward_snapshot_download_honors_local_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Kling video reward passes local-only mode to snapshot download."""
    from vrl.rewards.models.kling_video_reward import _resolve_model_root

    root = _video_reward_root(tmp_path)
    captured = {}

    def _fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    resolved = _resolve_model_root(
        {
            "reward_model_name": "KlingTeam/VideoReward@main",
            "local_files_only": True,
        },
    )

    assert resolved == root
    assert captured["repo_id"] == "KlingTeam/VideoReward"
    assert captured["revision"] == "main"
    assert captured["local_files_only"] is True


@pytest.mark.real_cover(
    "tests/rewards/kling_video_reward/test_model_loading.py"
    "::test_create_model_and_processor_runs_offline_on_a_tiny_repo",
    why=(
        "hf_hub falls back to the cache on a connection error, so whether "
        "local_files_only reached the loaders is unobservable in-process; only the "
        "recorded kwarg proves it, while the counterpart runs the real loaders offline"
    ),
)
def test_kling_video_reward_base_loader_honors_local_files_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks Kling video reward keeps base model loading offline when requested."""
    from vrl.rewards.models import kling_video_reward as kling_reward

    captured = {}

    class _FakeTokenizer:
        padding_side = "right"
        pad_token_id = 0

    class _FakeProcessor:
        tokenizer = _FakeTokenizer()

    class _FakeModelConfig:
        pass

    class _FakeModel:
        config = _FakeModelConfig()

        def to(self, dtype):
            captured["to_dtype"] = dtype
            return self

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["processor"] = (args, kwargs)
            return _FakeProcessor()

    def _fake_from_pretrained(*args, **kwargs):
        captured["model"] = (args, kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        kling_reward.KlingQwen2VLRewardModel,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor",
        _FakeAutoProcessor,
    )

    model_config = kling_reward._ModelConfig(
        model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        model_revision="main",
        torch_dtype="bfloat16",
    )
    peft_config = kling_reward._PeftLoraConfig(lora_enable=False)

    model, processor = kling_reward._create_model_and_processor(
        model_config,
        peft_config,
        dtype=kling_reward.torch.bfloat16,
        disable_flash_attn2=True,
        local_files_only=True,
    )

    assert isinstance(model, _FakeModel)
    assert isinstance(processor, _FakeProcessor)
    assert captured["processor"][1]["local_files_only"] is True
    assert captured["processor"][1]["revision"] == "main"
    assert captured["model"][1]["local_files_only"] is True
    assert captured["model"][1]["revision"] == "main"


def test_kling_video_reward_remaps_qwen2vl_checkpoint_keys() -> None:
    """Legacy Qwen2-VL keys remap into the transformers 5 layout: ``visual.*`` gains a ``model.``
    prefix, ``model.layers`` / ``embed_tokens`` move under ``model.language_model``, and
    ``lm_head`` is untouched.
    """
    from vrl.rewards.models.kling_video_reward import _remap_qwen2vl_key

    assert (
        _remap_qwen2vl_key(
            "base_model.model.visual.patch_embed.proj.weight",
        )
        == "base_model.model.model.visual.patch_embed.proj.weight"
    )
    assert _remap_qwen2vl_key(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
    ) == ("base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight")
    assert (
        _remap_qwen2vl_key(
            "base_model.model.model.embed_tokens.weight",
        )
        == "base_model.model.model.language_model.embed_tokens.weight"
    )
    assert (
        _remap_qwen2vl_key(
            "base_model.model.lm_head.weight",
        )
        == "base_model.model.lm_head.weight"
    )


def test_kling_normalize_scores_renames_drops_missing_and_never_leaks() -> None:
    """_normalize_scores renames raw VQ/MQ/TA/Overall to public aliases, drops a
    public key when its raw source is absent, and never leaks raw/undocumented keys.

    Expectations are written as literal dicts (not derived from ``_SCORE_KEY_MAP``)
    so the test fails if the mapping or the ``if model_key in raw`` filter regresses.
    """
    from vrl.rewards.models.kling_video_reward import _normalize_scores

    # Full raw set -> all four public aliases, values carried by the rename.
    assert _normalize_scores({"VQ": 1.0, "MQ": 2.0, "TA": 3.0, "Overall": 6.0}) == {
        "overall_reward": 6.0,
        "visual_quality": 1.0,
        "motion_quality": 2.0,
        "text_alignment": 3.0,
    }

    # Missing raw source -> the corresponding public key is dropped, not defaulted.
    assert _normalize_scores({"VQ": 1.0, "Overall": 4.0}) == {
        "visual_quality": 1.0,
        "overall_reward": 4.0,
    }

    # Undocumented raw key -> never crosses into the public scoring contract.
    assert _normalize_scores({"VQ": 1.0, "BOGUS": 9.0}) == {"visual_quality": 1.0}


def _lora_wrapped(seed: int):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        build_tiny_kling_reward_model(seed=seed),
        LoraConfig(r=2, target_modules=["q_proj", "v_proj"]),
    )


def _legacy_layout(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The pre-transformers-5 key layout: no ``language_model`` nesting, top-level ``visual``."""

    return {
        key.replace(
            "base_model.model.model.language_model.", "base_model.model.model.", 1
        ).replace("base_model.model.model.visual.", "base_model.model.visual.", 1): value
        for key, value in state.items()
    }


def test_checkpoint_loader_strict_loads_a_live_model_in_either_key_layout(tmp_path: Path) -> None:
    """``_remap_qwen2vl_state_dict`` compares against the LIVE model's keys and the
    loader then ``strict=True``-loads; both the current and the legacy layout must
    land on a fresh model bit-for-bit."""
    from vrl.rewards.models.kling_video_reward import (
        _remap_qwen2vl_state_dict,
        load_kling_video_reward_checkpoint,
    )

    source = _lora_wrapped(seed=0)
    checkpoint = tmp_path / "checkpoint-11352"
    checkpoint.mkdir()
    torch.save(source.state_dict(), checkpoint / "model.pth")

    loaded, step = load_kling_video_reward_checkpoint(_lora_wrapped(seed=1), tmp_path)
    assert step == "11352"
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key

    legacy = _legacy_layout(source.state_dict())
    assert legacy.keys() != source.state_dict().keys()
    assert set(_remap_qwen2vl_state_dict(legacy, source.state_dict())) == set(source.state_dict())
    torch.save(legacy, checkpoint / "model.pth")
    relocated, _ = load_kling_video_reward_checkpoint(_lora_wrapped(seed=2), tmp_path)
    for key, value in source.state_dict().items():
        assert torch.equal(relocated.state_dict()[key], value), key


def test_create_model_and_processor_runs_offline_on_a_tiny_repo(tmp_path: Path) -> None:
    """The real loaders read a tiny repo: special tokens are added and resized,
    the pad token and padding side reach the model config, and the shipped
    ``lora_namespan_exclude`` keeps LoRA off the reward head, embeddings and vision."""
    from vrl.rewards.models.kling_video_reward import (
        _SPECIAL_TOKENS,
        _create_model_and_processor,
        _find_target_linear_names,
        _ModelConfig,
        _PeftLoraConfig,
    )

    repo = build_tiny_qwen2vl_repo(tmp_path / "tiny-qwen2vl")

    model, processor = _create_model_and_processor(
        _ModelConfig(
            model_name_or_path=str(repo),
            model_revision="main",
            output_dim=1,
            use_special_tokens=True,
            reward_token="special",
        ),
        _PeftLoraConfig(lora_enable=True, lora_r=2, lora_namespan_exclude=_LORA_EXCLUDE),
        dtype=torch.float32,
        disable_flash_attn2=True,
        local_files_only=True,
    )

    base = model.base_model.model
    assert model.config.pad_token_id == processor.tokenizer.pad_token_id
    assert model.config.tokenizer_padding_side == "right"
    assert base.special_token_ids == processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKENS)
    assert len(base.special_token_ids) == 3
    assert base.reward_token == "special"
    targets = _find_target_linear_names(base, lora_namespan_exclude=_LORA_EXCLUDE)
    assert targets
    assert not any(
        bad in name for name in targets for bad in ("rm_head", "embed_tokens", "visual")
    )


def test_chat_payload_applies_the_checkpoint_frame_budget_and_frame_policy() -> None:
    """``max_pixels`` falls back to the checkpoint budget, ``min_pixels`` is written
    only when set, and ``nframes`` displaces ``fps``: these decide whether the
    reward scores in-distribution."""
    from vrl.rewards.models.kling_video_reward import _build_chat_payload, _DataConfig

    by_fps = _DataConfig(max_frame_pixels=200704, fps=2.0, eval_dim=["VQ", "MQ", "TA"])
    (conversation,) = _build_chat_payload(
        ["/tmp/clip.mp4"], ["a robot arm"], data_config=by_fps, max_pixels=None, min_pixels=None
    )
    (turn,) = conversation
    video, text = turn["content"]
    assert turn["role"] == "user"
    assert video == {
        "type": "video",
        "video": "file:///tmp/clip.mp4",
        "max_pixels": 200704,
        "fps": 2.0,
    }
    assert text["type"] == "text"
    assert "a robot arm" in text["text"]

    by_frames = _DataConfig(max_frame_pixels=200704, num_frames=8, fps=2.0)
    (conversation,) = _build_chat_payload(
        ["/tmp/clip.mp4"], ["p"], data_config=by_frames, max_pixels=1024, min_pixels=256
    )
    video = conversation[0]["content"][0]
    assert video["max_pixels"] == 1024
    assert video["min_pixels"] == 256
    assert video["nframes"] == 8
    assert "fps" not in video

    with pytest.raises(ValueError, match="uniform"):
        _build_chat_payload(
            ["/tmp/clip.mp4"],
            ["p"],
            data_config=_DataConfig(sample_type="random"),
            max_pixels=None,
            min_pixels=None,
        )


@pytest.mark.optional
def test_prepare_batch_decodes_a_real_clip_through_the_real_processor(tmp_path: Path) -> None:
    """The decode + chat-template half of ``_prepare_batch`` needs the ``[reward]``
    extra (qwen_vl_utils / decord); with it, an 8-frame mp4 goes through the tiny
    real processor and the tiny model scores it."""
    pytest.importorskip("qwen_vl_utils")
    from vrl.rewards.models.kling_video_reward import (
        KlingVideoRewardModel,
        _create_model_and_processor,
        _DataConfig,
        _ModelConfig,
        _PeftLoraConfig,
    )
    from vrl.utils.media import write_mp4

    repo = build_tiny_qwen2vl_repo(tmp_path / "tiny-qwen2vl", output_dim=3)
    clip = tmp_path / "clip.mp4"
    write_mp4(torch.rand(3, 8, 32, 32), clip, fps=4.0)
    model, processor = _create_model_and_processor(
        _ModelConfig(model_name_or_path=str(repo), model_revision="main", output_dim=3),
        _PeftLoraConfig(lora_enable=False),
        dtype=torch.float32,
        disable_flash_attn2=True,
        local_files_only=True,
    )
    reward_model = KlingVideoRewardModel.__new__(KlingVideoRewardModel)
    reward_model.model = model
    reward_model.processor = processor
    reward_model.device = "cpu"
    reward_model.data_config = _DataConfig(max_frame_pixels=28 * 28 * 4, num_frames=4)

    batch = reward_model._prepare_batch([str(clip)], ["a robot arm"])

    assert batch["input_ids"].shape[0] == 1
    assert batch["pixel_values_videos"].ndim == 2
    assert batch["video_grid_thw"].shape == (1, 3)
    with torch.no_grad():
        logits = model(return_dict=True, **batch)["logits"]
    assert logits.shape == (1, 3)
