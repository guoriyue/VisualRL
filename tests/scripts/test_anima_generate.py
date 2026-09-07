from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.config.precision import RolePrecision
from vrl.config.schema import parse_config
from vrl.scripts.families.cosmos.anima import generate
from vrl.trainers.data import PromptExample


def _minimal_generate_config():
    return OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2-anima",
                "path": "org/model",
                "use_lora": False,
                "lora": {"path": ""},
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
            "sampling": {
                "width": 8,
                "height": 8,
                "num_steps": 1,
                "guidance_scale": 4.5,
                "max_sequence_length": 8,
            },
        },
    )


def test_generate_disables_empty_training_lora_for_inference() -> None:
    """An empty ``lora.path`` with ``use_lora`` on means an untrained adapter: inference turns
    LoRA off instead of loading nothing.
    """
    cfg = OmegaConf.create({"model": {"use_lora": True, "lora": {"path": ""}}})

    assert generate._lora_overrides(cfg, lora_path="") == ["model.use_lora=false"]


def test_generate_accepts_checkpoint_dir_as_lora_path(tmp_path) -> None:
    """A checkpoint directory is accepted as the LoRA path and resolved to its ``lora_weights``
    export, switching LoRA on.
    """
    checkpoint = tmp_path / "checkpoint-final"
    exported = checkpoint / "lora_weights"
    exported.mkdir(parents=True)
    cfg = OmegaConf.create({"model": {"use_lora": False, "lora": {"path": ""}}})

    assert generate._lora_overrides(cfg, lora_path=str(checkpoint)) == [
        "model.use_lora=true",
        f"model.lora.path={exported}",
    ]


def test_lora_checkpoint_provenance_binds_progress_and_identity(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-final"
    adapter = checkpoint / "lora_weights"
    adapter.mkdir(parents=True)
    identity = {"schema": "test/v1", "sources": {"main": {}}, "build": {}}
    metadata = {
        "schema_version": 2,
        "family": "cosmos-predict2-anima",
        "model_identity": identity,
        "trainer_step": 20,
        "global_step": 19,
        "completed_epoch": 20,
        "next_epoch": 20,
        "uses_lora": True,
    }
    (checkpoint / "checkpoint_meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    provenance = generate._lora_checkpoint_provenance(str(adapter), identity)

    assert provenance is not None
    assert provenance["label"] == "checkpoint-final"
    assert {
        key: provenance[key]
        for key in ("global_step", "trainer_step", "completed_epoch", "next_epoch")
    } == {
        "global_step": 19,
        "trainer_step": 20,
        "completed_epoch": 20,
        "next_epoch": 20,
    }
    assert len(provenance["metadata_sha256"]) == 64


def test_generate_image_conversion_accepts_chw_float() -> None:
    """``image_to_uint8_hwc`` turns a float CHW image into uint8 HWC with the [0, 1] range scaled
    to 255.
    """
    from vrl.utils.media import image_to_uint8_hwc

    image = np.zeros((3, 2, 2), dtype=np.float32)
    image[0] = 1.0

    out = image_to_uint8_hwc(image)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert out[..., 0].max() == 255


def test_generate_sampling_defaults_follow_config() -> None:
    """With no CLI overrides the generation sampling (size, steps, guidance, sequence length) is
    taken from the config's sampling section.
    """
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "cosmos-predict2-anima"},
                "sampling": {
                    "width": 768,
                    "height": 512,
                    "num_steps": 12,
                    "guidance_scale": 4.0,
                    "max_sequence_length": 256,
                },
            },
        ),
    )
    args = generate.build_parser().parse_args(["--prompt", "adult anime portrait"])

    sampling = generate._resolve_sampling(args, root)

    assert sampling.width == 768
    assert sampling.height == 512
    assert sampling.num_steps == 12
    assert sampling.guidance_scale == 4.0
    assert sampling.max_sequence_length == 256


def test_generate_sampling_preserves_explicit_zero_guidance() -> None:
    """An explicit zero disables CFG instead of falling back to the config."""
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "cosmos-predict2-anima"},
                "sampling": {
                    "width": 768,
                    "height": 512,
                    "num_steps": 12,
                    "guidance_scale": 4.0,
                    "max_sequence_length": 256,
                },
            },
        ),
    )
    args = generate.build_parser().parse_args(
        ["--prompt", "adult anime portrait", "--guidance-scale", "0"],
    )

    sampling = generate._resolve_sampling(args, root)

    assert sampling.guidance_scale == 0.0


@pytest.mark.parametrize(
    ("device", "overrides", "expected_dtype", "expected_steps", "expected_precision"),
    [
        ("cpu", [], "float32", 10, "tf32"),
        ("cuda:0", [], "bfloat16", 10, "tf32"),
        (
            "cuda:0",
            [
                "+sampling/denoise=20_step_cfg_4_5",
                "precision.training.dtype=fp32",
                "precision.float32_precision=ieee",
            ],
            "float32",
            20,
            "ieee",
        ),
    ],
)
def test_generate_default_is_reward_independent_and_preserves_inference_policy(
    monkeypatch,
    tmp_path,
    device,
    overrides,
    expected_dtype,
    expected_steps,
    expected_precision,
) -> None:
    import torch

    loaded = []
    real_load_config = generate.load_config

    def tracked_load_config(path, *, overrides):
        cfg = real_load_config(path, overrides=overrides)
        loaded.append((cfg, list(overrides)))
        return cfg

    monkeypatch.setattr(generate, "load_config", tracked_load_config)
    monkeypatch.setattr(generate, "resolve_eval_device", lambda _: torch.device(device))
    output_dir = tmp_path / "generated"
    generate.main(
        [
            "--prompt",
            "adult anime portrait",
            "--dry-run",
            "--output-dir",
            str(output_dir),
            *overrides,
        ],
    )

    cfg, effective_overrides = loaded[0]
    assert not {"reward", "data", "actor", "trainer"}.intersection(cfg)
    record = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert record["config"] == "model/cosmos/anima_preview3"
    assert record["config_overrides"] == effective_overrides
    assert record["sampling"] == {
        "width": 512,
        "height": 512,
        "num_steps": expected_steps,
        "guidance_scale": 4.5,
        "max_sequence_length": 128,
    }
    assert record["execution"]["dtype"] == expected_dtype
    assert record["generation_policy"]["role_precision"]["float32_precision"] == (
        expected_precision
    )
    assert record["generation_policy"]["role_precision"]["outer_autocast"] is True
    assert record["model"]["use_lora"] is False
    assert not (output_dir / "images").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_generate_refuses_nonempty_output_before_writing(
    monkeypatch,
    tmp_path,
    dry_run,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    existing = output_dir / "run_config.json"
    existing.write_text('{"owner": "earlier run"}\n', encoding="utf-8")
    monkeypatch.setattr(
        generate,
        "load_config",
        lambda *_args, **_kwargs: _minimal_generate_config(),
    )
    argv = [
        "--prompt",
        "adult anime portrait",
        "--device",
        "cpu",
        "--dtype",
        "fp32",
        "--output-dir",
        str(output_dir),
    ]
    if dry_run:
        argv.append("--dry-run")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        generate.main(argv)

    assert existing.read_text(encoding="utf-8") == '{"owner": "earlier run"}\n'
    assert list(output_dir.iterdir()) == [existing]


def test_generate_persists_canonical_identity_for_component_overrides(
    monkeypatch,
    tmp_path,
) -> None:
    cfg = _minimal_generate_config()
    expected_sources = {}
    for component in ("transformer", "text_encoder", "vae"):
        artifact = tmp_path / f"{component}.safetensors"
        artifact.write_bytes(component.encode())
        cfg.model[component + "_path"] = str(artifact)
        expected_sources[component] = "local-file"
    monkeypatch.setattr(generate, "load_config", lambda *_args, **_kwargs: cfg)
    output_dir = tmp_path / "generated"

    generate.main(
        [
            "--prompt",
            "adult anime portrait",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ],
    )

    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert {
        name: source["kind"] for name, source in run_config["model_identity"]["sources"].items()
    } == expected_sources
    assert "path" not in run_config["model"]
    assert run_config["schema"] == "vrl.anima-generation/v1"
    assert run_config["base_seed"] == 20260520
    assert len(run_config["generator_runtime"]["vrl_python_tree_sha256"]) == 64
    assert set(run_config["generator_runtime"]["packages"]) == {
        "diffusers",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    }
    assert run_config["generation_policy"] == {
        "diffusion_math_dtype": "fp32",
        "family": "cosmos-predict2-anima",
        "generation_memory": None,
        "parameter_dtype": "float32",
        "prompt_encoder_dtype": "float32",
        "role_precision": {
            "dtype": "fp32",
            "float32_precision": "ieee",
            "outer_autocast": True,
            "quantization": None,
        },
        "rollout": {
            "base_weight_sync": True,
            "pipeline_offload_mode": "none",
        },
        "torch_compile": None,
    }
    assert not (output_dir / "images").exists()


def test_explicit_dtype_rejects_malformed_model_section(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        generate,
        "load_config",
        lambda *_args, **_kwargs: OmegaConf.create({"model": ["not", "a", "mapping"]}),
    )

    with pytest.raises(ValueError, match="model must be a mapping"):
        generate.main(
            [
                "--prompt",
                "adult anime portrait",
                "--dtype",
                "fp32",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "output"),
            ],
        )


def test_generate_records_the_batch_seed_for_every_sample(monkeypatch, tmp_path) -> None:
    """Rows share the one generator seed used for their prompt batch."""
    cfg = OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2-anima",
                "path": "unused",
                "use_lora": False,
                "lora": {"path": ""},
            },
            "sampling": {
                "width": 2,
                "height": 2,
                "num_steps": 1,
                "guidance_scale": 1.0,
                "max_sequence_length": 8,
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
            },
        },
    )

    class _Model:
        def eval(self):
            return self

    monkeypatch.setattr(generate, "load_config", lambda *_args, **_kwargs: cfg)
    build = object()
    entry = SimpleNamespace(
        resolve_model_build=lambda *_args, **_kwargs: build,
        build_rollout=lambda _build: SimpleNamespace(
            model=_Model(),
            precision=RolePrecision(
                dtype="fp32",
                float32_precision="ieee",
                outer_autocast=False,
            ),
        ),
    )
    monkeypatch.setattr(generate, "get_model_family_entry", lambda _family: entry)
    monkeypatch.setattr(
        generate,
        "resolve_checkpoint_model_identity",
        lambda resolved: {"schema": "test/v1", "resolved_build": resolved is build},
    )
    monkeypatch.setattr(generate, "_generation_policy", lambda *_args: {"test": True})
    monkeypatch.setattr(
        generate,
        "_load_prompts",
        lambda *_args, **_kwargs: [
            PromptExample(
                prompt='A sign reading "OPEN".',
                target_text="OPEN",
                metadata={"bucket": "sign"},
            ),
        ],
    )
    monkeypatch.setattr(
        generate,
        "generate_images",
        lambda *_args, **_kwargs: [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
    )

    output_dir = tmp_path / "generated"
    generate.main(
        [
            "--prompt",
            "adult anime portrait",
            "--samples-per-prompt",
            "2",
            "--seed",
            "37",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--output-dir",
            str(output_dir),
        ],
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["seed"] for row in rows] == [37, 37]
    assert {path.name for path in output_dir.iterdir()} == {
        "images",
        "run_config.json",
        "metadata.jsonl",
        "anchor_manifest.jsonl",
    }
    assert [row["prompt_metadata"] for row in rows] == [
        {"bucket": "sign"},
        {"bucket": "sign"},
    ]
    assert [row["reward_metadata"] for row in rows] == [
        {"bucket": "sign", "target_text": "OPEN"},
        {"bucket": "sign", "target_text": "OPEN"},
    ]
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["model_identity"] == {
        "resolved_build": True,
        "schema": "test/v1",
    }
    anchors = [
        json.loads(line)
        for line in (output_dir / "anchor_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["target_image"] for row in anchors] == [
        "images/anima_0000_00.png",
        "images/anima_0000_01.png",
    ]
    assert [row["metadata"]["anchor_source"] for row in anchors] == [
        "anima_base_synthetic",
        "anima_base_synthetic",
    ]


def test_generate_preserves_manifest_metadata_in_anchor_rows(tmp_path) -> None:
    """Anchor manifests retain prompt strata alongside generation provenance."""
    rows = [
        {
            "prompt": "An adult musician plays violin on stage.",
            "prompt_metadata": {
                "bucket": "object_interaction",
                "prompt_style": "natural_language",
                "source": "codex_curated",
            },
            "image_path": str(tmp_path / "images" / "anima_0000_00.png"),
            "sample_index": 0,
            "seed": 41,
        },
    ]

    generate._write_metadata(rows, tmp_path, anchor_source="anima_base_synthetic")

    anchor = json.loads((tmp_path / "anchor_manifest.jsonl").read_text(encoding="utf-8"))
    assert anchor["metadata"] == {
        "anchor_sample_index": 0,
        "anchor_seed": 41,
        "anchor_source": "anima_base_synthetic",
        "bucket": "object_interaction",
        "prompt_style": "natural_language",
        "source": "codex_curated",
    }


def test_generate_labels_lora_anchors_as_lora_outputs(tmp_path) -> None:
    rows = [
        {
            "prompt": "A sign reading OPEN.",
            "prompt_metadata": {},
            "image_path": str(tmp_path / "images" / "anima_0000_00.png"),
            "sample_index": 0,
            "seed": 41,
        },
    ]

    generate._write_metadata(rows, tmp_path, anchor_source="anima_lora_synthetic")

    anchor = json.loads((tmp_path / "anchor_manifest.jsonl").read_text(encoding="utf-8"))
    assert anchor["metadata"]["anchor_source"] == "anima_lora_synthetic"


def test_generate_cuda_device_fails_fast_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested but CUDA is unavailable"):
        generate.resolve_eval_device("cuda:0")
