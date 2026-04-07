# Wan serving scaffold

`wm-infra` includes a concrete `wan-video` backend for temporal sample production.
This doc exists to describe the implemented Wan 2.2 path inside a repo that is otherwise broader temporal infra.

## What it does

- accepts `text_to_video`, `image_to_video`, and `video_to_video` sample requests
- uses first-class `wan_config` fields instead of hiding key execution knobs in metadata
- computes a scheduler-friendly resource estimate aligned with the memory-aware rollout work
- batches queue-compatible Wan 2.2 requests by shared diffusion shape and near-shape execution hints
- tracks a warmed execution-profile pool keyed by resolution, frame count, step count, and CFG buckets
- returns quality-cost fallback hints (`auto_step_reduction`, `resolution_fallback`, `progressive_preview`) when admission rejects an oversized request
- persists a durable sample record through the existing manifest store
- can run in:
  - **stub mode**: no external runner configured, but request/log/output paths are still materialized
  - **shell mode**: `WM_WAN_SHELL_RUNNER` template is executed and its stdout/stderr is captured
  - **official mode**: local Wan 2.2 repo invocation, currently aligned with `wan2.2-t2v-A14B` and `wan2.2-i2v-A14B`

## API example

```json
POST /v1/samples
{
  "task_type": "text_to_video",
  "backend": "wan-video",
  "model": "wan2.2-t2v-A14B",
  "sample_spec": {
    "prompt": "a corgi surfing through a data center"
  },
  "wan_config": {
    "num_steps": 4,
    "frame_count": 9,
    "width": 832,
    "height": 480,
    "guidance_scale": 4.0,
    "shift": 12.0,
    "memory_profile": "low_vram"
  }
}
```

## Environment variables

- `WM_WAN_OUTPUT_ROOT=/path/to/wan-output`
- `WM_WAN_SHELL_RUNNER='python run_wan.py --prompt {prompt} --size {width}x{height} --frames {frame_count} --steps {num_steps} --out {output_path}'`

## Notes

This is still an early serving path, not a claim of complete production maturity.
The current batching and warm-profile logic lives at the queue/runtime layer; it does not yet mean Wan 2.2 is running as a fully in-process multi-stage diffusion engine.
The point is to make Wan 2.2 sample production a first-class backend and API path now so the repo can evolve around a real temporal workload without another schema rewrite later.
