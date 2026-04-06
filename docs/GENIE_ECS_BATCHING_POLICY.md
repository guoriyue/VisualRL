# Genie ECS Batching Policy

This document records the Step 2 batching-policy convergence for `genie-rollout`.

## Root Cause

The original `genie_cross_request_batching` switch did not represent a single policy surface.

- queue-level whole-request batching was controlled by `GenieJobQueue`
- transition-stage batching inside `GenieTransitionBatcher` stayed enabled even when queue batching was off
- multi-window workloads could still batch at the transition stage while the benchmark label said `off`
- single-window workloads could be grouped into large whole-request batches that increased head-of-line delay

That made the benchmark pairs hard to interpret and caused avoidable regressions.

## Policy After Convergence

The batching policy is now explicit:

1. `ControlPlaneConfig.genie_max_batch_size` and `genie_batch_wait_ms` configure the transition batcher directly.
2. `genie_cross_request_batching=off` now disables both queue-level whole-request batching and transition-stage batching because the backend receives `transition_max_batch_size=1` and `transition_batch_wait_ms=0`.
3. Whole-request queue batching is only allowed for single-window Genie workloads.
4. Whole-request queue batching is capped at batch size `2` to avoid head-of-line spikes for later samples in the queue.
5. Multi-window workloads such as checkpoint-heavy profile/heavy runs skip whole-request batching and rely on stage-local transition batching only.

Code paths:

- `wm_infra/backends/genie.py`
- `wm_infra/api/server.py`

## Post-Policy Benchmark Snapshot

| Artifact | Submit Mean (ms) | Submit P95 (ms) | Terminal Mean (ms) | Terminal P95 (ms) | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| `genie_default_baseline.json` | 2424.265 | 3473.730 | 2428.410 | 3478.440 | 1.0 |
| `genie_default_batched.json` | 5.141 | 5.336 | 2389.426 | 3456.351 | 1.0 |
| `genie_profile_baseline.json` | 9069.379 | 9069.532 | 9076.242 | 9076.321 | 1.0 |
| `genie_profile_batched.json` | 1961.656 | 1961.825 | 6084.445 | 8398.770 | 1.0 |
| `genie_heavy_off.json` | 4172.983 | 5083.019 | 4177.414 | 5088.084 | 1.0 |
| `genie_heavy_on.json` | 1626.891 | 2199.628 | 4373.812 | 5565.042 | 1.0 |

## Gate Result

Default workload gate:

- terminal mean ratio: `0.9840x`
- terminal p95 ratio: `0.9936x`
- verdict: pass

Heavy workload gate:

- terminal mean ratio: `1.0470x`
- terminal p95 ratio: `1.0937x`
- verdict: pass

Profile workload snapshot:

- terminal mean ratio: `0.6704x`
- terminal p95 ratio: `0.9253x`
- verdict: improved

## Step 2 Conclusion

The batching policy is now consistent and benchmark-comparable:

- `off` means no cross-request batching
- default workload stays within the cleanup gate
- heavy workload stays within the cleanup gate
- profile workload no longer shows the earlier head-of-line regression

The next step is to extend stage runtime scheduling and profiling so the same policy can be reasoned about directly from runtime metadata rather than inferred only from benchmark output.
