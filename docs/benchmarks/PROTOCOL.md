# Vestra benchmark protocol

## Locked hardware

- CPU: AMD Ryzen 9 9950X, 16 physical cores / 32 logical processors
- CPU benchmark thread budget: 16
- GPU: NVIDIA GeForce RTX 5080, 16 GiB
- RAM: 96 GiB

## Locked product workload

- 60-second 1080p phone video
- 120 deterministic selected frames
- 504×336 inference resolution
- chunk size 12, overlap 3
- relative-scale reconstruction
- identical confidence, ICP, loop-closure, TSDF, and export settings per arm

## Statistical contract

- At least 10 independent randomized trials per arm
- One warm-up followed by 10 measured iterations for steady-state operators
- Primary statistic: mean of trial medians with a 95% confidence interval
- Increase N when intervals or paired differences do not resolve the claim
- Preserve every raw sample; do not remove outliers without a prior rule
- Record source revisions, dirty diff hashes, compiler flags, binary hashes,
  model/input hashes, driver/runtime versions, hardware state, and temperatures

## Required comparisons

| Workload | Required Vestra result |
|---|---:|
| Single-image CPU F32 vs C++/ggml | Preserve at least 25% lower latency |
| 12-view CPU F32 window | At least 25% lower latency |
| 12-view GPU, identical precision | At least 15% lower latency |
| Complete 120-frame CPU reconstruction | At least 20% lower latency |
| Complete 120-frame GPU reconstruction | At least 15% lower latency |
| Product latency on RTX 5080 | Interactive <10 s; complete <15 s |

Quantized, mixed-precision, CPU, and GPU results are separate studies.
