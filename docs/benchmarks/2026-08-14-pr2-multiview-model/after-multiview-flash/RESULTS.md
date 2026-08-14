# Multi-view Flash Attention — 2026-08-14

## Hypothesis

The AVX-512 DA3 flash-attention implementation was guarded to exactly one
504×336 view (`865` tokens). PR #2 global layers concatenate complete views,
therefore a 12-view window is the same `[heads, tokens, 64]` workload with
`12 × 865 = 10,380` tokens. The fixed 64-wide kernel supports that shape, but
the guard forced the global layers into the generic row-wise fallback.

The candidate admits only a positive integral number of DA3-BASE rasters. It
does not change model precision, frame count, resolution, scheduling, weights,
or the attention arithmetic. `DA3_KERNELS_DISABLE_MULTIVIEW_FLASH=1` is a
narrow diagnostic control retaining the previous global fallback.

## Correctness

- An AVX-512 two-view (`1,730` token) regression test compares the candidate
  with the serial online-softmax reference: MAE <= `2e-5`, max error <=
  `2e-4`.
- The full 24-frame `MVO1` C++ differential remains unchanged: depth MAE
  `1.67e-6`, confidence MAE `3.27e-5`, extrinsics MAE `5.42e-7`, and
  intrinsics MAE `1.22e-4`. Shapes and the three PR #2 windows match.
- Alternating same-binary A/B (three fresh trials per arm) measured Rust
  fallback `14,117.368 ms` versus candidate `8,485.674 ms`: a `39.89%`
  reduction. This is a diagnostic acceptance gate, not the C++ comparison.

## Fair C++ comparison

Both arms used the same F32 GGUF, 24 canonical RGB24 PPM frames, 504×336
raster, three genuine PR #2 12/3 multi-view windows, and 16 CPU threads.
Model loading, PPM decoding, input-window assembly, and output serialization
were outside the timed scope. Each fresh process performed one excluded warmup
and one measured repetition.

| Arm | n | Mean (ms) | Median (ms) | SD (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | --: | ---: |
| C++ PR #2 F32 | 12 | 8589.448 | 8582.415 | 39.900 | [8564.096, 8614.799] |
| Vestra Rust F32 | 12 | 8506.431 | 8500.289 | 71.276 | [8461.144, 8551.718] |

Rust's observed mean is `0.966%` lower (`0.990335×` C++ wall time). The
intervals overlap, so this is **not** evidence for a statistically resolved
Rust win. It is evidence that the previous `1.632×` model-bound deficit has
been eliminated under the locked workload.

## Raw-data integrity

[`raw.jsonl`](raw.jsonl) contains all 24 fresh-process samples in execution
order. The initially imbalanced 11-C++/9-Rust schedule was corrected by adding
fresh trials to reach 12 per arm; no sample was removed or replaced. Exact
commits, binaries, hashes, and the correction record are in
[`metadata.json`](metadata.json).

## Decision

Retain the kernel. Increase the trial count only with explicit authorization
if a non-overlapping confidence interval is required for a public speed claim.
