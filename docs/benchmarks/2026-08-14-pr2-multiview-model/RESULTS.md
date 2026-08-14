# PR #2 multi-view model benchmark — 2026-08-14

## Scope

This is the model-bound counterpart to the geometry benchmark. Both arms use
the same 24 canonical RGB24 PPM frames, the identical
`depth-anything-base-f32.gguf` model, 504×336 raster, the PR #2 12-frame / 3
frame-overlap schedule, and a 16-thread CPU budget on the AMD Ryzen 9 9950X.
Each timed sample executes all three genuine multi-view
depth/confidence/pose windows, including reference-view selection and global
attention. Model loading, PPM decoding, input-window assembly and output
serialization happen before timing.

The C++ arm is `vestra_cpp_multiview_bench`, calling pinned PR #2
`Engine::depth_pose_multi`. The Rust arm is `vestra oracle-model-bench`, calling
Vestra Engine's `infer_multi_view`. Both retain a small output checksum so the
model outputs cannot be optimized away. Numerical comparison is independently
recorded in [`../../validation/STRICT_PR2_PROFILE_SMOKE_2026-08-14.md`](../../validation/STRICT_PR2_PROFILE_SMOKE_2026-08-14.md): depth, confidence,
intrinsics and extrinsics use the same direct MVO1 boundary.

## Protocol

- One excluded warm-up followed by one measured repeat inside each fresh
  process trial.
- Ten fresh process trials per arm in one randomized order (seed `20260817`).
- `OMP_NUM_THREADS=16` for C++; `RAYON_NUM_THREADS=16` for Rust.
- No outliers were removed. Raw samples and exact artifact hashes are retained
  in [`raw.jsonl`](raw.jsonl) and [`metadata.json`](metadata.json).

## Result

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 F32 | 10 | 8578.887 | 8582.875 | [8550.531, 8607.243] |
| Vestra Rust F32 | 10 | 14002.409 | 14005.276 | [13954.910, 14049.907] |

Rust is **1.632×** C++ wall time in this exact multi-view model tier (63.22%
slower). The intervals are clearly separated. This result must not be blended
with the accepted geometry-plus-TSDF win: the Rust geometry pipeline is 4.11%
faster on its locked model-free fixture, but the current multi-view transformer
implementation remains the dominant blocker for a complete end-to-end win.

The resolved follow-up is recorded in
[`final-30-per-arm/RESULTS.md`](final-30-per-arm/RESULTS.md). The AVX-512
multi-view Flash route removed this deficit and achieved a 1.089% statistically
resolved Rust win under the same locked runner. This baseline remains retained
as the pre-optimization record; it was not overwritten.
