# Final 30/30 Multi-view Model Study — 2026-08-14

## Scope and locked protocol

This final study evaluates the accepted AVX-512 multi-view Flash attention
route from Vestra Kernels commit `85c9dc2` against pinned
depth-anything.cpp PR #2 commit `f56e9be`.

Both arms use the identical `depth-anything-base-f32.gguf` model, 24 canonical
RGB24 PPM frames, a 504×336 raster, the PR #2 12-frame / 3-frame-overlap
schedule (three genuine multi-view windows), and 16 CPU threads on an AMD
Ryzen 9 9950X. Each fresh process loads the model and canonical inputs before
timing, performs one excluded warmup, then one measured complete model
forward. Model loading, PPM decoding, input-window assembly, and output
serialization are excluded in both runners.

The input, precision, frame count, windowing, threads, and measured work are
unchanged from the baseline. No samples were removed.

## Results

| Arm | n | Mean (ms) | Median (ms) | SD (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | --: | ---: |
| C++ PR #2 F32 | 30 | 8588.277 | 8586.590 | 43.751 | [8571.942, 8604.612] |
| Vestra Rust F32 | 30 | 8494.734 | 8500.278 | 75.923 | [8466.387, 8523.081] |

Vestra Rust is **1.089% faster** by mean wall time (`0.989108×` C++ time), a
gain of **93.543 ms** per complete 24-frame multi-view run.

The independently-sampled Welch difference is C++ minus Rust = `93.543 ms`,
SE `15.998 ms`, Welch df `46.35`, and 95% CI **[61.338, 125.748] ms**. The
interval excludes zero, so this establishes a statistically resolved Rust
speed win under the locked model-bound workload. Individual arm mean intervals
may overlap; that is not a valid test of an independent difference.

## Correctness gates

- The new kernel is constrained to a positive integral number of DA3-BASE
  865-token views and has an AVX-512 two-view online-softmax reference test.
- The real 24-frame `MVO1` differential remains at depth MAE `1.67e-6`,
  confidence MAE `3.27e-5`, extrinsics MAE `5.42e-7`, and intrinsics MAE
  `1.22e-4`; shapes and all three windows match the C++ model output.
- The pre-acceptance same-binary A/B reduced the old global-attention fallback
  from `14,117.368 ms` to `8,485.674 ms` (`39.89%`).

## Raw-data integrity

[`raw.jsonl`](raw.jsonl) contains all 60 fresh-process samples in execution
order. A detected initial 11-C++/9-Rust scheduling imbalance was corrected
only by appending fresh measurements, first to 12/12 and then through a fixed
18/18 random extension to 30/30. No value was replaced, selected, or removed.
Exact source, kernel, binary, model hashes, and the correction history are in
[`metadata.json`](metadata.json).

## Interpretation boundary

This is a CPU F32 **model-bound multi-view** result. It does not claim a GPU,
quantized-model, video decode, geometry-only, or browser end-to-end advantage.
