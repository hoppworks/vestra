# Multi-view S=2 oracle run — 2026-08-13

## Status

**Accepted with canonical input.** The initial JPEG run is retained below as a
decoder-boundary finding; it is not a model-inference mismatch. The accepted
contract supplies both implementations with identical FFmpeg-decoded PPM RGB
frames, exactly as Vestra's video pipeline already does.

## Environment

- Host: AMD Ryzen 9 9950X Workhorse.
- Reference source: `localai-org/depth-anything.cpp` PR #2 merge commit
  `2028b47ac75a8659c6a9aa617baf09be193eb55f`; its CLI working tree contained
  the existing timing-only modification to `examples/cli/main.cpp`.
- Oracle source contract: PR head
  `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`.
- Model: `depth-anything-base-f32.gguf`, 412,110,144 bytes.
- Inputs: PR fixtures `canyon.jpg` and `desk.jpg`, each 1024×680.
- Model output: 504×336, two views.
- Rust compilation: `RUSTFLAGS="-C target-cpu=znver5"`, release.
- Vestra Engine revision: `1562f8b70a1b35a9908feb88eaa38577b92f2a2a`.
- Vestra Kernels safe revision: `bde198958348fcb7a0a294e0d05cd8f2f7e93c5b`.

## Commands

```bash
da3-cli depth --model depth-anything-base-f32.gguf \
  --input canyon.jpg --input desk.jpg --out-prefix cpp_s2

vestra-engine infer-multi --model depth-anything-base-f32.gguf \
  --image canyon.jpg --image desk.jpg --out-prefix rust_s2

python3 scripts/compare_multiview_oracle.py \
  --cpp-prefix cpp_s2 --rust-prefix rust_s2 --views 2 --output s2-report.json
```

The acceptance gates are depth Pearson `r >= 0.9999`, depth MAE `<= 0.005`,
W2C extrinsics MAE `<= 0.005`, and intrinsics max absolute error `<= 1.5`
pixels for every view.

## Initial JPEG finding

| View | Depth r | Depth MAE | W2C MAE | Intrinsics max error | Accepted |
|---|---:|---:|---:|---:|---|
| 0 | 0.99999034 | 0.00965160 | 0.00000327 | 1.05414 px | No |
| 1 | 0.99996033 | 0.00021448 | 0.00428749 | 6.54682 px | No |

The safe generic Flash path produces materially identical figures, so the
remaining discrepancy was not caused by the packed-QT8 crash fix. Block traces
then showed the first mismatch at the input patch tokens while CLS/position
tokens were bit-identical. Different JPEG decoders produced different input
RGB bytes.

## Accepted PPM results

The same source JPG fixtures were decoded once with FFmpeg (`rgb24`) to PPM;
those PPM files were passed unchanged to both CLIs. All values below satisfy
the locked gates for **every view**.

| Window | Views | Worst depth r | Worst depth MAE | Worst W2C MAE | Worst intrinsics error |
|---|---:|---:|---:|---:|---:|
| S=2 | 2 | 0.999999999982 | 0.0000015403 | 0.0000019950 | 0.003965 px |
| S=3 | 3 | 0.999999999985 | 0.0000026587 | 0.0000082050 | 0.004754 px |
| S=12 | 12 | 0.999999999741 | 0.0000211174 | 0.0000199768 | 0.032229 px |

S=3 exercises the automatic saddle-balanced reference selection. S=12 uses a
fixed ordered, FFmpeg-decoded schedule of `canyon`, `desk`, `mountains`, and
`street`, repeated three times; this tests the actual 10,380-token global
attention shape without conflating JPEG decoder differences with inference.

## Safety finding

The previous default `packed QT8` AVX-512 Flash implementation crashed in the
actual S=2 route (`SIGSEGV`, 12/12 quick runs). A GDB trace placed the fault in
`vestra_kernels::specialized::flash_attention_avx512` on a Rayon worker.
`bde1989` makes that experimental implementation opt-in through
`DA3_KERNELS_FLASH_PACKED_QT8=1`; normal inference uses the validated generic
tile path and completed repeated S=2 runs.

## Follow-up requirement

The canonical PPM decode boundary is now part of every future parity and
benchmark fixture. Do not compare JPEG-decoding-inclusive timing or output
bytes across different decoder libraries and describe it as an inference
comparison.
