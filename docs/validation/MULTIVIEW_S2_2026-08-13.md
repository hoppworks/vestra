# Multi-view S=2 oracle run — 2026-08-13

## Status

**Not accepted.** This is a reproducible integration finding, not a quality or
performance claim. Vestra must not enable world fusion, stitching, or product
marketing based on this result.

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
- Vestra Engine revision: `4c74e1a0dc85cc3db379683c8b0c0f51a7499918`.
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

## Result

| View | Depth r | Depth MAE | W2C MAE | Intrinsics max error | Accepted |
|---|---:|---:|---:|---:|---|
| 0 | 0.99999034 | 0.00965160 | 0.00000327 | 1.05414 px | No |
| 1 | 0.99996033 | 0.00021448 | 0.00428749 | 6.54682 px | No |

The safe generic Flash path produces materially identical figures, so the
remaining discrepancy is not caused by the packed-QT8 crash fix.

## Safety finding

The previous default `packed QT8` AVX-512 Flash implementation crashed in the
actual S=2 route (`SIGSEGV`, 12/12 quick runs). A GDB trace placed the fault in
`vestra_kernels::specialized::flash_attention_avx512` on a Rayon worker.
`bde1989` makes that experimental implementation opt-in through
`DA3_KERNELS_FLASH_PACKED_QT8=1`; normal inference uses the validated generic
tile path and completed repeated S=2 runs.

## Next debugging seam

Matching post-block tensors now establish that the numerical drift begins at
block 0 (1,328,640 values; MAE `0.00246201`; max `0.0550864`) and grows
monotonically to block 11 (MAE `0.11778764`; max `7.1549683`). This rules out a
late head-only or output-ordering defect. The next authorised target is the
first block's operator sequence: LayerNorm, QKV projection, QK norm/RoPE,
attention, attention projection, FC1/GELU, and FC2/residual.
