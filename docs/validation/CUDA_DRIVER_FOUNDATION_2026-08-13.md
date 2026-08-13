# Native CUDA Driver Foundation — 2026-08-13

## Scope

This record validates the first native CUDA boundary in Vestra Kernels. It
does **not** validate GPU inference, multiview parity, or an end-to-end speed
claim.

## Locked implementation

| Item | Value |
| --- | --- |
| Repository | `https://github.com/hoppworks/vestra-kernels` |
| Revision | `688e2905f7bb8e4c9130e3d6faf5f8c4b1723508` |
| Feature | `cuda` |
| CUDA binding | `cudarc 0.19.9`, Driver API with dynamic loading and NVRTC |
| Host | Workhorse — AMD Ryzen 9 9950X, NVIDIA GeForce RTX 5080 |
| NVIDIA driver | `610.43.03` |
| CUDA driver support | 12.0 |

## Verified operation

The Workhorse intentionally has the NVIDIA Driver but no system-wide NVRTC.
The isolated runtime compiler is installed at
`/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib/libnvrtc.so.12`; it is
supplied only through `LD_LIBRARY_PATH` for native kernel development.

The test created CUDA device 0 through the Driver API, compiled and loaded the
native `vestra_add_f32` kernel, uploaded `[1.0, -2.0, 3.5]` and
`[0.5, 2.0, -1.0]`, executed device-resident addition, then downloaded and
exactly verified `[1.5, 0.0, 2.5]`. It passed on the RTX 5080:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib \
VESTRA_CUDA_TEST=1 cargo test --lib --features cuda \
  driver_round_trip_preserves_f32_values_when_explicitly_enabled -- --nocapture
```

## Boundary and next gate

`CudaRuntime` and `CudaTensorF32` live exclusively in Vestra Kernels behind
the opt-in `cuda` feature. `add_f32_in_place` is the first native,
device-resident residual building block. No Engine inference operation
currently calls it; therefore this evidence must not be used as a GPU
inference or performance result.

The first Engine-owned fixed-shape CUDA operator is recorded below. Every
subsequent CUDA operator must likewise have a CPU F32 oracle and the locked
DA3-BASE shape parity fixture before it is used by inference.

## Engine integration evidence

Vestra Engine revision `c07e08dbf03b17d19bd0b341d21da228283ec156` enables
`cuda-residual-oracle`. Its integration test loads the same BASE-F32 GGUF
twice, executes CPU inference once and an inference whose 24 transformer
residual additions run through native CUDA once, then compares depth and
confidence.

On the Workhorse it passed for `mountains.jpg` with the thresholds MAE
`<= 1e-6` and maximum absolute error `<= 1e-5` for both outputs:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib \
VESTRA_CUDA_MODEL=/var/roothome/da3-bench/models/depth-anything-base-f32.gguf \
VESTRA_CUDA_IMAGE=/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/mountains.jpg \
RUSTFLAGS='-C target-cpu=znver5' \
cargo test -p vestra-engine --features cuda-residual-oracle \
  cuda_residual_slice_matches_cpu_depth_and_confidence -- --nocapture
```

The test took 19.79 seconds in a debug test build. That is expected and is
not a timing study: each residual currently crosses PCIe in both directions.
The next CUDA slice must keep adjacent activations and weights resident on
device before a GPU performance benchmark is meaningful.
