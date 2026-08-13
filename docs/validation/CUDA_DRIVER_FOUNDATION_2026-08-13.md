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

From a fresh checkout of the locked Kernels revision on the Workhorse:

```sh
VESTRA_CUDA_TEST=1 cargo test --lib --features cuda \
  driver_round_trip_preserves_f32_values_when_explicitly_enabled -- --nocapture
```

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

The next implementation gate is an Engine-owned backend selection path plus
one fixed-shape CUDA operator. That operator must have a CPU F32 oracle and
the locked DA3-BASE shape parity fixture before it is used by inference.
