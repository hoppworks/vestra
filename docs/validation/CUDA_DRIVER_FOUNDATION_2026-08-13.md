# Native CUDA Driver Foundation — 2026-08-13

## Scope

This record validates the first native CUDA boundary in Vestra Kernels. It
does **not** validate GPU inference, multiview parity, or an end-to-end speed
claim.

## Locked implementation

| Item | Value |
| --- | --- |
| Repository | `https://github.com/hoppworks/vestra-kernels` |
| Revision | `2e4c31faf43991523ca378ff30785cdce17b20ac` |
| Feature | `cuda` |
| CUDA binding | `cudarc 0.19.9`, Driver API with dynamic loading |
| Host | Workhorse — AMD Ryzen 9 9950X, NVIDIA GeForce RTX 5080 |
| NVIDIA driver | `610.43.03` |
| CUDA driver support | 12.0 |

## Verified operation

From a fresh checkout of the locked Kernels revision on the Workhorse:

```sh
VESTRA_CUDA_TEST=1 cargo test --lib --features cuda \
  driver_round_trip_preserves_f32_values_when_explicitly_enabled -- --nocapture
```

The test created CUDA device 0 through the Driver API, uploaded `[1.0, -2.0,
3.5]` as F32, downloaded it on the ordered default stream, and verified the
exact values. It passed on the RTX 5080.

## Boundary and next gate

`CudaRuntime` and `CudaTensorF32` live exclusively in Vestra Kernels behind
the opt-in `cuda` feature. No Engine inference operation currently calls them;
therefore this evidence must not be used as a GPU inference or performance
result.

The next implementation gate is an Engine-owned backend selection path plus
one fixed-shape CUDA operator. That operator must have a CPU F32 oracle and
the locked DA3-BASE shape parity fixture before it is used by inference.
