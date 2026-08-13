# Native CUDA Driver Foundation — 2026-08-13

## Scope

This record validates the first native CUDA boundary in Vestra Kernels. It
does **not** validate GPU inference, multiview parity, or an end-to-end speed
claim.

## Locked implementation

| Item | Value |
| --- | --- |
| Repository | `https://github.com/hoppworks/vestra-kernels` |
| Revision | `f3b0df00555e8132b2878568e9d374d05f3b090d` |
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
device-resident residual building block. The separate opt-in Engine parity
slice below calls it, but the normal Engine path does not; therefore this
evidence must not be used as a GPU performance result.

Kernels revision `f3b0df00555e8132b2878568e9d374d05f3b090d` additionally
provides device-resident F32 row-major GEMM through dynamically loaded
CUBLAS, a native linear epilogue, and the same exact-erf GELU approximation
used by the CPU path. Its Workhorse oracle verified the hand-computed `2×3 ×
3×2` product `[58, 64, 139, 154]`, followed by `(C + bias) × gamma` with
`bias=[1,-4]`, `gamma=[0.5,2]`, yielding `[29.5,120,70,300]` exactly. It also
compared device GELU at `[-3,-1,0,1,3]` with the CPU reference at an absolute
tolerance of `2e-6`. The implementation invokes CUBLAS through the
equivalent column-major identity `Cᵀ = Bᵀ × Aᵀ`, avoiding a host-side
transpose. CUBLAS is isolated under
`/var/roothome/vestra-cuda-deps/nvidia/cublas/lib` alongside NVRTC.

The same test now builds two cached device linear plans and chains them:
`[2,3] → [2,2] → [2,1]`. Both weights, biases, and scales are uploaded only
when a plan is prepared; the first projection output stays on the device as
the second projection input. The final exact result is `[-58,-157]`. This is
the first verified device-resident multi-operator chain, not an end-to-end
inference or performance claim.

The first Engine-owned fixed-shape CUDA operator is recorded below. Every
subsequent CUDA operator must likewise have a CPU F32 oracle and the locked
DA3-BASE shape parity fixture before it is used by inference.

## Engine integration evidence

Vestra Engine revision `df6f8038e8bd1e44fcb904be0a07775c551a4439` enables
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

The same fixture also covers `infer_multi_view_ordered` with distinct
`canyon.jpg` and `desk.jpg` views. It compares every returned depth and
confidence map under the same numerical bounds, exercising PR #2's real
local/global multi-view backbone control flow with CUDA residual execution.
