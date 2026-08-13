# Native CUDA Driver Foundation — 2026-08-13

## Scope

This record validates the first native CUDA boundary in Vestra Kernels. It
does **not** validate GPU inference, multiview parity, or an end-to-end speed
claim.

## Locked implementation

| Item | Value |
| --- | --- |
| Repository | `https://github.com/hoppworks/vestra-kernels` |
| Revision | `48bc7b5b8fcd6b328e63c99bd7610cd30bb44f01` |
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

Kernels revision `e078b562d6e23d3a213b6a5858f54688c9fad6d3` additionally
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

It also provides two CUDA LayerNorm routes. The parallel fixed-geometry
version uses one 256-thread block per DA3-BASE token row and has a standalone
two-row F32 error envelope `<= 2e-5`. It is deliberately **not** used by the
Engine: in ordered multiview it produced depth MAE `1.1371911e-6`, exceeding
the locked `1e-6` end-to-end limit. The accepted Engine route is the strict
CPU-order oracle: one GPU thread per row with ascending column reductions,
which preserves the reference numerical sequence.

## CUDA attention oracle

Kernels revision `48bc7b5b8fcd6b328e63c99bd7610cd30bb44f01` adds a
device-resident F32 online-softmax attention primitive for head-major
`[heads,tokens,head_dim]` tensors with `head_dim <= 64`. It maps one GPU
thread to each `(head, query)` row and uses the same 64-key online-softmax
state machine as the CPU path: running maximum, correction, running sum, and
value accumulator. The Workhorse RTX 5080 test compares a two-head, five-token
case with the CPU attention oracle under an absolute error envelope `<= 5e-5`.
It additionally passed the actual DA3-BASE geometry—12 heads × 865 tokens ×
64 dimensions—against Vestra's production CPU attention path on the RTX 5080
in 0.39 seconds, with MAE `<= 5e-5` and maximum error `<= 5e-4`.

It is intentionally not wired into the Engine yet. DA3 has a specialised
AVX-512 CPU attention path and optional Q/K normalization plus RoPE; the next
gate is a dedicated actual-shape attention parity fixture, followed by the
complete single-view and ordered-multiview Engine gates. No GPU speed claim is
made from this small oracle.

The first Engine-owned fixed-shape CUDA operator is recorded below. Every
subsequent CUDA operator must likewise have a CPU F32 oracle and the locked
DA3-BASE shape parity fixture before it is used by inference.

## Engine integration evidence

Vestra Engine revision `9046c209565419d4b89266018659ab7db9748ba2` enables
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

## Cached CUDA MLP oracle

The same opt-in Engine feature now exposes a distinct `cuda_mlp_oracle`.
During enablement it uploads and retains strict LayerNorm, FC1/FC2, bias, and
LayerScale parameters for all 12 DA3-BASE blocks. For every MLP branch it
uploads the pre-MLP token tensor once, performs strict LayerNorm → FC1 →
bias/scale → exact-erf GELU → FC2 → bias/LayerScale entirely on the GPU, and
downloads only the FC2 result for the still-CPU residual. Thus the normalized
865×768 tokens and 865×3072 FC1 activation never cross PCIe. This is an
integration/parity result, not an end-to-end GPU performance claim: attention
and residuals still run on CPU in this mode.

On the Workhorse, both strict real-model gates passed with depth and
confidence MAE `<= 1e-6` and maximum absolute error `<= 1e-5`:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib:/var/roothome/vestra-cuda-deps/nvidia/cublas/lib \
VESTRA_CUDA_MODEL=/var/roothome/da3-bench/models/depth-anything-base-f32.gguf \
VESTRA_CUDA_IMAGE=/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/mountains.jpg \
RUSTFLAGS='-C target-cpu=znver5' \
cargo test -p vestra-engine --features cuda-residual-oracle \
  cuda_mlp_slice_matches_cpu_depth_and_confidence -- --nocapture
```

The ordered two-view gate used `canyon.jpg:desk.jpg` through
`VESTRA_CUDA_IMAGES` and passed in 37.45 seconds in a debug test build using
the strict CPU-order LayerNorm route:

```sh
cargo test -p vestra-engine --features cuda-residual-oracle \
  cuda_mlp_slice_matches_cpu_ordered_multiview -- --nocapture
```

## CUDA Q/K normalization and RoPE oracle

Vestra Kernels revision `22cdd0e` adds the locked DA3-BASE-shape parity gate
for the device-resident fused Q/K normalization and two-dimensional RoPE
operator. The test uses head-major Q and K tensors at 12 heads × 865 tokens ×
64 dimensions, the production 24 × 36 token-position layout, and the
production CPU operator as its oracle. It verifies Q and K independently with
MAE `<= 5e-5` and maximum absolute error `<= 5e-4`.

On the Workhorse RTX 5080 it passed in 0.33 seconds in a debug test build:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib:/var/roothome/vestra-cuda-deps/nvidia/cublas/lib \
VESTRA_CUDA_DA3_QK_ROPE_TEST=1 \
cargo test --lib --features cuda \
  cuda_qk_norm_rope_matches_production_da3_base_shape -- --nocapture
```

This closes the isolated numerical gate for the attention input preparation.
It does not yet make CUDA attention an Engine execution path; that next slice
must retain Q, K, V, Q/K normalization, RoPE, attention, and attention
projection on the device before it can make an end-to-end speed claim.

## Cached CUDA attention Engine slice

Vestra Engine revision `b4a8f50` wires the complete qualified attention
branch into inference: cached QKV weights, device QKV-to-head-major layout,
Q/K normalization and RoPE, online attention, device head-major-to-token
layout, and cached output projection. The CPU LayerNorm-1 input and the
projected branch output remain explicit host boundaries; therefore this is a
numerical integration gate and not a CUDA throughput claim.

The actual single-view `mountains.jpg` gate and the ordered PR #2 multi-view
gate with `canyon.jpg:desk.jpg` both passed on the RTX 5080 under the locked
depth/confidence thresholds (MAE `<= 1e-6`, maximum absolute error `<= 1e-5`).
The multi-view run exercised local per-view attention and flattened global
attention and completed in 38.83 seconds in a debug test build:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib:/var/roothome/vestra-cuda-deps/nvidia/cublas/lib \
VESTRA_CUDA_MODEL=/var/roothome/da3-bench/models/depth-anything-base-f32.gguf \
VESTRA_CUDA_IMAGES=/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/canyon.jpg:/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/desk.jpg \
RUSTFLAGS='-C target-cpu=znver5' \
cargo test -p vestra-engine --features cuda-residual-oracle \
  cuda_attention_slice_matches_cpu_ordered_multiview -- --nocapture
```

The next integration step is a single device-resident transformer-block
executor that combines the qualified attention and MLP branches with both
residual additions, eliminating their current host hand-off.

## Connected CUDA transformer tail

Vestra Engine revision `05c8717` adds that connected parity slice for the
blocks that have DA3 Q/K normalization and RoPE (the locked BASE model starts
them at block 4). It performs QKV through output projection, the first
residual, strict CPU-order LN2, FC1/GELU/FC2, and the second residual on one
CUDA stream. Attention results, residual states, and FC1 activations never
cross PCIe; the CPU-order LN1 input and the final token state remain explicit
per-block host boundaries.

It passed the locked single-view and `canyon.jpg:desk.jpg` ordered multiview
depth/confidence gates on the Workhorse RTX 5080. The multiview gate completed
in 32.12 seconds in a debug test build:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib:/var/roothome/vestra-cuda-deps/nvidia/cublas/lib \
VESTRA_CUDA_MODEL=/var/roothome/da3-bench/models/depth-anything-base-f32.gguf \
VESTRA_CUDA_IMAGES=/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/canyon.jpg:/var/roothome/da3-bench/src/depth-anything.cpp/assets/samples/desk.jpg \
RUSTFLAGS='-C target-cpu=znver5' \
cargo test -p vestra-engine --features cuda-residual-oracle \
  cuda_transformer_tail_matches_cpu_ordered_multiview -- --nocapture
```

The next CUDA work is therefore not another attention variant: it is an
entire device-resident backbone lifetime (LN1 and token ping-pong buffers)
followed by qualified DPT/pose-head operators.

## Device-side patch lowering

Vestra Kernels revision `493762a` adds the first input-side building block for
that lifetime: `patchify_nchw_f32`. It lowers an NCHW image into row-major,
non-overlapping patch rows in the exact `[channel, patch_y, patch_x]` order
consumed by the model's OIHW patch-projection weights. The output is ready for
a cached CUBLAS projection and avoids a host-side im2col allocation.

This is a layout-only kernel, not a CUDA speed claim and not yet wired into
Engine inference. Its explicit 3-channel, 4×4, patch-2 oracle passed on the
RTX 5080:

```sh
LD_LIBRARY_PATH=/var/roothome/vestra-cuda-deps/nvidia/cuda_nvrtc/lib:/var/roothome/vestra-cuda-deps/nvidia/cublas/lib \
VESTRA_CUDA_TEST=1 RUSTFLAGS='-C target-cpu=znver5' \
cargo test --lib --features cuda \
  driver_round_trip_preserves_f32_values_when_explicitly_enabled -- --nocapture
```

Vestra Engine revision `388167c` consumes the kernel through a cached CUBLAS
patch-projection plan and assembles DA3-BASE's CLS/position token sequence on
the GPU before its temporary download to the CPU backbone. It passed both a single-image and the ordered `canyon.jpg:desk.jpg`
multiview end-to-end Depth/Confidence oracle on the RTX 5080. CUBLAS changes
the F32 reduction tree, so this narrow seam has its own explicit bound: MAE
`<= 5e-6`, maximum absolute error `<= 1e-4`. This remains far tighter than
the product F32 parity gate (Pearson `>= 0.9999`, MAE `<= 0.005`) and does not
alter that gate.

The next dependent slice is device-side CLS/position-token assembly followed
by a device-resident backbone token lifetime. It must retain equivalent
CPU-F32 oracle coverage before it can be treated as a production backend.
