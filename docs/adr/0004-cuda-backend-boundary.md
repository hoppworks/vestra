# ADR 0004: CUDA is an Engine backend, not a Studio or pipeline shortcut

## Status

Accepted. Native CUDA now covers transfer-bound, real-model parity slices for
patch projection and the qualified transformer tail; no production Engine
inference backend is routed to CUDA yet.

## Context

The locked Workhorse has an NVIDIA GeForce RTX 5080 with CUDA 12.0 support.
Vestra Engine currently owns a `CpuBackend` directly and the graph executor,
weights, activation layout, and fixed-shape kernels are CPU-resident. Vestra
Studio and `vestra-core` consume Engine outputs only.

Vestra Kernels revision `ee925619d2cba3780a4af8daeb88e38671c890dd` provides
an opt-in `cuda` feature backed by dynamically loaded CUDA Driver API calls.
It owns a selected device context, its default ordered stream, and explicit
F32 host-to-device/device-to-host transfers. This was exercised on the
Workhorse GPU. It also carries a device-resident F32 residual-add kernel.

Engine revision `530c7ad26e9987963300677561305b330ef6678a` adds an opt-in
`cuda-residual-oracle` feature with cached patch projection and a qualified
transformer-tail core that can accept and return device-owned tokens. The
currently routed adapter still has explicit activation uploads/downloads. The
mode exists to establish Engine integration and F32 parity; it is deliberately
not exposed as a performance backend.

Calling the C++ PR implementation, a Python CLI, or a GPU viewer from Rust
would make a useful demo but would not establish a native Vestra CUDA backend
or support a fair Vestra-vs-reference speed claim.

## Decision

Implement CUDA below the Engine public inference API, with no CUDA types in
`vestra-core`, `.vestra` scene bundles, CLI quality profiles, or Studio.

The implementation order is:

1. Introduce an Engine-owned backend selection API and promote the typed
   device-resident tensor/weight store from the Kernels foundation while
   retaining the CPU F32 output API.
2. Port and oracle-test preprocessing, patch embedding, linear/QKV, LayerNorm,
   RoPE, attention, GELU, convolution/Winograd, resize, depth/confidence, and
   pose heads for the locked DA3-BASE shapes.
3. Establish C++ F32 multi-view parity at S=1, 2, 3, and 12 before enabling
   CUDA from Vestra CLI.
4. Add explicit device-transfer accounting and run the benchmark matrix using
   identical model, precision, frames, resolution, window schedule, and GPU.
5. Only then move geometry/fusion stages to GPU where a profile demonstrates
   an end-to-end benefit.

## Consequences

- The current native CPU world pipeline remains the product baseline.
- CPU and CUDA studies are separate benchmark arms; no current CPU result is
  compared to C++ CUDA/Vulkan or Q4 output.
- Any temporary GPU oracle must live in validation tooling and cannot become
  the production inference route.
