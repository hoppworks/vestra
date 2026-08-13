# ADR 0004: CUDA is an Engine backend, not a Studio or pipeline shortcut

## Status

Accepted for the next backend milestone; not implemented.

## Context

The locked Workhorse has an NVIDIA GeForce RTX 5080 with CUDA 12.0 support.
Vestra Engine currently owns a `CpuBackend` directly and the graph executor,
weights, activation layout, and fixed-shape kernels are CPU-resident. Vestra
Studio and `vestra-core` consume Engine outputs only.

Calling the C++ PR implementation, a Python CLI, or a GPU viewer from Rust
would make a useful demo but would not establish a native Vestra CUDA backend
or support a fair Vestra-vs-reference speed claim.

## Decision

Implement CUDA below the Engine public inference API, with no CUDA types in
`vestra-core`, `.vestra` scene bundles, CLI quality profiles, or Studio.

The implementation order is:

1. Introduce an Engine-owned backend selection API and a typed device-resident
   tensor/weight store while retaining the CPU F32 output API.
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
