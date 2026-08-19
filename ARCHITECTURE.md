# Vestra architecture

This document describes the current system boundary and the publication
rules. It is intentionally more precise than the product vision: a branch
that is experimental or rejected is not part of the default pipeline.

## Repository boundaries

Vestra is three independently versioned repositories:

- **`vestra`** owns capture ingestion, reconstruction jobs, scene bundles,
  fusion, exports, CLI behavior, and browser Studio.
- **`vestra-engine`** owns DA3 tensor semantics and the ordered multi-view
  model implementation.
- **`vestra-kernels`** owns narrow tensor and numerical kernels. It has no
  knowledge of videos, scenes, GGUF metadata, or product jobs.

The dependency revisions are pinned. Cross-repository compatibility is a
release concern, not an implicit local path contract. The accepted boundary
decision is recorded in
[`docs/adr/0001-repository-boundaries.md`](docs/adr/0001-repository-boundaries.md).

## Accepted local pipeline

```text
landscape video
  -> FFmpeg RGB24 candidate decode (currently 8 fps, safety-capped)
  -> deterministic geometry-keyframe selection
  -> 12-frame / 3-frame-overlap multi-view windows
  -> Vestra Engine DA3 depth, confidence, intrinsics, and local poses
  -> calibrated back-projection and confidence filtering
  -> robust relative Sim(3) window alignment
  -> optional verified revisit measurement and pose-graph optimization
  -> deterministic surfel/voxel fusion or normal-space TSDF
  -> immutable measured/fused `.vestra` layers
  -> PLY, GLB, `.splat`, camera evidence, and browser Studio
```

The candidate decode and keyframe selector preserve source candidate indices
and policy fingerprints. A resumed job cannot silently reuse rasters made by a
different selection policy. The keyframe selector controls image evidence;
it is not a pose estimator and does not establish parallax or global scale.

The local product is relative-scale. Sim(3) alignment may make a coherent
local world, but it does not assert metres. An independently verified scale
anchor is required before a scene can be labelled metric; the accepted v1
decision is in
[`docs/adr/0002-relative-scale-v1.md`](docs/adr/0002-relative-scale-v1.md).

## Model and geometry ownership

Vestra Engine exposes the ordered multi-view contract needed by the pinned PR
#2 reference: per-view local blocks, flattened global attention with explicit
reference/source camera tokens, repeated per-view RoPE boundaries, and
restoration of per-view outputs. Vestra Kernels exposes the primitives used by
that contract. `vestra-core` owns reconstruction state and geometry, but
does not reach into model or kernel internals.

Adjacent windows are aligned from direct shared-pixel evidence with quality
and degeneracy gates. A pose graph is considered only when an independently
verified revisit edge exists; otherwise sequential stitching remains the
defined result. Raw windows remain available after alignment and fusion.

Normal-space TSDF is implemented as a deterministic fused derivative. Its
semantics and tolerance gate are recorded in
[`docs/validation/TSDF_ORACLE_2026-08-14.md`](docs/validation/TSDF_ORACLE_2026-08-14.md).
TSDF reduces surface duplication; it does not repair a bad camera trajectory,
create a mesh, or prove global room geometry.

## Optional global-pose and dense-MVS branch

Global providers are additive, never silent replacements for the local world.
They consume the exact immutable raster manifest and emit a versioned pose
solution. Vestra validates the provider, raster fingerprint, rigid W2C camera
matrices, registration coverage, and every window's camera-fit residual before
allowing global fusion. Missing frames are not interpolated to make the gate
pass. The sidecar contract is documented in
[`docs/providers/GLOBAL_POSE_SIDECAR.md`](docs/providers/GLOBAL_POSE_SIDECAR.md).

The pinned COLMAP path can also supply a dense-MVS control. MVS geometry and
DA3 geometry remain separately labelled; an MVS-guided derivative may replace
depth only where its evidence exists. The current COLMAP and provider studies
show why these outputs are research derivatives: registration fragments and
residual failures correctly prevent global publication.

## Scene storage and durability

A `.vestra` bundle is a directory of immutable, content-addressed artifacts.
Measured window chunks, raster manifests, pose solutions, fused payloads, and
progressive binary point chunks are hashed before publication. The manifest is
atomically replaced only after referenced chunks are durable. Fusion never
overwrites raw evidence, and repeating the same derivation is idempotent.

Job identity includes the input, model, engine and kernel revisions, and
normalized settings. Completed windows are durable checkpoints. Cancellation
and resume operate at those atomic boundaries, with provenance mismatch
refused rather than guessed. The real-video evidence and remaining browser
validation boundary are summarized in
[`docs/validation/REAL_VIDEO_IMG_2269_2026-08-13.md`](docs/validation/REAL_VIDEO_IMG_2269_2026-08-13.md)
and
[`docs/validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md`](docs/validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md).

## Product and lab surfaces

The product binary exposes the small path a user can rely on: local app,
reconstruction, serving, inspection, and exports. `vestra-lab` exposes oracle
fixtures, benchmark runners, pose imports, dense-MVS imports, and experimental
derivatives. The binaries reject commands from the other surface; this prevents
a research command from becoming an accidental product promise.

## Benchmark boundary

Performance claims are workload-specific. The locked studies use an AMD Ryzen
9 9950X; the available validation machine also has an RTX 5080 and 96 GiB of
RAM. The canonical claims are:

| Workload | C++ reference | Vestra Rust | Samples | Scoped result |
| --- | ---: | ---: | ---: | --- |
| DA3-BASE single-image CPU F32 | 238.789 ms | 171.141 ms | 20 | 28.3% lower latency; 39.5% higher throughput |
| PR #2 multi-view model CPU F32 | 8588.277 ms | 8494.734 ms | 30 | 1.089% lower wall time |
| PR #2 geometry + TSDF, model-free | 867.421 ms | 831.797 ms | 10 | 4.11% lower wall time |

The multi-view study is documented in
[`docs/benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md`](docs/benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md);
the geometry-plus-TSDF study is documented in
[`docs/benchmarks/2026-08-14-pr2-geometry/RESULTS.md`](docs/benchmarks/2026-08-14-pr2-geometry/RESULTS.md).
These are not stage speedups that can be added together. None is an
end-to-end video, GPU, browser, or complete product-world performance claim.
