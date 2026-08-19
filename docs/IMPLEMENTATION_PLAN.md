# Vestra implementation plan

This plan separates accepted foundations from work that still needs evidence.
The default path remains local, relative-scale, and CPU F32; research branches
must not widen the product promise without their own gates.

## 1. Repository and model contract — complete

- Keep `vestra`, `vestra-engine`, and `vestra-kernels` independently
  versioned.
- Pin compatible revisions in the Vestra workspace.
- Complete the ordered multi-view local/global transformer contract and camera
  token semantics.
- Preserve numerical parity against the pinned PR #2 F32 oracle for the
  accepted S=2, S=3, and S=12 model cases.

Evidence: [ADR 0001](adr/0001-repository-boundaries.md),
[multi-view validation](validation/MULTIVIEW_S2_2026-08-13.md), and
[multi-view benchmark](benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md).

## 2. Local capture and relative reconstruction — baseline complete

- Decode a landscape video to RGB24 candidates at a fixed rate (currently
  8 fps) under a high safety ceiling.
- Select geometry keyframes from temporal baseline, luma novelty, sharpness,
  and a maximum gap; retain first and final candidates.
- Run the PR #2-style 12-frame / 3-frame-overlap local pipeline.
- Persist calibrated depth evidence, confidence, relative camera transforms,
  robust Sim(3) seams, and deterministic fusion.
- Keep the scene explicitly relative-scale.

The first real-video path, durable checkpoints, and local relative-world
evidence exist. The remaining product-quality work is not to restore a fixed
total-frame contract; it is to validate adaptive selection on representative
captures and tune only against measured geometry quality.

Evidence: [ADR 0002](adr/0002-relative-scale-v1.md),
[ADR 0005](adr/0005-candidate-rate-and-geometry-keyframes.md), and
[real-video validation](validation/REAL_VIDEO_IMG_2269_2026-08-13.md).

## 3. Durable scene and TSDF layers — implemented; product hardening ongoing

- Store measured and derived layers as immutable, content-addressed chunks.
- Publish manifests atomically and preserve raw evidence during re-fusion.
- Support deterministic surfel/voxel fusion and normal-space TSDF derivatives.
- Export open PLY, GLB, `.splat`, and camera-evidence artifacts.
- Keep exact TSDF oracle parity as a correctness gate.

Remaining work is a broader regression corpus for capture-dependent quality:
connectedness, drift, surface retention, ghost thickness, and a deliberate
revisit. TSDF is implemented; it is not a substitute for those scene-level
gates.

Evidence: [TSDF oracle](validation/TSDF_ORACLE_2026-08-14.md) and
[closed-loop oracle](validation/CLOSED_LOOP_ORACLE_2026-08-14.md).

## 4. Product CLI and Studio — product/lab split complete

- Keep `vestra` focused on app, reconstruct, demo, serve, inspect, and export.
- Keep `vestra-lab` explicit for oracle, benchmark, pose, MVS, and
  architecture experiments.
- Keep the two binaries mutually exclusive with parser-level regression tests.
- Use one durable job contract for CLI and browser intake.
- Finish real F32 browser validation for cancellation, restart, resume, and
  final provenance on a room video.

The local Studio path and resumable job state exist. The real browser flow on
the locked Workhorse remains an operational validation item, not a simulated
test claim.

Evidence: [end-to-end smoke](validation/END_TO_END_SMOKE_2026-08-13.md) and
[resumable jobs](validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md).

## 5. Global pose and dense MVS — research branch

- Keep the local product selected when global evidence is incomplete.
- Import provider solutions only through the exact raster/pose sidecar
  contract.
- Require registration coverage and a passing per-window residual gate before
  global fusion.
- Evaluate a pinned COLMAP global pose and dense-MVS control as separate,
  provider-labelled products.
- Improve retrieval, correspondence, and global bundle adjustment before
  considering a production global world.

The current COLMAP, DROID-SLAM, hybrid, and VGGT provider attempts are useful
diagnostics but do not satisfy the publication gate for the difficult
`IMG_2323` capture. Dense MVS and DA3-guided derivatives remain additive
controls, not the default world.

Evidence: [global pose sidecar](providers/GLOBAL_POSE_SIDECAR.md),
[provider evaluation](validation/GLOBAL_POSE_PROVIDERS_IMG_2323_2026-08-17.md),
and [MVS hybrid](validation/DA3_MVS_HYBRID_IMG_2323_2026-08-17.md).

## 6. GPU and measured performance — pending production path

- Preserve the CPU F32 parity path as the reference implementation.
- Use the Ryzen 9 9950X / RTX 5080 / 96 GiB Workhorse for reproducible studies.
- Profile before moving geometry or inference phases to GPU.
- Establish numerical parity and a complete protocol before any CUDA
  throughput claim.
- Report model, geometry, and end-to-end measurements separately.

The canonical current results are 171.141 ms versus 238.789 ms for the
single-image CPU F32 path (N=20), 8494.734 ms versus 8588.277 ms for the
multi-view model path (N=30), and 831.797 ms versus 867.421 ms for the
model-free geometry-plus-TSDF fixture (N=10). These claims are narrowly
scoped and must never be summed into an end-to-end speedup.

## 7. Deferred product goals

Metric scale, semantic architecture, generated/unseen geometry, a production
GPU backend, cinematic rendering, and a universally coherent global room
model remain deferred. Each requires an explicit evidence contract and a
separate acceptance record before entering the product CLI.
