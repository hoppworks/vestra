# Vestra project status

Status snapshot: 2026-08-19. This is a curated view for engineers and
reviewers. The linked validation records remain the authority for individual
measurements.

## Executive summary

Vestra has an accepted local-first path from landscape video through DA3
multi-view inference to a relative-scale, browser-viewable world. The path
has durable measured windows, deterministic fusion, implemented TSDF output,
and content-addressed scene artifacts. Separate `vestra` and `vestra-lab`
binaries enforce the product and engineering command surfaces.

The hard unresolved problem is global coherence on difficult, long captures.
Global pose providers and dense MVS are retained as labelled experiments;
their residual and coverage gates correctly prevent them from replacing the
local product. Metric scale, semantic architecture, generated geometry, and a
production GPU performance claim are not current release capabilities.

## Accepted baseline

| Area | Current truth | Evidence |
| --- | --- | --- |
| Repositories | Three independently versioned projects: Vestra product, Vestra Engine, Vestra Kernels | [ADR 0001](adr/0001-repository-boundaries.md) |
| Capture | 8 fps candidate decode, safety ceiling, adaptive geometry keyframes; no fixed total-frame contract | [ADR 0005](adr/0005-candidate-rate-and-geometry-keyframes.md) |
| Local reconstruction | PR #2-style 12/3 multi-view windows, calibrated back-projection, relative Sim(3), deterministic fusion | [real-video validation](validation/REAL_VIDEO_IMG_2269_2026-08-13.md) |
| Scale | Relative only; no metres claim | [ADR 0002](adr/0002-relative-scale-v1.md) |
| Surface fusion | Normal-space TSDF is implemented and passes the pinned fixture oracle | [TSDF oracle](validation/TSDF_ORACLE_2026-08-14.md) |
| Storage | Measured and derived layers are immutable/content-addressed; manifest publication is atomic | [end-to-end smoke](validation/END_TO_END_SMOKE_2026-08-13.md) |
| Delivery | Local browser Studio serves progressive scene data and evidence; real F32 restart/resume browser validation remains pending | [resumable jobs](validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md) |
| Command surfaces | `vestra` exposes six product commands; `vestra-lab` owns oracle, provider, MVS, and architecture experiments | [architecture](../ARCHITECTURE.md) |

## Canonical performance evidence

All numbers below are CPU F32 or model-free CPU studies on the AMD Ryzen 9
9950X Workhorse. The machine also has an RTX 5080 and 96 GiB of RAM. The
comparison arms, model, precision, thread budget, and excluded work are part
of each study.

| Workload | C++ reference (ms) | Vestra Rust (ms) | N | Result |
| --- | ---: | ---: | ---: | --- |
| DA3-BASE single-image model path | 238.789 | 171.141 | 20 | 28.3% lower latency; 39.5% higher throughput |
| PR #2 multi-view model path | 8588.277 | 8494.734 | 30 | 1.089% lower wall time |
| PR #2 geometry + TSDF, model-free | 867.421 | 831.797 | 10 | 4.11% lower wall time |

The multi-view protocol and raw artifacts are in
[final-30-per-arm/RESULTS.md](benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md).
The geometry-plus-TSDF protocol and raw artifacts are in
[RESULTS.md](benchmarks/2026-08-14-pr2-geometry/RESULTS.md). The single-image
number is a companion DA3-BASE study, not an end-to-end Vestra video result.
These measurements are separate scopes: no stage speedup is additive, and
none implies end-to-end, GPU, or browser throughput.

## Research and known limits

### Global pose and MVS

The global-pose sidecar and pinned COLMAP path are implemented as optional
imports. On IMG_2323, COLMAP, DROID-SLAM, hybrid, and VGGT candidates fail
the current registration/residual publication gate. The local PR #2-relative
world therefore remains selected. Dense COLMAP MVS and DA3/MVS hybrids are
separate controls; TSDF can reduce double sheets but cannot repair a bad
trajectory.

Evidence: [global provider evaluation](validation/GLOBAL_POSE_PROVIDERS_IMG_2323_2026-08-17.md),
[COLMAP spike](validation/COLMAP_GLOBAL_POSE_IMG_2323_2026-08-17.md), and
[MVS hybrid](validation/DA3_MVS_HYBRID_IMG_2323_2026-08-17.md).

### Geometry quality

The real-video path proves local wiring, durable evidence, relative fusion,
and progressive delivery. It does not prove metric accuracy, a universally
connected room, successful loop closure for arbitrary captures, semantic
meshing, or generated geometry. The next quality corpus needs representative
room fixtures, an intentional revisit, and measured gates for drift, holes,
surface retention, and ghost thickness.

### Product hardening

The product/lab CLI split is complete. Real F32 Studio cancel/restart/resume
validation is still being completed. Current evidence for the shared durable
job contract is [here](validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md); it
explicitly marks the Workhorse browser flow as operational validation still
required.

### Deferred claims

Metric scale, semantic architecture, a production CUDA backend, cinematic
rendering, and global coherent-room publication remain pending. No current
document should imply that any of these is accepted merely because a research
derivative or isolated oracle exists.

## Next milestones

1. Run adaptive candidate/keyframe selection over representative captures and
   publish geometry-quality evidence, not a frame-count target.
2. Complete real F32 Studio cancellation and resume validation.
3. Improve global camera registration and bundle adjustment before retrying
   global fusion.
4. Profile and verify any GPU path against the CPU F32 reference before making
   a throughput claim.
