# Vestra implementation plan

The plan is parity-first. A stage does not advance because it looks plausible
in the viewer; it advances when its isolated C++ oracle, Rust tests, and scene
quality measurements pass.

## Slice 1 — Multi-view inference parity (complete for the locked F32 contract)

Completed:

- independent repository boundaries and pinned revisions
- real ordered multi-view local/global transformer execution
- per-view reference/source camera token injection
- repeated per-view RoPE boundaries during flattened global attention
- per-view depth, confidence, pose, and intrinsics output API
- bitwise `S=1` equivalence and synthetic cross-view coupling tests
- saddle-balanced reference scoring and reference-first permutation contract

Accepted gate:

- automatic `S>=3` reference selection is part of the locked engine path
- the temporary C++ oracle dump command was used only to establish parity and
  is not a product dependency
- canonical RGB24 C++ oracle comparisons passed at `S=2`, `S=3`, and `S=12`
  for depth/confidence and camera outputs; see `docs/validation/`

## Slice 2 — Points and sliding windows (first usable implementation complete)

- calibrated back-projection with confidence, color, and radius
- immutable per-window scene chunks
- exact 12/3 window schedule and progressive frame ownership
- robust relative Sim(3) on dense shared-pixel overlap correspondences,
  including rank-one rejection and quality gates
- deterministic confidence-weighted voxel surfel fusion, with immutable raw
  evidence and an atomically published fused layer
- controlled MP4 → Studio Workhorse smoke evidence
- first-observer ownership: an overlapping source frame contributes to seam
  alignment in every retained raw window but votes exactly once into fusion

Remaining:

- three more representative captured-room regression fixtures, including a
  deliberate revisit that validates an accepted loop edge
- scene-level quality gates for connectedness, drift, surface retention, and
  ghost thickness

## Slice 3 — Reconstruction cleanup (core pose work complete)

- one bounded point-to-plane refinement over direct overlap matches, using
  fused surfel normals (complete)
- automatic revisit proposal, geometric loop measurement, relative Sim3
  pose-graph optimization, and deferred final fusion (complete)
- surfel normals, voxel fusion, conservative sampling, and confidence reveal
  ordering (complete)

Remaining:

- spatial-hash ICP only where a profile and fixture demonstrate that bounded
  seam refinement is insufficient
- conservative TSDF de-ghosting, backed by a regression corpus and an
  explicit measured-versus-fused provenance boundary
- scene-level quality gates for connectedness, drift, surface retention, and
  ghost thickness

## Slice 4 — Scene, service, and studio (first local product path complete)

- content-addressed `.vestra` measured/fused JSON chunks and atomic manifest
- local CLI (`reconstruct`, `fuse`, `export`, `serve`) and browser Studio
- interactive WebGL point-world inspection with fused-layer preference
- ASCII PLY export containing relative positions, RGB, confidence, radius,
  and contributor count
- content-addressed binary 40-byte surfel chunks with progressive GPU upload
- GLB points, compact `.splat`, and composable camera JSON export

Remaining:

- WebGL2/WebGPU cinematic modes and a reproducible flythrough export
- Workhorse validation of browser cancellation and restart/resume against a
  real F32 room-video run; see
  `docs/validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md`

## Slice 5 — CUDA and product performance

- Engine-owned native RTX 5080 backend; the current Engine is CPU-bound and
  has no GPU fallback or claim (see `docs/adr/0004-cuda-backend-boundary.md`)
- native CUDA inference kernels in Vestra Kernels
- GPU geometry phases where profiling justifies them
- progressive time-to-first-world optimization
- full randomized C++/Vestra CPU and GPU benchmark matrix

The first real video validation is recorded in
`docs/validation/REAL_VIDEO_IMG_2269_2026-08-13.md`. It validates durable
evidence and relative-world fusion, not metric accuracy, semantic meshing, or
the end-to-end performance target.

Metric scale and true multi-frame Gaussian reconstruction remain explicitly
outside v1.
