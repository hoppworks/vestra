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

Remaining:

- a real captured-room seam/oracle corpus
- correspondence ownership policy for overlapping source frames

## Slice 3 — Reconstruction cleanup (next geometry milestones)

- one bounded point-to-plane refinement over direct overlap matches, using
  fused surfel normals (complete); spatial-hash ICP remains next
- loop detection and Sim3 pose graph
- TSDF de-ghosting and first-observer ownership
- surfel normals, voxel mode, thinning, and confidence reveal ordering
- quality gates for connectedness, drift, overlap error, surface retention,
  and ghost thickness

## Slice 4 — Scene, service, and studio (first local product path complete)

- content-addressed `.vestra` measured/fused JSON chunks and atomic manifest
- local CLI (`reconstruct`, `fuse`, `export`, `serve`) and browser Studio
- interactive WebGL point-world inspection with fused-layer preference
- ASCII PLY export containing relative positions, RGB, confidence, radius,
  and contributor count

Remaining:

- progressive binary chunks and GPU uploads
- WebGL2/WebGPU diagnostic and cinematic modes
- `.splat`, GLB, camera JSON, and flythrough export

## Slice 5 — CUDA and product performance

- native RTX 5080 inference kernels in Vestra Kernels
- GPU geometry phases where profiling justifies them
- progressive time-to-first-world optimization
- full randomized C++/Vestra CPU and GPU benchmark matrix

Metric scale and true multi-frame Gaussian reconstruction remain explicitly
outside v1.
