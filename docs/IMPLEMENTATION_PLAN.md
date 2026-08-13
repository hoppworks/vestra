# Vestra implementation plan

The plan is parity-first. A stage does not advance because it looks plausible
in the viewer; it advances when its isolated C++ oracle, Rust tests, and scene
quality measurements pass.

## Slice 1 — Multi-view inference parity (active)

Completed:

- independent repository boundaries and pinned revisions
- real ordered multi-view local/global transformer execution
- per-view reference/source camera token injection
- repeated per-view RoPE boundaries during flattened global attention
- per-view depth, confidence, pose, and intrinsics output API
- bitwise `S=1` equivalence and synthetic cross-view coupling tests
- saddle-balanced reference scoring and reference-first permutation contract

Remaining hard gate:

- integrate the preliminary local CLS pass for automatic `S>=3` selection
- add an oracle dump command to pinned C++ PR #2
- compare captures and final results at `S=2`, `S=3`, and `S=12`
- require Pearson `r >= 0.9999` and MAE `<= 0.005` for F32 tensors unless an
  operator-specific stricter bitwise contract applies
- benchmark one 12-view window with identical CPU-F32 work

## Slice 2 — Points and sliding windows

- calibrated back-projection with confidence, color, and radius
- immutable per-window scene chunks
- exact 12/3 window schedule and progressive frame ownership
- weighted Huber/IRLS Umeyama Sim3 on dense overlap correspondences
- C++ analytic and real-window seam oracles

## Slice 3 — Reconstruction cleanup

- point-to-plane ICP with spatial hash
- loop detection and Sim3 pose graph
- TSDF de-ghosting and first-observer ownership
- surfel normals, voxel mode, thinning, and confidence reveal ordering
- quality gates for connectedness, drift, overlap error, surface retention,
  and ghost thickness

## Slice 4 — Scene, service, and studio

- streamable content-addressed `.vestra` scene format
- resumable local job service and CLI
- progressive WebGPU studio with WebGL2 fallback
- cinematic world mode and diagnostic inspect mode
- `.splat`, PLY, GLB, camera JSON, and flythrough export

## Slice 5 — CUDA and product performance

- native RTX 5080 inference kernels in Vestra Kernels
- GPU geometry phases where profiling justifies them
- progressive time-to-first-world optimization
- full randomized C++/Vestra CPU and GPU benchmark matrix

Metric scale and true multi-frame Gaussian reconstruction remain explicitly
outside v1.
