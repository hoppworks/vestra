# Vestra

Vestra turns an ordinary phone video into an explorable spatial world. It is a
local-first native Rust reconstruction pipeline with a browser studio for
progressive reveal, cinematic flythroughs, inspection, and open export.

Vestra is inspired by the video-to-world pipeline merged in
`localai-org/depth-anything.cpp` PR #2, but it is not a C++ wrapper. The neural
path uses the separately versioned Vestra Engine and Vestra Kernels projects;
the reconstruction and scene pipeline are native Rust owned here.

## Product contract

- One continuous room video is the v1 target.
- Relative scale is the v1 truth; metric scale is explicitly deferred.
- Raw measured geometry, fused/interpolated geometry, and generated geometry
  are separate provenance layers.
- Surfels are the reliable default representation. A cinematic presentation
  mode adds lighting, atmospheric depth, progressive reveal, and camera motion.
- Processing is local and offline by default. A remote Workhorse mode is
  explicit and shows its destination and deletion policy.
- CLI and browser studio consume the same durable Rust scene contract.

## Sporting performance target

Locked product workload:

- 60-second 1080p phone video
- 120 selected frames
- 504×336 reconstruction resolution
- 12-frame windows with 3-frame overlap
- AMD Ryzen 9 9950X and NVIDIA RTX 5080
- first interactive world in less than 10 seconds
- completed fused world in less than 15 seconds

Performance claims require identical model, precision, frame set, resolution,
window schedule, geometry settings, thread budget, and backend. CPU/GPU or
Q4/F32 comparisons are never presented as implementation speedups.

## Repository boundary

| Repository | Owns |
|---|---|
| `vestra` | Video orchestration, reconstruction, `.vestra` scenes, CLI, service, studio |
| `vestra-engine` | Model loading, preprocessing, single/multi-view inference, backend selection |
| `vestra-kernels` | Qualified fixed-shape CPU and CUDA kernels |

Exact revisions are recorded in `vestra.lock.toml`. Local path overrides exist
only to support coordinated development without copying code.

## Current state

The base relative-scale world pipeline is implemented and independently
validated against the pinned C++ reference. The remaining work is production
backend completion, broader regression coverage, and explicitly optional
branches—not a claim that those features already exist:

1. PR #2's streaming window schedule is represented in Rust.
2. Vestra imports Vestra Engine through the repository boundary.
3. Vestra Engine implements genuine ordered multi-view local/global attention.
4. `S=1` is bitwise equal to its established single-view path.
5. The automatic saddle-balanced reference-selection and restoration path is
   implemented for eligible multi-view windows.
6. Calibrated depth, confidence, RGB, and a W2C camera pose deterministically
   produce relative-scale measured surfel points. Invalid or low-confidence
   pixels never become geometry.
7. Each measured window can be written to an atomic, content-addressed
   `.vestra` bundle. Repeating an identical window write is idempotent; a
   crash cannot publish a partial manifest as a completed scene.
8. C++ oracle parity for canonical RGB24 input is accepted for `S=2`, `S=3`,
   and `S=12` multi-view windows, including depth, confidence, intrinsics, and
   W2C camera poses. The durable evidence is in
   [the validation record](docs/validation/MULTIVIEW_S2_2026-08-13.md).
9. Overlapping measured windows can now be aligned by robust relative Sim(3),
   then confidence-fused into a voxel-deduplicated surfel world. The derived
   chunk is immutable, content-addressed, and atomically referenced by the
   manifest; raw evidence is never overwritten.
10. Vestra automatically proposes only non-adjacent camera revisits, measures
    them with tight spatial geometry gates, and redistributes accepted loop
    constraints through a relative Sim(3) pose graph before fusion. Failed or
    weak candidates are ignored; they never create an identity fallback.
11. A full 120-frame real-phone-video run is evidenced on the Ryzen 9
    Workhorse in [the real-video validation record](docs/validation/REAL_VIDEO_IMG_2269_2026-08-13.md).
    The earlier 14-window run is retained as historical pipeline evidence but
    is superseded for PR #2 parity after the terminal-window correction; its
    stated limitations remain part of the result. The earlier
    [end-to-end](docs/validation/END_TO_END_SMOKE_2026-08-13.md) and
    [motion-video](docs/validation/MOTION_VIDEO_SMOKE_2026-08-13.md) smokes
    remain narrower wiring evidence.
12. Fused points are additionally persisted in ordered, content-addressed
    chunks (50,000 surfels each). Studio prefers compact 40-byte binary
    surfel chunks for progressive GPU upload, while JSON chunks and the
    canonical fused payload remain available for compatibility and export.
13. The separately versioned Engine and Kernels repositories have a qualified
    native CUDA transformer-tail parity slice for the DA3-BASE F32 model. It
    has passed both single-image and ordered multi-view CPU-F32 oracle tests
    on the RTX 5080. It is deliberately not exposed as a production speed
    backend yet: preprocessing, token preparation, early blocks, feature
    captures, and the DPT/pose heads still cross or remain on the CPU.
14. Studio can show the decoded source frames already retained inside the
    local bundle as a picture-in-picture diagnostic. It dynamically converts
    only an indexed local RGB24 cache frame for the loopback browser request;
    it exposes neither an upload endpoint nor arbitrary filesystem paths.
15. The pinned C++ PR #2 base stream now has real-fixture, pre-voxel parity:
    identical 12/3 window schedule, dense overlap correspondences, Huber-IRLS
    Sim(3), first-owner point emission, 9,931,557 ordered points, per-frame
    ownership and RGB. The numerical cloud comparison is recorded in
    [the streaming oracle record](docs/validation/CPP_PR2_STREAMING_ORACLE_2026-08-13.md).
16. The PR #2 normal-space TSDF branch has accepted identity and closed-loop
    oracle tiers. The closed-loop result has identical ordered ownership and
    RGB for all 25,434 surfels, with a position MAE of `3.046e-7` relative
    units. The acceptance boundary and reproduction commands are in
    [the TSDF oracle record](docs/validation/TSDF_ORACLE_2026-08-14.md).
17. The corrected current Rust product pipeline has also reconstructed the
    same 120-frame capture into 296,596 finite fused surfels across 13 windows
    and served six progressive binary chunks through local Studio. The
    reproducible run is documented in
    [the current product-world record](docs/validation/CURRENT_PRODUCT_WORLD_IMG_2269_2026-08-13.md).
18. In the locked model-free PR #2 closed-loop geometry-plus-TSDF workload,
    Vestra now completes ten fresh-process trials at 831.797 ms mean versus
    the pinned C++ reference at 867.421 ms (4.11% faster; non-overlapping
    95% t intervals). The fixture, output oracle, raw samples, binary hashes,
    16-thread budget and scoped claim are recorded in
    [the geometry benchmark](docs/benchmarks/2026-08-14-pr2-geometry/RESULTS.md).
    This is not an inference, quantized-model, GPU, or end-to-end video claim.
19. In the separately locked CPU F32 PR #2 multi-view model stage (24 frames,
    504×336, three 12/3 windows, 16 threads), Vestra's AVX-512 multi-view
    Flash route completes 30 fresh-process trials at 8494.734 ms mean versus
    8588.277 ms for the pinned C++ reference: 1.089% faster. The independent
    C++ minus Rust 95% Welch interval is [61.338, 125.748] ms. Input/model
    preparation and output serialization are excluded for both arms; raw
    samples and hashes are recorded in
    [the final model benchmark](docs/benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md).
    This is a model-stage claim, not an end-to-end video, quantized-model, or
    GPU claim.

The next hard performance gate is a device-resident end-to-end backbone and
DPT/pose-head path with the same oracle discipline. Only then can a CUDA
throughput claim or a fair CUDA comparison with the C++ reference be made.
Studio already prefers the fused world when a manifest references one, while
retaining the measured-layer fallback for inspection and recovery.

## Scene bundles

A `.vestra` scene is a local directory while processing. Its manifest records
the engine, kernel, model, and settings identities. Measured window chunks are
immutable JSON payloads addressed by their SHA-256 content hash. The manifest
is replaced atomically after a chunk is durable, which lets a viewer reveal
completed windows without confusing them with fused geometry. Automatic resume
of an interrupted `reconstruct` command is a remaining job-lifecycle feature;
persisted raw chunks are nevertheless retained for inspection and re-fusion.

V1 scenes are explicitly relative-scale. A point's position is evidence from
the depth map and camera calibration; it is not a claim in metres. Metric-scale
anchors are a future opt-in phase.

Fused windows use shared source-frame pixels as direct correspondence evidence
for a robust Sim(3). Degenerate (rank-one) overlaps are rejected rather than
being assigned an invented rotation. The current fusion stores global points;
the window cameras remain raw diagnostic evidence. Their final local-to-fused
world transforms are persisted with the derived world and can be exported for
inspection; they are not silently rewritten into a misleading global W2C pose.

There is no claim yet that the complete video-to-published-world pipeline is
faster than the C++ PR. The qualified CPU F32 evidence now covers both the
locked PR #2 multi-view model stage and the locked geometry-plus-TSDF stage;
their separate timings must not be summed into an end-to-end claim.

## Development

```bash
cargo test --workspace
cargo run -p vestra-cli -- plan --frames 120 --chunk-size 12 --overlap 3
```

## Local first run

For the local browser flow, start the intake with the model and a dedicated
job directory. It binds only to loopback, streams the selected video directly
to that directory, starts the same durable `reconstruct` command, and opens a
second local Studio port for the completed world:

```bash
cargo run -p vestra-cli -- app \
  --model depth-anything-base-f32.gguf \
  --jobs ./vestra-jobs
```

Open `http://127.0.0.1:4317`, choose one `.mov`, `.mp4`, `.m4v`, or `.avi`
file, then select **Start reconstruction**. The intake persists the job's
video filename and reconstruction settings atomically in its job directory.
Use **Cancel safely** to request the same interrupt that the CLI handles; only
complete window checkpoints remain published. On restart, the most recent
running job becomes **interrupted** and can be resumed from the browser with
its original settings. The intake deliberately accepts one job at a time; the
job owns its copied input, scene bundle, `job.json`, and `reconstruct.log`
under `./vestra-jobs/job-000001/`.

With FFmpeg and FFprobe installed, create a relative-scale scene locally:

```bash
cargo run -p vestra-cli -- reconstruct \
  --video room.mov \
  --model depth-anything-base-f32.gguf \
  --output room.vestra
```

Vestra samples up to 120 frames at 504×336 by default, retains one measured
surfel per 8×8 depth pixels for the local JSON/WebGL v1, reconstructs each
12/3 overlapping window, checkpoints measured evidence, then automatically
publishes a derived fused world. Use `--pixel-stride 1` only for small
diagnostic captures. To continue a stopped run, repeat exactly the same
command with `--resume`; Vestra verifies input/model/engine/kernel/settings
provenance and reuses only complete schedule-compatible windows. To rebuild
the derived layer without another model inference run:

```bash
cargo run -p vestra-cli -- reconstruct \
  --video room.mov \
  --model depth-anything-base-f32.gguf \
  --output room.vestra \
  --resume
```

Press `Ctrl-C` to cancel a CLI reconstruction immediately. Vestra exits with
code `130`; the atomic scene contract guarantees that only complete measured
windows remain manifest-referenced. Restart with the identical command and
`--resume` to continue. The in-process API currently exposes window-boundary
completion, while this CLI path supplies the bounded cancellation guarantee.

For a strict PR #2-relative geometry run, capture a new dense evidence bundle
with `--cpp-pr2-relative`. This stores all finite-depth samples and the
per-window confidence percentile used by PR #2 for loop keys and first-owner
emission. A legacy sparse scene is intentionally refused by strict fusion,
rather than being relabelled as reference-compatible:

```bash
cargo run -p vestra-cli -- reconstruct \
  --video room.mov \
  --model depth-anything-base-f32.gguf \
  --output room-pr2.vestra \
  --cpp-pr2-relative
```

Before inference, Vestra records a lightweight adjacent-frame luma-motion
indicator as `ready`, `review`, or `recapture` in the manifest and Studio HUD.
It is a capture-risk warning—not a claim that a `ready` capture is geometrically
correct.

```bash
cargo run -p vestra-cli -- fuse --scene room.vestra
```

Inspect the persisted provenance and evidence signals before making a quality
claim or sharing a capture. The report distinguishes a measured-only bundle
from a fused relative-scale world, checks that published surfels are finite,
and records alignment, loop-graph, capture-risk, and progressive-delivery
facts. It deliberately does not infer metric accuracy from these signals.

```bash
cargo run -p vestra-cli -- inspect --scene room.vestra
```

Export the same fused relative-scale layer for inspection in standard 3D
software:

```bash
cargo run -p vestra-cli -- export --scene room.vestra --output room.ply
```

For a glTF 2.0 point-cloud asset (positions, normals, and vertex colors), use:

```bash
cargo run -p vestra-cli -- export-glb --scene room.vestra --output room.glb
```

For compact oriented surfels compatible with the common 32-byte `.splat`
layout, use `export-splat`. This is a visualization of measured surfels, not
Gaussian-splat training or generated geometry.

```bash
cargo run -p vestra-cli -- export-splat --scene room.vestra --output room.splat
```

For camera-path diagnostics, export the raw per-window W2C camera evidence
alongside the final local-to-fused relative Sim(3) transform. Consumers must
compose these explicitly; the output intentionally does not pretend the W2C
matrix is already global.

Newly fused bundles also retain the exact sequential seam edges and every
accepted non-sequential loop edge that informed the final pose graph. This is
provenance for local inspection, not an assertion that every proposed revisit
was accepted.

```bash
cargo run -p vestra-cli -- export-cameras --scene room.vestra --output room.cameras.json
```

Each new surfel carries a world-space normal estimated from its local depth
stencil. Fused normals are confidence-weighted; missing normals in older
bundles are treated as unknown for backward compatibility.

Open the result in the local browser studio:

```bash
cargo run -p vestra-cli -- serve --scene room.vestra
```

The server binds only to `127.0.0.1:4317`. It serves no upload endpoint and
never sends the scene to a remote service.

Press `C` or use **Show camera rays** in the scene ledger to overlay the
captured camera directions and calibrated image-plane frustums. They are
derived from the stored window-local W2C calibrations plus their persisted
final local-to-fused relative Sim(3) poses. Their rendered length is scaled to
the currently displayed relative world, so they are diagnostic evidence—not a
metric camera trajectory.

When a fused world is present, **Show measured evidence** switches between the
derived voxel-fused surfels and the immutable per-window measurements. This is
a local diagnostic control: it makes seam or ghost inspection possible without
presenting the fused layer as new source evidence.

Use **Show seams / loops** (or `L`) to inspect persisted seam evidence. Teal
links are sequential overlap seams; amber links are independently verified
loop edges. Legacy bundles remain readable and can still expose their ordered
sequential seams, but only bundles produced by current Vestra revisions retain
explicit loop-edge provenance.

When a bundle retains its decode cache, Studio also shows a local
**Source frame** picture-in-picture. Use **Next frame** (or `F`) to step through
the frames that contributed to the reconstruction. These are diagnostic source
images served only through the existing loopback Studio process; they are not
embedded in exported worlds or uploaded anywhere.

See [VISION.md](VISION.md), [ARCHITECTURE.md](ARCHITECTURE.md), and the ADRs in
`docs/adr/` for the locked decisions and implementation order.
