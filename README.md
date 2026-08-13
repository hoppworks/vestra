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
- CLI and browser studio drive the same resumable Rust job engine.

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

The first tracer bullet is active. Vestra can already represent the parts of a
world pipeline that do not depend on an unverified model claim:

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
    It completed 14 persisted windows and fused 300,906 finite relative-scale
    surfels; its stated limitations remain part of the result. The earlier
    [end-to-end](docs/validation/END_TO_END_SMOKE_2026-08-13.md) and
    [motion-video](docs/validation/MOTION_VIDEO_SMOKE_2026-08-13.md) smokes
    remain narrower wiring evidence.
12. Fused points are additionally persisted in ordered, content-addressed
    chunks (50,000 surfels each). Studio prefers compact 40-byte binary
    surfel chunks for progressive GPU upload, while JSON chunks and the
    canonical fused payload remain available for compatibility and export.

The next hard gate is a real-video end-to-end run followed by validation of
window alignment, revisit/loop handling, and dense fusion quality. Studio
prefers the fused world when a manifest references one, while retaining the
measured-layer fallback for inspection and recovery.

## Scene bundles

A `.vestra` scene is a local directory while processing. Its manifest records
the engine, kernel, model, and settings identities. Measured window chunks are
immutable JSON payloads addressed by their SHA-256 content hash. The manifest
is replaced atomically after a chunk is durable, which lets a job resume or a
viewer reveal completed windows without confusing them with fused geometry.

V1 scenes are explicitly relative-scale. A point's position is evidence from
the depth map and camera calibration; it is not a claim in metres. Metric-scale
anchors are a future opt-in phase.

Fused windows use shared source-frame pixels as direct correspondence evidence
for a robust Sim(3). Degenerate (rank-one) overlaps are rejected rather than
being assigned an invented rotation. The current fusion stores global points;
the window cameras remain raw diagnostic evidence and are not yet exposed as
global camera-path geometry.

There is no claim yet that the complete Vestra world pipeline is faster than
the C++ PR. The existing qualified speed result covers single-image CPU F32.

## Development

```bash
cargo test --workspace
cargo run -p vestra-cli -- plan --frames 120 --chunk-size 12 --overlap 3
```

## Local first run

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
diagnostic captures. To rebuild that derived layer without another model
inference run:

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

Each new surfel carries a world-space normal estimated from its local depth
stencil. Fused normals are confidence-weighted; missing normals in older
bundles are treated as unknown for backward compatibility.

Open the result in the local browser studio:

```bash
cargo run -p vestra-cli -- serve --scene room.vestra
```

The server binds only to `127.0.0.1:4317`. It serves no upload endpoint and
never sends the scene to a remote service.

See [VISION.md](VISION.md), [ARCHITECTURE.md](ARCHITECTURE.md), and the ADRs in
`docs/adr/` for the locked decisions and implementation order.
