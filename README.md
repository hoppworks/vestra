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

The first tracer bullet is active:

1. PR #2's streaming window schedule is represented in Rust.
2. Vestra imports Vestra Engine through the repository boundary.
3. Vestra Engine implements genuine ordered multi-view local/global attention.
4. `S=1` is bitwise equal to its established single-view path.
5. The saddle-balanced reference scoring and reordering contract is ported.
6. Real C++ oracle parity for `S=2,3,12` is the next hard gate.

There is no claim yet that the complete Vestra world pipeline is faster than
the C++ PR. The existing qualified speed result covers single-image CPU F32.

## Development

```bash
cargo test --workspace
cargo run -p vestra-cli -- plan --frames 120 --chunk-size 12 --overlap 3
```

See [VISION.md](VISION.md), [ARCHITECTURE.md](ARCHITECTURE.md), and the ADRs in
`docs/adr/` for the locked decisions and implementation order.
