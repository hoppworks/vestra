# Vestra

> A depth map is not a world.

Vestra is a local-first spatial reconstruction system written in Rust. It
turns a handheld room video into an inspectable `.vestra` scene while keeping
model output, camera evidence, deterministic fusion, and presentation as
separate, attributable layers.

The project is deliberately not a C++ wrapper and not a screenshot-only AI
demo. Its inference runtime, fixed-shape kernels, reconstruction pipeline,
scene format, local service, and WebGL Studio are independently testable
modules. Optional global-pose and dense-MVS providers are pinned, labelled, and
gated rather than hidden behind the renderer.

## What this repository demonstrates

- **Native systems work:** ordered multi-view DA3 inference through separately
  versioned Rust Engine and Kernels repositories.
- **Geometry with failure semantics:** calibrated back-projection, robust
  relative Sim(3), revisit constraints, pose-graph optimization, surfel fusion,
  and normal-space TSDF without an identity fallback.
- **Durable product design:** immutable content-addressed evidence, atomic
  manifests, resumable windows, progressive binary delivery, and open exports.
- **Evidence before claims:** numerical C++ parity gates, raw randomized
  benchmark samples, provider coverage/residual gates, and explicit limits.

## Public demo

The release demo uses the CC-BY-4.0 TUM RGB-D `freiburg1_room` sequence: a
real handheld indoor loop, reconstructed without consuming its ground-truth
trajectory. The original video, derived scene, checksums, and attribution are
distributed as release assets rather than committed binaries.

After downloading and extracting the latest `vestra-demo-freiburg1-room`
scene from [Releases](https://github.com/hoppworks/vestra/releases), open it
without a model download or inference run:

```bash
cargo run --release --locked -p vestra-cli -- \
  demo --scene /path/to/freiburg1_room.vestra
```

Then open `http://127.0.0.1:4317`. The demo command only validates and serves
the precomputed bundle. The source and transformation contract lives in
[`demo/`](demo/README.md).

## From a video

Vestra keeps the user-facing command surface intentionally small:

```text
vestra app          local browser intake
vestra reconstruct  video -> relative-scale scene
vestra demo         serve a precomputed release scene
vestra serve        serve an existing scene
vestra inspect      print provenance and quality evidence
vestra export       export the selected world as PLY
```

Start the browser intake with a DA3-BASE F32 GGUF file:

```bash
cargo run --release --locked -p vestra-cli -- app \
  --model /path/to/depth-anything-base-f32.gguf \
  --jobs ./vestra-jobs
```

Or reconstruct directly:

```bash
cargo run --release --locked -p vestra-cli -- reconstruct \
  --video room.mp4 \
  --model /path/to/depth-anything-base-f32.gguf \
  --output room.vestra
```

The default capture contract decodes RGB candidates at 8 fps under a high
safety ceiling, then selects geometry keyframes from temporal baseline, luma
novelty, sharpness, and maximum gap. It does not use a fixed total-frame target
that becomes too sparse for long videos.

Research commands—reference oracles, provider imports, MVS controls, specialist
exports, and architecture experiments—live in the explicit `vestra-lab`
binary. They are not hidden product aliases.

## Architecture

```text
video
  |
  +-- candidate decode + deterministic geometry keyframes
  |
  +-- Vestra Engine ---- Vestra Kernels
  |      depth / confidence / intrinsics / local camera evidence
  |
  +-- immutable measured windows
  |      relative Sim(3) / verified loops / pose graph
  |
  +-- derived products
  |      surfels / voxel fusion / normal-space TSDF
  |      optional pinned global-pose + dense-MVS controls
  |
  +-- content-addressed .vestra scene
         Studio / PLY / GLB / .splat / camera evidence
```

| Repository | Responsibility |
| --- | --- |
| [`hoppworks/vestra`](https://github.com/hoppworks/vestra) | Video jobs, geometry, scene contract, CLI, local service, Studio |
| [`hoppworks/vestra-engine`](https://github.com/hoppworks/vestra-engine) | GGUF loading, preprocessing, DA3 single/multi-view model semantics |
| [`hoppworks/vestra-kernels`](https://github.com/hoppworks/vestra-kernels) | Qualified fixed-shape CPU and CUDA kernel slices |

Exact compatible revisions are recorded in [`vestra.lock.toml`](vestra.lock.toml).
A fresh clone resolves the pinned repositories without sibling checkouts;
[`scripts/use-local-deps.sh`](scripts/use-local-deps.sh) creates
an uncommitted Cargo override for coordinated development.

The optional COLMAP branch is not described as pure Rust. COLMAP supplies a
pinned global camera/dense-MVS provider; Vestra owns the raster contract,
validation gates, product labelling, scene import, provenance, and rendering.

## Performance evidence

These are three different CPU studies on an AMD Ryzen 9 9950X with a 16-thread
budget. The validation machine also has an NVIDIA RTX 5080 and 96 GiB RAM.

| Locked workload | C++ reference | Vestra Rust | N | Result |
| --- | ---: | ---: | ---: | --- |
| DA3-BASE single image, F32, 504x336 | 238.789 ms | 171.141 ms | 20 | 28.3% lower latency; 39.5% higher throughput |
| PR #2 multi-view model, F32, 24 frames | 8588.277 ms | 8494.734 ms | 30 | 1.089% lower wall time |
| PR #2 geometry + TSDF, model-free | 867.421 ms | 831.797 ms | 10 | 4.11% lower wall time |

The stages, exclusions, source revisions, model/input hashes, raw samples, and
confidence intervals are part of each study. The results are not additive and
none is an end-to-end video, browser, quantized-model, or GPU speed claim. See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) and the
[`docs/benchmarks/`](docs/benchmarks/) evidence.

## Scene truth and reliability

A `.vestra` scene is a local directory while processing and may be archived
for transport. Immutable chunks are hashed and made durable before an atomic
manifest replacement references them. Raw measurements are never overwritten
by fusion. Provenance binds the video, raster policy, model, engine, kernels,
settings, pose provider, and derived products.

The accepted v1 scene is relative-scale. TSDF may consolidate duplicate
surface evidence; it cannot repair a bad camera trajectory or establish
metres. Global provider output stays a separate product unless coverage and
residual gates pass. Generated or unseen geometry is not presented as measured
evidence.

## Current limits

- Arbitrary long or low-parallax captures do not yet guarantee one globally
  coherent room.
- Metric scale requires an independent anchor and is not a default claim.
- Semantic walls, doors, floors, and a watertight architectural mesh are
  research products, not release output.
- CUDA has qualified parity slices, not a device-resident end-to-end product
  path or public GPU speed claim.
- Model weights are not distributed by this repository; checkpoint terms are
  model-specific.

That boundary is intentional. Failed pose providers and unsupported surfaces
remain visible evidence rather than being converted into a prettier claim.

## Development

Rust 1.93.0 is pinned. The complete local gate is:

```bash
./scripts/verify.sh
```

It runs formatting, strict Clippy, Rust and Python tests, browser-control tests,
documentation checks, and repository integrity checks against the locked
dependency graph. CI runs the same source contract.

Start with the [documentation index](docs/README.md),
[architecture](ARCHITECTURE.md), [vision](VISION.md), and
[contribution guide](CONTRIBUTING.md). Third-party source and dataset notices
are retained in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Vestra is licensed under Apache-2.0.
