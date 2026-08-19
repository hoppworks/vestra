# Vestra vision

Vestra turns a handheld room video into an inspectable spatial record on the
operator's own machine. The goal is not a decorative point cloud. It is a
reconstruction whose visible result can be traced back to captured pixels,
camera evidence, model revisions, and deterministic fusion choices.

## The product we are building

The intended experience is simple: select a landscape room video, let Vestra
check capture risk and select useful geometry keyframes, then inspect a world
as it becomes available. The browser Studio should expose both the useful
view and the evidence behind it: source frames, camera rays, seams, layer
provenance, and confidence diagnostics.

The product has two command surfaces. `vestra` is the small, stable product
CLI and local Studio entrypoint. `vestra-lab` is the explicit engineering
surface for oracle comparisons, pose-provider imports, dense-MVS experiments,
and other work that is not yet a product promise. The split is enforced by
separate binaries so experimental commands cannot become accidental product
promises.

## Truth model

Every published geometric sample belongs to a declared layer:

1. **Measured** — directly supported by decoded pixels, inferred depth, and
   the camera evidence available to that product.
2. **Fused** — deterministic consolidation of measured evidence, such as
   relative alignment, TSDF surface fusion, confidence filtering, or bounded
   hole handling.
3. **Generated** — content that is not sufficiently supported by the capture.

The accepted v1 path may publish measured and deterministic fused layers. It
must not present generated or unseen content as measured geometry. Semantic
labels, architectural interpretations, and future completion methods remain
separate products until they pass their own evidence gates.

## What is accepted now

The release baseline is local-first video-to-DA3 inference followed by the
relative-scale PR #2 reconstruction path. It includes durable measured
windows, relative Sim(3) alignment, deterministic fusion, implemented
normal-space TSDF output, content-addressed `.vestra` layers, open point-cloud
exports, and a local browser Studio. Metric scale is intentionally not
assumed. The implementation and evidence map is maintained in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

Frame volume is not defined by a product-wide total. Vestra currently decodes
at a fixed candidate rate (8 fps) behind a high safety ceiling, then retains
geometry keyframes using temporal baseline, luma novelty, sharpness, and a
maximum temporal gap. The first and final candidates are retained. This keeps
long captures from becoming sparse and short captures from becoming
needlessly dense; it is a selection policy, not a claim of global camera
quality.

## What remains aspirational

Vestra should eventually offer a globally coherent world for difficult
captures, with stronger camera trajectories, metric-scale anchors, semantic
architecture products, and GPU-backed production inference. Those are goals,
not current release guarantees. A global-pose fusion product is published only
as a separate, provider-labelled derivative after every local window passes
the global-fit gate. Dense-MVS controls may be published separately, including
partial results, but remain explicitly labelled and do not replace the local
world. The current provider findings are recorded
in
[`docs/validation/GLOBAL_POSE_PROVIDERS_IMG_2323_2026-08-17.md`](docs/validation/GLOBAL_POSE_PROVIDERS_IMG_2323_2026-08-17.md)
and
[`docs/validation/COLMAP_DENSE_MVS_CONTROL_IMG_2323_2026-08-17.md`](docs/validation/COLMAP_DENSE_MVS_CONTROL_IMG_2323_2026-08-17.md).

The architecture product is likewise a future, evidence-backed derivative;
the geometry-only RANSAC prototype is explicitly experimental. No vision
statement here should be read as a claim that arbitrary room videos already
produce a globally coherent mesh, semantic floor plan, or metric survey.

## Definition of a successful release

- A local capture produces a resumable, relative-scale `.vestra` bundle.
- Raw measured evidence remains immutable while fused layers are derived.
- Studio can progressively inspect the world and its provenance.
- CLI and Studio use the same durable job contract.
- Model, geometry, and TSDF behavior are checked against pinned reference
  artifacts before performance is discussed.
- Any benchmark claim names its exact workload, precision, host, sample size,
  and excluded stages.

The aspiration is a trustworthy spatial tool. The accepted release is the
smaller promise above, with research paths kept visible but clearly labelled.
