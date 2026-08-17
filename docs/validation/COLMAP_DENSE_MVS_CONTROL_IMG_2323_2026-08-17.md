# COLMAP dense-MVS control — IMG_2323

## Purpose

This is an isolated geometry-control experiment for the frame-global Vestra
route. It publishes a separately labelled, immutable browser product; it does
not modify DA3 measurements or replace a Vestra-derived world.

It answers a narrow causal question: **does the accepted COLMAP global bundle
adjustment support a coherent dense reconstruction for this capture?**

- A coherent MVS cloud means the camera authority is viable and the remaining
  Vestra problem is locally predicted DA3 geometry. The next experiment is
  pose-conditioned DA3 depth using these cameras.
- A warped or fragmented MVS cloud means the input trajectory/correspondence
  solution is not good enough. Do not try to conceal that with TSDF, point
  size, or another local Sim(3) chain.

## Frozen inputs

| Item | Value |
| --- | --- |
| Source video | `IMG_2323.mov` evidence cache |
| Source raster contract | centred 1920×1080 → 1620×1080 crop, then Vestra 504×336 inference raster |
| COLMAP model | `/var/roothome/vestra-runs/img-2323-keyframes-da05305.colmap-source-1620-r2/global-ba` |
| Registered cameras | 220 |
| Vestra accepted camera subset | 200 |
| Execution | pinned `docker.io/colmap/colmap:latest` CUDA container on the RTX 5080 |
| Output workspace | `/tmp/vestra-mvs-img2323-dense-r1` |

The `/tmp` workspace is deliberate: the source tree lives below a protected
remote home directory that the SELinux-confined container cannot traverse. The
workspace is a copied, read-only input; it must never overwrite the source
pose solution or a Vestra artifact.

## Procedure

1. `image_undistorter` consumes the accepted global BA model and produces a
   COLMAP dense workspace.
2. `patch_match_stereo` creates the initial photometric depth maps for every
   registered image.
3. Run the geometric-consistency pass after neighbouring depth maps exist.
4. `stereo_fusion --input_type geometric` writes `fused.ply`.
5. `vestra import-colmap-mvs --scene <scene> --ply fused.ply --pose-solution <hash>` publishes a
   verified geometric cloud as the independent `colmap-mvs-geometric` Studio
   product. If the provider emitted only photometric maps, the caller must add
   `--photometric`; the result is then named `colmap-mvs-photometric-control`.
   Neither route mutates DA3 measurements nor replaces a Vestra-derived product.
6. The pose solution attaches only its registered source frames and calibrated
   camera rays to the MVS product. Studio keeps the DA3 replay hidden: MVS has
   no DA3 depth replay to show.
7. Inspect the raw MVS cloud from multiple original-camera views before any
   smoothing, TSDF, or conversion to a Vestra product.

For a reproducible, non-browser-only inspection, render at least four
temporally separated registered cameras through the immutable pose solution:

```sh
python3 tools/inspect_colmap_mvs.py \
  --ply /tmp/vestra-mvs-img2323-dense-r1/fused-geometric.ply \
  --pose-solution <scene>/chunks/pose-<pose-solution-hash>.json \
  --frames 0 55 110 165 \
  --output /tmp/vestra-mvs-img2323-camera-inspection \
  --maximum-points 6000000
```

The tool writes colour PPM reprojections plus `inspection.json` containing
depth-tested coverage per source camera.  Those images are inspection evidence
only; the tool cannot publish, alter, or smooth the cloud.

The first PatchMatch pass must be recorded as photometric initialization if
the installed COLMAP build reports `geom_consistency: 0` before neighbour maps
exist. A later geometric fusion may only be labelled geometric when the log
explicitly reports `geom_consistency: 1` and a geometric fused PLY exists.

## Current geometric control run

The initial photometric-only control remains preserved as
`colmap-mvs-photometric-control`. It was followed by a separate geometric
PatchMatch/fusion pass in the same immutable copied workspace:

| Item | Result |
| --- | --- |
| Geometric fusion command | `stereo_fusion --input_type geometric` |
| Fused points | 1,517,276 |
| Published product | `colmap-mvs-geometric` |
| Source-camera evidence | 220 registered cameras and source frames |
| Camera reprojection coverage | 50.43% (frame 0), 30.47% (55), 27.41% (110), 15.24% (165) |
| Product classification | **Partial** |

The four depth-tested MVS reprojections show stable room structure in the
well-observed areas: ceiling beams, chair/table geometry, floor, door/window
frames, and furniture appear in compatible locations under their original
calibrated cameras. The final sampled view has much lower coverage, so the
cloud is not complete enough to claim a coherent full-room reconstruction.
It is nevertheless decisive negative evidence against the old explanation
that the global COLMAP trajectory itself necessarily produces a spiral: a
global multi-view geometry control can preserve substantial room structure
without any window-to-window Sim(3) chaining.

Studio now keeps this as a distinct raw MVS surfel product and its `match 3D
camera` action uses the exact COLMAP pose, roll, vertical field of view, and
3:2 camera aspect. The visual point size is intentionally reduced in this
mode so the MVS evidence is not hidden by oversized surfel discs. This is a
viewer-only projection rule; it does not alter the PLY or camera authority.

The held-out sparse-track/dense-depth comparison remains unexecuted. That
means this run is a quality-control and trajectory decision, not an accuracy
certificate or a metric-scale claim.

## Acceptance evidence

The control is useful only with all of the following evidence:

- `fused.ply` exists and its vertex count is recorded.
- Original-camera reprojections show walls, floor, door openings, and furniture
  in mutually consistent locations across temporally separated frames.
- Held-out COLMAP tracks reproject to compatible dense depth.
- The cloud is inspected from at least four original camera poses, not solely
  from a free orbit camera.
- The result is explicitly classified as `coherent`, `partial`, or `failed`.

No screenshot, point count, or TSDF derivative by itself satisfies this gate.

## Non-goals

- This does not benchmark Vestra Engine.
- This does not replace the Rust renderer or the raw/surfel/TSDF product model.
- This does not prove metric scale.
- This does not change the existing frame-global COLMAP-derived world.
