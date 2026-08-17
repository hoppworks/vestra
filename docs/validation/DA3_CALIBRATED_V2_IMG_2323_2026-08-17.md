# Calibrated DA3 V2 world — IMG_2323

## Decision

**Accepted as a separate relative-scale DA3 product; not accepted as proof of
a fully coherent room.** The calibrated product is deliberately additive: it
does not replace the immutable raw DA3, COLMAP MVS, or local-relative worlds.

## Locked input and contract

| Field | Value |
| --- | --- |
| Raw artifact | `img-2323-da3-pose-conditioned-Kscaled-90e1813` |
| Raster / pose | Immutable 504×336 raster; global BA pose `e931ea6…dee5e9` |
| Registered frames | 220 |
| Calibrated schema | `vestra.da3-pose-conditioned-calibration/v2` |
| Pixel mapping | `pixel-center-resize/v1` |
| Track split | `sha256-track-id-fold/v1` |
| Reprojection maximum | 2.5 px |
| Minimum train / held-out tracks | 24 / 6 |
| Per-frame held-out median maximum | 0.20 log-depth |
| Minimum accepted coverage | 85% |
| Overlap p95 maximum | 22% relative depth |

The candidate selection uses only training-landmark median residual, training
support, batch index, and source slot. Held-out landmarks do not choose an
overlap candidate. The one selected prediction per accepted frame is then
calibrated and emitted; rejected frames and non-canonical overlap copies do
not enter PLY, replay rasters, or TSDF.

## Outcome

| Metric | Raw DA3 | V2 calibrated | Result |
| --- | ---: | ---: | --- |
| Published source frames | 220 | 195 / 220 | 88.64%, gate passed |
| Raw surfels | 5,543,193 | 4,949,833 | accepted frames only |
| Held-out global median log-depth error | 0.09479 | **0.05620** | better |
| Held-out global p95 log-depth error | 0.72233 | **0.65535** | better |
| Overlap median relative error | 6.689% | **3.490%** | better |
| Overlap p95 relative error | 21.638% | **16.538%** | gate passed |
| Calibrated TSDF observations | — | 989,967 | bounded, regular frame/pixel sample |
| Calibrated TSDF surfels | — | 52,383 | rendering derivative |

Published IDs:

- `da3-pose-conditioned-colmap-calibrated-surfel`
- `da3-pose-conditioned-colmap-calibrated-tsdf`

The Studio verification loaded both products through `/world/`, showed 195
source-frame evidence for the raw calibrated product, and reported no browser
console warnings or errors.

## Limit

V2 removes a measured per-frame scale bias under a fixed global COLMAP camera
trajectory. It cannot reconstruct occluded content, undo local depth-shape
bias, or establish metric scale. The TSDF layer must not be interpreted as a
mesh or as evidence that floors/walls are globally planar.

The next geometry step is a controlled fusion comparison where COLMAP's
geometric MVS depth is trusted where it exists and pose-conditioned DA3 only
fills explicitly unsupported regions. It must score held-out sparse tracks,
camera reprojection coverage, and floor/wall residuals before publication.
