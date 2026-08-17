# MVS-guided DA3 hybrid — IMG_2323

## Purpose

This is a third, independently selectable global world product. It does not
replace calibrated DA3, geometric COLMAP MVS, or their TSDF derivatives.

For each accepted DA3 frame, COLMAP's geometric MVS PLY is depth-tested through
the exact global W2C/K supplied to pose-conditioned DA3. A finite MVS Z-buffer
pixel replaces DA3 depth only at that pixel; DA3 remains unchanged everywhere
MVS has no sample. This exact policy is recorded as
`mvs-zbuffer-where-observed-else-da3/v1`.

The separately selectable guided derivative first estimates a robust depth
ratio in each 24x24 raster tile with at least 16 MVS samples, applies that
ratio to DA3 depth in the supported tile, and still lets exact MVS samples win
at their pixels. Its policy is
`mvs-zbuffer-plus-coarse-local-ratio/v1`. This is local depth guidance only:
it never adjusts COLMAP cameras or invents MVS values in unobserved pixels.

## Locked evidence

| Field | Value |
| --- | ---: |
| Input DA3 product | Accepted calibrated V2, 195 source frames |
| MVS control | Geometric `stereo_fusion` PLY, 1,517,276 vertices |
| Exact / guided raw surfels | 4,962,416 / 4,962,416 |
| Median per-frame MVS pixel coverage | 32.92% |
| Sparse-track depth samples | 214,962 |
| Calibrated / exact / guided global p95 log-depth error | 0.65802 / 0.46634 / 0.44480 |
| Calibrated / exact / guided global median log-depth error | 0.05632 / 0.00957 / 0.00824 |

The lower p95 and median are evidence that globally posed MVS corrects a
material part of DA3's depth error; local guidance improves the exact hybrid
again. The inspector maps COLMAP pixel centres to the 504x336 DA3 raster with
the same half-pixel resize convention used by inference. This is not a
full-room acceptance result: most pixels still originate in DA3, MVS coverage
varies strongly by camera, and the product has not yet passed an independent
floor/wall-planarity gate.

## Publication contract

The importer accepts `vestra.da3-mvs-hybrid/v1` only when all of these bind:

- the accepted V2 source artifact and its canonical selected batch hash;
- the immutable raster and global COLMAP pose hash;
- an MVS PLY SHA-256 and positive vertex count;
- finite positive MVS coverage and an allow-listed exact or local-guidance
  replacement policy;
- hashed PLY and replay depth assets.

The published Studio products are `da3-mvs-hybrid-colmap-surfel` (exact) and
`da3-mvs-guided-colmap-surfel` (guided), with authorities
`colmap-mvs-geometric-plus-da3` and
`colmap-mvs-geometric-plus-da3-local-guidance`. They are intentionally
separate from raw DA3, calibrated DA3, and MVS-only products so visual
smoothing cannot hide a regression.

## Current decision

**Guided is the current visual candidate, but not accepted as a coherent-room
default.** The next quality gate is a global floor/wall residual and
source-camera replay comparison against the calibrated DA3 and MVS-only
controls. Only a winning result can become the default product or receive a
TSDF derivative.
