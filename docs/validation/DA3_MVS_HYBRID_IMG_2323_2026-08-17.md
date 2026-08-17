# MVS-guided DA3 hybrid — IMG_2323

## Purpose

This is a third, independently selectable global world product. It does not
replace calibrated DA3, geometric COLMAP MVS, or their TSDF derivatives.

For each accepted DA3 frame, COLMAP's geometric MVS PLY is depth-tested through
the exact global W2C/K supplied to pose-conditioned DA3. A finite MVS Z-buffer
pixel replaces DA3 depth only at that pixel; DA3 remains unchanged everywhere
MVS has no sample. The policy is recorded as
`mvs-zbuffer-where-observed-else-da3/v1`.

## Locked evidence

| Field | Value |
| --- | ---: |
| Input DA3 product | Accepted calibrated V2, 195 source frames |
| MVS control | Geometric `stereo_fusion` PLY, 1,517,276 vertices |
| Hybrid raw surfels | 4,962,416 |
| Median per-frame MVS pixel coverage | 32.92% |
| Sparse-track depth samples | 214,927 |
| Hybrid global p95 log-depth error | 0.46282 |
| Calibrated DA3 global p95 log-depth error | 0.65535 |

The lower p95 is evidence that globally posed MVS corrects a material part of
DA3's depth error. It is not a full-room acceptance result: most pixels still
originate in DA3, MVS coverage varies strongly by camera, and the product has
not yet passed an independent floor/wall-planarity gate.

## Publication contract

The importer accepts `vestra.da3-mvs-hybrid/v1` only when all of these bind:

- the accepted V2 source artifact and its canonical selected batch hash;
- the immutable raster and global COLMAP pose hash;
- an MVS PLY SHA-256 and positive vertex count;
- finite positive MVS coverage and the exact replacement policy;
- hashed PLY and replay depth assets.

The published Studio product is
`da3-mvs-hybrid-colmap-surfel`, with authority
`colmap-mvs-geometric-plus-da3`. It is intentionally separate from raw DA3,
calibrated DA3, and MVS-only products so visual smoothing cannot hide a
regression.

## Current decision

**Promising but not accepted as a coherent-room default.** The next quality
gate is a global floor/wall residual and source-camera replay comparison against
the calibrated DA3 and MVS-only controls. Only a winning result can become the
default product or receive a TSDF derivative.
