# Frame-global COLMAP rebase — IMG_2323

## Purpose

This validation records the first Vestra world that does **not** compose local
DA3 windows with sequential Sim(3) transforms. COLMAP global bundle adjustment
is the camera authority. Vestra Engine supplies dense per-frame relative depth;
COLMAP sparse tracks calibrate depth scale independently for each accepted
frame, and each dense observation is reprojected through that frame's global
camera.

The result is still relative-scale reconstruction. It is not a metric claim and
it is not evidence that every room surface is correct.

## Locked input

| Field | Value |
| --- | --- |
| Source | `IMG_2323.mov` |
| Source raster | 1920×1080, 114.427 s |
| COLMAP raster | central 1620×1080 crop, matching Vestra's 504×336 3:2 raster contract |
| Canonical DA3 frames | 230 |
| COLMAP global-BA registered frames | 220 |
| COLMAP sparse tracks retained | 50,887 |
| Pose solution | `e931ea6a82a354e46e308aa3146ca99064112d1c3ccb4f2b9f5b4459a5dee5e9` |

## Per-frame acceptance contract

For each canonical frame, Vestra maps source pixels through the exact
half-pixel resize contract, compares dense local camera depth to COLMAP track
depth, fits a robust scale on a deterministic train split, and validates it on
the held-out split.

| Gate | Value |
| --- | --- |
| Minimum independent scale samples | 12 |
| Maximum sparse-track reprojection error | 2.5 px |
| Maximum held-out median log-depth error | 0.20 |
| Minimum accepted frame coverage | 85% |

Results from `inspect-colmap-frame-global`:

| Result | Value |
| --- | --- |
| Accepted global frames | 200 / 230 (86.96%) |
| Median held-out log-depth error | 0.081776 |
| p95 held-out log-depth error | 0.253264 |
| Frames with no COLMAP registration | 131, 132, 151, 193–197, 202, 203 |

Frames failing either registration, evidence, or held-out scale quality are
omitted. Vestra does not interpolate camera poses or silently substitute the
legacy sequential window path.

## Published derived product

| Field | Value |
| --- | --- |
| Product ID | `colmap-ba-frame-global-active` |
| Camera authority | `colmap-ba-frame-global` |
| Surface | normal-space TSDF surfels |
| TSDF evidence budget | 6,000,000 deterministic frame/pixel observations maximum |
| Published surfels | 76,108 |
| Browser chunks | 2 binary surfel chunks |

The complete raw measured scene remains immutable. The older
`local-pr2-relative` product is still selectable in Studio for direct visual
comparison; it was not overwritten.

## Browser integration result

Studio's primary intake route is `/world/`. Scene assets must be resolved
relative to that route; absolute root paths load the wrong job-world manifest.
The current Studio implementation therefore loads `manifest.json`, chunks,
replay frames, source frames, and camera controls relative to `/world/`.

The binary surfel loader is syntax-checked with Node and tested through the
Studio route tests. Its radius attribute is explicitly one float per vertex.

## Interpretation

This run eliminates the known 79-seam sequential-window trajectory from the
published global product. It does not, by itself, prove that the floor is flat:
the next quality check must compare held-out geometric residuals and inspect
the global product at fixed camera views. If this COLMAP gate becomes unstable
on another interior capture, evaluate DROID-SLAM or VGGT as a pose provider;
do not use DA3 Streaming alone as a substitute for global SLAM/SfM.
