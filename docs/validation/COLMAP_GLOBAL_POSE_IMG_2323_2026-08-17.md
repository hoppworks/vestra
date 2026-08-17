# COLMAP global-pose spike — IMG_2323

## Purpose

Determine whether a global COLMAP trajectory can safely replace Vestra's
window-chained local Sim(3) poses for the existing `IMG_2323` reconstruction.
This is a geometry experiment, not a visual-quality claim.

## Locked evidence

- Source: `IMG_2323.mov`, 1920×1080, 114.426667 seconds.
- DA3 raster cache: 720 immutable `frame-*.ppm` images at 504×336.
- Candidate rate used to recreate the raster contract: 6.292239552865767 fps.
- Raster manifest: `2d163880c229db3c98969a53062cccfb58a05bf81eb55b053f69c16ec297bb76`.
- COLMAP container digest:
  `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`.
- COLMAP settings: CPU SIFT, `SIMPLE_RADIAL`, 16 feature/matching threads,
  sequential overlap 20, quadratic overlap enabled, guided matching enabled.
- Published pose solution:
  `8e06c572624aac739e44922932ba48fdf4c70a8db7a74626e1e1a68898b0b6da`.

## Result

COLMAP's largest sparse component registers 576 of 720 rasters. Vestra's
per-window camera-centre fit reports 80 windows total:

| Metric | Result |
| --- | ---: |
| Windows with a fit (at least 3 registered cameras) | 65 |
| Windows without a valid fit | 15 |
| Normalized camera-fit RMS, median | 0.0608 |
| Normalized camera-fit RMS, p95 | 0.2278 |
| Normalized camera-fit RMS, maximum | 0.3131 |
| Windows over the 0.15 acceptance gate | 19, 23, 26, 32, 38, 57, 69, 70 |
| Windows with fewer than three registered cameras | 46–51, 71–79 |

The TSDF global-world publication was correctly refused. Lowering the gate or
silently interpolating these windows would produce a visually smoother but
geometrically untrustworthy world.

## Decision

Keep the local PR#2-relative product selected. Retain the COLMAP pose solution
and window-fit report as immutable diagnostic evidence. The next pose-provider
spike must improve camera registration/loop closure before it can feed global
fusion. Candidate paths are a COLMAP run with stronger loop/retrieval and
keyframe selection, followed by a GPU SLAM provider if that still fragments the
trajectory. Any provider must import through the same raster/pose contract and
pass the existing per-window global-fit gate before a new product is published.
