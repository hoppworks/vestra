# COLMAP dense-MVS control — IMG_2323

## Purpose

This is an isolated geometry-control experiment for the frame-global Vestra
route. It does not modify a `.vestra` scene and it does not count as a Vestra
product until its geometry has been inspected.

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
5. Inspect the raw MVS cloud from multiple original-camera views before any
   smoothing, TSDF, or conversion to a Vestra product.

The first PatchMatch pass must be recorded as photometric initialization if
the installed COLMAP build reports `geom_consistency: 0` before neighbour maps
exist. A later geometric fusion may only be labelled geometric when the log
explicitly reports `geom_consistency: 1` and a geometric fused PLY exists.

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
