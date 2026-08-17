# DA3 pose-conditioned COLMAP world — IMG_2323

## Purpose

This experiment tests the next causal link after the IMG_2323 global-pose
control: whether official DA3 can predict depth while the accepted global
COLMAP bundle-adjustment cameras are already authoritative. It does **not**
chain local DA3 windows through sequential Sim(3) registrations.

The resulting surfel world is a separately named Studio product. It does not
replace raw measurements, the local relative reconstruction, the COLMAP MVS
control, or any TSDF derivative.

## Frozen evidence

| Item | Value |
| --- | --- |
| Source raster / COLMAP pose | IMG_2323 immutable 504×336 raster, pose solution `e931ea6a82a354e46e308aa3146ca99064112d1c3ccb4f2b9f5b4459a5dee5e9` |
| Registered frames | 220 |
| Model | Official `depth-anything/DA3-BASE` Python API |
| Camera input | The exact per-frame globally bundle-adjusted COLMAP W2C and pinhole K approximation, rescaled from the 1620×1080 COLMAP crop into the immutable 504×336 DA3 raster |
| DA3 process resolution | 504×336, `upper_bound_resize` |
| External geometry mode | `align_to_input_ext_scale=True` |
| GPU | NVIDIA GeForce RTX 5080, CUDA 12.8, PyTorch `2.7.1+cu128` |
| Batch policy | 16 views, 4-view overlap, first-owner emission |
| Surface sampling | Confidence percentile 40, pixel stride 2 |
| Artifact | `/var/roothome/vestra-runs/img-2323-da3-pose-conditioned-Kscaled-90e1813` |

## Result

The sidecar completed 18 bounded batches and produced a SHA-256-bound binary
PLY with **5,543,193** valid surfels. The Rust importer verified the raster
fingerprint, pose-solution hash, external-scale flag, PLY name, PLY digest,
and exact registered source-frame list before publishing:

```text
Studio product: da3-pose-conditioned-colmap-surfel
Pose authority: da3-base-pose-conditioned-colmap
Surface mode: surfel
Source cameras: 220
```

Studio exposes the original-camera evidence for this independent world. It
does not show legacy DA3 replay geometry as if that were an equivalent global
raw layer.

The product additionally retains **220 colourized 504×336 depth rasters** as
display derivatives of the real DA3 float32 depth outputs. Studio serves them
through the selected product only and synchronizes them to the original video;
they are not reconstructed from the point cloud. The raw surfel and TSDF
products remain independently selectable.

## Verification performed

1. A three-frame GPU smoke completed model inference under supplied cameras
   and generated 76,141 valid surfels. It exposed and fixed the documented
   DA3 output convention (`[N,3,4]` W2C) by normalizing it to homogeneous
   `[N,4,4]` before backprojection.
2. The initial full artifact was deliberately superseded: it passed the
   1620×1080 COLMAP intrinsics directly to 504×336 images. Revision `90e1813`
   scales `fx`/`cx` by `504 / 1620` and `fy`/`cy` by `336 / 1080` before any
   inference. A GPU smoke verifies the resulting `K` (for example,
   `fx≈272.811`, `cx=252`, `cy=168`), and the replacement artifact is the only
   one referenced by the published product.
3. The full runner validated every selected decoded-raster SHA-256 before GPU
   work; its manifest records every batch hash and every supplied camera.
4. The Rust import command revalidated artifact identity before publication.
5. The live Studio endpoints report the selected product, 5,543,193 points,
   80 spatial-preview chunks, 220 camera rays, and 220 source frames.
6. The live product endpoint verifies all 220 retained depth frames and serves
   frame zero as a 504×336 RGB BMP (`508,086` bytes) for browser replay.

## Remaining acceptance gates

This run proves an end-to-end pose-conditioned product, **not** that the room
is coherent. Assess it against the existing MVS control and frame-global
baseline at four source-camera poses using held-out COLMAP-track depth,
cross-batch boundary continuity, floor-plane residual, and wall/door
alignment. A separately named normal-space TSDF product has been published
only as a rendering derivative: it sampled 923,866 of the raw observations
and contains 56,367 surfels. It cannot count as evidence that the trajectory
or depth is geometrically correct.

Metric scale remains out of scope: the world is explicitly relative-scale.
