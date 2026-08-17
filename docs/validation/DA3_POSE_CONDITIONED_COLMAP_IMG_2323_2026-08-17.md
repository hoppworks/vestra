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
| Camera input | The exact per-frame globally bundle-adjusted COLMAP W2C and pinhole K approximation |
| DA3 process resolution | 504×336, `upper_bound_resize` |
| External geometry mode | `align_to_input_ext_scale=True` |
| GPU | NVIDIA GeForce RTX 5080, CUDA 12.8, PyTorch `2.7.1+cu128` |
| Batch policy | 16 views, 4-view overlap, first-owner emission |
| Surface sampling | Confidence percentile 40, pixel stride 2 |
| Artifact | `/var/roothome/vestra-runs/img-2323-da3-pose-conditioned-dff3b27` |

## Result

The sidecar completed 18 bounded batches and produced a SHA-256-bound binary
PLY with **5,388,583** valid surfels. The Rust importer verified the raster
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

## Verification performed

1. A three-frame GPU smoke completed model inference under supplied cameras
   and generated 76,141 valid surfels. It exposed and fixed the documented
   DA3 output convention (`[N,3,4]` W2C) by normalizing it to homogeneous
   `[N,4,4]` before backprojection.
2. The full runner validated every selected decoded-raster SHA-256 before GPU
   work; its manifest records every batch hash and every supplied camera.
3. The Rust import command revalidated artifact identity before publication.
4. The live Studio endpoints report the selected product, 5,388,583 points,
   80 spatial-preview chunks, 220 camera rays, and 220 source frames.

## Remaining acceptance gates

This run proves an end-to-end pose-conditioned product, **not** that the room
is coherent. Assess it against the existing MVS control and frame-global
baseline at four source-camera poses using held-out COLMAP-track depth,
cross-batch boundary continuity, floor-plane residual, and wall/door
alignment. Only then should a TSDF derivative be published for this product.

Metric scale remains out of scope: the world is explicitly relative-scale.
