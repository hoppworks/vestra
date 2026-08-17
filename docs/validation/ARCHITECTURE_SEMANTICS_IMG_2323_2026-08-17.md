# Architecture semantics — IMG_2323

## Purpose

This is an evidence-generation validation, not a completed architecture-world
claim.  It validates that a semantic sidecar can label exactly the decoded
Vestra rasters before any geometry is selected or meshed.

## Input

- Scene: `img-2323-keyframes-da05305.vestra`
- Decoded frames: 230 × 504 × 336 RGB PPM
- Runner: `vestra.tools.run_architecture_semantics/1`
- Model: `nvidia/segformer-b5-finetuned-ade-640-640`
- Resolved revision: `739f5d4692954e4a185eac280dec1ba5a7d52f1d`
- Recorded terms: `research-only`
- Device: NVIDIA GeForce RTX 5080

## Result

The runner wrote the compressed sidecar
`/var/roothome/vestra-runs/img-2323-architecture-semantics-segformer-b5/`
on Workhorse:

| Class | Pixels across 230 frames |
|---|---:|
| Floor | 5,153,302 |
| Wall | 18,545,415 |
| Ceiling / roof | 5,595,240 |
| Door / opening | 2,543,824 |
| Window | 1,503,810 |
| Non-architectural | 5,607,529 |

`masks.npz` stores 78,898,240 bytes of class/confidence data before
compression.  Its raster dimensions match the decoded geometry contract.

## Negative control

The prior geometry-only RANSAC product
`colmap-mvs-geometric-architecture` was inspected in Studio and rejected.  It
emitted 12 fitted plane fragments / 6,923 surface cells, but did not represent
continuous architectural surfaces reliably.  It remains an experimental,
non-default product and is not evidence that walls or doors have been solved.

## Next gate

Before publishing a new architecture product, reproject the global MVS points
through the registered COLMAP cameras, require multi-view semantic agreement,
then fit and polygonise per-class supported planes.  Door masks only cut a
hole where a wall plane is otherwise supported and the geometric depth also
shows an opening.
