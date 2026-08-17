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

## Rejected larger temporal-context control

A second, non-published sidecar used the identical raster, model, external
COLMAP cameras, confidence threshold, and first-owner policy, but increased
the temporal batch to 32 views with 8 repeated views. It completed nine
batches and emitted 5,585,928 valid surfels. The comparison is deliberately
kept out of Studio because it does not meet the continuity gate:

| Metric | Accepted 16 / 4 run | 32 / 8 control | Direction |
| --- | ---: | ---: | --- |
| Sparse-track samples | 288,969 | 303,973 | coverage only |
| Median absolute log-depth error after one global scale | 0.09479 | 0.08004 | better |
| p95 absolute log-depth error after one global scale | 0.72233 | 0.69433 | slightly better |
| Median repeated-frame relative error | 6.689% | 6.538% | slightly better |
| p95 repeated-frame relative error | 21.638% | 26.898% | **worse** |

The 32 / 8 run is rejected as a production candidate. More consecutive video
frames are not inherently better context: after a turn, they can show a
different room sector even while remaining temporally close. The next
quality-focused experiment must construct DA3 multi-view groups from the
global COLMAP covisibility graph (shared sparse tracks and compatible camera
directions), not from a fixed temporal window length.

## Covisibility-context control — mixed, not published

A third sidecar replaced temporal grouping with deterministic, bounded groups
selected from shared accepted COLMAP-track observations. It retained the same
220 frames, official model, supplied global cameras, 16-view maximum, four
context views, first-owner policy, confidence threshold, and pixel stride. It
formed 19 groups and emitted a separate artifact at
`/var/roothome/vestra-runs/img-2323-da3-pose-conditioned-covisibility-b16-o4`.

| Metric | Accepted temporal 16 / 4 | Covisibility 16 / 4 | Direction |
| --- | ---: | ---: | --- |
| Sparse-track samples | 288,969 | 329,098 | coverage only |
| Median absolute log-depth error after one global scale | 0.09479 | 0.07791 | better |
| p95 absolute log-depth error after one global scale | 0.72233 | 0.74071 | **worse** |
| Median repeated-frame relative error | 6.689% | 4.311% | better |
| p95 repeated-frame relative error | 21.638% | 14.971% | better |

The new grouping materially improves agreement for repeated views, which is
the desired causal effect, but it does **not** dominate the established
temporal run on both robust global-depth statistics. It is therefore retained
as an immutable diagnostic artifact and is **not imported or published to
Studio**. The next acceptance pass must add held-out depth-track scoring by
context/room sector and reject bad contexts before dense fusion; selecting a
better batch layout alone cannot repair the model's worst local depth failures.

## Direction-gated covisibility control — rejected

The covisibility selector was then constrained so every pair in a DA3 context
had an optical-axis dot product of at least `0.25` (at most 75 degrees apart).
The preflight covered all 220 registered views in 20 groups and verified a
minimum observed pairwise dot product of `0.25033`; this was a valid input
policy, not an execution failure. The resulting artifact is
`/var/roothome/vestra-runs/img-2323-da3-pose-conditioned-covisibility-dir075-b16-o4`.

| Metric | Temporal 16 / 4 | Covisibility | Direction-gated covisibility | Decision |
| --- | ---: | ---: | ---: | --- |
| Median global log-depth error | 0.09479 | **0.07791** | 0.08352 | worse than covisibility |
| p95 global log-depth error | **0.72233** | 0.74071 | 0.74441 | worse |
| Median repeated-frame relative error | 6.689% | **4.311%** | 5.126% | worse than covisibility |
| p95 repeated-frame relative error | 21.638% | **14.971%** | 21.130% | loses the main gain |

This rejects the direction gate as the next production policy. Shared sparse
observations are more predictive than an arbitrary view-angle threshold for
this capture; enforcing a uniform angle splits useful context without fixing
the hard depth outliers. The runner keeps the option for controlled future
experiments, but it is not used by any published product.

## Withdrawn calibration prototype

An initial post-inference sparse-track scale prototype improved the aggregate
diagnostic values, but it was **withdrawn before acceptance**. It did not yet
select one overlap prediction using train-only evidence, did not use the exact
half-pixel raster mapping, and originally allowed failed first-owner frames to
remain in the PLY. Its temporary browser import was immediately restored to
the immutable raw sidecar; it must not be used as a quality result.

The replacement is a separate calibrated-artifact contract: stable track-ID
split, exact resize mapping, train-only overlap candidate selection, held-out
per-frame/context/product gates, accepted-frame-only assets, and a dedicated
Rust publisher that cannot overwrite the raw DA3 product. Until that V2
contract passes, the published DA3 products remain the uncalibrated raw and
TSDF diagnostic layers.
