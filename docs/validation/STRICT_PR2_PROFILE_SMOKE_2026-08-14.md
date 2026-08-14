# Strict PR #2-relative capture smoke — 2026-08-14

## Purpose

This is an end-to-end execution gate for Vestra's dense PR #2-relative capture
profile. It verifies that a real landscape phone video can pass through frame
selection, multi-view inference, dense raw evidence persistence, PR #2-relative
seam/loop preparation, normal-space TSDF fusion, and durable scene publication.
It is **not** a C++ differential result: the C++ pipeline has not yet been run
on this exact decoded-frame set.

## Environment

- Host: AMD Ryzen 9 9950X Workhorse, `RAYON_NUM_THREADS=16`.
- Vestra revision: `9a23b17f814b702882accff8d6c62af7019c5694`.
- Engine revision: `e6c8d0fd566a37f8bc15d030d9ac99370e748df9`.
- Kernels revision: `9740a98dffc3aedf6611f987c1f67ff272f894d7`.
- Model: `depth-anything-base-f32.gguf`, SHA-256
  `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`.
- Input: local landscape `IMG_2310.mov`, SHA-256
  `839dd5d18b2f78f28703c688df5569f9d6bb33786f1e299051df99f52ed0b264`.

## Command

```bash
RAYON_NUM_THREADS=16 ./target/release/vestra reconstruct \
  --video IMG_2310.mov \
  --model depth-anything-base-f32.gguf \
  --output /tmp/vestra-pr2-profile-smoke.vestra \
  --frames 24 \
  --cpp-pr2-relative \
  --tsdf
```

## Result

| Stage | Result |
|---|---:|
| Decoded frames | 24 from 21.993 s |
| Raster and schedule | 504×336, 12-frame windows, 3-frame overlap |
| Immutable measured windows | 3 |
| Dense measured points | 5,080,320 |
| Final normal-space TSDF surfels | 16,722 |
| Capture indicator | `ready` (mean adjacent luma delta `0.1452706754`) |
| Output size | 929 MiB |

`vestra inspect` confirmed finite fused points, a relative-scale coordinate
contract, and two sequential alignments with 1,016,064 correspondences each.
No pose-graph loop was proposed in this short 24-frame sample, which is
expected: it is a profile and persistence smoke, not a closed-walk regression.

## Interpretation and next gate

The strict mode deliberately emits one raw sample per finite-depth pixel and
persists each window's PR #2 confidence percentile. This makes the scene much
larger than the sparse Studio default, but prevents confidence-selection drift
between loop keys and final first-owner emission.

The next acceptance gate is a paired C++/Rust run using an identical locked
decoded-frame directory, model, precision, 12/3 schedule, geometry flags, and
thread budget. It must compare pre-voxel ordered emission, trajectory, TSDF
ordering, and stage timings before any end-to-end speed claim is made.

## First paired C++ probe — not accepted

The pinned C++ C API was subsequently run over the exact persisted `decoded/`
directory using the same F32 model, 16 threads, `chunk=12`, `overlap=3`,
55th-percentile confidence selection, seam ICP, loop closure, and TSDF fusion.
The versioned `CPS1` harness reported 16,643 surfels; Vestra reported 16,722.
Both runs found no loop in this short clip. This 79-surfel difference means
end-to-end video parity is **open**. The harness and output SHA-256 are retained
for diagnosis (`df2436cacff65f81df324a131929f0a4262c2aa7dfacbc8985068bf83cfa2704`);
no relative performance or parity claim is made from this probe.

## Raw-geometry differential — resolved 2026-08-14

The C++ C-API probe alone mixes two distinct boundaries: DA3 inference and
streaming geometry.  To prevent a cloud-count difference from being attributed
to the wrong layer, Vestra captured a `VPS1` fixture from its exact 24-frame
Rust inference output and replayed that immutable fixture through both
implementations of the PR #2 streaming geometry.

| Input / result | Value |
|---|---:|
| Frames / schedule | 24 / 12-frame windows with 3-frame overlap |
| C++ raw points from the shared `VPS1` | 1,902,168 |
| Rust raw points from the shared `VPS1` | 1,902,168 |
| Per-frame ownership counts | Exact match |
| RGB mismatches in ordered emission | 0 |
| Ordered position MAE / max | 5.99e-9 / 2.38e-7 |
| Radius MAE / max | 0 / 0 |
| Camera-position MAE / max | 6.42e-9 / 1.19e-7 |
| Camera-forward MAE / max | 1.11e-8 / 5.96e-8 |

This accepts the Rust PR #2 **raw geometry** path for the captured tensor
fixture: confidence percentile, inverse/backprojection boundary, sequential
Sim(3), first-owner scheduling, radii, colours and frame trajectory agree with
the pinned C++ implementation to F32 output precision.

The live C-API comparison remains deliberately open.  With fresh inference in
each runtime, C++ emitted 1,902,167 raw points and Rust emitted 1,902,168;
their camera positions still agreed within 2.39e-6.  The one-pixel-per-frame
selection differences shift ordered cloud indices, so the resulting raw
position/RGB mismatch is not evidence against the accepted geometry replay.
It is evidence that the next parity gate belongs at the DA3 model-output
boundary (depth, confidence, intrinsics and extrinsics), before more work is
done on the stitcher or TSDF.

The comparison is executable, not a manual analysis:

```bash
vestra oracle-compare --fixture real.vps --reference cpp-raw.vpo
vestra oracle-compare-capi --fixture real.vps --reference cpp-raw.cps
```

`CPS1` is produced by `tools/cpp-pr2-oracle/capi_stream_dump.cpp`; its command
now records the fusion, ICP and loop switches explicitly, so a reference run
cannot silently use a different geometry branch.

## Multi-view model-output differential — open numerical work

`MVO1` records the actual `Engine::depth_pose_multi` output for every C++
window, before thresholding or geometry. The matching Rust input is the same
`VPS1` fixture, so the comparison does exercise cross-view global attention
rather than a misleading single-image path.

On the same 24-frame F32 workload (three windows; 5,080,320 depth/confidence
values), the current result is:

| Tensor family | MAE | Maximum absolute delta |
|---|---:|---:|
| Depth | 1.67e-6 | 8.49e-5 |
| Confidence | 3.27e-5 | 7.51e-4 |
| Extrinsics | 5.42e-7 | 8.46e-6 |
| Intrinsics | 1.22e-4 | 1.31e-3 |

All tensor shapes and the 24-frame / 12-3 schedule match. The largest
user-visible divergence is confidence near its percentile boundary: C++ and
Rust selected respectively 914,459 and 914,458 pixels in the first window;
the threshold difference was 7.44e-5. The next two windows selected equal
totals but not necessarily the same boundary pixels. This explains the one
point difference in live first-owner emission without assigning a false fault
to geometry.

The next implementation gate is therefore **DPT confidence numerical parity**.
It must improve the `MVO1` confidence values without degrading depth or pose,
then demonstrate identical per-window confidence selection and exact ordered
raw CAPI cloud counts. The direct command is:

```bash
vestra oracle-compare-model --fixture real.vps --reference cpp.mvo
```

The C++ producer is `tools/cpp-pr2-oracle/multiview_dump.cpp`; it calls
`Engine::depth_pose_multi` directly, with no C-API cloud/geometry work.
