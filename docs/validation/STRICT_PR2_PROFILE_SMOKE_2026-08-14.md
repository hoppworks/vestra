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
