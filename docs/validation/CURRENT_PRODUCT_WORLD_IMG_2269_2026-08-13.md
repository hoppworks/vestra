# Current product-world validation — IMG_2269 — 2026-08-13

## Scope

This is the current end-to-end product run after the PR #2 terminal-window
schedule correction and the streaming-oracle parity work. It validates the
local Rust pipeline from the user-provided phone video through durable scene
publication and local Studio delivery. It does not claim metric scale, a mesh,
loop closure, ICP, TSDF, or GPU throughput.

## Locked run

| Field | Value |
| --- | --- |
| Input | local user-provided `IMG_2269.MOV`, 40.365 seconds, 1920×1080 |
| Selected frames / reconstruction raster | 120 / 504×336 RGB24 |
| Schedule | 12 views per window, 3-view overlap, **13** terminal-rule windows |
| Retained geometry | minimum confidence `1.0`, pixel stride `8` |
| Model | `depth-anything-base-f32.gguf`, SHA-256 `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da` |
| Host | AMD Ryzen 9 9950X, `RAYON_NUM_THREADS=16`, `-C target-cpu=znver5` |
| Vestra | `554707b01a734c3bfd8b49afb8018d26087d2088` |
| Vestra Engine / Kernels | `e6c8d0fd566a37f8bc15d030d9ac99370e748df9` / `9740a98dffc3aedf6611f987c1f67ff272f894d7` |

The scene is local to the Workhorse at
`/var/roothome/vestra-runs/img-2269-current-13w.vestra`; it is not committed
because it contains user-video-derived geometry and decoded imagery.

## Reproduction command

```bash
RAYON_NUM_THREADS=16 ./target/release/vestra reconstruct \
  --video IMG_2269.MOV \
  --model depth-anything-base-f32.gguf \
  --output img-2269-current-13w.vestra
```

## Observed result

```text
decoded_frames: 120
inferred_windows: 13
measured_points: 412,776
fused_points: 296,596
capture: ready (mean adjacent luma delta 0.0970028564)
```

`vestra inspect` reported a `fused_relative_world` with finite geometry, 12
sequential seams, 95,256 direct source-pixel correspondences, 93,050 accepted
inliers, scale range `0.1247817…1.3250973`, and six progressive binary surfel
chunks. No loop was accepted for this capture, so `pose_graph: null` is the
correct conservative result.

## Local Studio delivery

The current release served the fresh scene on `127.0.0.1` and was queried
directly:

| Endpoint | Result |
| --- | --- |
| `/manifest.json` | `vestra.scene/v1`, 13 measured chunks, 6 binary fused chunks |
| first fused binary chunk | 2,000,012 bytes (`12`-byte header + 50,000 `40`-byte surfels) |
| `/evidence.json` | 156 finite camera rays and 12 sequential seam links |

This confirms the current Rust pipeline publishes a browser-consumable local
relative-scale surfel world. Visual/semantic quality remains capture-dependent
and is not inferred from these delivery checks.

## Relationship to the C++ oracle

The same corrected 120-frame / 13-window model evidence was independently run
through the pinned `depth-anything.cpp` PR #2 base stream. Its 9,931,557
ordered pre-voxel points match the Rust reference-only oracle emitter with
exact ownership and RGB, position MAE `1.0073384563168364e-7`, and radius MAE
`9.551295881469472e-11`. See
[the streaming-oracle record](CPP_PR2_STREAMING_ORACLE_2026-08-13.md).
