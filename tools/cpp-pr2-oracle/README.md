# C++ PR #2 streaming oracle

This standalone CMake target executes the exact `stream_points_core` implementation
from the pinned `depth-anything.cpp` PR #2 revision. It deliberately receives
precomputed depth, confidence, intrinsics, extrinsics, and RGB frames instead of
loading a model. That isolates sliding-window backprojection, confidence gating,
Sim3 seams, frame ownership, and deferred emission from inference parity.

It is an oracle harness, not a production dependency and never changes the C++
checkout passed through `CPP_PR2_SOURCE`.

## Build

```bash
cmake -S tools/cpp-pr2-oracle -B /tmp/vestra-cpp-pr2-build \
  -DCPP_PR2_SOURCE=/path/to/pristine/depth-anything.cpp-pr2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vestra-cpp-pr2-build --target vestra_cpp_stream_fixture_dump --parallel
```

The locked revision is `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`; verify it before
running a comparison.

## Fixture (`VPS1`)

All integers and floats are little-endian. Frames are serialized in global video
order. `depth`, `confidence`, and `RGB` use the exact same processed raster, so
pixel `(u,v)` is a valid overlap correspondence across windows.

| Field | Type |
| --- | --- |
| magic | 4 bytes: `VPS1` |
| version | `u32`, currently 1 |
| frame count, height, width | three `u32` |
| chunk size, overlap | two `u32` |
| confidence percentile | `f64` |
| point-size multiplier | `f32` |
| minimum overlap points | `u32` |
| per frame intrinsics | 9 × `f32`, row-major 3×3 |
| per frame extrinsics | 12 × `f32`, row-major W2C 3×4 |
| per frame depth | `height × width` × `f32` |
| per frame confidence | `height × width` × `f32` |
| per frame colour | `height × width × 3` bytes, RGB row-major |

The harness intentionally locks `global_budget=0`, `icp_refine=false`,
`loop_close=false`, and `metric=false`. These belong to later oracle tiers. The
first tier proves the deterministic sequential stitch path without conflating it
with optional policy/optimization branches.

`make_identity_fixture.py` creates the smallest deterministic control fixture:
four identical calibrated plane views, two 3/1 windows, and one identity seam.
Run it before using model-derived data:

```bash
python3 tools/cpp-pr2-oracle/make_identity_fixture.py /tmp/identity.vps
/tmp/vestra-cpp-pr2-build/vestra_cpp_stream_fixture_dump /tmp/identity.vps /tmp/identity.vpo
```

## Output (`VPO1`)

The header is `VPO1`, `u32 version`, `u32 frame_count`, `u32 height`, `u32 width`,
`u32 point_count`, `u32 window_count`, `i32 warnings`, `i32 loops_found`, and
`f32 metric_scale`. It is followed by C++ `StreamCloud` arrays in this order:

1. `xyz`: `point_count × 3` `f32` in the global OpenCV frame
2. `rgb`: `point_count × 3` bytes
3. `radius`: `point_count` `f32`
4. `counts`: `frame_count` `i32` prefix contribution counts
5. `window_pos`: `window_count × 3` `f32`
6. `window_mid_frame`: `window_count` `i32`
7. `frame_pos`: `frame_count × 3` `f32`
8. `frame_fwd`: `frame_count × 3` `f32`

`VPO1` is an evidence artifact. A Vestra-side reader/comparator must compare it
before voxel fusion: point ownership/counts and seam trajectory first, then
point-cloud distances. Do not claim end-to-end PR parity from model-output parity
alone.
