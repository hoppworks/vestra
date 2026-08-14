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

All integers and floats are little-endian. `depth`, `confidence`, and `RGB`
use the exact same processed raster, so pixel `(u,v)` is a valid overlap
correspondence across windows. Payloads are window-scoped: an overlapping
source frame is re-inferred in each DA3 multi-view window and must therefore
not be represented by one global prediction.

| Field | Type |
| --- | --- |
| magic | 4 bytes: `VPS1` |
| version | `u32`, currently 3 (`2` remains readable) |
| frame count, height, width | three `u32` |
| chunk size, overlap | two `u32` |
| confidence percentile | `f64` |
| point-size multiplier | `f32` |
| minimum overlap points | `u32` |
| branch bitmap | `u32`, V3 only: bit 0 ICP, bit 1 loop closure |
| window count | `u32`, must equal the schedule-derived count |
| per window view count | `u32`, must equal that window's length |
| per window/view intrinsics | 9 × `f32`, row-major 3×3 |
| per window/view extrinsics | 12 × `f32`, row-major W2C 3×4 |
| per window/view depth | `height × width` × `f32` |
| per window/view confidence | `height × width` × `f32` |
| per window/view colour | `height × width × 3` bytes, RGB row-major |

The harness locks `global_budget=0` and `metric=false`. V2 fixtures and V3
fixtures with a zero branch bitmap prove the deterministic sequential path. V3
can opt into the reference's ICP and loop-closure branches; the executable does
not accept an arbitrary policy flag that is absent from the evidence artifact.

`make_identity_fixture.py` creates the smallest deterministic control fixture:
four identical calibrated plane views, two 3/1 windows, and one identity seam.
Run it before using model-derived data:

```bash
python3 tools/cpp-pr2-oracle/make_identity_fixture.py /tmp/identity.vps
/tmp/vestra-cpp-pr2-build/vestra_cpp_stream_fixture_dump /tmp/identity.vps /tmp/identity.vpo

# Optional normal-space TSDF branch used by the PR #2 C API.
/tmp/vestra-cpp-pr2-build/vestra_cpp_stream_fixture_dump /tmp/identity.vps /tmp/identity-tsdf.vpo --tsdf

# Diagnostic-only phase timing. This does not change the VPO1 output.
/tmp/vestra-cpp-pr2-build/vestra_cpp_stream_fixture_dump /tmp/identity.vps /tmp/identity-tsdf.vpo --tsdf --profile
```

`--profile` writes fixture-read, PR #2 stream, TSDF, VPO-write, and total
durations to stderr. It is for phase attribution only; use the locked
fresh-process benchmark protocol for comparative wall-clock claims.

## Model-only multi-view benchmark

`vestra_cpp_multiview_bench` and `vestra oracle-model-bench` are paired
diagnostic runners. Both load the F32 model and canonical RGB24 PPM cache once
before timing, then repeatedly execute only the same PR #2 multi-view
depth/confidence/pose windows. Neither runner writes model outputs in the
timed interval.

```bash
vestra_cpp_multiview_bench MODEL.gguf decoded/ 16 12 3 1 10
vestra oracle-model-bench --model MODEL.gguf --decoded decoded/ \
  --frames 24 --width 504 --height 336 --chunk-size 12 --overlap 3 \
  --warmup 1 --repeat 10
```

The runner returns raw per-repeat milliseconds plus a checksum to keep the
computed tensors observable. It is a stage benchmark; run the two binaries in
fresh randomized process trials and retain their full JSON outputs before
claiming an implementation speed comparison.

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

Passing `--tsdf` applies the exact PR #2 `fuse_tsdf` default profile after
streaming. The result remains `VPO1`; its point sequence and per-frame counts
describe the frame-major, first-observing TSDF surface. This makes TSDF a
separate, differential parity tier rather than an unrecorded viewer-only step.
