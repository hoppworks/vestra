# Closed-loop geometry oracle — 2026-08-14

## Purpose

This is a model-free differential fixture for the pinned
`depth-anything.cpp` PR #2 geometry path. It prevents a world viewer from
masking global trajectory drift with a visually plausible point cloud.

The fixture serializes the deterministic synthetic-room regression from the
reference's `tests/test_stream_loop.cpp` as window-scoped `VPS1` v3 evidence:

- 60 frames, orbit period 48; frames 48–59 revisit the first sector;
- 160×120 calibrated raycast views;
- 12-frame windows with 4-frame overlap (seven windows);
- 0.5 percent deterministic depth noise;
- a 0.6 degree yaw plus 2 cm overlap-only pose perturbation per later window;
- loop closure enabled, metric scale disabled.

No neural inference, metre claim, or generated geometry is involved.

## Commands

```bash
cmake -S tools/cpp-pr2-oracle -B /tmp/vestra-cpp-pr2-build-loop \
  -DCPP_PR2_SOURCE=/tmp/vestra-pr2-reference-f56e9be \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vestra-cpp-pr2-build-loop \
  --target vestra_cpp_closed_loop_fixture vestra_cpp_stream_fixture_dump --parallel 8

/tmp/vestra-cpp-pr2-build-loop/vestra_cpp_closed_loop_fixture /tmp/vestra-closed-loop.vps
/tmp/vestra-cpp-pr2-build-loop/vestra_cpp_stream_fixture_dump \
  /tmp/vestra-closed-loop.vps /tmp/vestra-closed-loop.vpo
cargo run -q -p vestra-cli -- oracle-stitch --input /tmp/vestra-closed-loop.vps
```

The C++ source must resolve to
`f56e9be43a22c12ef575584d2fa57a6a5d5be7ae` before any result is accepted.

## Baseline result

The C++ reference accepted two non-adjacent loop edges (`0<->5`, `0<->6`) and
reduced its pose-graph cost from `0.16` to `0.01521`. It emitted 921,600
pre-voxel first-owner points without warnings.

Vestra's older automatic loop policy accepted zero edges on the exact same
recorded views. That policy remains deliberately separate from the reference
oracle: it uses different radius-relative gates and must not be presented as
PR #2 parity.

The dedicated Rust oracle now accepts the same two edges (`0<->5`, `0<->6`)
after first-owner key sampling, many-to-one seed matching, scale-locked
Umeyama, and iterative rank-truncated point-to-plane ICP. Its pose-graph cost
is `0.158411 -> 0.014370`; it is close to, but not numerically identical with,
the C++ result because this fixture currently uses F32 stored evidence in Rust
and F64 geometry in the reference. Exact trajectory/point tolerances are the
next gate. This is oracle-tier evidence only; it is not wired into Vestra's
product reconstruction settings yet.

## Fixture interchange

`VPS1` v3 adds a branch bitmap after `minimum_overlap_points`:

- bit 0: per-seam point-to-plane ICP;
- bit 1: loop closure and relative Sim(3) pose graph.

V2 evidence remains readable and decodes with both branches disabled. `VPO1`
already records loop count and post-optimization window/frame trajectories, so
no output-format fork was necessary.
