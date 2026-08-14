# TSDF differential oracle — 2026-08-14

## Reader and action

This record is for an engineer adding or changing Vestra surface fusion. After
reading it, they can reproduce the two TSDF oracle tiers and know which result
is a hard acceptance gate versus an unresolved trajectory-sensitive result.

## Contract

The pinned C++ PR #2 revision is
`f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`. Its normal-space TSDF is the
reference behavior: PCA normals are oriented toward the nearest camera, colour
is accumulated in linear light, weights default to inverse point radius, and
output is frame-major by first observing frame.

The oracle harness accepts `--tsdf` after its `VPS1` input and writes a normal
`VPO1` artifact. It deliberately does not turn TSDF into an unrecorded viewer
post-process.

```bash
cmake -S tools/cpp-pr2-oracle -B /tmp/vestra-cpp-pr2-build-loop \
  -DCPP_PR2_SOURCE=/tmp/vestra-pr2-reference-f56e9be \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vestra-cpp-pr2-build-loop \
  --target vestra_cpp_stream_fixture_dump --parallel 8

/tmp/vestra-cpp-pr2-build-loop/vestra_cpp_stream_fixture_dump \
  INPUT.vps OUTPUT.vpo --tsdf
cargo run --release -q -p vestra-cli -- oracle-compare \
  --fixture INPUT.vps --reference OUTPUT.vpo --tsdf
```

## Tier 1: identity control — accepted

The four-frame identity-plane fixture is an exact equality gate. Its source
cloud is already pre-voxel identical, so it isolates TSDF semantics from seam
and loop trajectory differences.

```bash
python3 tools/cpp-pr2-oracle/make_identity_fixture.py /tmp/vestra-identity.vps
/tmp/vestra-cpp-pr2-build-loop/vestra_cpp_stream_fixture_dump \
  /tmp/vestra-identity.vps /tmp/vestra-identity-tsdf.vpo --tsdf
cargo run --release -q -p vestra-cli -- oracle-compare \
  --fixture /tmp/vestra-identity.vps \
  --reference /tmp/vestra-identity-tsdf.vpo --tsdf
```

Result: 12 points on both sides, identical per-frame ownership, RGB, positions,
and radii (all absolute error metrics are zero).

This tier caught and fixed two incompatible Vestra defaults:

- a forced `0.03` minimum voxel size for every small valid scene; PR #2 only
  uses `0.03` when the bounding box is degenerate;
- confidence weighting; PR #2's default TSDF path uses inverse point radius.

## Tier 2: closed room trajectory — open

The 60-frame closed-loop fixture is the meaningful world-level gate. C++ emits
25,434 TSDF surfels; Vestra emits 25,413. Raw pre-TSDF ownership and RGB match,
but the C++ and Rust pose graphs have a small accepted trajectory difference.
At the fine PR #2 voxel size that difference changes voxel membership and hence
the ordered zero-crossing surface.

The current ordered comparison reports position MAE `0.19647` and maximum
absolute difference `3.31914`; it is **not accepted parity**. Do not use this
result as a visual-quality or performance claim.

The trajectory comparator localizes the preceding cause: window-centre position
MAE is `0.004343` (maximum `0.011424`), per-frame position MAE is `0.004050`
(maximum `0.018469`), and forward-direction MAE is `0.000677`. Window midpoint
selection itself matches exactly. These values must fall below a declared
voxel-stability tolerance before this tier can be accepted.

## Next action

Before changing the TSDF kernel again, make the closed-loop window/frame poses
match the C++ reference within a voxel-stability tolerance, then rerun Tier 2.
Only after counts, frame ownership, colours, and a declared spatial surface
tolerance pass may TSDF be called PR #2-parity complete.
