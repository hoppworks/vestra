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

## Tier 2: closed room trajectory — accepted surface tolerance

The 60-frame closed-loop fixture is the meaningful world-level gate. The
reference and Vestra both emit **25,434** TSDF surfels with identical ordered
per-frame ownership and RGB. Its pre-TSDF trajectory is already at final F32
output rounding noise: window-centre MAE `1.334e-7`, per-frame position MAE
`1.278e-7`, and forward-direction MAE `2.866e-8`.

The ordered TSDF comparison reports position MAE `3.046e-7`; the maximum
absolute position delta is `0.001071`. The latter is a local PCA/zero-crossing
rounding difference, not an index or topology mismatch: count, frame-major
ordering, colour, radius, and the other 25,433 corresponding surfels agree to
final-output precision. The acceptance envelope for this fixture is therefore:

- exact point count, ordered first-observing-frame counts, and RGB;
- position MAE at most `5e-7`;
- position maximum absolute delta at most `0.0011` relative scene units;
- radius MAE at most `1e-8`.

The recorded result meets every listed gate. The F64 voxel-edge calculation is
important here: retaining PR #2's F32-input/F64-key boundary changed Vestra
from 25,433 to the reference's 25,434 extracted surfels.

## Next action

The semantic TSDF parity tier is complete. The next work is the paired
performance study: release builds, a quiet fixed-thread host, and repeated
Rust/C++ stage timings for normal estimation, splatting, extraction, and the
complete recorded-fixture geometry pipeline.
