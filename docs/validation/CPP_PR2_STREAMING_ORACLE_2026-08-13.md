# C++ PR #2 streaming oracle bootstrap — 2026-08-13

## Status

**Infrastructure accepted; Vestra geometry parity is not claimed yet.** This
record establishes a reproducible, model-free executable boundary around the
actual `stream_points_core` implementation used by the reference project. The
next milestone is a Vestra-generated fixture and a pre-voxel comparator.

## Locked reference

- Reference: `localai-org/depth-anything.cpp` PR #2 head
  `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`.
- The source was checked out detached at `/tmp/vestra-cpp-pr2`; its ggml
  submodule is locked at `eced84c86f8b012c752c016f7fe789adea168e1e`.
- Host: local Apple Silicon development machine. This was a correctness build,
  not a performance benchmark.
- The separate user checkout was deliberately left untouched because it has
  unrelated local modifications.

## Executed commands

```bash
git -C /tmp/vestra-cpp-pr2 worktree add --detach /tmp/vestra-cpp-pr2 \
  f56e9be43a22c12ef575584d2fa57a6a5d5be7ae
git -C /tmp/vestra-cpp-pr2 submodule update --init --recursive
cmake -S tools/cpp-pr2-oracle -B /tmp/vestra-cpp-pr2-build \
  -DCPP_PR2_SOURCE=/tmp/vestra-cpp-pr2 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/vestra-cpp-pr2-build --target vestra_cpp_stream_fixture_dump --parallel 8
python3 tools/cpp-pr2-oracle/make_identity_fixture.py /tmp/vestra-identity.vps
/tmp/vestra-cpp-pr2-build/vestra_cpp_stream_fixture_dump \
  /tmp/vestra-identity.vps /tmp/vestra-identity.vpo
```

The final oracle output was:

```text
[da3] stream: window @2 stitch M=12 s=1.0000 rms=0
[da3] stream timing: ... windows=2 loops=0 pts=48
VPO1 points=48 windows=2 warnings=0 loops=0
```

The fixture is four 3×4 calibrated plane views, with a 3-frame window and a
one-frame overlap. It proves the versioned binary contract executes a valid
identity Sim3 seam and preserves first-owner frame emission. It does **not**
prove behaviour on model-generated geometry, translations, noise, failure
seams, loop closures, ICP, TSDF, or metric scale.

## Required next comparison

1. Have Vestra serialize identical model outputs to `VPS1` using its canonical
   FFmpeg RGB24 frames and the same 12/3 schedule.
2. Add a Vestra diagnostic stage that exposes aligned, first-owner points
   **before voxel fusion**. Current `fuse_scene_bundle` is too late because it
   unconditionally fuses surfels.
3. Parse `VPO1` and gate the sequential tier on window count/order, per-frame
   owned-point counts, warnings, window/frame trajectory, and bidirectional
   cloud distance. Persist raw fixtures, output hashes, revisions and numeric
   thresholds in a new validation record.
4. Only then add controlled synthetic translation/noise fixtures, seam failure
   semantics, ICP, loop/pose-graph, and optional metric/TSDF tiers.

The C++ source currently falls back to an identity seam on a degenerate
overlap. Vestra's quality-gated rejection/quarantine policy is intentionally
stricter, so a failed-seam test must record a policy difference rather than
force superficial equality.
