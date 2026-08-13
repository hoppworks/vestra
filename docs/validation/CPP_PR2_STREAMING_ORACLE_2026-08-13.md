# C++ PR #2 streaming oracle bootstrap — 2026-08-13

## Status

**Infrastructure accepted; Vestra geometry parity is not claimed yet.** This
record establishes a reproducible, model-free executable boundary around the
actual `stream_points_core` implementation used by the reference project. The
next milestone is a Vestra-generated fixture and a pre-voxel comparator.

The original bootstrap interchange was corrected before it was used for a
model-derived comparison: `VPS1` v2 stores outputs **per inference window**,
not only per global frame. That distinction is required because DA3 computes
overlap frames again in each multi-view window, with different context.

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

The fixture is four 3×4 calibrated plane views, with two 3-frame windows and a
one-frame overlap. It proves the v2 window-scoped binary contract executes a
valid identity Sim3 seam and preserves first-owner frame emission. It does **not**
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

## First real fixture: IMG_2269

**Status: model evidence and the C++ reference stream ran successfully; Sim(3)
parity is not accepted.** The run used the existing canonical 120-frame FFmpeg
RGB24 cache at 504×336, the BASE F32 model, 12-frame chunks and 3-frame overlap
on the AMD Ryzen 9 9950X Workhorse with 16 Rust worker threads.

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| Rust window-scoped input (`oracle-rust.vps`) | `0317211fc5059df582fa22d42e72304f6d3c3038f126bfc262d713ffd0fce242` | 120 frames, 13 terminal-rule windows, 278 MiB |
| C++ PR #2 stream output (`oracle-cpp.vpo`) | `1693e4987113f9c32440b88ef444e7c09dd3a71ece9a65eb93c8b9762c62ae40` | 9,931,557 finite positive-radius pre-voxel points, 13 windows, 12 seams, 0 warnings, 0 loops |

The C++ executable was built from exactly `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`
and ggml submodule `eced84c86f8b012c752c016f7fe789adea168e1e`. Its seam log
reported all twelve dense correspondence sets at `M=508,032`.

This run found and fixed a product schedule defect before comparison:
Vestra's old planner emitted an extra `[117,120)` overlap-only window after
the terminal `[108,120)` window. PR #2 stops as soon as a window reaches the
last source frame. `0da55af` changes the default 120-frame schedule from 14 to
13 windows and marks the earlier 14-window browser/demo record as historical.

The first transform-tier Rust replay initially filtered seam evidence by the
55th confidence percentile; that was also wrong. PR #2 uses dense finite-depth
overlap points for Sim(3), with confidence only as a weight, and applies the
percentile gate only when emitting the final cloud. `94b2742` corrects that
oracle path. Both implementations then use the same 508,032 direct pixel
correspondences per seam, but their transforms still differ materially.

The remaining concrete mismatch is algorithmic and intentional in the current
Vestra product path:

- PR #2: weighted Umeyama followed by eight confidence-weighted Huber-IRLS
  iterations; no point-to-plane correction in the base stream tier.
- Vestra: trimmed robust similarity followed by point-to-plane refinement.

The next required implementation is a separately named PR #2-compatible
weighted Huber-IRLS Sim(3) estimator, selected only by the oracle/parity path.
It must be validated against synthetic known transforms and this fixture before
changing the product's stricter seam policy or claiming reconstruction parity.
