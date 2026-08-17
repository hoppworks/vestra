# Coherent-world reconstruction options for Vestra

## Executive recommendation

The current failure is primarily a **global camera/registration** problem, not a
point-rendering problem. The reported `0.53–2.02` range of successive local
similarity scales is already enough to bend floors and turn a real circular
path into a spiral. More frames make the local depth observations denser; they
do not make a chained sequence of locally estimated coordinate systems globally
consistent.

Build and evaluate an optional **global-pose provider** before changing the
visual renderer:

1. Use **COLMAP SfM with global bundle adjustment** as the first production
   candidate. It is a local command-line dependency, not a Rust reimplementation
   project. Its registered camera poses become the global coordinate system.
2. Keep Vestra Engine for high-quality dense depth. Estimate each DA3/Vestra
   window's transform against the global pose solution, optimize all windows
   jointly, and only then project/fuse the dense depth maps.
3. Use the current point/surfel/TSDF layer only *after* pose acceptance. A
   better splat or mesh may make a correct world look better; it cannot make a
   geometrically inconsistent trajectory correct.

This is a high-leverage architectural change, but not a claim that COLMAP will
solve every room. Sparse, repetitive, glossy or motion-blurred indoor footage
can still fail. The job must surface that as a pose-quality failure rather than
silently publishing a plausible-looking but bent world.

## What the PR #2 reference actually does

The pinned `localai-org/depth-anything.cpp` PR #2 is a strong **local streaming
and fusion** reference, not a global SfM/SLAM solution:

| Stage | Pinned C++ PR #2 | Current Vestra strict profile | Consequence |
| --- | --- | --- | --- |
| Local geometry | A 12-frame DA3 window predicts local depth, confidence, intrinsics and poses. | Same measured-window concept and validated 12/3 scheduling/parity. | Individual objects can look very good. |
| Sequential registration | Three repeated source frames are aligned with weighted Sim(3); the result is composed onto the preceding window. | Same PR #2-compatible sequential oracle path. | A single biased seam affects every later window. |
| Loop handling | It proposes non-adjacent revisits from the predicted path, applies tight geometric matching and ICP, locks loop scale, then optimizes a Sim(3) pose graph. | Same closed-loop design is represented in Rust. | It helps only when a true revisit is proposed and passes geometry checks. |
| Emission/fusion | Local points are retained until the final poses are known, then emitted once (with optional normal-space TSDF). | Vestra likewise defers emission and supports the TSDF derivative. | This prevents duplicate points from being permanently baked before optimization; it does not create missing global constraints. |

The C++ source is explicit about the key limitation: loop scale is locked because
a near-planar revisit cannot observe it reliably. A floor/wall-dominated room
is therefore an adverse case even in the reference implementation. The source
also deliberately rejects broad loop gates because they can align opposite sides
of a small room with a false transform. These are good safeguards, but they do
not turn the sequential chain into bundle adjustment.

Consequently, the fact that Vestra is PR #2-compatible does **not** imply a
good global world for every phone capture. The reference can show a coherent
demo when its seams and loops are well constrained; it has the same fundamental
failure mode when they are not. This is why copying more of that PR or merely
matching its voxel/TSDF output is not the primary fix.

Relevant local evidence:

- `/tmp/vestra-pr2-reference-f56e9be/src/stream.cpp` keeps `WindowRec` points
  in local space, composes sequential `Sim3 G`, and only measures loop edges
  after the sequential chain exists.
- `crates/vestra-core/src/reconstruction.rs` uses the same pattern: it obtains
  `optimized_window_poses`, then applies them while emitting first-owned frames.
- `docs/research/2026-08-14-video-capture-warp-diagnosis.md` correctly
  distinguishes locally credible depth from accumulated global drift.

## What will and will not improve the result

| Change | Helps | Does not help |
| --- | --- | --- |
| More source frames | Temporal coverage, local depth continuity, selecting less blurred keyframes. | A drifted global trajectory by itself. It can also add many highly correlated observations. |
| Smaller pixels / more points | Visible point density and small holes. | Camera pose, scale drift, or a bowed floor. |
| Bigger point sprites, confidence filtering, depth testing | A much nicer browser presentation; removal of outliers/overdraw. | False separation of repeated surfaces or a spiral trajectory. |
| TSDF/surfel fusion after good registration | Local surface continuity, denoising, a more solid visual world. | Systematic pose/depth bias. Fusing conflicting views can make a wrong world look more confidently wrong. |
| A global pose solver plus verified revisits | The actual cause: every dense point is put into one jointly constrained coordinate system. | Textureless/blurred footage with insufficient constraints; this must remain diagnosable. |

The 720-frame result should therefore remain useful as a dense **measurement
cache**, but it should not be considered a quality reference merely because it
contains 55.7 million points. For the global solver, selected, sharp keyframes
with meaningful baseline and deliberate revisits are preferable to treating all
near-duplicate video frames as independent constraints.

## Candidate global-pose providers

| Candidate | What it supplies | Product fit | Recommendation |
| --- | --- | --- | --- |
| COLMAP | Feature tracks, globally registered cameras from SfM, final global bundle adjustment; can then run dense MVS and fusion. | Local executable; mature open pipeline. It needs image texture and careful phone-camera calibration/priors. | **First implementation and acceptance baseline.** |
| DROID-SLAM | A learned visual-SLAM trajectory and optional full-resolution depth reconstruction. Official code supports monocular/stereo/RGB-D and requires camera calibration for own footage. | BSD-3-Clause source; CUDA and at least 11 GB VRAM according to its README. | **Second experiment** if COLMAP loses tracking or registration in indoor video. Treat it as a local pose provider, not a Rust port. |
| DA3-Streaming | Long-video DA3 inference that persists state, poses/intrinsics and a combined point cloud. | Directly relevant model family, but its own authors state it is not SLAM. Its reported evaluation uses 120-frame chunks, 50% overlap and loop closure. | **Useful oracle/prototype**, not the automatic answer. Evaluate as an external local runner before any Rust implementation. |
| VGGT | Direct predictions of cameras, point/depth maps and tracks for one to hundreds of views; official runner can export COLMAP cameras/points and optionally run BA. | Powerful global learned geometry route; requires a GPU/PyTorch sidecar and model-license/checkpoint review. | **Best learned global-alignment experiment** after COLMAP baseline. Do not promise production licensing until the selected checkpoint is approved. |
| MASt3R / DUSt3R | Learned point maps, matching and global alignment. | Official DUSt3R checkpoint terms include CC-BY-NC-SA; this is unsuitable for a commercial product without separate permission. | Research comparison only unless licensing is cleared. |
| ORB-SLAM3 | Conventional visual/visual-inertial SLAM with loop closure and multi-map support. | GPLv3 in the public repository; calibration is required. | Do not embed in Vestra's Apache-2.0 product path. It is a diagnostic/reference option only unless commercially licensed. |

### Why COLMAP first

COLMAP's official reconstruction pipeline makes the missing separation
explicit: SfM first recovers sparse 3D structure **and camera poses**, and
multi-view stereo then uses those registered cameras to obtain dense geometry.
It also provides a final global bundle-adjustment stage and documents a path
for dense reconstruction from known camera poses. This is exactly the missing
global coordinate-system boundary in the current chained-window architecture.

For Vestra, COLMAP should initially be a *pose authority*, not a replacement
for Vestra Engine's depth. This preserves the existing native model work and
lets an A/B test isolate whether global poses, rather than depth quality, remove
the curved floor.

## Recommended architecture

```text
video
  ├─ deterministic crop / undistortion / sharp keyframe selection
  ├─ COLMAP (or optional learned pose provider)
  │     └─ global cameras + tracks + registration diagnostics
  ├─ Vestra Engine local multi-view depth on dense selected frames
  ├─ robust window-to-global alignment
  │     └─ jointly optimize all window Sim(3) variables
  ├─ pose-quality gate
  │     ├─ reject / ask for recapture when disconnected or high-error
  │     └─ publish only accepted global geometry
  └─ confidence-aware surfel/TSDF fusion → point/splat/mesh presentation
```

The integration boundary should be intentionally small and reproducible:

- A versioned `PoseSolution` artifact contains input-frame IDs, crop/resize
  transform, camera model/intrinsics, W2C poses, tracks, registered-image mask,
  reprojection diagnostics and provider provenance.
- Rust invokes a pinned local provider through a narrow command-line adapter
  (`std::process::Command`), receives this artifact, validates every ID and
  matrix, then owns all durable `.vestra` persistence and all dense fusion.
  No Python/C++ bindings belong in the Rust hot path.
- Preserve the raw Vestra measured windows. A new pose provider produces a new
  derived world; it must never overwrite or reinterpret measured evidence.
- Estimate the per-window DA3-local to provider-global transform using all
  compatible camera centres/tracks, with robust residuals, and optimize the
  *entire* window graph at once. Do not derive the transform by composing only
  the previous seam.
- Use the same crop and camera model in frame extraction, provider input,
  Vestra inference and replay. A crop/intrinsics mismatch can imitate geometric
  drift and must fail validation.

This approach is still "Rust-first": the product contract, storage, validation,
fusion, viewer and native depth engine remain Rust. Reusing a well-tested
global-geometry executable is a better engineering boundary than attempting to
reimplement SfM/SLAM before proving the problem is solved.

## Concrete investigation sequence

### Gate 0 — make the current failure measurable

Before replacing anything, add/persist a report for every world:

1. Sequential and optimized camera trajectories with cumulative scale,
   yaw/pitch/roll per window.
2. Per-seam correspondence count, inlier ratio, RMS, scale and rotation.
3. Loop candidates, rejection reasons, accepted ICP correspondence count and
   post-optimization residual.
4. A local-window floor-plane fit versus fused-world floor-plane residual.
5. Exact crop, raster and intrinsics values for each processed frame.

This distinguishes (a) locally flat floor + global drift, (b) bad local
depth/pose, and (c) a crop/calibration bug. No visual smoothing work should
claim to solve case (a) or (b).

### Gate 1 — COLMAP pose-provider spike

Run a local, pinned COLMAP pipeline on one existing room video:

1. Extract sharp, baseline-aware keyframes from the already decoded frames;
   preserve their original IDs and use the same 3:2 crop policy.
2. Use sequential matching plus verified non-adjacent/revisit candidates;
   keep phone intrinsics fixed when calibration is known rather than allowing
   unconstrained camera-model changes.
3. Require one connected model over the intended interval. Persist registered
   versus rejected keyframes, tracks and reprojection/BA diagnostics.
4. Reproject a held-out set of tracks. If the global solution is disconnected
   or has visibly inconsistent reprojection/trajectory evidence, mark the job
   `pose_review`, not `complete`.
5. Transform the *existing* measured depth windows into the COLMAP frame and
   compare local/fused floor-plane residuals, duplicate-surface separation and
   loop-end displacement against the existing PR #2 path.

Success is not "a prettier screenshot". It is a connected pose solution plus
a lower global plane/loop residual while the local-depth evidence remains
unchanged. This validates the causal hypothesis.

### Gate 2 — choose the provider with evidence

If COLMAP succeeds on a deliberately revisited room capture, productize it as
the default optional global-pose stage. If it fails because of poor indoor
feature coverage, run the identical cached keyframes through DROID-SLAM and
VGGT (with their model/license constraints recorded) and compare the same
trajectory/plane/loop metrics. Do not compare point density or screenshot
appearance alone.

### Gate 3 — improve presentation only after pose acceptance

Once geometry is accepted, render oriented surfels/splats with a depth buffer,
confidence-weighted opacity and adaptive radius; optionally build a mesh/TSDF
preview. Clearly label these as presentation derivatives. Keep the raw point
world selectable in Studio so a visually smooth layer cannot hide registration
errors.

## Source notes

All external sources below are first-party repositories or official
documentation. Local C++ references name the pinned source used for Vestra's
existing parity work.

- ByteDance Seed, [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3): DA3 model outputs and multi-view geometry/pose capabilities.
- ByteDance Seed, [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md): stateful chunking, persisted geometry, reported 120/60 overlap/loop-closure evaluation, and the explicit non-SLAM limitation.
- COLMAP, [official tutorial](https://colmap.github.io/tutorial.html): sparse SfM camera registration before dense MVS/fusion.
- COLMAP, [official FAQ](https://colmap.github.io/faq.html): final global BA and dense reconstruction from known cameras.
- Princeton VL, [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM): supported sensor modes, calibration requirement and local reconstruction output.
- Facebook Research, [VGGT](https://github.com/facebookresearch/vggt): direct camera/depth/point/track inference and COLMAP export with optional BA.
- Naver Labs, [DUSt3R](https://github.com/naver/dust3r): global-alignment API and checkpoint-license constraint.
- UZ-SLAMLab, [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3): visual/visual-inertial SLAM capability and public GPLv3 licensing.
- Pinned C++ snapshot `f56e9be`, `/tmp/vestra-pr2-reference-f56e9be/src/stream.cpp`: local source of truth for the PR #2 schedule, sequential Sim(3), deferred emission and loop measurement.
