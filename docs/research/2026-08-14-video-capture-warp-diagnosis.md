# Video capture warp diagnosis for DA3-style local worlds

## Question and scope

This note investigates why a world reconstructed from an inward/forward phone
walkthrough can show a visibly curved or tilted floor, even when individual
objects look plausible. It does **not** diagnose the next video, which has not
yet been supplied. The conclusion is based on the pinned local
`depth-anything.cpp` PR #2 reference source, the current Vestra source and
validation records, and the official Depth Anything 3 (DA3) source/docs.

## Short answer

The proposed explanation is partly right: a straight, mostly forward capture
usually provides less lateral parallax and fewer repeated observations than a
loop or orbit. That makes depth, focal length, pitch, and pose errors harder to
separate. But it is not the only plausible cause, and it is not evidence that
the user recorded the video "wrong".

For the current Vestra/PR #2-derived path, a warped floor is most likely a
**global registration/drift problem**, magnified by a capture with weak
side-to-side constraint. Each window receives a locally consistent multi-view
prediction, then adjacent windows are chained through a relative Sim(3).
Small systematic pitch, scale, or depth errors can therefore accumulate. A
loop can constrain that accumulated error only if the system actually finds and
accepts a geometrically verified revisit; a merely circular-looking camera path
does not guarantee one.

This is an inference from the architecture and visible symptom, not a claim
that one failure cause is proven without the new capture's seam/trajectory
diagnostics.

## What the implemented systems actually assume

### DA3

DA3 predicts multi-view depth and camera geometry from an image set. Its
official API exposes optional camera extrinsics/intrinsics, a predicted-pose
path, an Umeyama alignment stage, and reference-view selection. The official
documentation specifically recommends the middle frame as reference for a
temporally ordered video sequence. [DA3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)

The any-view BASE model used by Vestra is relative-scale rather than metric.
DA3's official model cards distinguish it from the separate `DA3Metric` and
`DA3Nested` variants. The project further states that reliable supplied poses
improve pose-conditioned depth consistency and describes ray-pose as generally
slower but more accurate than the default camera decoder. [DA3 model cards and
FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards)

The official DA3-Streaming project is not a SLAM system. It chunks long video,
maintains state between chunks, writes poses/intrinsics and a combined point
cloud, and reports odometry error even when evaluated with substantial overlap
and loop closure. Its published video extraction example is 5 fps at 640-pixel
width; its reported comparison uses 120-frame chunks, 60-frame overlap and
loop closure. [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)

This establishes two important limits: DA3 can infer useful geometry and pose
from arbitrary views, but it does not make camera trajectory drift or global
registration disappear automatically.

The official benchmark notes also attribute inaccurate ScanNet++ poses to
motion blur and textureless iPhone frames, and describe blur filtering,
fisheye calibration and exhaustive COLMAP matching as their remedy. That is
direct primary-source support for treating blur, texture and calibration as
plausible competing causes, rather than assuming that trajectory shape alone
explains a warped room. [DA3 benchmark notes](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md#scannet)

### Pinned C++ PR #2 reference

The PR #2 path is a host-side sliding-window reconstruction: 12 views per
window, 3 shared frames, a weighted Umeyama Sim(3) seam solve, optional ICP,
and optional loop-closure pose graph. The model itself does not create the
global world. See the local pinned source:

- `/tmp/vestra-pr2-reference-f56e9be/src/stream.hpp:2-43`
- `/tmp/vestra-pr2-reference-f56e9be/src/stream.cpp:212-282`
- `/tmp/vestra-pr2-reference-f56e9be/src/stream.cpp:361-417`

Adjacent windows are aligned from the *same source pixels* in their three
overlap frames. Confidence is a weight rather than a hard correspondence
filter. The relative transform is composed onto the previous global pose. A
bad but accepted seam consequently affects every later window. PR #2 only
tries loop closure after the sequential chain exists; it first gates candidates
by predicted camera proximity and viewing direction, then requires a tight
geometry match and ICP correspondences. The source explicitly avoids a large
match gate because a small room could otherwise align across the room and fit a
garbage transform.

The C++ source also explicitly locks loop-edge scale because a near-planar
revisit cannot observe it reliably (`stream.cpp:131-132`). That is direct
evidence that planar geometry, such as a floor/wall-dominated room, has an
observability limitation; it is not a Vestra-only theory.

### Current Vestra product

Vestra's strict PR #2-relative fusion takes durable measured windows, builds
the same sequential/loop solution, then emits first-observed frames into the
fused world. It is relative-scale, not metric geometry. See
`crates/vestra-core/src/reconstruction.rs:132-209`.

The prior 120-frame real-video validation was an integrity/delivery result,
not a visual-quality guarantee. It had 12 sequential seams and **no accepted
loop**, despite finite output and high seam inlier counts. It also explicitly
states that visual/semantic quality remains capture-dependent. See
`docs/validation/CURRENT_PRODUCT_WORLD_IMG_2269_2026-08-13.md:37-66`.

## Diagnosis matrix

| Observed symptom | Most likely mechanism | Evidence to inspect on the next run | Capture implication | Reconstruction response |
| --- | --- | --- | --- | --- |
| Whole floor bows smoothly while objects remain locally credible | Accumulated sequential Sim(3) scale/pitch drift, or depth bias consistent within each window | Per-seam scale, rotation/pitch change, RMS residual and trajectory; whether later seams carry the same signed pitch error | Forward-only motion and a floor-dominant field of view give little cross-track constraint | Diagnose/plot raw window trajectory first; add a globally constrained pose stage only if diagnostics confirm drift |
| World tilts progressively after a particular time | One bad accepted seam is composed into all later window poses | Identify first discontinuity in seam residual/scale/rotation; compare raw local window point clouds either side | Fast turn, motion blur, exposure jump, or weak texture near that seam | Reject/down-weight that seam or re-estimate it; do not hide it with cosmetic world rotation |
| A turn becomes a spiral or a circle fails to close | No accepted loop closure, or revisit candidate does not meet geometric match gates | Proposed vs accepted loop edges; camera distance/direction gate; ICP correspondence count | The same wall must be seen again with overlap and reasonably similar view direction; avoid filming only blank/reflective wall | Use verified loop edges and pose-graph optimization; report "no loop accepted" rather than inventing a closure |
| Flat floor becomes rippled but global orientation is stable | Per-frame depth noise / confidence leakage, amplified by dense point emission | Depth/confidence maps; floor-plane residual by frame/window; correlation with low confidence or motion blur | Keep the floor and textured static features in view, move steadily, avoid blur and automatic focus/exposure jumps | Plane-aware confidence/normal filtering or a robust surface fusion stage; this changes quality, not camera pose truth |
| Floor bends near video borders only | Intrinsics / crop-resize mismatch or weak depth at the image edges | Check processed raster, crop transform, predicted intrinsics and location of residuals | Keep important walls/floor away from extreme frame edges; use the same orientation throughout capture | Verify reversible crop/intrinsics propagation before altering geometry code |
| Parallel walls/floor all deform coherently, including early frames | Incorrect camera calibration/pose prediction, not simply accumulated stitching | Local single-window geometry and camera-ray overlay; compare camera-decoder versus ray-pose diagnostic if available | More diverse viewing angles help; capture alone cannot guarantee recovery from a systematic pose error | Treat as pose-model/calibration validation; do not blame fusion until a local-window check fails |
| Objects double, but floor is mostly flat | Registration residual or duplicate observations, rather than planar curvature | Nearest-neighbour sheet separation, seam residual and TSDF/voxel evidence | Revisit objects with overlap; keep them static | Improve registration/fusion; this is distinct from a bowed floor |

## Why a forward walkthrough is weaker than a loop/orbit

The key distinction is not "straight is invalid, circular is valid." It is
**constraint diversity**.

1. Moving forward while looking forward often changes image scale more than
   viewpoint. Features near the optical axis have limited lateral parallax.
   A small focal/depth/translation mistake can explain similar imagery, so the
   pose and depth estimates have less independent evidence.
2. A sideways arc around visible, static, textured geometry creates cross-track
   parallax. It observes a wall/floor/object from different positions and
   supplies richer geometry to the overlap seam.
3. Returning to a previously seen area can add a non-adjacent constraint. In
   this architecture that only helps after the system accepts a loop edge;
   otherwise the sequential chain remains open-loop.
4. A slow turn in place has strong rotational coverage but near-zero
   translational parallax. It can improve visual coverage and loop appearance,
   but by itself is not a substitute for moving sideways.

Thus, "looking around much more" is useful only when combined with controlled
translation and overlap. Aggressive back-and-forth camera rotations can instead
create blur, rolling-shutter distortion, auto-exposure changes and gaps between
the frames selected for the fixed 12/3 schedule.

## Recommended Version-1 capture, before changing algorithms

Use this as a diagnostic-quality capture rather than a perfect production
protocol:

1. Hold the phone level and keep its orientation fixed. Do not alternate
   portrait/landscape.
2. Walk one slow, closed path near the perimeter, approximately parallel to the
   walls. Keep the camera pointed 30–45° inward rather than exclusively at the
   opposite wall.
3. Keep a stable mix of floor, wall junctions, doors/windows and textured
   objects in every view. Corners and wall-floor boundaries are much more useful
   than a large blank floor or blank wall.
4. Make one deliberate revisiting segment: end by looking again at the exact
   same corner/door from a similar height and direction. Pause briefly there.
5. Avoid rapid pans, walking while rotating quickly, people/pets, mirrors,
   glossy windows, and changing lighting. These create changing image evidence
   that a static scene model cannot reconcile.
6. For a round room, use a **single ring** first: one perimeter lap, then a
   short repeat of the starting sector. Do not create an inward spiral for the
   diagnostic run. An inward spiral changes both radius and viewing geometry
   continuously, making it hard to separate genuine layout from global scale
   drift.

The new video should be treated as an A/B capture test. We should inspect its
seam and loop diagnostics before declaring the old capture or current pipeline
to be the primary cause.

## Falsifiable next-run checks

Before any quality algorithm change, retain and inspect these artifacts:

1. **Window trajectory:** raw sequential transforms and final transforms, with
   cumulative yaw/pitch/roll and scale plotted per window.
2. **Seam table:** correspondences, inlier ratio, RMS, scale and incremental
   rotation for every adjacency. Mark the first seam at which floor curvature
   becomes visible.
3. **Loop table:** all candidates, rejection reason, and accepted ICP
   correspondence count. A circular capture that has no accepted loop must not
   be described as globally constrained.
4. **Local-versus-global floor fit:** fit a plane independently to each local
   window's high-confidence floor points, then to the fused points. A flat
   local floor but curved fused floor isolates registration; curvature already
   local points to depth/pose/crop issues.
5. **Crop/calibration audit:** verify that every source-to-504×336 transform
   and predicted intrinsic is applied consistently in both back-projection and
   replay.

## Decision rule

Do not select a new model merely because the next capture looks bad. First use
the five checks above.

- If local windows are flat and the sequential trajectory drifts, the highest
  value work is a stronger global pose/loop-closure or SfM/BA constraint.
- If local windows already curve, investigate DA3 pose/depth, crop/intrinsics,
  motion blur, and rolling shutter before changing fusion.
- If the new deliberate loop produces no verified loop edge, improve loop
  candidate generation/measurement or use an external global SfM/SLAM pose
  source; a better point-fusion kernel cannot recover the missing constraint.
- If the loop is accepted but the floor is still warped, compare the
  sequential and optimized trajectories and test whether the loop measurement
  itself is geometrically valid.

## Primary sources

- ByteDance-Seed, [Depth Anything 3 API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md): multi-view inputs, pose conditioning, Umeyama alignment and video reference-view strategy.
- ByteDance-Seed, [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md): chunked long-video pipeline, pose/intrinsic/point-cloud outputs and evaluation conditions.
- ByteDance-Seed, [DA3 network source](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/model/da3.py): depth, confidence and camera-pose network outputs.
- ByteDance-Seed, [DA3 benchmark notes](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md#scannet): motion blur, low texture and calibration countermeasures in official evaluation.
- Pinned local `depth-anything.cpp` PR #2 snapshot `f56e9be`, `src/stream.hpp` and `src/stream.cpp`: exact reference schedule, seam chaining and loop measurement. Upstream review: [PR #2](https://github.com/localai-org/depth-anything.cpp/pull/2).
- Local Vestra current product: `crates/vestra-core/src/reconstruction.rs` and `docs/validation/CURRENT_PRODUCT_WORLD_IMG_2269_2026-08-13.md`.
