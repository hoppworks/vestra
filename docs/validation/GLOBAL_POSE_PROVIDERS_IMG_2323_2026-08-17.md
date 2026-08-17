# Global pose-provider evaluation — IMG_2323

## Purpose

Evaluate global camera trajectories for the existing DA3 measured windows
without changing the immutable raster cache or relaxing Vestra's publication
gate. A provider may create a new coherent-world product only when every
measured window has enough registered cameras and a normalized camera-fit RMS
of at most `0.15`.

This is a geometry decision. A more densely rendered point cloud is not
evidence of a better camera trajectory.

## Common contract

- Scene: `img-2323-keyframes-da05305.vestra`.
- Geometry keyframes: 230 selected from 915 decoded raster candidates.
- Camera convention: OpenCV world-to-camera, row-major 3×4.
- All solutions are bound to the scene raster manifest and are imported through
  the same `vestra.pose-solution/v1` contract.
- A window requires at least six registered cameras before its camera fit can
  be used for publication.

## Results

| Provider | Registered frames | Outcome | Reason publication is refused |
| --- | ---: | --- | --- |
| COLMAP retrieval-wide | 215 / 230 | Rejected | Windows 21 and 25 had fewer than six cameras; several fitted windows exceeded the RMS gate. |
| DROID-SLAM | 174 / 230 | Rejected | Window 14 normalized RMS was `1.1909952`, above `0.15`. |
| Hybrid COLMAP + DROID | 227 / 230 | Rejected | Alignment filled frame IDs but degraded geometric agreement; examples include windows 14 (`0.6195`) and 21 (`1.95`). |
| VGGT overlap stitch | 230 / 230 | Rejected | All camera matrices passed rigid-rotation validation, but many per-window fits still exceed the gate: window 6 `0.7929`, 14 `0.6583`, 21 `1.1132`, and 24 `0.6366`. |

### VGGT evidence

The stitched VGGT file was imported as pose solution
`5873336055fe80bc8b528ff3b4696097166e09d2301bdb1c2180d04702659e5d`.
It contains 230 registered frames. Before import, each bf16-derived rotation
was projected to the nearest proper SO(3) matrix; this fixed a real file-format
validity issue, but it did not alter the residual gate or make the trajectory
acceptable.

## Decision

No `fuse-global-pose` result is published for this scene. The selected local
product remains the only available world; it must be labelled as local/chained
rather than coherent-global.

Lowering the RMS threshold, interpolating bad windows, or selecting only good
fragments would conceal drift rather than solve it.

## Next architecture step

The failed providers all estimate poses in disconnected or independently
aligned chunks. The next candidate must optimize one global trajectory across
the entire capture:

1. retain a globally connected correspondence graph across chunk boundaries;
2. jointly optimize camera poses and, where necessary, intrinsics with robust
   bundle adjustment or a SLAM pose graph;
3. re-run the existing per-window fit gate on the resulting immutable
   `PoseSolution`;
4. publish a separate TSDF/surfel product only after every window passes.

The implementation must preserve the raw local product and report provider,
revision, model/checkpoint hash, raster fingerprint, all residuals, and the
accept/reject decision.
