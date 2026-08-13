# Vestra goal completion audit — 2026-08-13

## Goal

> Complete Vestra: a local, coherent 3D world pipeline in Rust that ingests
> video and produces an interactive browser-viewable relative-scale
> point/surfel world, using Vestra Engine and Vestra Kernels with validated
> multiview reconstruction parity against the depth-anything.cpp PR #2
> reference.

## Requirement audit

| Requirement | Current implementation and direct evidence | Status |
| --- | --- | --- |
| Local video ingestion | `vestra app` serves a loopback-only file picker, streams the selected video to a job-owned directory, and starts the existing durable Rust `reconstruct` command. A Workhorse loopback upload of `IMG_2269.MOV` completed successfully. | Accepted |
| Rust reconstruction pipeline | `vestra-core` owns FFmpeg frame extraction, ordered multi-view inference, calibrated backprojection, sequential Sim(3), quality gates, deferred fusion, progressive scene publication, and exports. The current 120-frame real-video run completed all 13 windows. | Accepted |
| Coherent relative-scale point/surfel world | Current real run produced 296,596 finite fused surfels, with 12 accepted sequential seams and a versioned relative-scale scene manifest. No metric claim is made. | Accepted |
| Interactive browser delivery | `vestra-studio` serves progressive binary surfel chunks, measured/fused-layer controls, camera/frustum and seam overlays, source-frame PiP, and local evidence. The current scene served a manifest, six binary chunks, 156 camera rays, and 12 seam links; the intake success path also opened a viewer for a completed job. | Accepted |
| Vestra Engine and Kernels ownership | Root Cargo patches import the separately versioned `vestra-engine` and `vestra-kernels` projects. The current run records Engine `e6c8d0f` and Kernels `9740a98` in scene provenance. | Accepted |
| Validated PR #2 multiview reconstruction parity | S=2/S=3/S=12 inference parity is recorded in `MULTIVIEW_S2_2026-08-13.md`. The pinned PR #2 base stream replay additionally matched all 12 real-fixture Huber-IRLS seam scales/RMS values and 9,931,557 ordered pre-voxel emitted points: exact count, ownership, RGB; position MAE `1.0073384563168364e-7`; radius MAE `9.551295881469472e-11`. | Accepted |

## Verification commands

```bash
cargo test --workspace
vestra app --model depth-anything-base-f32.gguf --jobs ./vestra-jobs
vestra reconstruct --video room.mov --model depth-anything-base-f32.gguf --output room.vestra
vestra serve --scene room.vestra
```

The final source verification completed with 54 `vestra-core`, 7
`vestra-studio`, and 4 CLI unit tests passing. The current real product-world
and browser-intake validations are recorded in
[CURRENT_PRODUCT_WORLD_IMG_2269_2026-08-13.md](CURRENT_PRODUCT_WORLD_IMG_2269_2026-08-13.md).

## Explicit non-goals

This closure does not claim metric measurements, an editable floorplan, a
semantic mesh, generated/unobserved geometry, TSDF, accepted loop closure for
this particular capture, or an end-to-end CUDA throughput result. Those are
future, separately validated extensions and are not required by the stated
relative-scale point/surfel world goal.
