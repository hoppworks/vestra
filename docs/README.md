# Vestra documentation

This index is for a reader who needs to distinguish product guarantees from
engineering evidence and future work.

## Authority order

When documents disagree, use this order:

1. **Code and pinned revisions** define what the current build can execute.
2. **Accepted ADRs** define intentional contracts and boundaries.
3. **Validation records** define whether a behavior or quality gate was
   actually observed.
4. **Benchmark result files** define measured performance for their named
   workload only.
5. **This index, PROJECT_STATUS.md, and VISION.md** summarize or frame the
   evidence; they do not widen it.

The implementation roadmap is the planning view, not an acceptance record:
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Start here

- [Vision](../VISION.md) — product intent, truth model, and the boundary
  between accepted release behavior and aspiration.
- [Architecture](../ARCHITECTURE.md) — repository boundaries, pipeline,
  scene durability, and optional global branches.
- [Current project status](PROJECT_STATUS.md) — curated milestone state,
  benchmark claims, and known limits.

## Design contracts

- [Repository boundaries](adr/0001-repository-boundaries.md)
- [Relative scale v1](adr/0002-relative-scale-v1.md)
- [Parity before beautification](adr/0003-parity-before-beautification.md)
- [CUDA backend boundary](adr/0004-cuda-backend-boundary.md)
- [Candidate rate and geometry keyframes](adr/0005-candidate-rate-and-geometry-keyframes.md)
- [Global pose sidecar contract](providers/GLOBAL_POSE_SIDECAR.md)

## Evidence navigation

### Reconstruction and durability

- [Real-video relative-world validation](validation/REAL_VIDEO_IMG_2269_2026-08-13.md)
- [End-to-end local-world smoke](validation/END_TO_END_SMOKE_2026-08-13.md)
- [Resumable Studio jobs](validation/RESUMABLE_STUDIO_JOBS_2026-08-13.md)
- [Closed-loop geometry oracle](validation/CLOSED_LOOP_ORACLE_2026-08-14.md)
- [TSDF differential oracle](validation/TSDF_ORACLE_2026-08-14.md)

### Global pose, MVS, and semantic research

- [Global pose-provider evaluation](validation/GLOBAL_POSE_PROVIDERS_IMG_2323_2026-08-17.md)
- [COLMAP global-pose spike](validation/COLMAP_GLOBAL_POSE_IMG_2323_2026-08-17.md)
- [COLMAP dense-MVS control](validation/COLMAP_DENSE_MVS_CONTROL_IMG_2323_2026-08-17.md)
- [Calibrated DA3 V2](validation/DA3_CALIBRATED_V2_IMG_2323_2026-08-17.md)
- [MVS-guided DA3 hybrid](validation/DA3_MVS_HYBRID_IMG_2323_2026-08-17.md)
- [Architecture semantics](validation/ARCHITECTURE_SEMANTICS_IMG_2323_2026-08-17.md)

### Performance

- [Benchmark protocol](benchmarks/PROTOCOL.md)
- [PR #2 multi-view model study](benchmarks/2026-08-14-pr2-multiview-model/final-30-per-arm/RESULTS.md)
- [PR #2 geometry-plus-TSDF study](benchmarks/2026-08-14-pr2-geometry/RESULTS.md)

## Reading rule for performance claims

The canonical numbers are separate studies: DA3-BASE single-image CPU F32 is
171.141 ms versus 238.789 ms for N=20; the PR #2 multi-view model is
8494.734 ms versus 8588.277 ms for N=30; and model-free geometry plus TSDF is
831.797 ms versus 867.421 ms for N=10. The first is a companion model-path
measurement; the latter two have the evidence files above. No stage result is
an end-to-end video, GPU, or browser speed claim, and the percentages must not
be added together.
