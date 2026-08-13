# ADR 0003: C++ oracle parity before beautification

Status: accepted — 2026-08-13

The pinned oracle is `localai-org/depth-anything.cpp` PR #2 at
`f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`.

Vestra first reproduces multi-view inference, back-projection, Sim3 stitching,
ICP, loop closure, TSDF fusion, surfels, voxels, progressive ownership, camera
paths, and exports. Visual enhancements begin only after each phase has an
isolated oracle and the full reconstruction quality gate holds.
