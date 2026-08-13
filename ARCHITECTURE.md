# Vestra architecture

## Pipeline

```text
video
  -> decode and capture-quality analysis
  -> deterministic frame selection
  -> overlapping 12/3 multi-view windows
  -> Vestra Engine depth, confidence, intrinsics, extrinsics
  -> back-projection and confidence filtering
  -> robust weighted Sim3 window alignment
  -> bounded point-to-plane seam refinement
  -> revisit proposal and geometric loop measurement
  -> verified-loop Sim3 pose-graph optimization
  -> confidence-weighted voxel surfel fusion
  -> [next] TSDF de-ghosting
  -> measured/fused scene chunks
  -> surfel/voxel render data and open exports
  -> browser studio
```

Metric-scale inference and multi-frame Gaussian reconstruction are deferred.

## Module ownership

`vestra-core` owns the reconstruction job state, deterministic window schedule,
geometry and scene contracts. It calls Vestra Engine but never depends on model
internals or kernel APIs.

Vestra Engine owns the tensor semantics. Its ordered multi-view implementation
matches the pinned C++ oracle: local blocks execute per view, global blocks
flatten `[view, token, channel]`, reference and source camera tokens differ,
and per-view outputs are restored after global attention.

Vestra Kernels exposes primitive slices, explicit shapes, and kernel-owned
types. It does not know about videos, scenes, GGUF metadata, or product jobs.

## Scene storage

A `.vestra` scene is a directory bundle during processing and may be archived
for transport. Immutable content-addressed chunks are written first; the
manifest is atomically replaced last. This supports progressive viewing,
deduplication, checkpoint resume, partial reprocessing, and migrations.

The manifest identifies source fingerprints, engine/kernel revisions, model
and settings hashes, coordinate conventions, and its immutable measured and
derived fused chunk identities. Fused data is published only after its chunk
is durable; raw evidence is never replaced. The canonical fused payload is
also segmented into ordered SHA-addressed point chunks, allowing Studio to
render a world progressively without waiting for a single huge JSON response.

## Reliability

- Job identity derives from input content and normalized settings.
- Each window is an independently durable checkpoint.
- A crash cannot make a partial bundle appear complete.
- Re-running a completed fusion is deterministic and idempotent.
- A seam needs direct shared-pixel evidence, non-degenerate geometry, and the
  configured correspondence/inlier/scale gate; there is no identity fallback.
- Raw windows remain local and immutable until all accepted pose constraints
  have been applied. A pose graph is only run when it receives at least one
  independently verified loop edge; otherwise sequential stitching is a
  strict no-op equivalent.
- PLY export preserves relative coordinates rather than asserting metres.

## Benchmark boundary

The benchmark DAG records decode, preprocessing, backbone, depth/pose heads,
back-projection, Sim3, ICP, loop closure, TSDF, encoding, and viewer readiness.
Ten randomized independent trials per arm are the minimum. Sample size grows
when variance leaves the confidence interval too broad for the declared win.
