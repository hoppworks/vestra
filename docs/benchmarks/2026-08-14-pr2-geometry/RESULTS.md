# PR #2 Geometry Benchmark — 2026-08-14

## Scope

This is a **geometry-oracle benchmark**, not an inference benchmark. Both arms
consume the identical closed-loop `VPS1` evidence fixture (60 frames at
160×120), perform sequential Sim(3) stitching, two loop closures, pose-graph
optimization, deferred emission, and normal-space TSDF surfel fusion. DA3
inference, video decoding, fixture generation, model loading, and output
comparison are outside the timed interval.

The fixture enables both loop closure and TSDF. It produces 25,434 ordered
surfel outputs in both implementations. The reference-output comparison is
recorded separately in
[`../../validation/TSDF_ORACLE_2026-08-14.md`](../../validation/TSDF_ORACLE_2026-08-14.md).

## Protocol

- Host: Workhorse, AMD Ryzen 9 9950X.
- Thread budget: 16 (`OMP_NUM_THREADS=16` for C++, `RAYON_NUM_THREADS=16` for Rust).
- Arms: pinned C++ PR #2 source `f56e9be`; Vestra `caa8114`.
- Warm-up: one fresh invocation of each arm, excluded.
- Measurement: ten fresh-process invocations per arm in randomized alternating
  order; wall-clock process duration measured with `date +%s%N`.
- C++ writes its compact VPO oracle output; Rust's `oracle-run` writes only a
  short JSON summary. This small output asymmetry is disclosed: the benchmark
  is useful for identifying the Rust geometry gap, but is not yet a claim of
  end-to-end product superiority.
- Source, binary and fixture hashes are retained in `metadata.json`; every raw
  timing is retained in `raw.jsonl`. No outliers were removed.

## Results

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 864.080 | 864.388 | [861.738, 866.422] |
| Vestra Rust | 10 | 1840.159 | 1837.694 | [1833.158, 1847.161] |

Rust is currently **2.130×** the C++ wall time in this bounded geometry tier.
This is an honest regression target, not a completed performance objective.

## Follow-up: TSDF normal-orientation parallelization

The phase profiler then identified that Rust oriented 921,600 independent PCA
normals serially, whereas PR #2 uses an OpenMP loop. The fix parallelizes that
independent pass and reuses per-worker neighbour scratch. It retained the exact
ordered output count, RGB, frame ownership and the TSDF oracle tolerances.

The repeated, fresh 10× trial is stored in
[`after-normal-orientation/raw.jsonl`](after-normal-orientation/raw.jsonl):

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 862.679 | 863.054 | [861.175, 864.183] |
| Vestra Rust | 10 | 1437.229 | 1434.143 | [1429.892, 1444.567] |

This reduces the Rust geometry wall time by **21.90%** relative to the first
series (1840.159 ms → 1437.229 ms) and narrows the C++ ratio from 2.130× to
**1.666×**. It is a real improvement, but Rust is still slower; the next
optimization must be justified by a new phase profile rather than assumed.

## Follow-up: eliminate repeated sequential seam alignment

The loop-closed TSDF adapter first built the sequential seam reports for
emission and then recomputed the same six alignments inside the loop oracle.
PR #2 carries that sequential trajectory forward once. Reusing the reports is
semantically neutral and retains the same C++ differential result.

The final current series is stored in
[`after-deduplicated-seams/raw.jsonl`](after-deduplicated-seams/raw.jsonl):

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 865.059 | 864.023 | [861.996, 868.121] |
| Vestra Rust | 10 | 1223.294 | 1220.570 | [1216.176, 1230.413] |

Rust is now **1.414×** the C++ wall time in this defined tier, a total
**33.52% reduction** versus the initial 1840.159 ms measurement. The remaining
work should focus on the now-measured loop/ICP trajectory work and the PCA
normal kernel; no comparison outcome has been weakened to obtain this result.

## Next action

Profile the Rust TSDF stage separately from seam/loop/emit work before changing
algorithms. The F64 TSDF cell-boundary port has closed output parity, but this
result shows that its implementation and/or normal-estimation execution path
needs a measured optimization campaign before any end-to-end speed claim.
