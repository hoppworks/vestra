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

## Follow-up: parallel seam estimation

The six adjacent-window seam estimates are mathematically independent; only
their ordered Sim(3) composition is sequential. Computing the reports with
Rayon, then retaining their indexed order for composition, lowers the measured
seam phase from about 200 ms to about 60 ms without changing the reference
cloud.

The 10× confirmation is retained in
[`after-parallel-seams/raw.jsonl`](after-parallel-seams/raw.jsonl):

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 865.517 | 864.325 | [861.881, 869.153] |
| Vestra Rust | 10 | 1083.438 | 1079.696 | [1074.107, 1092.770] |

Vestra is now **41.12% faster than the original Rust geometry baseline**
(1840.159 ms → 1083.438 ms), while still **1.252×** the pinned C++ reference.
The next serious candidate is the remaining ~600 ms PCA normal-estimation
kernel, not another seam or allocation tweak.

## Rejected experiment

A fixed-shape 3×3 maximum-pivot Jacobi kernel was bitwise-identical to the
previous generic solver, but its Workhorse phase profile was approximately
1053 ms versus approximately 1047 ms for the established path. It was reverted
in `9987c0d`; no unproven microkernel is retained as an optimization.

Fusing PCA and camera-normal orientation into one Rayon pass also preserved
output parity but regressed the same profile to approximately 1073 ms, likely
through cache pressure and less favourable load distribution. It was reverted
in `9d7e56a`.

## Next action

Profile the Rust TSDF stage separately from seam/loop/emit work before changing
algorithms. The F64 TSDF cell-boundary port has closed output parity, but this
result shows that its implementation and/or normal-estimation execution path
needs a measured optimization campaign before any end-to-end speed claim.

## Follow-up: deterministic spatial-grid hasher

The current phase profile showed that the normal-estimation spatial grid still
used Rust's cryptographic default `SipHash`, although its only keys are private
finite `(i32, i32, i32)` voxel coordinates. The pinned C++ reference uses a
non-cryptographic numeric voxel hash. Vestra now uses a local deterministic
FNV-1a hasher for that private grid only; equality checks, cell vectors,
neighbour traversal and floating-point accumulation order are unchanged.

This is a smoke A/B, not a replacement for the full 10× study. Both release
builds used `-C target-cpu=znver5`, the same `VPS1` closed-loop+TSDF fixture,
and `RAYON_NUM_THREADS=16` on the idle Ryzen 9 9950X. Three profiled runs per
arm reported:

| Arm | PCA normals mean (ms) | TSDF total mean (ms) | Output surfels |
| --- | ---: | ---: | ---: |
| `0738451` baseline | 581.09 | 822.51 | 25,434 |
| `dbb16fb` fast grid hash | 556.81 | 794.24 | 25,434 |

The candidate reduces the measured PCA phase by **4.18%** and the full TSDF
phase by **3.44%**. All 70 `vestra-core` tests passed; the exact shared-VPS1
C++ geometry oracle remains the correctness gate. A full randomized 10× study
is required before claiming an end-to-end performance improvement.

## Follow-up: packed spatial-grid keys

The C++ PR #2 spatial hash uses one packed 21-bit-per-axis integer key rather
than a compound key. Vestra now uses the same bounded relative-coordinate key
layout and filters radius candidates as each populated neighbour cell is read,
without changing the 27-cell traversal order or any floating-point operation.

On the same quiet Workhorse, release build (`-C target-cpu=znver5`), closed
`VPS1` fixture and `RAYON_NUM_THREADS=16`, three fresh profiled invocations
reported:

| Revision | PCA normals mean (ms) | TSDF total mean (ms) | Output surfels |
| --- | ---: | ---: | ---: |
| `dbb16fb` FNV grid hash | 556.81 | 794.24 | 25,434 |
| `13d8fa2` packed keys | 486.62 | 726.07 | 25,434 |

This is a further **12.61%** reduction in the PCA phase and **8.58%** in the
complete TSDF phase versus the accepted FNV-hash revision. The exact C++ TSDF
oracle still has 25,434 ordered surfels, zero RGB mismatches and identical
frame ownership; position MAE is `3.05e-7`, with a maximum of `0.0010701`.
This accepts the representation change for the locked fixture. It remains a
three-run smoke result; the full fresh-process randomized 10× C++/Rust study
is still required for a product-performance claim.

## Full confirmation after packed-grid optimization

The promised full series is now retained in
[`after-packed-grid/raw.jsonl`](after-packed-grid/raw.jsonl), with exact source,
binary, fixture, CPU and protocol provenance in its adjacent
[`metadata.json`](after-packed-grid/metadata.json). Each arm had one excluded
fresh-process warm-up, followed by ten fresh process invocations in a single
randomized sequence (seed `20260814`). No samples were removed.

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 866.205 | 867.098 | [863.745, 868.664] |
| Vestra Rust | 10 | 964.487 | 962.225 | [959.329, 969.645] |

For this locked geometry-plus-TSDF tier, Rust is now **1.113×** C++ wall time
(11.35% slower). Relative to the initial 1840.159 ms Rust series, this is a
**47.59% reduction**. The confidence interval still sits entirely above the
C++ interval, so this is not a speed win, but it is a fair and reproducible
improvement with exact C++ TSDF-output validation.

## Full confirmation after parallel window backprojection

The prior phase comparison exposed 60 ms of sequential, independent fixture
backprojection. Building each immutable window in Rayon while collecting by
index preserves the C++ schedule and the ordered downstream trajectory, but
releases the available 16-thread CPU budget for that phase. A representative
profile reduced it from 60.715 ms to 11.290 ms; the exact C++ TSDF differential
remained unchanged (25,434 surfels, zero RGB mismatches and identical
frame ownership).

The full randomized, fresh-process confirmation is in
[`after-parallel-windows/raw.jsonl`](after-parallel-windows/raw.jsonl), with
complete provenance in its adjacent
[`metadata.json`](after-parallel-windows/metadata.json). It again uses one
excluded warm-up per arm and ten measurements per arm; no outliers were
removed.

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 866.475 | 866.133 | [864.118, 868.832] |
| Vestra Rust | 10 | 911.959 | 912.094 | [909.121, 914.797] |

Vestra is now **1.052×** C++ wall time in the locked geometry-plus-TSDF tier
(5.25% slower), a **50.44%** reduction from its 1840.159 ms starting point.
The intervals remain separate, so this is an honest near-parity result rather
than a performance win. Future work must target an observed phase gap; it may
not alter the workload, model tensors, precision, schedule or thread budget.

## Accepted win: PR #2 TSDF field hash

The final remaining high-cost serial field operation used Rust's general-purpose
hashing while PR #2 uses a three-axis numeric voxel mixer. Vestra now uses the
same mixer for the private TSDF field. This does not change key equality,
insertion order, cell accumulation, floating-point arithmetic, output sorting,
or the TSDF representation. A unit oracle locks the mixer for signed voxel
coordinates; the closed-loop C++ output oracle still verifies all 25,434
surfels, frame ownership and RGB exactly, with position MAE `3.05e-7`.

The full confirmation is retained in
[`after-tsdf-pr2-hash/raw.jsonl`](after-tsdf-pr2-hash/raw.jsonl) and
[`metadata.json`](after-tsdf-pr2-hash/metadata.json). The protocol is unchanged:
one excluded fresh-process warm-up per arm, 10 fresh process trials per arm,
one randomized order (seed `20260816`), no outlier removal, identical
closed-loop VPS1 fixture and 16-thread budget.

| Arm | n | Mean (ms) | Median (ms) | 95% t CI of mean (ms) |
| --- | --: | --: | --: | ---: |
| C++ PR #2 | 10 | 867.421 | 867.965 | [865.416, 869.426] |
| Vestra Rust | 10 | 831.797 | 827.120 | [819.473, 844.120] |

Vestra wins this locked PR #2 geometry-plus-TSDF workload by **4.11%** on the
mean of trial medians (`0.959×` C++ wall time). The 95% intervals do not
overlap, including the retained 870.656 ms Rust sample; this is a documented
speed win, not an outlier-filtered claim. It is narrowly scoped to this
model-free geometry fixture. It must not be presented as an end-to-end DA3
inference or GPU result.
