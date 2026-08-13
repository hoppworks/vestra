# Relative Seam Gate — Real-Video Regression, 2026-08-13

## Purpose

This validation exercises the production seam-quality gate on the existing
`IMG_2269.MOV` room bundle without changing its immutable measured evidence.
It prevents a numerically finite Sim(3) alignment with a visibly excessive
relative error from being fused automatically.

## Gate

Each `AlignmentReport` now records `normalized_rms_residual`:

```text
RMS point residual / RMS spatial extent of matched target observations
```

The value is dimensionless and therefore valid for Vestra's relative-scale
worlds. Default fusion requires it to be finite and no greater than `1.0`.
That broad product guard is deliberately supplemented by the tighter,
capture-specific raw-residual threshold in `room-relative-v1.json`.

The initial `0.10` proposal was rejected by the real scene: a valid cluttered
room seam measured `0.237`. A second independent re-fusion observed `0.561`.
The correct safe policy is a broad coherence bound plus the named regression
profile, not an uncalibrated universal precision promise.

## Reproducible Workhorse run

The run used a fresh three-repository checkout of Vestra `6beeaf5`, Engine
`1562f8b70a1b35a9908feb88eaa38577b92f2a2a`, and Kernels
`2e4c31faf43991523ca378ff30785cdce17b20ac`; it copied the existing scene
before fusion:

```sh
RUSTFLAGS='-C target-cpu=znver5' cargo build --release -p vestra-cli
./target/release/vestra fuse --scene copied-img-2269.vestra
./target/release/vestra inspect --scene copied-img-2269.vestra
```

Result:

| Signal | Value |
| --- | ---: |
| Measured windows | 14 |
| Fused points | 300,906 |
| Sequential seams | 13 |
| Inlier ratio, min / median / max | 0.8811 / 0.9952 / 1.0000 |
| Raw RMS residual, maximum | 0.14025 |
| Fused point positions finite | yes |
| Loop closure accepted | no (expected for this capture) |

The resulting derivative hash was
`92adff6ebf7c346daddab49e5d8809498de368fa27d9852e73193bce570516ba`.
The copied bundle preserves the original raw provenance; that provenance is
evidence of the input run, not a claim that its old kernel revision executed
this later fusion-only validation.
