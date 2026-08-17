# ADR 0005: Fixed-rate candidates, evidence-selected geometry frames

Status: accepted — 2026-08-17

## Context

An arbitrary total frame count makes long videos temporally sparse and short
videos needlessly dense. Conversely, forwarding every high-rate video frame
into chained local reconstruction adds near-duplicate observations and more
opportunities for registration drift.

## Decision

Vestra decodes at a fixed candidate rate (currently 8 fps) with a safety
ceiling. It then retains geometry inputs deterministically using a 0.4-second
minimum temporal baseline, thumbnail luma novelty, a local sharpness floor,
and a maximum 0.6-second temporal gap. The lower bound prevents a smooth pan
from treating every candidate as a new geometry view; the upper bound keeps a
slow walk-through from developing large temporal holes. The first and final
candidate frames are always retained.

The selected canonical rasters are renumbered for the engine, while
`decoded/selection.json` records the original candidate indices. Raster
manifests derive timestamps from those indices. The selection policy/version
is part of the reconstruction provenance fingerprint, so a resumed job cannot
silently use cache rasters produced by a different policy.

## Consequences

- The final count scales with capture motion and duration rather than a
  product-wide target number.
- This is not a claim of geometric parallax or calibrated pose; global
  COLMAP/DROID/VGGT providers remain responsible for that stronger evidence.
- Candidate caches remain sufficient for a future provider-specific selector,
  provided it publishes a new, fingerprinted raster manifest.
