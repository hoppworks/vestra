# Calibrated DA3 artifact V2

## Decision

`vestra.da3-pose-conditioned/v1` remains an immutable, raw official-DA3
artifact. Calibration is a separate `vestra.da3-pose-conditioned-calibration/v2`
artifact and can never overwrite or be accepted by the raw importer.

The purpose is narrow: correct frame-local dense-depth scale bias using the
already accepted global COLMAP camera/landmark solution. It does not change a
camera pose, estimate a Sim(3), or create metric scale.

## Evidence contract

The V2 manifest binds the raw input by SHA-256 and records the exact source
slot selected for every published frame.

```json
{
  "schema": "vestra.da3-pose-conditioned-calibration/v2",
  "source": {
    "raw_manifest_sha256": "…",
    "raster_fingerprint": "…",
    "pose_solution_hash": "…",
    "batch_files": [{"file": "batch-0000.npz", "sha256": "…"}]
  },
  "contract": {
    "pixel_mapping": "pixel-center-resize/v1",
    "track_split": "sha256-track-id-fold/v1",
    "reprojection_error_px_max": 2.5,
    "minimum_train_tracks": 24,
    "minimum_heldout_tracks": 6,
    "maximum_heldout_median_log_error": 0.20,
    "minimum_accepted_frame_fraction": 0.85
  },
  "frames": [{
    "frame_index": 42,
    "source_batch": "batch-0003.npz",
    "source_slot": 5,
    "status": "accepted",
    "scale": 1.037,
    "train": {"tracks": 312, "median_abs_log_error": 0.06},
    "held_out": {"tracks": 78, "median_abs_log_error": 0.08}
  }],
  "decision": "accepted"
}
```

`pixel-center-resize/v1` maps a COLMAP observation `(u, v)` from its source
raster to a DA3 raster of width `W`, height `H` as:

```text
u' = (u + 0.5) * W / source_width - 0.5
v' = (v + 0.5) * H / source_height - 0.5
```

## Selection and gates

1. A SHA-256 bucket of `(pose_solution_hash, track_id)` selects train versus
   held-out. A track always remains in the same split in every observation.
2. For each duplicate DA3 prediction of one frame, choose the candidate using
   only train median residual, train-track count, batch index, and source slot.
   Held-out values never choose a candidate.
3. Fit `median(log(z_colmap / z_da3))` on the selected train tracks. Apply its
   exponent only to that selected frame's depth raster.
4. Require train >= 24, held-out >= 6, per-frame held-out median <= 0.20, and
   accepted-frame coverage >= 85%. A failed run writes diagnostics only.
5. Generate PLY, replay rasters, and TSDF input from accepted selected frames
   only. Rejected or non-canonical overlap predictions have no geometry asset.
6. An independent inspector recalculates held-out values from the frozen
   source artifact and V2 manifest before the Rust publisher accepts it.

## Product boundary

The Rust command `import-da3-pose-conditioned-calibrated` accepts only V2
artifacts with `decision: accepted`. It publishes
`da3-pose-conditioned-colmap-calibrated-surfel`; its TSDF derivative has its
own calibrated ID. The raw DA3, COLMAP MVS, local relative, and existing TSDF
products remain untouched and selectable.

## Tests

- exact half-pixel mapping and stable track split;
- held-out edits cannot affect train-only overlap selection;
- rejected frames create no PLY vertices or replay image;
- raw manifest/NPZ/raster/pose/K/W2C tampering is rejected;
- insufficient support, coverage, or context quality produces a rejected V2
  diagnostic with no importable geometry;
- synthetic end-to-end fixture proves calibrated coordinates and source-camera
  replay against the selected raw slot.
