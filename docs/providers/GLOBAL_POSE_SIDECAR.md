# Global pose sidecar contract

Vestra keeps DA3 depth evidence immutable and may derive an additional world
only from a separately published global camera trajectory.  A sidecar does not
replace the local PR#2-relative product and it must never synthesize poses for
frames it did not register.

## Input contract

The sidecar consumes the exact PPM rasters named by the bundle's
`raster.manifest.json`.  It must honour that manifest's crop, output dimensions,
ordering, per-raster names, timestamps, and `raster_fingerprint`.  Any camera
calibration passed to the provider must be transformed after the same crop and
resize; feeding uncropped video calibration is invalid.

## Output contract

Write one JSON document that deserializes as `vestra.pose-solution/v1`:

```json
{
  "schema": "vestra.pose-solution/v1",
  "provider": {
    "kind": "droid-slam",
    "version": "pinned-upstream-revision",
    "settings_fingerprint": "sha256-of-command-and-settings"
  },
  "raster_fingerprint": "exact-manifest-fingerprint",
  "coordinate_convention": "OpenCV world; W2C row-major 3x4 f64",
  "frames": [
    {
      "frame_index": 0,
      "image_name": "frame-000001.ppm",
      "registered": true,
      "world_to_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    }
  ],
  "diagnostics": {
    "input_frames": 720,
    "registered_frames": 1,
    "duplicate_images": 0
  }
}
```

`world_to_camera` is a rigid, right-handed, row-major 3x4 OpenCV W2C matrix in
f64. Frames not tracked by the provider may be omitted; Vestra will report
their windows as insufficient evidence and refuse global publication rather
than interpolate them.

## Import and gates

```sh
vestra-lab pose-import-json --scene world.vestra --solution provider.json
vestra-lab inspect-global-pose --scene world.vestra --pose-solution <hash>
vestra-lab fuse-global-pose --scene world.vestra --pose-solution <hash>
```

The historical command names retain compatibility with the first COLMAP spike;
the actual provider is recorded in the imported artifact and product authority.
Import validates the allow-listed provider (`colmap`, `droid-slam`, or `vggt`),
exact raster identity/fingerprint, rigid W2C values, and diagnostics. Fusion
then independently fits every local DA3 window to registered provider camera
centres. It publishes a separate surfel/TSDF product only if every window meets
the global-fit gate; otherwise the local product remains selected.
