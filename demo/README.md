# Public demo fixture

The golden Vestra demo uses the `freiburg1_room` RGB sequence from the TUM
RGB-D Benchmark. It is a real, handheld indoor loop with a stable public source
and a redistribution-compatible license.

Run the preparation script from the repository root:

```bash
./scripts/prepare-demo-input.sh
```

The former v0.1.0 H.264 input was withdrawn from public distribution. A source
audit downloads and verifies the original AVI from its publisher before making
a non-canonical local transcode:

```bash
./scripts/prepare-demo-input.sh .demo-assets --rebuild-from-source
```

FFmpeg/libx264 versions can change the rebuilt MP4 bytes. The `.rebuilt.mp4`
output is therefore not a substitute for a future verified release input.

The finished `.vestra` scene, screenshots, and person-free hero video are
release assets rather than Git objects. Source RGB footage and decoded source
frames are intentionally excluded from public release bundles. Their release
checksum file is the final
distribution authority for each artifact it lists. The source metadata in
`source.json`, standalone `ATTRIBUTION.md`, and attribution in the repository
notice remain versioned with the code.

The machine-readable `release.json`, standalone `ATTRIBUTION.md`, and the
[public-demo validation record](../docs/validation/PUBLIC_DEMO_FREIBURG1_ROOM_2026-08-20.md)
record the exact release-asset hashes, local-reconstruction revisions,
public-product revision, reconstruction counts, global-pose identity, and
COLMAP dense-MVS derivative. Attach `ATTRIBUTION.md` to any redistributed demo
asset. The release manifest is provenance metadata; it does not replace any
asset digest in the distributed checksum file.

To serve an already extracted scene directly:

```bash
cargo run --release --locked -p vestra-cli -- demo --scene /path/to/vestra-demo.vestra
```

`demo` only validates and serves that existing bundle on localhost. Scene
preparation and reconstruction remain separate, explicit workflows.

TUM ground-truth poses are an evaluation oracle only. Vestra does not consume
them while reconstructing the public demo.
