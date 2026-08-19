# Public demo fixture

The golden Vestra demo uses the `freiburg1_room` RGB sequence from the TUM
RGB-D Benchmark. It is a real, handheld indoor loop with a stable public source
and a redistribution-compatible license.

Run the preparation script from the repository root:

```bash
./scripts/prepare-demo-input.sh
```

By default the script downloads the exact v0.1.0 H.264 MP4 release asset and
refuses it unless its SHA-256 matches `release.json`. This keeps the canonical
640×480 input byte-identical. Vestra's normal intake pipeline then records and
applies its deterministic 3:2 model crop; the release scene manifest is the
authority for that inference geometry.

An optional source audit downloads and verifies the original AVI before making
a non-canonical local transcode:

```bash
./scripts/prepare-demo-input.sh .demo-assets --rebuild-from-source
```

FFmpeg/libx264 versions can change the rebuilt MP4 bytes. The `.rebuilt.mp4`
output is therefore not a substitute for the verified release input.

The input, finished `.vestra` scene, screenshots, and hero video are release
assets rather than Git objects. Their release checksum file is the final
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

Download, verify, extract, and open the precomputed release scene without model
download or inference:

```bash
./scripts/run-public-demo.sh
```

To serve an already extracted scene directly:

```bash
cargo run --release --locked -p vestra-cli -- demo --scene /path/to/vestra-demo.vestra
```

`demo` only validates and serves that existing bundle on localhost. Scene
preparation and reconstruction remain separate, explicit workflows.

TUM ground-truth poses are an evaluation oracle only. Vestra does not consume
them while reconstructing the public demo.
