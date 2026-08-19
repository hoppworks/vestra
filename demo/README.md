# Public demo fixture

The golden Vestra demo uses the `freiburg1_room` RGB sequence from the TUM
RGB-D Benchmark. It is a real, handheld indoor loop with a stable public source
and a redistribution-compatible license.

Run the preparation script from the repository root:

```bash
./scripts/prepare-demo-input.sh
```

The script downloads the original AVI, verifies its recorded SHA-256, and
creates a browser-compatible H.264 MP4 without cropping or rescaling. This
keeps the distributed 640×480 input intact. Vestra's normal intake pipeline
then records and applies its deterministic 3:2 model crop; the release scene
manifest is the authority for that inference geometry.

The input, finished `.vestra` scene, screenshots, and hero video are release
assets rather than Git objects. Their release checksum file is the final
distribution authority. The source metadata in `source.json` and attribution
in the repository notice remain versioned with the code.

Once the precomputed release scene is present, open it without model download
or inference:

```bash
cargo run --release --locked -p vestra-cli -- demo --scene /path/to/freiburg1_room.vestra
```

`demo` only validates and serves that existing bundle on localhost. Scene
preparation and reconstruction remain separate, explicit workflows.

TUM ground-truth poses are an evaluation oracle only. Vestra does not consume
them while reconstructing the public demo.
