# Public demo validation — TUM RGB-D `freiburg1_room`

Date: 2026-08-20

This record identifies the inputs, code revisions, reconstruction counts, and
provider boundary for the public Vestra demo. It is intended for a reviewer who
needs to verify what the release contains without inferring provenance from a
screenshot or point count.

The record is specific to this release run. It is not a performance benchmark,
a metric-accuracy certificate, or a claim that every derived product was
implemented entirely in Rust.

## Release identity

| Item | Recorded value |
| --- | --- |
| Release | `v0.1.0` |
| Fixture | TUM RGB-D Benchmark `freiburg1_room` RGB sequence |
| Local-reconstruction Vestra revision | [`4b0630e9606d816fe38a65b6e2d74fd10a6c825d`](https://github.com/hoppworks/vestra/commit/4b0630e9606d816fe38a65b6e2d74fd10a6c825d) |
| Local-reconstruction Engine revision | [`ec10ae38e8ceff3da4778fc2b47ad8f868dac311`](https://github.com/hoppworks/vestra-engine/commit/ec10ae38e8ceff3da4778fc2b47ad8f868dac311) |
| Local-reconstruction Kernels revision | [`1ad85305de14ea76ddd878af6dac80f19bdf2bc3`](https://github.com/hoppworks/vestra-kernels/commit/1ad85305de14ea76ddd878af6dac80f19bdf2bc3) |
| Public product revision | [`d626d9b0c255cc1d7c0397276554622f325c9478`](https://github.com/hoppworks/vestra/commit/d626d9b0c255cc1d7c0397276554622f325c9478) |
| Validation host | AMD Ryzen 9 9950X, NVIDIA RTX 5080, 96 GiB RAM |

The measured and local-TSDF products were generated at the earlier release
baseline. The later public product revision contains the exact-timestamp and
bridge-frame COLMAP provider tooling as well as the import, presentation, and
serving surface used for the release. The release therefore combines artifacts
from this recorded lineage; it must not be attributed entirely to either
revision. The public product's newer Engine and Kernels lock revisions must not
be substituted into the historical local-reconstruction provenance.

## Input provenance

The source is the RGB AVI distributed for the
[TUM RGB-D Benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset). It is
used under CC-BY-4.0; dataset attribution and license terms are retained in the
repository's third-party notice.

| Artifact | Format | Recorded properties | SHA-256 |
| --- | --- | --- | --- |
| [Original `freiburg1_room` RGB video](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room-rgb.avi) | AVI | 13,765,996 bytes | `904f2c932e82e1aa0acf0682800993803b5089b25e424421074ef4f27df7721a` |
| Prepared demo input | H.264 MP4 | 24,218,662 bytes; 640×480; 30 fps; 45.4 s | `0447ecc3033fa8ef125820f4c53a48b3ba0ec11ebbd3dae310d38769a6063f9f` |

The preparation step transcodes the AVI to a browser-compatible H.264 MP4,
removes audio and container metadata, and does not crop or rescale the video.
Vestra applies its recorded inference-raster policy later in the reconstruction
pipeline.

The MP4 digest identifies the canonical release derivative. A local transcode made
with a different FFmpeg or libx264 build is not assumed to be byte-identical;
only the distributed 24,218,662-byte asset with the recorded digest is the
release input.

## Vestra reconstruction record

| Measurement | Recorded value |
| --- | ---: |
| Candidate sampling rate | 8 fps |
| Candidate frames | 363 |
| Selected geometry frames | 77 |
| Measured windows | 9 |
| Measured points | 4,275,936 |
| Local TSDF points | 40,980 |

These counts describe persisted evidence and derived local geometry. They do
not by themselves establish global coherence, completeness, metric scale, or
accuracy against the TUM reference trajectory.

## Global pose and dense-MVS derivative

The release also contains a separately attributable COLMAP-derived global
camera and dense-MVS result:

| Item | Recorded value |
| --- | --- |
| Provider | COLMAP 4.2.0.dev0 |
| Vestra provider tooling | [`d626d9b0c255cc1d7c0397276554622f325c9478`](https://github.com/hoppworks/vestra/commit/d626d9b0c255cc1d7c0397276554622f325c9478) |
| Container image digest | `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f` |
| Cameras in the global bundle adjustment | 150 |
| Cameras selected for the release product | 55 |
| Pose-solution hash | `3891cbb96dc89e75280feacd8c813fe1ef0a44e42dcc66f2af77208175d7b14b` |
| Dense-MVS points | 355,581 |
| Dense-MVS PLY | 9,600,921 bytes; SHA-256 `be34da0370ba102543e18bf2f0d42b6c44d5371e6acb088474decc99279af969` |

COLMAP performs the global bundle adjustment and dense multi-view stereo that
produce this PLY. The 355,581-point MVS artifact is therefore a COLMAP
derivative, not a pure-Rust Vestra reconstruction. Vestra's Rust components
validate and import the provider artifacts, retain their provenance, publish
them as a separately labelled product, and serve and render that product.

The reconstruction did **not** consume the TUM ground-truth poses. The
benchmark trajectory remains an external evaluation reference and is not an
input to either the local reconstruction or the COLMAP provider run.

No separate provider-settings or bridge-schedule fingerprint was supplied for
this record. The provider result is bounded here by the Vestra tooling revision,
COLMAP image digest, camera counts, pose-solution hash, and PLY digest; this
record does not invent a stronger identity.

## Integrity boundary

The versioned [demo release manifest](../../demo/release.json) repeats the
machine-readable values in this record. The source contract remains in the
[demo source manifest](../../demo/source.json), and the standalone
[attribution notice](../../demo/ATTRIBUTION.md) is suitable for distribution
beside the downloadable assets.

| Release asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `vestra-demo-freiburg1-room-v0.1.0.tar.zst` | 380,358,870 | `01400b0596456eda44e52561b33d139f75af043efc460682e48768876b3f2f12` |
| `freiburg1_room-rgb.avi` | 13,765,996 | `904f2c932e82e1aa0acf0682800993803b5089b25e424421074ef4f27df7721a` |
| `vestra-demo-input.mp4` | 24,218,662 | `0447ecc3033fa8ef125820f4c53a48b3ba0ec11ebbd3dae310d38769a6063f9f` |
| `vestra-demo-freiburg1-room-mvs.ply` | 9,600,921 | `be34da0370ba102543e18bf2f0d42b6c44d5371e6acb088474decc99279af969` |
| `capture-depth-replay.mp4` | 1,911,875 | `c86cbf037478d929996bac299e0d8dd2e9bb9223ce464713a171d6f4bf7bfe66` |
| `global-world-orbit.mp4` | 1,436,711 | `dd0b7ab160dd24f21e2652756d700e6bf22cf267e71de973273568105021d4b6` |
| `capture-depth-poster.png` | 557,373 | `6d798907f5f7ecbb2673b3fd4649eef53f0aa9d0b82e893ddcf2e7ccbbc55a66` |
| `global-world-poster.png` | 505,483 | `d6fb72ab51ecf51cf87c830ed8b13bab851dface8f0b3a3120e0762cb3bd1a7d` |

Neither versioned manifest authenticates an asset whose bytes are not recorded
above. The checksum file published beside the release assets is the authority
only for the individual artifacts it lists. A screenshot or hero video has no
checksum claim unless it has its own entry there.

## Verification

Download and verify the exact release MP4 with the repository helper:

```bash
./scripts/prepare-demo-input.sh
```

The default mode refuses an MP4 whose SHA-256 differs from the recorded release
digest. To audit the original AVI and make a deliberately non-canonical local
transcode, use:

```bash
./scripts/prepare-demo-input.sh .demo-assets --rebuild-from-source
```

That optional mode verifies the AVI digest and writes
`vestra-demo-input.rebuilt.mp4`. Its encoder-dependent digest is not expected
to equal the canonical MP4 digest.

Release assets can also be checked directly:

```bash
shasum -a 256 /path/to/freiburg1_room-rgb.avi
shasum -a 256 /path/to/vestra-demo-input.mp4
shasum -a 256 /path/to/vestra-demo-freiburg1-room-v0.1.0.tar.zst
shasum -a 256 /path/to/vestra-demo-freiburg1-room-mvs.ply
```

Each command must report the corresponding digest recorded above. On Linux,
`sha256sum` may be used instead of `shasum -a 256`. Do not substitute a local
FFmpeg/libx264 transcode for the signed MP4 solely because its decoded video
properties match.

After separately checking the downloaded release archive against its published
checksum, inspect and serve the precomputed scene without model download or
inference:

```bash
./scripts/run-public-demo.sh
```

The helper verifies the recorded archive digest before extraction. To inspect
or serve an already extracted scene directly:

```bash
cargo run --release --locked -p vestra-cli -- \
  inspect --scene /path/to/vestra-demo.vestra
cargo run --release --locked -p vestra-cli -- \
  demo --scene /path/to/vestra-demo.vestra
```

The demo command validates the bundle before serving it on loopback. It does
not rerun reconstruction or COLMAP.
