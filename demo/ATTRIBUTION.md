# Vestra public demo attribution — `freiburg1_room`

This notice accompanies the Vestra v0.1.0 public demo and its derived media and
3D artifacts.

## TUM RGB-D Benchmark

The demo is based on the `freiburg1_room` RGB sequence provided by the Computer
Vision Group at the Technical University of Munich.

- Dataset: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>
- Source RGB AVI: <https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room-rgb.avi>
- Source identity: 13,765,996 bytes; SHA-256
  `904f2c932e82e1aa0acf0682800993803b5089b25e424421074ef4f27df7721a`
- License: Creative Commons Attribution 4.0 International
- License text: <https://creativecommons.org/licenses/by/4.0/>

Please cite the dataset publication:

J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers,
“A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012.

Vestra transcoded the distributed AVI to H.264 MP4, removed audio and container
metadata, and processed the RGB frames into depth evidence, 3D scene artifacts,
screenshots, and video. TUM ground-truth poses were not used to reconstruct the
demo. No endorsement by the original authors or Technical University of Munich
is implied.

## COLMAP

The separately labelled global camera and dense-MVS derivative was generated
with COLMAP 4.2.0.dev0 from the official container image identified by digest
`sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`.

- Source: <https://github.com/colmap/colmap>
- License: BSD 3-Clause

The 355,581-point PLY is a COLMAP derivative, not a pure-Rust Vestra output.
Vestra does not redistribute the COLMAP container or binaries.

## Vestra

Vestra supplies the local reconstruction, scene format, provenance and import
gates, local service, and renderer. Vestra source code is licensed under
Apache-2.0: <https://github.com/hoppworks/vestra>.
