# Third-party notices

Vestra is licensed under Apache-2.0. The following notices cover upstream
software, model code, and data whose ideas or implementation are represented
in this repository or its published demo artifacts.

## Depth Anything 3

Vestra Engine implements the Depth Anything 3 architecture and consumes the
Apache-2.0 `DA3-BASE` checkpoint family.

- Source: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- License: Apache License 2.0
- Copyright 2025 The Depth Anything 3 Team

The repository's root `LICENSE` contains the Apache License 2.0 text. Model
licenses are checkpoint-specific: this notice applies to `DA3-BASE`, not to
the separately published Large, Giant, or Nested checkpoints.

## depth-anything.cpp

Vestra's pinned geometry-oracle modules adapt portions of the multi-view
streaming, Sim(3), ICP, pose-graph, and normal-space TSDF contracts from
`depth-anything.cpp` pull request 2 at commit
`f56e9be43a22c12ef575584d2fa57a6a5d5be7ae`.

- Source: <https://github.com/localai-org/depth-anything.cpp/tree/f56e9be43a22c12ef575584d2fa57a6a5d5be7ae>
- License: MIT

MIT License

Copyright (c) 2026 the depth-anything.cpp authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## ggml

Vestra Kernels contains specialized Rust kernels whose numerical contracts
were compared with, and in limited cases adapted from, ggml at commit
`eced84c86f8b012c752c016f7fe789adea168e1e`.

- Source: <https://github.com/ggml-org/ggml/tree/eced84c86f8b012c752c016f7fe789adea168e1e>
- License: MIT

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## TUM RGB-D Benchmark — `freiburg1_room`

Public demo media and derived 3D artifacts are based on the
`freiburg1_room` sequence provided by the Computer Vision Group at the
Technical University of Munich.

Dataset publication:

J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers,
“A Benchmark for the Evaluation of RGB-D SLAM Systems,” IROS 2012.

- Source: <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>
- License: Creative Commons Attribution 4.0 International
- License text: <https://creativecommons.org/licenses/by/4.0/>

Changes made by the Vestra project:

- transcoded the distributed AVI to H.264 MP4;
- removed audio and container metadata;
- processed the RGB frames into derived depth previews, 3D scene artifacts,
  screenshots, and videos.

No endorsement by the original authors or Technical University of Munich is
implied.

## COLMAP

The public demo's optional globally registered dense-MVS control is generated
with COLMAP 4.2.0.dev0 from the official container image pinned by digest in
the demo validation record. Vestra does not redistribute the container or
COLMAP binaries.

- Source: <https://github.com/colmap/colmap>
- License: BSD 3-Clause

The resulting scene remains labelled as a COLMAP MVS derivative. COLMAP does
not supply Vestra's scene format, provenance gates, product selection, local
service, or renderer.
