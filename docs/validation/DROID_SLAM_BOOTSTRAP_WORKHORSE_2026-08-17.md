# DROID-SLAM provider bootstrap — Workhorse

## Purpose

Prepare an isolated GPU pose-provider spike for the immutable `IMG_2323`
raster contract after both COLMAP trajectories failed Vestra's global-fit gate.
This document records environment compatibility only; no DROID trajectory or
world was published.

## Pinned environment

- Host GPU: NVIDIA GeForce RTX 5080, 16,303 MiB, driver 610.43.03.
- Sidecar location: `/var/roothome/vestra-providers/droid-env`; it does not
  alter the host Python 3.14 environment.
- Python: 3.11.15.
- PyTorch: `2.7.1+cu128`; CUDA `12.8`; device capability `[12, 0]` verified.
- DROID-SLAM: `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99` with pinned recursive
  `lietorch` and `pytorch_scatter` submodules.
- CUDA toolkit: 12.8 in the sidecar, including SM 12.0 compilation support.

Both upstream extensions successfully compiled and imported on the RTX 5080:
`import lietorch`, `import torch_scatter`, and `import droid` all passed.

## Model-weight blocker

The official upstream bootstrap script refers to Google Drive object
`1PpqVt1H4maBa_GbPJp4NwxRsd9jk-el`. On 2026-08-17 the official `gdown` request
was rejected by Drive before download, so no `droid.pth` is present and no
trajectory smoke test has been run. This is an external model-hosting access
failure, not a fallback to an unverified checkpoint.

Once the official checkpoint is available locally, run
`tools/droid_slam_export.py` on a short raster subset first, then import it
with `vestra pose-import-json` and evaluate it with
`vestra inspect-global-pose`. A full world may only be emitted if the existing
per-window coverage and normalized camera-fit gates pass.
