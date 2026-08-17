# COLMAP retrieval and global-BA protocol

Status: active experiment — 2026-08-17

This protocol tests whether a sparse, globally optimized camera trajectory can
replace the chained local Sim(3) trajectory for one Vestra capture. It does
not permit a global world to be published merely because COLMAP produced a
sparse model.

## Input invariants

- An existing `.vestra` bundle with its immutable raster manifest and selected
  `decoded/frame-*.ppm` rasters.
- A pinned COLMAP executable/container and a versioned vocabulary tree.
- The scene's selected rasters, not arbitrary frames from the original video.

The runner verifies every raster SHA-256 against the bundle before feature
extraction. It then uses CPU SIFT, sequential matching and vocabulary-tree
retrieval, maps the resulting graph and performs a final global bundle
adjustment. It refuses to overwrite a previous run directory.

```sh
python3 tools/run_colmap_global_pose.py \
  --scene /path/to/world.vestra \
  --output /path/to/colmap-retrieval-run \
  --vocabulary-tree /path/to/vocab_tree_flickr100K_words256K.bin \
  --container-image docker.io/colmap/colmap:latest \
  --threads 16
```

`--colmap /path/to/pinned-colmap` remains available for a directly installed
binary. Container mode binds only the scene, vocabulary-tree and output
directories at their original paths and runs without network access.

`run.json`, `colmap.log`, the optimized COLMAP model and text `images.txt` are
all retained. Import only the final global-BA output:

```sh
vestra pose-import-colmap \
  --scene /path/to/world.vestra \
  --images-txt /path/to/colmap-retrieval-run/sparse-text/images.txt \
  --provider-version <pinned-colmap-version> \
  --settings-fingerprint "$(jq -r .settings_fingerprint /path/to/colmap-retrieval-run/run.json)"
```

## Pre-committed acceptance gates

A downstream `vestra fuse-global-pose` is allowed only if the imported
trajectory meets all of the following:

1. one connected primary model;
2. at least 90% of selected rasters registered;
3. every DA3 window has at least six registered cameras;
4. every window has normalized local-to-global camera RMS at most `0.15`;
5. at least one geometrically verified non-sequential match spanning at least
   23 seconds for a deliberately revisited room.

The Rust fusion command enforces the per-window gates. The runner records the
other evidence; do not lower these thresholds after seeing a result. A failure
remains useful diagnostic evidence and leaves the local and TSDF products
selected.
