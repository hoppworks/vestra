# End-to-end local-world smoke — 2026-08-13

## Purpose

This is a wiring and provenance smoke test, not a reconstruction-quality or
speed claim. It exercises the complete native Vestra path:

`MP4 → FFmpeg RGB24 frames → Vestra Engine multiview → measured surfels →
relative Sim(3) → voxel fusion → atomic scene manifest → localhost Studio`.

## Environment

- Host: AMD Ryzen 9 9950X Workhorse
- Model: `depth-anything-base-f32.gguf`
- Model SHA-256: `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`
- Engine revision: `1562f8b70a1b35a9908feb88eaa38577b92f2a2a`
- Kernel revision: `bde198958348fcb7a0a294e0d05cd8f2f7e93c5b`
- `RAYON_NUM_THREADS=16`

## Input and command

The input was a controlled 0.8-second, 5fps MP4 made by repeating the public
`canyon.jpg` sample, scaled to 504×336. Repetition makes overlap correspondence
easy to validate; it does not establish room-scale quality.

```text
vestra reconstruct \
  --video vestra-canyon-loop.mp4 \
  --model depth-anything-base-f32.gguf \
  --output vestra-canyon.vestra \
  --frames 4 --width 504 --height 336 \
  --chunk-size 3 --overlap 1 --pixel-stride 32
```

## Result

- decoded frames: 4
- measured windows: 2 (`[0..3)`, `[2..4)`)
- measured points: 880
- fused points: 697
- alignment reports: 1
- fused voxel size: `0.0050612893` relative units
- Studio served both `/manifest.json` and the content-addressed fused chunk
  successfully over `127.0.0.1:4319`.

The resulting manifest contains two immutable raw chunk hashes plus one
`fused_chunk_hash`. This proves fusion does not overwrite evidence.

## Limits

This test deliberately uses repeated imagery and a sparse stride. The next
quality gate is a real capture with camera translation and revisits, assessed
for seam residuals, drift, holes, duplicate surfels, and visual coherence.
