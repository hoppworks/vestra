# Motion-video local-world smoke — 2026-08-13

## Purpose

This is a native Vestra wiring and provenance smoke run over moving footage. It
does not claim room-reconstruction accuracy, loop closure, metric scale, or
visual quality. Its purpose is to exercise more than two overlapping windows
through the deferred pose collection and relative surfel fusion route.

## Environment

- Host: AMD Ryzen 9 9950X Workhorse
- Thread budget: `RAYON_NUM_THREADS=16`
- Build: `RUSTFLAGS=-C target-cpu=znver5`, Vestra `275210c`
- Model: `depth-anything-base-f32.gguf` (F32), SHA-256
  `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da`
- Engine revision: `1562f8b70a1b35a9908feb88eaa38577b92f2a2a`
- Kernel revision: `bde198958348fcb7a0a294e0d05cd8f2f7e93c5b`

## Input and command

The input was the public `robot_unitree.mp4` example shipped by
Depth-Anything-3. It is not a room and does not contain a revisited loop.

```text
vestra reconstruct \
  --video robot_unitree.mp4 \
  --model depth-anything-base-f32.gguf \
  --output robot-motion.vestra \
  --frames 12 --width 504 --height 336 \
  --chunk-size 6 --overlap 2 --pixel-stride 32
vestra export --scene robot-motion.vestra --output world.ply
```

## Result

- decoded frames: 12 from 3.495 s
- capture indicator: `ready` (mean adjacent luma delta `0.05544386`)
- measured windows: 3 (`[0..6)`, `[4..10)`, `[8..12)`)
- measured points: 2,816
- fused relative-scale surfels: 2,710
- exported ASCII PLY rows: 2,710
- fused chunk SHA-256:
  `bc72dec49dc41cac2b278779a8ca20fa4cd66315e03cd1fe693ed677a00774c2`

The successful run confirms that raw windows are checkpointed independently,
sequential poses are collected before a single final fusion pass, and the
derived fused chunk remains content-addressed. It is not evidence of a
geometrically coherent room: that requires a real capture and the pending
revisit candidate/measurement path.
