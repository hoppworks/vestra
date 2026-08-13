# Browser intake validation — IMG_2269

Date: 2026-08-13  
Scope: current Vestra browser upload, full CPU reconstruction, and local Studio
delivery for the user-provided `IMG_2269.mov`. Scale remains relative.

## Environment

| Field | Value |
| --- | --- |
| Vestra revision | `0581fe1` (`style: align Studio with Hoppworks visual language`) |
| Model | `depth-anything-base-f32.gguf` |
| Model SHA-256 | `1b13b166e8a8b4f2c862f42d36edb2f9aab995a18cc527a52b9f160b99c6b8da` |
| Host | Workhorse, AMD Ryzen 9 9950X, `RAYON_NUM_THREADS=16`, `-C target-cpu=znver5` |
| Input | `IMG_2269.mov`, 81,709,361 bytes, 40.367 seconds |
| Reconstruction | 120 frames, 504x336 RGB24, 12-frame windows, 3-frame overlap, confidence 1.0, pixel stride 8 |
| Browser | local `agent-browser`, through loopback-only SSH forwarding |

## Browser flow actually exercised

1. Opened the loopback-only Vestra intake.
2. Selected and uploaded the Desktop `IMG_2269.mov` through its file input.
3. Pressed **Start reconstruction** from the real browser UI.
4. Verified the job-owned input copy, the running process, and browser job state.
5. Waited for all checkpoints and the fused-world publication.
6. Opened the resulting local Studio in a browser.
7. Enabled camera rays, seam/loop diagnostics, and advanced the source-frame
   picture-in-picture. Browser console errors were empty.

## Published result

```text
decoded frames: 120
inferred windows: 13
measured points: 412,776
fused points: 298,577
capture: Ready (mean adjacent luma delta 0.0974588692188263)
```

`vestra inspect` additionally confirmed finite fused geometry, six progressive
binary surfel chunks, 12 sequential seams, 95,256 source-pixel
correspondences, 92,688 accepted inliers, a minimum seam inlier ratio of
89.3046%, and a maximum seam RMS residual of 0.140133. No loop closure was
accepted; that conservative result is expected for this capture.

## Presentation result

The current Studio and intake use the Hoppworks portfolio visual language:
warm light capture surface, restrained green accents, rounded panels, display
serif wordmark, and compact mono technical labels. The 3D canvas deliberately
remains dark so real surfel colors, camera frustums, and seam links retain
contrast. This was verified against the live fused scene rather than a mocked
fixture.

## Deliberate limits

This proves the local relative-scale point/surfel world workflow. It does not
claim metre accuracy, a watertight mesh, semantic completion, or an accepted
loop closure for this particular video.
