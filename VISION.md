# Vestra vision

Vestra should make spatial capture feel immediate and expressive without
confusing visual polish with measured truth.

A user selects a room video. Vestra checks capture quality, chooses useful
frames, and begins reconstruction automatically. Raw geometry appears while
multi-view windows complete. The camera path and fused surfaces stabilize in
front of the user. The finished scene opens as a restrained cinematic world,
with an inspection mode available for confidence, seams, camera frustums, loop
closures, and provenance.

## Truth model

Every geometric primitive belongs to exactly one layer:

1. **Measured** — directly supported by inferred depth and camera evidence.
2. **Fused** — deterministic consolidation, smoothing, interpolation, or small
   hole filling derived from measured evidence.
3. **Generated** — content not sufficiently supported by capture evidence.

V1 may create only deterministic fused geometry: small-hole filling, surface
smoothing, short plane continuation, supported floor/ceiling completion, color
interpolation, and isolated-floater removal. Generative objects and unseen
textures are deferred research features and must never masquerade as measured.

## Definition of done

- Native Rust parity with the pinned PR #2 multi-view world pipeline
- CPU reference and native CUDA production backends
- Browser studio and CLI using one resumable job engine
- Versioned streamable `.vestra` scene bundle
- `.splat`, PLY, GLB, camera JSON, and optional flythrough export
- Raw, fused, and generated layer controls
- Four representative room-video regression fixtures
- Published randomized benchmark matrix with raw data
- First interactive world under 10 seconds and complete fused world under
  15 seconds on the locked Ryzen 9 9950X / RTX 5080 workload
- Cancellation within one second, atomic finalization, and crash-safe resume
- English code and technical documentation
