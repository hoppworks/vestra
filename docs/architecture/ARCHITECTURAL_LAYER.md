# Vestra architectural layer

## Status

The architectural layer is a separate, derived product.  It does not alter the
measured or globally fused MVS surfel world.  A viewer must always be able to
switch back to the source world that supplied its evidence.

The first geometry-only RANSAC prototype is deliberately **experimental**.  It
is useful to validate point-cloud plumbing, but it is not a surface-recognition
solution: on `IMG_2323` it splits visually continuous surfaces into several
nearby planes.  It must not become the default product or be presented as a
finished room model.

The first supported product is deliberately narrower: **floor and wall
surfaces only**.  The current attic capture has beams and an irregular roof,
so ceiling and roof labels are explicitly excluded rather than forced into a
misleading planar mesh.

## Accepted design

The production path combines two independent evidence sources:

1. **Global geometry** provides the 3D position, plane fit, spatial support,
   and the boundary of every emitted cell or triangle.
2. **Per-frame semantic masks** label the corresponding observed raster pixels
   as floor, wall, ceiling/roof, door/opening, window, or non-architectural.

Neither source may fabricate unobserved space.  Semantic masks select and
classify geometry; they cannot extend a wall through an unobserved region.

## Output rules

- Floors and walls are planar regions assembled only from spatially connected,
  observed support.  Ceiling and roof labels are excluded in this release.
- Furniture stays in the Reality product and is excluded from Architecture.
- A door or opening is a hole in a supported wall polygon, never a painted
  rectangle.  A traversal through the candidate opening is provisional
  evidence; observations from both sides confirm it.
- Windows remain optional and need strong geometry plus semantic support.
- The architecture product is relative-scale unless an independently verified
  scale anchor is present.

## Evidence contract

The semantic sidecar uses `vestra.architecture-semantics/v1`.  Each frame has
the exact decoded raster size, a model identifier and immutable model revision,
and dense class/confidence rasters.  A frame is rejected if dimensions or
classes do not match the raster contract used for geometry.  The core type is
defined in `crates/vestra-core/src/architecture.rs`; model runners are adapters
and do not own geometry.

## Quality gates before publishing an architecture product

1. Every emitted cell/triangle has direct global geometry support and
   high-confidence compatible semantic observations from at least two views,
   unless it is on an observed scene boundary.
2. Plane residual, connected-component extent, and multi-view label agreement
   are recorded per surface.
3. Door/opening candidates retain their holes unless both mask and geometric
   depth discontinuity agree.  Traversal is recorded as additional evidence.
4. Reality and Architecture remain distinct selectable products; an
   Architecture preview can never replace the source world silently.

## Current extraction order

`vestra extract-architecture --semantic-volume <masks.vsem>` first projects a
bounded, spatially distributed sample of the global surfel world into every
registered camera.  It requires two agreeing high-confidence floor or wall
labels, rejects ceiling/roof outright, and vetoes candidates with repeated
door/opening evidence.  Only then are floor and wall candidates fitted in
separate RANSAC passes (one floor plane, up to the requested wall-plane
budget).  Directly supported cells become triangles; no hole is bridged.

## Model boundary

Many high-quality indoor semantic checkpoints are trained on ADE20K.  The
official class set contains wall, floor, ceiling, windowpane, and door, but its
dataset terms are research-only.  Vestra therefore records model licence and
revision in every sidecar and does not make an unreviewed checkpoint a default
product dependency.  For local research, an explicitly selected checkpoint can
be used; any distributed/commercial default requires a separately verified
licence.
