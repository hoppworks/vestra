# Vestra room regression protocol

This protocol validates a **relative-scale surfel world**, not room dimensions,
semantic topology, or a watertight mesh.

## Fixture set

Run four local phone-video fixtures with the locked 120-frame, 504×336, 12/3
schedule and the same F32 model:

1. `room-loop` — continuous perimeter pass plus a deliberate revisit to the
   starting viewpoint; it must be checked with a profile requiring a verified
   loop closure.
2. `room-clutter` — furnished room with occluders and door openings.
3. `room-open` — mostly clear room with long planar surfaces.
4. `room-curved` — circular or curved-wall room; inspect seams visually rather
   than forcing a planar topology interpretation.

Store only source hashes, normalized capture settings, derived `.vestra`
bundle hashes, revisions, and result JSON in version control. Do not add the
private videos to the repository.

## Evidence command

```bash
vestra-lab verify \
  --scene room.vestra \
  --profile docs/quality-profiles/room-relative-v1.json \
  > room.verification.json
```

The baseline profile requires complete window coverage, finite fusion, enough
fused surfels, and supported sequential seams. It deliberately does not gate
the sparse stride-8 world on voxel connectedness: that diagnostic is reported
by `vestra inspect` and is expected to be fragmented before dense/TSDF work.

For `room-loop`, copy the profile and set `require_loop_closure` to `true`.
A rejection is useful evidence: retain the JSON violations and recapture or
tighten the candidate geometry rather than lowering the profile silently.

## Acceptance record

For every fixture, record:

- video SHA-256, model SHA-256, engine/kernel/Vestra revisions and build flags;
- capture quality and all `verify` JSON evidence;
- Studio inspection notes for seams, duplicate sheets, holes, and openings;
- whether a loop was required, proposed, accepted, or rejected;
- timing only under the separate randomized benchmark protocol.

The fixture becomes a regression only after its immutable result and review
record are accepted. Do not turn a capture-risk indicator or a relative Sim(3)
residual into a metric accuracy claim.
