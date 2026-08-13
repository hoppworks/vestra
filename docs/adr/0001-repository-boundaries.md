# ADR 0001: Independent product, engine, and kernel repositories

Status: accepted — 2026-08-13

## Decision

Vestra is split into `vestra`, `vestra-engine`, and `vestra-kernels`.
Dependencies are versioned and pinned. Local path overrides are development
conveniences, not permission to reach through repository boundaries.

## Consequences

- Vestra can evolve its scene and UI without destabilizing model execution.
- Engine performance work remains independently benchmarkable.
- Kernels remain reusable, narrow, and free of product/model dependencies.
- Cross-repository changes require explicit compatibility commits and revision
  updates in `vestra.lock.toml`.
