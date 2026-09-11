# Changelog

All notable changes to Vestra are documented here. The project follows
[Semantic Versioning](https://semver.org/) once the first public release is
tagged.

## Unreleased

### Fixed

- Studio keyboard navigation: `A`/`D` and `W`/`S` turned and pitched in the
  opposite direction of their labels, and the home view looked up at the
  world from below. The look sign convention is now documented and pinned to
  the rendered forward vector by tests.
- Holding a navigation key no longer speeds up after the OS auto-repeat delay,
  a long "look up" no longer banks invisible pitch that "look down" had to
  unwind before the view responded, and diagonal movement is no longer faster
  than straight movement.
- Navigation keys can no longer stick: browser modifier chords (for example
  `⌘←`) are ignored and clear held keys, Shift/CapsLock changing the
  character mid-press still releases the key, keys are tracked by physical
  code, and window blur, hidden tabs and the replay overlay drop held state.
- Arrow keys no longer move the hidden 3D camera or fight the video's native
  seeking while the "video + depth" replay is open, and `R` no longer stops
  keys that are still physically held.

### Changed

- Studio movement speed is derived from the world extent and elapsed time only
  (no per-frame minimum step), so it is identical on 60 Hz and 144 Hz
  displays and for small relative-scale worlds.
- The Studio test gate runs every `crates/vestra-studio/tests/*.test.js`
  suite: camera math, physical-key controller invariants, and first-person
  stepping semantics.

## 0.1.0 - 2026-08-20

### Added

- Reproducible release toolchain and verification gate.
- Public, attribution-bound golden demo fixture contract.
- CI and public repository hardening across the Vestra repository family.
- `vestra demo` for validating and serving a precomputed scene without model
  download or inference.
- Exact-timestamp bridge-frame support for the pinned COLMAP global-pose
  provider while retaining a selected-frame-only import model.

### Changed

- Portfolio documentation now separates accepted production behavior,
  qualified experiments, rejected approaches, and future work.
- The six-command `vestra` product surface is separate from the explicit
  28-command `vestra-lab` engineering and oracle surface.
- Engine and kernel dependencies are pinned to their public-release commits.
