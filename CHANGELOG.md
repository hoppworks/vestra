# Changelog

All notable changes to Vestra are documented here. The project follows
[Semantic Versioning](https://semver.org/) once the first public release is
tagged.

## Unreleased

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
