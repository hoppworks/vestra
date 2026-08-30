# Licensing and provenance decision

**Scope:** Vestra, Vestra Engine, and Vestra Kernels as inspected on
2026-08-30. This is an engineering compliance record, not legal advice.

## Short answer

Yes: an Apache-2.0 licence can cover the repositories' **Daniel-owned original
work** and can coexist with the upstream work used in them. It must not be
represented as changing the licence of upstream code or model weights. In
particular, an MIT-derived Rust port remains subject to the MIT notice and an
Apache-derived port remains subject to Apache-2.0's redistribution conditions.

The recommended policy is therefore:

| Repository | Current primary licence | Recommended primary licence | Required provenance retained |
| --- | --- | --- | --- |
| `vestra` | Apache-2.0 | Apache-2.0 | `THIRD_PARTY_NOTICES.md` |
| `vestra-engine` | MIT | Apache-2.0 for the next release, only for Daniel-owned contributions | `THIRD_PARTY_NOTICES.md` and the bundled MIT/Apache texts |
| `vestra-kernels` | Apache-2.0 | Apache-2.0 | `THIRD_PARTY_NOTICES.md` |

This is not a conversion of third-party material to Apache-2.0. It is an
Apache-2.0 licence for the repository's own contributions, with the listed
upstream licences continuing to apply to the corresponding material.

## DA3 code and weights

- The official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
  publishes its code under Apache-2.0: [root LICENSE](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE)
  and [project metadata](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/pyproject.toml).
- The official publisher's [DA3-BASE model card](https://huggingface.co/depth-anything/DA3-BASE/blob/main/README.md)
  identifies the checkpoint as Apache-2.0. That permits this project to use
  DA3-BASE under those terms, but does not make Vestra the author or owner of
  the architecture or weights.
- This finding is checkpoint-specific. The official DA3 model table lists
  different terms for Large, Giant, and Nested variants; do not copy the
  DA3-BASE claim to those models.
- None of the three inspected Vestra repositories vendors the DA3-BASE weights.
  A future release that bundles a converted GGUF must preserve the checkpoint
  licence/provenance; converting a file format does not change the model's
  licence.

## What Apache-2.0 requires here

Apache-2.0 section 4 requires distributors of an upstream derivative to:

1. provide a copy of Apache-2.0;
2. mark modified upstream files prominently;
3. retain relevant copyright, patent, trademark, and attribution notices; and
4. reproduce relevant upstream `NOTICE` contents if the upstream distribution
   includes a `NOTICE` file.

See the [official Apache-2.0 text, section 4](https://www.apache.org/licenses/LICENSE-2.0).
The inspected DA3 checkout contains a root `LICENSE` but no project-level
`NOTICE`, so there is no DA3 NOTICE-file text to copy. This does **not** remove
the obligation to preserve relevant source notices when source is copied or
adapted.

## MIT material and a move from MIT to Apache-2.0

MIT permits use, modification, distribution, and sublicensing, provided its
copyright and permission notice accompany all copies or substantial portions.
See the [official MIT licence text](https://opensource.org/license/mit).

Accordingly, direct ports/adaptations from `depth-anything.cpp` or `ggml` may
be distributed in an Apache-2.0 project only if their MIT notices remain. They
are not thereby relicensed exclusively to Apache-2.0. The existing notices do
this by retaining the full MIT texts and the named upstream revisions.

An owner may publish **their own** prior MIT contribution under Apache-2.0 in a
later version (or dual-license it). This leaves the already granted MIT rights
in place and requires ownership or authorisation from every copyright holder;
commit metadata alone is evidence of repository history, not a legal ownership
assignment. Apache defines the licensor as the copyright owner or an authorised
entity in its [official text](https://www.apache.org/licenses/LICENSE-2.0), and
the [U.S. Copyright Office overview](https://www.copyright.gov/what-is-copyright/)
describes the owner-controlled exclusive rights.

## Local provenance observed

All three reachable Git histories currently show one listed author, Daniel
Hopp (361 commits in `vestra`, 113 in `vestra-engine`, and 65 in
`vestra-kernels`), with no Git submodules or vendored source directories in
those repositories. That supports a clean, single-maintainer release process,
but does not prove assignment of any work created for someone else.

The repositories already document the material that needs to remain attributed:

- `vestra` names DA3 Apache-2.0, `depth-anything.cpp` MIT, ggml MIT, the TUM
  RGB-D dataset CC BY 4.0, and COLMAP BSD-3-Clause.
- `vestra-engine` states that model semantics and several execution surfaces
  are direct Rust ports or structure-preserving translations of the pinned
  MIT-licensed `depth-anything.cpp`, and includes both MIT texts plus the
  Apache-2.0 text.
- `vestra-kernels` identifies adapted ggml Q8_0 code and retains full ggml MIT
  and `depth-anything.cpp` MIT notices; it also identifies DA3 as Apache-2.0.

## Release checklist for the Engine licence change

Before changing `vestra-engine` to Apache-2.0, make one explicit licence commit
that changes its root `LICENSE`, workspace `Cargo.toml` metadata, README, and
release notes together. Keep `THIRD_PARTY_NOTICES.md` and
`LICENSES/MIT-depth-anything.cpp.txt`, `LICENSES/MIT-ggml.txt`, and
`LICENSES/Apache-2.0.txt` unchanged except for factual provenance corrections.
Keep existing copied-file source headers and mark future direct ports as
modified/adapted. Do not claim that all files are exclusively Apache-2.0.

For a binary or archive release, include `LICENSE`, `THIRD_PARTY_NOTICES.md`,
and the referenced licence texts alongside the artefact. For any future model
download or bundle, record the exact model identifier, revision, licence, and
publisher in the release manifest.
