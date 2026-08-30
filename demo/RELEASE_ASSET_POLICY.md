# Public release-asset policy

Public Vestra releases are reproducibility artifacts, not source-capture
archives. A release may contain a person-free rendered or reconstructed scene,
its provenance metadata, checksums, documentation, and attribution.

Do not publish source videos, decoded source frames, EXIF-bearing captures,
audio, private captures, credentials, model weights, or material that could
identify a person without an explicit distribution review. If a dataset has a
permissive copyright license but contains identifiable people, remove or
appropriately redact the material before publishing it in a Vestra release.

Every release archive must include `ATTRIBUTION.md`, `source.json`,
`release.json`, and `SHA256SUMS` at its top level. `ATTRIBUTION.md` must name
the source, link the source and license, name the license, and state the
transformation.
