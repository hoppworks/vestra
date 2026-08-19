# Contributing to Vestra

Vestra accepts focused changes that preserve reconstruction evidence and make
the local product easier to verify. Start with a small issue or discussion
before changing model semantics, scene schemas, benchmark boundaries, or
cross-repository APIs.

## Development contract

- Use the pinned Rust toolchain.
- Keep raw measurements immutable. Derived geometry must retain provenance.
- Reject weak pose or fusion evidence instead of silently substituting an
  identity transform or a cosmetically plausible result.
- Compare performance only when model, precision, input, resolution, measured
  work, thread budget, and backend are identical.
- Add a regression test when observable behavior changes.

Run the complete local gate before opening a pull request:

```bash
./scripts/verify.sh
```

Changes spanning Vestra Engine or Vestra Kernels must update the exact
component revisions recorded by the product repository. The pull request must
link the matching component changes and their parity evidence.

## Pull requests

Describe the user-visible outcome first. Then record:

1. the invariant or failure being addressed;
2. the evidence used to accept the change;
3. the exact verification commands and results;
4. any intentionally deferred limitation;
5. benchmark and numerical-parity data when the hot path changes.

Do not commit model weights, private captures, generated scene bundles, build
outputs, or local dependency overrides.
