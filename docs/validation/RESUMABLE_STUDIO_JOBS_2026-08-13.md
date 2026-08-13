# Resumable Studio jobs — 2026-08-13

## Scope

This milestone makes the local browser intake use the same durable
reconstruction contract as the CLI. It does not change inference, geometry,
or the `.vestra` scene format.

## Contract

- Each upload receives a private `job-NNNNNN` directory.
- `job.json` is atomically replaced and records the fixed video filename,
  reconstruction settings, and lifecycle state.
- **Cancel safely** sends an interrupt to the reconstruction process group.
  The CLI retains its existing atomic-window checkpoint rule.
- On the next `vestra app` start, a previously `running` or
  `cancel_requested` job becomes `interrupted`; no process is assumed to have
  survived the intake restart.
- **Resume job** invokes the same `reconstruct` command with `--resume` and
  the persisted settings, so the CLI's provenance validation remains the
  authority for checkpoint reuse.
- Studio allows one active job at a time. A completed job can launch its local
  viewer; a canceled, interrupted, or failed job can be resumed.

## Verification

```bash
cargo test -p vestra-studio
cargo test --workspace
```

The test suite includes recovery of a persisted running job: its state becomes
`interrupted`, its original settings survive, and a subsequent allocation uses
the next job identifier. The full workspace passed with 54 core, 8 Studio, and
4 CLI tests.

## Operational validation still required

Run the browser flow on the locked Workhorse with the actual F32 model and a
room video: start a job, cancel during reconstruction, restart `vestra app`,
resume it, and inspect the final provenance for reused windows. That is an
environment validation, not something that should be simulated with a fake
model.
