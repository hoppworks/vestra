# Security policy

## Supported versions

Security fixes target the latest release and the `main` branch. Vestra is a
local-first application; its browser studio binds to loopback by default and
must not be exposed directly to an untrusted network.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for the Vestra repository.
Include the affected revision, reproduction steps, impact, and any suggested
mitigation. Do not open a public issue before a fix is available.

## Security boundaries

The project treats the following as security-sensitive:

- video and scene input validation;
- filesystem path containment;
- local HTTP routing and upload handling;
- model and derived-artifact provenance;
- content-addressed scene publication;
- external command invocation;
- GPU driver and dynamically loaded backend boundaries.

Model files, videos, and scene bundles are untrusted inputs. A successful
reconstruction does not establish that the input or model is safe to share.
