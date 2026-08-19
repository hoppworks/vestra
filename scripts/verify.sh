#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo metadata --locked --format-version 1 >/dev/null
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
python3 -m unittest discover -s tests -p 'test_*.py' -v
node --test crates/vestra-studio/tests/camera_controls.test.js
cargo doc --locked --workspace --no-deps
git diff --check
