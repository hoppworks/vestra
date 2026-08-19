#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
engine_root="${1:-$repo_root/../vestra-engine}"
kernels_root="${2:-$repo_root/../vestra-kernels}"

engine_crate="$engine_root/crates/da-engine"
if [[ ! -f "$engine_crate/Cargo.toml" || ! -f "$kernels_root/Cargo.toml" ]]; then
  echo "expected Vestra Engine and Vestra Kernels checkouts were not found" >&2
  exit 1
fi

mkdir -p "$repo_root/.cargo"
printf '%s\n' \
  '[patch."https://github.com/hoppworks/vestra-engine"]' \
  "vestra-engine = { path = \"$engine_crate\" }" \
  '' \
  '[patch."https://github.com/hoppworks/vestra-kernels"]' \
  "vestra-kernels = { path = \"$kernels_root\" }" \
  >"$repo_root/.cargo/config.toml"

echo "Wrote the ignored local dependency override to .cargo/config.toml"
