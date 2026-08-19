#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_dir="${1:-$repo_root/.demo-assets/public-v0.1.0}"
port="${2:-4317}"
archive_name="vestra-demo-freiburg1-room-v0.1.0.tar.zst"
archive_sha256="01400b0596456eda44e52561b33d139f75af043efc460682e48768876b3f2f12"
release_url="https://github.com/hoppworks/vestra/releases/download/v0.1.0/$archive_name"
archive_path="$asset_dir/$archive_name"
scene_path="$asset_dir/vestra-demo.vestra"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v cargo >/dev/null || { echo "cargo is required" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar is required" >&2; exit 1; }
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 1; }

sha256_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$asset_dir"
if [[ ! -f "$archive_path" ]]; then
  echo "Downloading the 380 MB public scene archive..."
  curl --fail --location --retry 3 --output "$archive_path.part" "$release_url"
  mv "$archive_path.part" "$archive_path"
fi

actual_sha256="$(sha256_file "$archive_path")"
if [[ "$actual_sha256" != "$archive_sha256" ]]; then
  echo "scene archive SHA-256 mismatch: expected $archive_sha256, got $actual_sha256" >&2
  exit 1
fi

if [[ ! -f "$scene_path/manifest.json" ]]; then
  echo "Extracting the verified scene..."
  zstd --quiet --decompress --stdout "$archive_path" | tar -xf - -C "$asset_dir"
fi

echo "Opening the verified public demo at http://127.0.0.1:$port"
cd "$repo_root"
exec cargo run --release --locked -p vestra-cli -- \
  demo --scene "$scene_path" --port "$port"
