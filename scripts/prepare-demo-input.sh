#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_dir="$repo_root/.demo-assets"
mode="release"
if [[ "${1:-}" == "--rebuild-from-source" ]]; then
  mode="$1"
elif [[ -n "${1:-}" ]]; then
  asset_dir="$1"
  mode="${2:-release}"
fi
release_url="https://github.com/hoppworks/vestra/releases/download/v0.1.0/vestra-demo-input.mp4"
source_url="https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room-rgb.avi"
source_sha256="904f2c932e82e1aa0acf0682800993803b5089b25e424421074ef4f27df7721a"
output_sha256="0447ecc3033fa8ef125820f4c53a48b3ba0ec11ebbd3dae310d38769a6063f9f"
source_path="$asset_dir/freiburg1_room-rgb.avi"
output_path="$asset_dir/vestra-demo-input.mp4"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
sha256_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$asset_dir"

if [[ "$mode" == "release" ]]; then
  if [[ ! -f "$output_path" ]]; then
    curl --fail --location --retry 3 --output "$output_path.part" "$release_url"
    mv "$output_path.part" "$output_path"
  fi

  actual_output_sha256="$(sha256_file "$output_path")"
  if [[ "$actual_output_sha256" != "$output_sha256" ]]; then
    echo "demo input SHA-256 mismatch: expected $output_sha256, got $actual_output_sha256" >&2
    exit 1
  fi

  echo "Prepared exact release input $output_path"
  echo "SHA-256 $actual_output_sha256"
  exit 0
fi

if [[ "$mode" != "--rebuild-from-source" ]]; then
  echo "usage: $0 [asset-dir] [--rebuild-from-source]" >&2
  exit 2
fi

command -v ffmpeg >/dev/null || { echo "ffmpeg is required for --rebuild-from-source" >&2; exit 1; }

if [[ ! -f "$source_path" ]]; then
  curl --fail --location --retry 3 --output "$source_path.part" "$source_url"
  mv "$source_path.part" "$source_path"
fi

actual_sha256="$(sha256_file "$source_path")"
if [[ "$actual_sha256" != "$source_sha256" ]]; then
  echo "demo source SHA-256 mismatch: expected $source_sha256, got $actual_sha256" >&2
  exit 1
fi

rebuilt_path="$asset_dir/vestra-demo-input.rebuilt.mp4"
ffmpeg -hide_banner -loglevel error -y \
  -i "$source_path" \
  -map 0:v:0 -an -map_metadata -1 \
  -c:v libx264 -preset slow -crf 14 \
  -pix_fmt yuv420p -movflags +faststart \
  "$rebuilt_path"

echo "Rebuilt $rebuilt_path from the verified source AVI"
echo "SHA-256 $(sha256_file "$rebuilt_path")"
echo "Encoder versions can change bytes; use release mode for the canonical input." >&2
