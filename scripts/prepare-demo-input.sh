#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_dir="${1:-$repo_root/.demo-assets}"
source_url="https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room-rgb.avi"
source_sha256="904f2c932e82e1aa0acf0682800993803b5089b25e424421074ef4f27df7721a"
source_path="$asset_dir/freiburg1_room-rgb.avi"
output_path="$asset_dir/vestra-demo-input.mp4"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

sha256_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir -p "$asset_dir"
if [[ ! -f "$source_path" ]]; then
  curl --fail --location --retry 3 --output "$source_path.part" "$source_url"
  mv "$source_path.part" "$source_path"
fi

actual_sha256="$(sha256_file "$source_path")"
if [[ "$actual_sha256" != "$source_sha256" ]]; then
  echo "demo source SHA-256 mismatch: expected $source_sha256, got $actual_sha256" >&2
  exit 1
fi

ffmpeg -hide_banner -loglevel error -y \
  -i "$source_path" \
  -map 0:v:0 -an -map_metadata -1 \
  -c:v libx264 -preset slow -crf 14 \
  -pix_fmt yuv420p -movflags +faststart \
  "$output_path"

echo "Prepared $output_path"
echo "SHA-256 $(sha256_file "$output_path")"
