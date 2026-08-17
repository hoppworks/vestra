#!/usr/bin/env python3
"""Publish an auditable DA3 + geometric-PatchMatch depth derivative.

Unlike a fused MVS PLY, COLMAP PatchMatch maps retain one dense depth sample
per source camera. This tool only accepts the *geometric* map suffix, resamples
those depth values onto the immutable DA3 504x336 pixel-centre grid, and uses
DA3 only where the geometric map has no finite observation. It never changes
the global W2C/K supplied by the accepted pose-conditioned DA3 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inspect_colmap_patchmatch_depth import read_colmap_array
from run_da3_pose_conditioned import sha256_file, write_depth_frames, write_ply


SCHEMA = "vestra.da3-mvs-hybrid/v1"
PIXEL_POLICY = "colmap-patchmatch-geometric-resample-else-da3/v1"


def resample_nearest_depth(depth: Any, width: int, height: int, np: Any) -> Any:
    """Map source pixel centres to a target raster without depth interpolation."""
    scalar = depth[..., 0] if depth.ndim == 3 else depth
    source_height, source_width = scalar.shape
    x = np.rint((np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5).astype(np.int64)
    y = np.rint((np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5).astype(np.int64)
    x = np.clip(x, 0, source_width - 1)
    y = np.clip(y, 0, source_height - 1)
    sampled = scalar[y[:, None], x[None, :]].astype(np.float32, copy=True)
    sampled[~np.isfinite(sampled) | (sampled <= 0)] = np.nan
    return sampled


def depth_map_path(directory: Path, frame_index: int) -> Path:
    return directory / f"frame-{frame_index + 1:06d}.ppm.geometric.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True,
                        help="accepted vestra.da3-pose-conditioned-calibration/v2 artifact")
    parser.add_argument("--geometric-depth-maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.pixel_stride <= 0 or not 0 <= args.confidence_percentile <= 100:
        parser.error("pixel stride and confidence percentile are invalid")

    import numpy as np

    source_path = args.artifact / "manifest.json"
    source = json.loads(source_path.read_text())
    if source.get("schema") != "vestra.da3-pose-conditioned-calibration/v2" or source.get("decision") != "accepted":
        raise ValueError("input must be an accepted calibrated DA3 V2 artifact")
    batches = source.get("batches", [])
    if len(batches) != 1 or batches[0].get("file") != "selected.npz":
        raise ValueError("calibrated source must contain exactly selected.npz")
    selected_path = args.artifact / "selected.npz"
    if sha256_file(selected_path) != batches[0].get("sha256"):
        raise ValueError("calibrated selected.npz does not match its manifest hash")
    with np.load(selected_path) as arrays:
        depth = arrays["depth"].copy()
        conf = arrays["conf"].copy()
        rgb = arrays["rgb"].copy()
        intrinsics = arrays["intrinsics"].copy()
        extrinsics = arrays["extrinsics"].copy()
        frame_indices = arrays["frame_indices"].copy()
    if depth.ndim != 3 or depth.shape[1:] != (336, 504):
        raise ValueError("calibrated source has an invalid canonical depth tensor")

    evidence = []
    map_hashes = []
    for ordinal, raw_frame_index in enumerate(frame_indices):
        frame_index = int(raw_frame_index)
        path = depth_map_path(args.geometric_depth_maps, frame_index)
        if not path.is_file():
            raise ValueError(f"missing geometric PatchMatch map for frame {frame_index}: {path}")
        map_depth = read_colmap_array(path, np)
        sampled = resample_nearest_depth(map_depth, 504, 336, np)
        observed = np.isfinite(sampled) & (sampled > 0)
        depth[ordinal][observed] = sampled[observed]
        if observed.any():
            floor = float(np.percentile(conf[ordinal][np.isfinite(conf[ordinal])], 75))
            if np.isfinite(floor):
                conf[ordinal][observed] = np.maximum(conf[ordinal][observed], floor)
        record = {
            "frame_index": frame_index,
            "file": path.name,
            "sha256": sha256_file(path),
            "source_width": int(map_depth.shape[1]),
            "source_height": int(map_depth.shape[0]),
            "mvs_pixels": int(observed.sum()),
            "mvs_coverage": float(observed.mean()),
        }
        evidence.append(record)
        map_hashes.append(record)
    index = {"schema": "vestra.colmap-patchmatch-geometric-index/v1", "maps": map_hashes}
    index_blob = json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")
    index_sha256 = hashlib.sha256(index_blob).hexdigest()

    args.output.mkdir(parents=True)
    selected_output = args.output / "selected.npz"
    np.savez_compressed(selected_output, depth=depth, conf=conf, rgb=rgb,
                        intrinsics=intrinsics, extrinsics=extrinsics,
                        frame_indices=frame_indices)
    manifest = dict(source)
    manifest["schema"] = SCHEMA
    manifest["source"] = {
        "raw_manifest_sha256": sha256_file(source_path),
        "raster_fingerprint": source["raster_fingerprint"],
        "pose_solution_hash": source["pose_solution_hash"],
        "batch_files": [{"file": "selected.npz", "sha256": sha256_file(selected_path)}],
    }
    manifest["batches"] = [{"file": selected_output.name, "sha256": sha256_file(selected_output),
                            "depth_shape": list(depth.shape)}]
    manifest["hybrid"] = {
        "pixel_policy": PIXEL_POLICY,
        "median_mvs_coverage": float(np.median([row["mvs_coverage"] for row in evidence])),
        "mvs_depth_map_index_sha256": index_sha256,
        "mvs_depth_map_count": len(evidence),
        "per_frame": evidence,
    }
    manifest["decision"] = "accepted"
    manifest["depth_frames"] = write_depth_frames(args.output / "depth-frames", [selected_output], np)
    ply = args.output / "world.ply"
    points = write_ply(ply, [selected_output], args.confidence_percentile, args.pixel_stride, np)
    manifest["ply"] = {"schema": source["ply"]["schema"], "file": ply.name,
                       "sha256": sha256_file(ply), "points": points,
                       "confidence_percentile": args.confidence_percentile}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": SCHEMA, "output": str(args.output), "points": points,
                      "median_mvs_coverage": manifest["hybrid"]["median_mvs_coverage"],
                      "mvs_depth_maps": len(evidence)}))


if __name__ == "__main__":
    main()
