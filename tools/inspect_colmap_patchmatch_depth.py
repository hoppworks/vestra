#!/usr/bin/env python3
"""Inspect a raw COLMAP PatchMatch depth map without changing reconstruction.

COLMAP stores dense arrays as an ASCII ``width&height&channels&`` header plus
little-endian f32 samples. Keeping this reader separate from fusion lets a
candidate prove coverage and finite-depth behaviour before it becomes input to
any DA3 completion or Studio product.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_colmap_array(path: Path, np: Any) -> Any:
    raw = path.read_bytes()
    header_end = 0
    separators = 0
    while header_end < len(raw) and separators < 3:
        if raw[header_end] == ord("&"):
            separators += 1
        header_end += 1
    if separators != 3:
        raise ValueError("COLMAP dense array has no three-field header")
    try:
        width, height, channels = (int(value) for value in raw[:header_end].decode("ascii").split("&")[:3])
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("invalid COLMAP dense array header") from error
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError("COLMAP dense array dimensions must be positive")
    expected = width * height * channels
    values = np.frombuffer(raw, dtype="<f4", offset=header_end)
    if values.size != expected:
        raise ValueError(f"COLMAP dense array payload has {values.size} values; expected {expected}")
    return values.reshape((height, width, channels))


def summarize_depth(depth: Any, np: Any) -> dict[str, Any]:
    scalar = depth[..., 0] if depth.ndim == 3 else depth
    valid = scalar[np.isfinite(scalar) & (scalar > 0)]
    return {
        "width": int(scalar.shape[1]),
        "height": int(scalar.shape[0]),
        "valid_pixels": int(valid.size),
        "coverage": float(valid.size / scalar.size),
        "depth_median": float(np.median(valid)) if valid.size else None,
        "depth_p95": float(np.percentile(valid, 95)) if valid.size else None,
    }


def sample_bilinear(image: Any, x: float, y: float) -> float | None:
    height, width = image.shape
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    tx, ty = x - x0, y - y0
    value = (
        float(image[y0, x0]) * (1 - tx) * (1 - ty)
        + float(image[y0, x1]) * tx * (1 - ty)
        + float(image[y1, x0]) * (1 - tx) * ty
        + float(image[y1, x1]) * tx * ty
    )
    return value if math.isfinite(value) and value > 0 else None


def track_depth_report(
    pose: dict[str, Any], frame_index: int, depth: Any, source_width: int, source_height: int, np: Any,
) -> dict[str, Any]:
    """Compare a single globally posed dense map with independent sparse tracks."""
    cameras = {
        int(frame["frame_index"]): frame["world_to_camera"]
        for frame in pose.get("frames", [])
        if frame.get("registered") and isinstance(frame.get("world_to_camera"), list)
    }
    w2c = cameras.get(frame_index)
    if w2c is None or len(w2c) != 12:
        raise ValueError("requested frame has no registered global W2C")
    scalar = depth[..., 0] if depth.ndim == 3 else depth
    errors: list[float] = []
    observations = 0
    for track in pose.get("global_trajectory", {}).get("tracks", []):
        point = track.get("position")
        if not isinstance(point, list) or len(point) != 3:
            continue
        for observation in track.get("observations", []):
            if int(observation.get("frame_index", -1)) != frame_index:
                continue
            xy = observation.get("image_xy")
            if not isinstance(xy, list) or len(xy) != 2:
                continue
            observations += 1
            x = (float(xy[0]) + 0.5) * scalar.shape[1] / source_width - 0.5
            y = (float(xy[1]) + 0.5) * scalar.shape[0] / source_height - 0.5
            measured = sample_bilinear(scalar, x, y)
            expected = float(w2c[8]) * point[0] + float(w2c[9]) * point[1] + float(w2c[10]) * point[2] + float(w2c[11])
            if measured is not None and math.isfinite(expected) and expected > 0:
                errors.append(abs(math.log(measured / expected)))
    return {
        "observations": observations,
        "covered_observations": len(errors),
        "coverage": len(errors) / observations if observations else 0.0,
        "median_abs_log_depth_error": float(np.median(errors)) if errors else None,
        "p95_abs_log_depth_error": float(np.percentile(errors, 95)) if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-solution", type=Path,
                        help="optional Vestra COLMAP pose chunk for sparse-track validation")
    parser.add_argument("--frame-index", type=int,
                        help="required with --pose-solution; frame index of this dense map")
    parser.add_argument("--source-width", type=int, default=1620)
    parser.add_argument("--source-height", type=int, default=1080)
    args = parser.parse_args()
    import numpy as np
    if (args.pose_solution is None) != (args.frame_index is None):
        parser.error("--pose-solution and --frame-index must be supplied together")
    if args.source_width <= 0 or args.source_height <= 0:
        parser.error("source dimensions must be positive")
    depth = read_colmap_array(args.depth_map, np)
    report = {"schema": "vestra.colmap-patchmatch-depth-inspection/v1", "depth_map": str(args.depth_map), **summarize_depth(depth, np)}
    if args.pose_solution:
        pose = json.loads(args.pose_solution.read_text())
        report["sparse_track_depth"] = track_depth_report(
            pose, args.frame_index, depth, args.source_width, args.source_height, np
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
