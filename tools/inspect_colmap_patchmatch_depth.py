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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    depth = read_colmap_array(args.depth_map, np)
    report = {"schema": "vestra.colmap-patchmatch-depth-inspection/v1", "depth_map": str(args.depth_map), **summarize_depth(depth, np)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
