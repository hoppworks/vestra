#!/usr/bin/env python3
"""Create an auditable global-depth hybrid from verified DA3 and COLMAP MVS.

MVS is used only at depth-tested pixels that it actually observes through the
same authoritative COLMAP camera as DA3.  Pose-conditioned DA3 remains intact
where MVS has no measurement.  This is deliberately a separate product: it
never mutates the calibrated DA3 evidence or presents MVS incompleteness as a
complete mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

from run_da3_pose_conditioned import sha256_file, write_depth_frames, write_ply


SCHEMA = "vestra.da3-mvs-hybrid/v1"
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_mvs_positions(path: Path, np: Any) -> Any:
    """Read only XYZ from a standard binary-little-endian COLMAP fused PLY."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("MVS input is not a PLY")
        encoding = None
        count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY header")
            fields = line.decode("ascii").strip().split()
            if fields == ["end_header"]:
                break
            if fields[:1] == ["format"]:
                encoding = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                count = int(fields[2])
                in_vertex = True
            elif fields[:1] == ["element"]:
                in_vertex = False
            elif in_vertex and fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in PLY_TYPES:
                    raise ValueError("unsupported MVS PLY vertex property")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
        if encoding != "binary_little_endian" or count is None:
            raise ValueError("MVS PLY must be binary_little_endian")
        names = {name for name, _ in properties}
        if not {"x", "y", "z"} <= names:
            raise ValueError("MVS PLY has no XYZ vertices")
        dtype = np.dtype([(name, "<" + kind) for name, kind in properties])
        payload = np.fromfile(handle, dtype=dtype, count=count)
    positions = np.stack((payload["x"], payload["y"], payload["z"]), axis=1).astype(np.float32)
    finite = np.isfinite(positions).all(axis=1)
    return positions[finite]


def project_mvs_depth(
    positions: Any, intrinsics: Any, w2c: Any, width: int, height: int, np: Any,
) -> Any:
    """Depth-test global MVS vertices into one authoritative DA3 raster."""
    output = np.full(width * height, np.inf, dtype=np.float32)
    rotation = np.asarray(w2c[:3, :3], dtype=np.float32)
    translation = np.asarray(w2c[:3, 3], dtype=np.float32)
    fx, fy, cx, cy = (float(intrinsics[0, 0]), float(intrinsics[1, 1]),
                      float(intrinsics[0, 2]), float(intrinsics[1, 2]))
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("invalid DA3 intrinsics in verified source artifact")
    # Batching bounds temporary arrays while keeping Z-buffer ownership exact.
    for start in range(0, len(positions), 200_000):
        world = positions[start:start + 200_000]
        camera = world @ rotation.T + translation
        z = camera[:, 2]
        valid = np.isfinite(z) & (z > 0)
        if not valid.any():
            continue
        camera, z = camera[valid], z[valid]
        x = np.rint(fx * camera[:, 0] / z + cx).astype(np.int64)
        y = np.rint(fy * camera[:, 1] / z + cy).astype(np.int64)
        visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if visible.any():
            np.minimum.at(output, y[visible] * width + x[visible], z[visible])
    return output.reshape(height, width)


def percentile(values: Any, q: float, np: Any) -> float | None:
    valid = values[np.isfinite(values)]
    return float(np.percentile(valid, q)) if valid.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True,
                        help="accepted vestra.da3-pose-conditioned-calibration/v2 artifact")
    parser.add_argument("--mvs-ply", type=Path, required=True,
                        help="geometric COLMAP stereo_fusion PLY in the same global frame")
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
        raise ValueError("calibrated source must contain exactly its canonical selected.npz")
    selected_path = args.artifact / "selected.npz"
    if sha256_file(selected_path) != batches[0].get("sha256"):
        raise ValueError("calibrated selected.npz does not match its manifest hash")
    positions = read_mvs_positions(args.mvs_ply, np)
    if positions.size == 0:
        raise ValueError("MVS PLY contains no finite positions")
    with np.load(selected_path) as arrays:
        depth = arrays["depth"].copy()
        conf = arrays["conf"].copy()
        rgb = arrays["rgb"].copy()
        intrinsics = arrays["intrinsics"].copy()
        extrinsics = arrays["extrinsics"].copy()
        frame_indices = arrays["frame_indices"].copy()
    if depth.ndim != 3 or depth.shape[1:] != (336, 504) or len(depth) != len(source.get("published_frames", [])):
        raise ValueError("calibrated source has an invalid canonical depth tensor")
    coverage: list[dict[str, Any]] = []
    for ordinal in range(len(depth)):
        mvs_depth = project_mvs_depth(positions, intrinsics[ordinal], extrinsics[ordinal], 504, 336, np)
        observed = np.isfinite(mvs_depth) & (mvs_depth > 0)
        source_conf = conf[ordinal]
        # Retain MVS only where it has a real Z-buffer sample.  The confidence
        # floor ensures those observations survive the established percentile
        # emission policy without deleting untouched DA3 support.
        floor = percentile(source_conf, 75, np)
        depth[ordinal][observed] = mvs_depth[observed]
        if floor is not None:
            source_conf[observed] = np.maximum(source_conf[observed], floor)
        coverage.append({
            "frame_index": int(frame_indices[ordinal]),
            "mvs_pixels": int(observed.sum()),
            "mvs_coverage": float(observed.mean()),
            "mvs_depth_median": percentile(mvs_depth[observed], 50, np),
        })
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
        "mvs_ply_sha256": sha256_file(args.mvs_ply),
        "mvs_vertices": int(len(positions)),
        "pixel_policy": "mvs-zbuffer-where-observed-else-da3/v1",
        "per_frame": coverage,
        "median_mvs_coverage": percentile(np.asarray([row["mvs_coverage"] for row in coverage]), 50, np),
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
                      "median_mvs_coverage": manifest["hybrid"]["median_mvs_coverage"]}))


if __name__ == "__main__":
    main()
