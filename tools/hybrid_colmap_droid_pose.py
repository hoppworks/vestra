#!/usr/bin/env python3
"""Create one auditable global pose solution from COLMAP plus DROID-SLAM.

COLMAP remains authoritative where it registered a raster.  DROID supplies
only missing COLMAP observations after its world is robustly aligned to the
COLMAP camera centres with one Sim(3).  This is deliberately a provider
combination, not interpolation: frames missing from both inputs remain absent
and Vestra's existing per-window fit gate is still authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


KIND = "hybrid-colmap-droid"
CONVENTION = "OpenCV world; W2C row-major 3x4 f64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-solution", type=Path, required=True)
    parser.add_argument("--droid-solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    payload = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            payload.update(block)
    return payload.hexdigest()


def load_solution(path: Path, expected_kind: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "vestra.pose-solution/v1":
        raise ValueError(f"{path} is not a Vestra pose solution")
    if value.get("provider", {}).get("kind") != expected_kind:
        raise ValueError(f"{path} is not a {expected_kind} solution")
    if value.get("coordinate_convention") not in {
        "COLMAP world; W2C row-major 3x4 f64", CONVENTION,
    }:
        raise ValueError(f"{path} has an unsupported W2C convention")
    return value


def frame_map(solution: dict) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for frame in solution.get("frames", []):
        if frame.get("registered"):
            index = frame.get("frame_index")
            matrix = frame.get("world_to_camera")
            if not isinstance(index, int) or not isinstance(matrix, list) or len(matrix) != 12:
                raise ValueError("registered pose frame is malformed")
            if index in result:
                raise ValueError("duplicate registered frame")
            result[index] = frame
    return result


def w2c(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(frame["world_to_camera"], dtype=np.float64).reshape(3, 4)
    return matrix[:, :3], matrix[:, 3]


def centre(frame: dict) -> np.ndarray:
    rotation, translation = w2c(frame)
    return -rotation.T @ translation


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("need at least three paired 3D camera centres")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if not np.isfinite(variance) or variance <= 1e-12:
        raise ValueError("degenerate DROID camera-centre geometry")
    covariance = (target_zero.T @ source_zero) / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1, -1] = -1.0
    rotation = left @ sign @ right_t
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid similarity scale")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def robust_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    active = np.ones(len(source), dtype=bool)
    for _ in range(6):
        scale, rotation, translation = fit_similarity(source[active], target[active])
        residual = np.linalg.norm((scale * (rotation @ source.T).T + translation) - target, axis=1)
        median = float(np.median(residual[active]))
        extent = float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1)))
        threshold = max(3.0 * median, 0.03 * extent, 1e-5)
        next_active = residual <= threshold
        if next_active.sum() < 12:
            raise ValueError("too few consistent COLMAP/DROID camera pairs")
        if np.array_equal(active, next_active):
            return scale, rotation, translation, active
        active = next_active
    scale, rotation, translation = fit_similarity(source[active], target[active])
    return scale, rotation, translation, active


def transformed_droid_w2c(frame: dict, scale: float, rotation: np.ndarray, translation: np.ndarray) -> list[float]:
    source_rotation, source_translation = w2c(frame)
    aligned_rotation = source_rotation @ rotation.T
    aligned_translation = scale * source_translation - aligned_rotation @ translation
    return np.column_stack((aligned_rotation, aligned_translation)).reshape(-1).astype(float).tolist()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    colmap = load_solution(args.colmap_solution, "colmap")
    droid = load_solution(args.droid_solution, "droid-slam")
    if colmap["raster_fingerprint"] != droid["raster_fingerprint"]:
        raise ValueError("providers were not run on the same immutable raster evidence")
    colmap_frames = frame_map(colmap)
    droid_frames = frame_map(droid)
    common = sorted(set(colmap_frames) & set(droid_frames))
    source = np.vstack([centre(droid_frames[index]) for index in common])
    target = np.vstack([centre(colmap_frames[index]) for index in common])
    scale, rotation, translation, inliers = robust_similarity(source, target)

    frames: list[dict] = []
    source_names = {frame["frame_index"]: frame["image_name"] for frame in colmap["frames"]}
    for index in sorted(set(colmap_frames) | set(droid_frames)):
        if index in colmap_frames:
            frame = dict(colmap_frames[index])
            frame["world_to_camera"] = [float(value) for value in frame["world_to_camera"]]
        else:
            droid_frame = droid_frames[index]
            frame = {
                "frame_index": index,
                "image_name": source_names.get(index, droid_frame["image_name"]),
                "registered": True,
                "world_to_camera": transformed_droid_w2c(droid_frame, scale, rotation, translation),
            }
        frames.append(frame)

    settings = {
        "colmap_solution_sha256": digest(args.colmap_solution),
        "droid_solution_sha256": digest(args.droid_solution),
        "common_registered_frames": len(common),
        "similarity_inliers": int(inliers.sum()),
        "droid_to_colmap_scale": scale,
        "droid_to_colmap_rotation": rotation.reshape(-1).astype(float).tolist(),
        "droid_to_colmap_translation": translation.astype(float).tolist(),
        "fallback_policy": "colmap-where-registered-else-aligned-native-droid",
    }
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema": "vestra.pose-solution/v1",
        "provider": {"kind": KIND, "version": "colmap+droid-slam-v1", "settings_fingerprint": fingerprint},
        "raster_fingerprint": colmap["raster_fingerprint"],
        "coordinate_convention": CONVENTION,
        "frames": frames,
        "diagnostics": {
            "input_frames": colmap["diagnostics"]["input_frames"],
            "registered_frames": len(frames),
            "duplicate_images": 0,
        },
        "provenance": settings,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "registered_frames": len(frames), **settings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
