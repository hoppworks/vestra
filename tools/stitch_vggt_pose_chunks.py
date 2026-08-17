#!/usr/bin/env python3
"""Align overlapping official-VGGT pose chunks into one Vestra trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONVENTION = "OpenCV world; W2C row-major 3x4 f64"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "vestra.pose-solution/v1" or data.get("provider", {}).get("kind") != "vggt":
        raise ValueError(f"{path} is not a VGGT Vestra pose chunk")
    if data.get("coordinate_convention") != CONVENTION:
        raise ValueError(f"{path} has a non-OpenCV W2C convention")
    return data


def registered(data: dict) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for frame in data["frames"]:
        if frame.get("registered"):
            index = frame["frame_index"]
            if index in result:
                raise ValueError("duplicate VGGT frame")
            result[index] = frame
    return result


def decompose(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(frame["world_to_camera"], dtype=np.float64).reshape(3, 4)
    return project_rotation(value[:, :3]), value[:, 3]


def project_rotation(value: np.ndarray) -> np.ndarray:
    """Projects a bf16/float camera matrix onto the nearest proper rotation."""
    left, _, right_t = np.linalg.svd(value)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def centre(frame: dict) -> np.ndarray:
    rotation, translation = decompose(frame)
    return -rotation.T @ translation


def fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if variance <= 1e-12:
        raise ValueError("degenerate overlapping VGGT camera centres")
    left, singular, right_t = np.linalg.svd((target_zero.T @ source_zero) / len(source))
    sign = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        sign[-1, -1] = -1.0
    rotation = left @ sign @ right_t
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    return scale, rotation, target_mean - scale * rotation @ source_mean


def robust_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, int]:
    active = np.ones(len(source), dtype=bool)
    for _ in range(6):
        scale, rotation, translation = fit(source[active], target[active])
        residual = np.linalg.norm(scale * (rotation @ source.T).T + translation - target, axis=1)
        threshold = max(3.0 * float(np.median(residual[active])), 1e-5)
        candidate = residual <= threshold
        if candidate.sum() < 12:
            raise ValueError("fewer than 12 geometrically consistent overlap cameras")
        if np.array_equal(active, candidate):
            return scale, rotation, translation, int(active.sum())
        active = candidate
    scale, rotation, translation = fit(source[active], target[active])
    return scale, rotation, translation, int(active.sum())


def transform(frame: dict, scale: float, world_rotation: np.ndarray, world_translation: np.ndarray) -> dict:
    rotation, translation = decompose(frame)
    aligned_rotation = rotation @ world_rotation.T
    aligned_translation = scale * translation - aligned_rotation @ world_translation
    return {
        "frame_index": frame["frame_index"],
        "image_name": frame["image_name"],
        "registered": True,
        "world_to_camera": np.column_stack((aligned_rotation, aligned_translation)).reshape(-1).astype(float).tolist(),
    }


def main() -> int:
    options = args()
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite {options.output}")
    chunks = [read(path) for path in options.chunk]
    fingerprint = chunks[0]["raster_fingerprint"]
    if any(chunk["raster_fingerprint"] != fingerprint for chunk in chunks):
        raise ValueError("all VGGT chunks must use the exact same raster manifest")
    identity = np.eye(3)
    zero = np.zeros(3)
    output = {index: transform(frame, 1.0, identity, zero) for index, frame in registered(chunks[0]).items()}
    if not output:
        raise ValueError("first VGGT chunk has no registered frames")
    alignments = []
    for ordinal, chunk in enumerate(chunks[1:], start=1):
        current = registered(chunk)
        overlap = sorted(set(output) & set(current))
        if len(overlap) < 12:
            raise ValueError(f"chunk {ordinal} has fewer than 12 overlapping VGGT cameras")
        source = np.vstack([centre(current[index]) for index in overlap])
        target = np.vstack([centre(output[index]) for index in overlap])
        scale, rotation, translation, inliers = robust_fit(source, target)
        for index, frame in current.items():
            output.setdefault(index, transform(frame, scale, rotation, translation))
        alignments.append({
            "chunk": ordinal,
            "overlap": len(overlap),
            "inliers": inliers,
            "scale": scale,
            "rotation": rotation.reshape(-1).astype(float).tolist(),
            "translation": translation.astype(float).tolist(),
        })
    diagnostics = chunks[0]["diagnostics"]
    provenance = {"chunk_sha256": [sha256(path) for path in options.chunk], "alignments": alignments}
    settings_fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    solution = {
        "schema": "vestra.pose-solution/v1",
        "provider": {"kind": "vggt", "version": "official-vggt-overlap-stitch-v1", "settings_fingerprint": settings_fingerprint},
        "raster_fingerprint": fingerprint,
        "coordinate_convention": CONVENTION,
        "frames": [output[index] for index in sorted(output)],
        "diagnostics": {"input_frames": diagnostics["input_frames"], "registered_frames": len(output), "duplicate_images": 0},
        "provenance": provenance,
    }
    options.output.write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(options.output), "registered_frames": len(output), "alignments": alignments}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
