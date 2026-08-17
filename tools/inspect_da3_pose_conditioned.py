#!/usr/bin/env python3
"""Measure a pose-conditioned DA3 artifact against its COLMAP authority.

The report deliberately separates two evidence channels: held-out sparse
COLMAP landmark depths test the global-camera/depth agreement, while repeated
overlap frames test whether independent DA3 batches disagree at their seam.
Neither metric is a claim of metric scale or a substitute for visual review.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "vestra.da3-pose-conditioned-inspection/v1"
RAW_SCHEMA = "vestra.da3-pose-conditioned/v1"
CALIBRATED_SCHEMA = "vestra.da3-pose-conditioned-calibration/v2"
MVS_HYBRID_SCHEMA = "vestra.da3-mvs-hybrid/v1"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


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


def camera_depth(w2c: list[float], position: list[float]) -> float:
    return w2c[8] * position[0] + w2c[9] * position[1] + w2c[10] * position[2] + w2c[11]


def inspect(scene: Path, artifact: Path, pose_hash: str) -> dict[str, Any]:
    import numpy as np

    sidecar = json.loads((artifact / "manifest.json").read_text())
    pose = json.loads((scene / "chunks" / f"pose-{pose_hash}.json").read_text())
    if sidecar.get("schema") not in {RAW_SCHEMA, CALIBRATED_SCHEMA, MVS_HYBRID_SCHEMA} or sidecar.get("pose_solution_hash") != pose_hash:
        raise ValueError("artifact does not bind the requested pose solution")
    if sidecar.get("schema") in {CALIBRATED_SCHEMA, MVS_HYBRID_SCHEMA} and sidecar.get("decision") != "accepted":
        raise ValueError("calibrated artifact was not accepted by its own evidence contract")
    trajectory = pose.get("global_trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("pose solution has no global trajectory")
    cameras = {int(frame["frame_index"]): frame["world_to_camera"] for frame in pose["frames"] if frame.get("registered")}
    observed: dict[int, list[tuple[list[float], float, float]]] = defaultdict(list)
    # COLMAP landmarks live in the preserved 1620×1080 source crop. Their
    # raster observations are mapped into the immutable 504×336 DA3 grid.
    source_width, source_height = 1620.0, 1080.0
    for track in trajectory.get("tracks", []):
        point = track.get("position")
        if not isinstance(point, list) or len(point) != 3:
            continue
        for observation in track.get("observations", []):
            frame_index = int(observation.get("frame_index", -1))
            xy = observation.get("image_xy")
            if frame_index not in cameras or not isinstance(xy, list) or len(xy) != 2:
                continue
            observed[frame_index].append((point, float(xy[0]) * 504.0 / source_width, float(xy[1]) * 336.0 / source_height))

    ratios: list[float] = []
    per_frame: dict[int, list[float]] = defaultdict(list)
    duplicates: dict[int, tuple[Any, Any]] = {}
    seam_errors: list[float] = []
    seam_by_frame: dict[int, list[float]] = defaultdict(list)
    for batch in sidecar.get("batches", []):
        file_name = batch.get("file")
        if not isinstance(file_name, str):
            raise ValueError("artifact batch has no file")
        with np.load(artifact / file_name) as arrays:
            depth, confidence, indices = arrays["depth"], arrays["conf"], arrays["frame_indices"]
            for offset, frame_index_raw in enumerate(indices):
                frame_index = int(frame_index_raw)
                frame_depth, frame_confidence = depth[offset], confidence[offset]
                for point, x, y in observed.get(frame_index, []):
                    expected = camera_depth(cameras[frame_index], point)
                    measured = sample_bilinear(frame_depth, x, y)
                    confidence_sample = sample_bilinear(frame_confidence, x, y)
                    if expected > 0 and measured is not None and confidence_sample is not None and confidence_sample > 0:
                        ratio = measured / expected
                        if math.isfinite(ratio) and ratio > 0:
                            ratios.append(ratio)
                            per_frame[frame_index].append(ratio)
                previous = duplicates.get(frame_index)
                if previous is None:
                    duplicates[frame_index] = (frame_depth.copy(), frame_confidence.copy())
                else:
                    old_depth, old_confidence = previous
                    valid = np.isfinite(old_depth) & np.isfinite(frame_depth) & (old_depth > 0) & (frame_depth > 0) & np.isfinite(old_confidence) & np.isfinite(frame_confidence) & (old_confidence > 0) & (frame_confidence > 0)
                    if valid.any():
                        relative = np.abs(old_depth[valid] - frame_depth[valid]) / np.maximum(old_depth[valid], frame_depth[valid])
                        values = [float(value) for value in relative if math.isfinite(float(value))]
                        seam_errors.extend(values)
                        seam_by_frame[frame_index].extend(values)

    scale = percentile(ratios, 0.5)
    log_errors = [abs(math.log(ratio / scale)) for ratio in ratios] if scale else []
    frame_rows = []
    for frame_index, frame_ratios in sorted(per_frame.items()):
        normalized = [abs(math.log(ratio / scale)) for ratio in frame_ratios] if scale else []
        frame_rows.append({
            "frame_index": frame_index,
            "track_samples": len(frame_ratios),
            "median_ratio": percentile(frame_ratios, 0.5),
            "median_abs_log_error_after_global_scale": percentile(normalized, 0.5),
            "p95_abs_log_error_after_global_scale": percentile(normalized, 0.95),
        })
    return {
        "schema": SCHEMA,
        "pose_solution_hash": pose_hash,
        "artifact": str(artifact),
        "registered_frames": len(cameras),
        "colmap_track_depth": {
            "samples": len(ratios),
            "global_depth_ratio_median": scale,
            "median_abs_log_error_after_global_scale": percentile(log_errors, 0.5),
            "p95_abs_log_error_after_global_scale": percentile(log_errors, 0.95),
            "frames": frame_rows,
        },
        "cross_batch_depth_continuity": {
            "overlapped_frames": len(seam_by_frame),
            "samples": len(seam_errors),
            "median_relative_error": percentile(seam_errors, 0.5),
            "p95_relative_error": percentile(seam_errors, 0.95),
            "per_frame_p95_relative_error": {
                str(index): percentile(values, 0.95) for index, values in sorted(seam_by_frame.items())
            },
        },
        "interpretation": "Sparse COLMAP tracks test depth agreement after one global relative scale; overlap error tests DA3 batch-boundary continuity. Neither proves metric scale or room-planarity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pose-solution", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.scene, args.artifact, args.pose_solution)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "track_samples": report["colmap_track_depth"]["samples"],
        "track_p95_log_error": report["colmap_track_depth"]["p95_abs_log_error_after_global_scale"],
        "overlap_frames": report["cross_batch_depth_continuity"]["overlapped_frames"],
        "overlap_p95_relative_error": report["cross_batch_depth_continuity"]["p95_relative_error"],
    }))


if __name__ == "__main__":
    main()
