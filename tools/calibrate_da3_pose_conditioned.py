#!/usr/bin/env python3
"""Calibrate DA3 pose-conditioned depth with held-out COLMAP landmarks.

This is a post-inference evidence transform. It never changes COLMAP cameras
or estimates a trajectory: it applies one robust positive depth multiplier per
source-frame prediction, fitted on sparse train landmarks and reported against
the deterministic held-out fifth of those landmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_da3_pose_conditioned import SCHEMA, batch_records, sha256_file, write_depth_frames, write_ply


CALIBRATION_SCHEMA = "vestra.da3-pose-conditioned-calibration/v2"
PIXEL_MAPPING = "pixel-center-resize/v1"
TRACK_SPLIT = "sha256-track-id-fold/v1"


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) * 0.5


def bilinear(depth: Any, x: float, y: float) -> float | None:
    height, width = depth.shape
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    tx, ty = x - x0, y - y0
    value = (
        float(depth[y0, x0]) * (1 - tx) * (1 - ty)
        + float(depth[y0, x1]) * tx * (1 - ty)
        + float(depth[y1, x0]) * (1 - tx) * ty
        + float(depth[y1, x1]) * tx * ty
    )
    return value if math.isfinite(value) and value > 0 else None


def camera_depth(w2c: list[float], point: list[float]) -> float:
    return sum(w2c[8 + axis] * point[axis] for axis in range(3)) + w2c[11]


def held_out_track(pose_hash: str, point_id: int) -> bool:
    """Assign one 20% hold-out bucket per stable COLMAP landmark ID."""
    payload = f"{TRACK_SPLIT}:{pose_hash}:{point_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5 == 0


def scale_evidence(scene: Path, pose_hash: str) -> tuple[dict[int, list[tuple[int, list[float], float, float, int, int]]], dict[int, list[float]]]:
    pose = json.loads((scene / "chunks" / f"pose-{pose_hash}.json").read_text())
    trajectory = pose.get("global_trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("pose solution has no global trajectory")
    cameras = {int(frame["frame_index"]): list(map(float, frame["world_to_camera"])) for frame in pose["frames"] if frame.get("registered")}
    models = {int(model["camera_id"]): model for model in trajectory.get("camera_models", [])}
    ids = {int(index): int(camera) for index, camera in trajectory.get("frame_camera_ids", {}).items()}
    samples: dict[int, list[tuple[int, list[float], float, float, int, int]]] = defaultdict(list)
    for track in trajectory.get("tracks", []):
        point = track.get("position")
        if not isinstance(point, list) or len(point) != 3 or float(track.get("reprojection_error_px", math.inf)) > 2.5:
            continue
        point_id = int(track.get("point_id", -1))
        for observation in track.get("observations", []):
            frame = int(observation.get("frame_index", -1))
            xy = observation.get("image_xy")
            model = models.get(ids.get(frame, -1))
            if frame not in cameras or model is None or not isinstance(xy, list) or len(xy) != 2:
                continue
            expected = camera_depth(cameras[frame], point)
            if expected <= 0 or not math.isfinite(expected):
                continue
            samples[frame].append((
                point_id,
                [float(value) for value in point],
                float(xy[0]),
                float(xy[1]),
                int(model["width"]),
                int(model["height"]),
            ))
    return samples, cameras


def calibrate_depth(
    depth: Any,
    samples: list[tuple[int, list[float], float, float, int, int]],
    w2c: list[float],
    pose_hash: str,
    minimum_samples: int,
    minimum_held_out_samples: int,
    np: Any,
) -> dict[str, Any]:
    train: list[float] = []
    held_out: list[float] = []
    height, width = depth.shape
    for point_id, point, x, y, source_width, source_height in samples:
        # This matches the immutable raster's half-pixel resize contract.
        measured = bilinear(depth, (x + 0.5) * width / source_width - 0.5, (y + 0.5) * height / source_height - 0.5)
        expected = camera_depth(w2c, point)
        if measured is None or expected <= 0 or not math.isfinite(expected):
            continue
        log_ratio = math.log(expected / measured)
        (held_out if held_out_track(pose_hash, point_id) else train).append(log_ratio)
    log_scale = median(train)
    held_out_error = median([abs(value - log_scale) for value in held_out]) if log_scale is not None else None
    train_error = median([abs(value - log_scale) for value in train]) if log_scale is not None else None
    accepted = log_scale is not None and len(train) >= minimum_samples and len(held_out) >= minimum_held_out_samples and held_out_error is not None
    return {
        "scale": math.exp(log_scale) if log_scale is not None else 1.0,
        "train_samples": len(train),
        "train_median_log_error": train_error,
        "held_out_samples": len(held_out),
        "held_out_median_log_error": held_out_error,
        "accepted": accepted,
    }


def run(args: argparse.Namespace) -> None:
    import numpy as np

    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    source = json.loads((args.artifact / "manifest.json").read_text())
    if source.get("schema") != SCHEMA or source.get("pose_solution_hash") != args.pose_solution:
        raise ValueError("source artifact does not bind this pose solution")
    evidence, cameras = scale_evidence(args.scene, args.pose_solution)
    args.output.mkdir(parents=True)
    source_batches = [
        {"file": str(batch["file"]), "sha256": str(batch["sha256"])}
        for batch in source["batches"]
    ]
    manifest = dict(source)
    manifest["schema"] = CALIBRATION_SCHEMA
    manifest["source"] = {
        "raw_manifest_sha256": sha256_file(args.artifact / "manifest.json"),
        "raster_fingerprint": source["raster_fingerprint"],
        "pose_solution_hash": args.pose_solution,
        "batch_files": source_batches,
    }
    manifest["contract"] = {
        "minimum_training_samples": args.minimum_training_samples,
        "minimum_held_out_samples": args.minimum_held_out_samples,
        "pixel_mapping": PIXEL_MAPPING,
        "track_split": TRACK_SPLIT,
        "reprojection_error_px_max": 2.5,
        "maximum_held_out_median_log_error": args.maximum_held_out_median_log_error,
        "minimum_accepted_frame_fraction": 0.85,
    }
    # Preserve the raw registered-frame identity at the top level; the
    # calibration evidence has its own collection to avoid weakening import
    # binding semantics.
    manifest["frames"] = list(source["frames"])
    manifest["calibration_frames"] = []
    paths: list[Path] = []
    for batch_index, batch in enumerate(manifest["batches"]):
        source_file = args.artifact / str(batch["file"])
        with np.load(source_file) as arrays:
            depth = arrays["depth"].copy()
            conf = arrays["conf"].copy()
            rgb = arrays["rgb"].copy()
            intrinsics = arrays["intrinsics"].copy()
            extrinsics = arrays["extrinsics"].copy()
            frame_indices = arrays["frame_indices"].copy()
        for offset, raw_index in enumerate(frame_indices):
            frame_index = int(raw_index)
            result = calibrate_depth(
                depth[offset], evidence.get(frame_index, []), cameras[frame_index],
                args.pose_solution, args.minimum_training_samples,
                args.minimum_held_out_samples, np,
            )
            result.update({
                "frame_index": frame_index,
                "source_batch": source_file.name,
                "source_slot": offset,
                "calibrated_batch": f"batch-{batch_index:04d}.npz",
                "batch_index": batch_index,
            })
            if result["accepted"]:
                depth[offset] *= float(result["scale"])
            manifest["calibration_frames"].append(result)
        target = args.output / f"batch-{batch_index:04d}.npz"
        np.savez_compressed(target, depth=depth, conf=conf, rgb=rgb, intrinsics=intrinsics, extrinsics=extrinsics, frame_indices=frame_indices)
        batch["file"] = target.name
        batch["sha256"] = sha256_file(target)
        paths.append(target)
    rows = manifest["calibration_frames"]
    canonical = {}
    for row in rows:
        frame_index = row["frame_index"]
        current = canonical.get(frame_index)
        # Held-out data deliberately does not participate in this ordering.
        key = (
            row["train_median_log_error"] if row["train_median_log_error"] is not None else math.inf,
            -row["train_samples"], row["batch_index"], row["source_slot"],
        )
        if current is None or key < current[0]:
            canonical[frame_index] = (key, row)
    published = [
        frame_index
        for frame_index in source["frames"]
        if (candidate := canonical.get(frame_index)) is not None
        and (row := candidate[1]) is not None
        and row["accepted"]
        and row["held_out_median_log_error"] <= args.maximum_held_out_median_log_error
    ]
    if len(published) / max(len(source["frames"]), 1) < 0.85:
        raise ValueError("fewer than 85% of registered frames passed held-out depth calibration")
    manifest["published_frames"] = published
    manifest["decision"] = "accepted"
    manifest["summary"] = {
        "accepted_predictions": len([row for row in rows if row["accepted"] and row["held_out_median_log_error"] <= args.maximum_held_out_median_log_error]),
        "total_predictions": len(rows),
        "accepted_frames": len(published),
        "total_registered_frames": len(source["frames"]),
    }
    selected = [canonical[frame][1] for frame in published]
    selected_arrays = []
    for row in selected:
        with np.load(args.output / row["calibrated_batch"]) as arrays:
            slot = row["source_slot"]
            selected_arrays.append(tuple(arrays[name][slot] for name in ("depth", "conf", "rgb", "intrinsics", "extrinsics", "frame_indices")))
    selected_path = args.output / "selected.npz"
    np.savez_compressed(
        selected_path,
        depth=np.stack([item[0] for item in selected_arrays]),
        conf=np.stack([item[1] for item in selected_arrays]),
        rgb=np.stack([item[2] for item in selected_arrays]),
        intrinsics=np.stack([item[3] for item in selected_arrays]),
        extrinsics=np.stack([item[4] for item in selected_arrays]),
        frame_indices=np.asarray([item[5] for item in selected_arrays], dtype=np.int64),
    )
    manifest["batches"] = [{"file": selected_path.name, "sha256": sha256_file(selected_path), "depth_shape": [len(selected), 336, 504]}]
    manifest["depth_frames"] = write_depth_frames(args.output / "depth-frames", [selected_path], np)
    ply = args.output / "world.ply"
    points = write_ply(ply, [selected_path], args.confidence_percentile, args.pixel_stride, np)
    manifest["ply"] = {"schema": source["ply"]["schema"], "file": ply.name, "sha256": sha256_file(ply), "points": points, "confidence_percentile": args.confidence_percentile}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pose-solution", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-training-samples", type=int, default=12)
    parser.add_argument("--minimum-held-out-samples", type=int, default=6)
    parser.add_argument("--maximum-held-out-median-log-error", type=float, default=0.20)
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_training_samples <= 0 or args.minimum_held_out_samples <= 0 or args.maximum_held_out_median_log_error < 0:
        parser.error("calibration thresholds must be positive")
    run(args)


if __name__ == "__main__":
    main()
