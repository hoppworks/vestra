#!/usr/bin/env python3
"""Run pinned DROID-SLAM and export only native tracked poses for Vestra.

This is intentionally a sidecar, not a second reconstruction pipeline.  It
accepts Vestra's immutable raster manifest and writes the provider-neutral
``vestra.pose-solution/v1`` JSON consumed by ``vestra pose-import-json``.
Non-keyframe pose filling is deliberately not used: a derived world must fail
its coverage gate rather than silently interpolate cameras it did not track.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--droid-root", type=Path, required=True)
    parser.add_argument("--raster-manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--calibration", type=Path, help="fx fy cx cy [distortion...] text file")
    calibration.add_argument(
        "--colmap-cameras",
        type=Path,
        help="validated COLMAP cameras.txt; derives the DROID pinhole calibration",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-thresh", type=float, default=2.4)
    parser.add_argument("--keyframe-thresh", type=float, default=4.0)
    parser.add_argument("--frontend-window", type=int, default=25)
    parser.add_argument("--backend-thresh", type=float, default=22.0)
    parser.add_argument("--backend-radius", type=int, default=2)
    return parser.parse_args()


def native_git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def calibration_from_colmap(path: Path) -> tuple[float, float, float, float]:
    """Reads the first calibrated camera from COLMAP's text model.

    DROID consumes the undistorted pinhole component only.  Keeping this
    derivation here avoids a hand-entered focal length between the COLMAP and
    DROID experiments while retaining the exact source file in provenance.
    """
    rows = [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]
    if len(rows) != 1 or len(rows[0]) < 8:
        raise SystemExit("expected exactly one valid COLMAP camera in cameras.txt")
    _, model, _, _, *params = rows[0]
    values = [float(value) for value in params]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and len(values) >= 3:
        return values[0], values[0], values[1], values[2]
    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV"} and len(values) >= 4:
        return values[0], values[1], values[2], values[3]
    raise SystemExit(f"unsupported COLMAP camera model {model!r}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.raster_manifest.read_text())
    if manifest.get("schema") != "vestra.raster/v1":
        raise SystemExit("expected vestra.raster/v1 manifest")
    frames = manifest["frames"]
    image_paths = [args.images_dir / frame["file_name"] for frame in frames]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing exact raster(s): {missing[:3]}")

    # Import only after the explicit root was supplied; this keeps Vestra's
    # Python-free default installation intact.
    sys.path.insert(0, str(args.droid_root / "droid_slam"))
    import cv2  # noqa: PLC0415
    import lietorch  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from droid import Droid  # noqa: PLC0415

    if args.colmap_cameras:
        fx, fy, cx, cy = calibration_from_colmap(args.colmap_cameras)
        calibration_source = args.colmap_cameras
    else:
        calibration = np.loadtxt(args.calibration, delimiter=" ")
        if calibration.shape[0] < 4:
            raise SystemExit("calibration must contain fx fy cx cy")
        fx, fy, cx, cy = (float(value) for value in calibration[:4])
        calibration_source = args.calibration

    def stream():
        for ordinal, path in enumerate(image_paths):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"OpenCV could not decode {path}")
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]
            intrinsics = torch.tensor(
                [fx * w1 / w0, fy * image.shape[0] / h0, cx * w1 / w0, cy * image.shape[0] / h0]
            )
            yield ordinal, torch.as_tensor(image).permute(2, 0, 1)[None], intrinsics

    options = argparse.Namespace(
        weights=str(args.weights),
        buffer=max(512, len(frames) + 8),
        image_size=[240, 320],
        disable_vis=True,
        beta=0.3,
        filter_thresh=args.filter_thresh,
        warmup=8,
        keyframe_thresh=args.keyframe_thresh,
        frontend_thresh=16.0,
        frontend_window=args.frontend_window,
        frontend_radius=2,
        frontend_nms=1,
        backend_thresh=args.backend_thresh,
        backend_radius=args.backend_radius,
        backend_nms=3,
        stereo=False,
    )
    torch.multiprocessing.set_start_method("spawn", force=True)
    droid = None
    for timestamp, image, intrinsics in stream():
        if droid is None:
            options.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(options)
        droid.track(timestamp, image, intrinsics=intrinsics)
    if droid is None:
        raise SystemExit("manifest has no rasters")

    # This is DROID's terminating global optimization, minus its optional
    # PoseTrajectoryFiller interpolation.  `video.poses` is internal W2C;
    # convert it through lietorch to the external OpenCV W2C matrix contract.
    del droid.frontend
    torch.cuda.empty_cache()
    droid.backend(7)
    torch.cuda.empty_cache()
    droid.backend(12)
    count = droid.video.counter.value
    keyframe_indices = droid.video.tstamp[:count].cpu().numpy().astype(int)
    w2c = lietorch.SE3(droid.video.poses[:count].clone()).matrix().cpu().numpy()

    if len(set(keyframe_indices.tolist())) != len(keyframe_indices):
        raise RuntimeError("DROID returned duplicate native keyframe timestamps")
    output_frames = []
    for index, matrix in zip(keyframe_indices.tolist(), w2c, strict=True):
        if index < 0 or index >= len(frames):
            raise RuntimeError(f"DROID keyframe timestamp {index} is outside raster contract")
        output_frames.append(
            {
                "frame_index": index,
                "image_name": frames[index]["file_name"],
                "registered": True,
                "world_to_camera": matrix[:3, :4].astype(float).reshape(-1).tolist(),
            }
        )
    output_frames.sort(key=lambda frame: frame["frame_index"])
    settings = {
        "droid_revision": native_git_revision(args.droid_root),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration_source.read_bytes()).hexdigest(),
        "filter_thresh": args.filter_thresh,
        "keyframe_thresh": args.keyframe_thresh,
        "frontend_window": args.frontend_window,
        "backend_thresh": args.backend_thresh,
        "backend_radius": args.backend_radius,
        "native_keyframes_only": True,
    }
    result = {
        "schema": "vestra.pose-solution/v1",
        "provider": {
            "kind": "droid-slam",
            "version": settings["droid_revision"],
            "settings_fingerprint": sha256_json(settings),
        },
        "raster_fingerprint": manifest["raster_fingerprint"],
        "coordinate_convention": "OpenCV world; W2C row-major 3x4 f64",
        "frames": output_frames,
        "diagnostics": {
            "input_frames": len(frames),
            "registered_frames": len(output_frames),
            "duplicate_images": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "registered_frames": len(output_frames), "settings": settings}))


if __name__ == "__main__":
    main()
