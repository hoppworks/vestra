#!/usr/bin/env python3
"""Run official DA3 with Vestra's immutable COLMAP camera authority.

This is an evidence-producing sidecar, not an alternate Vestra renderer. It
validates the raster/pose binding before GPU work, runs bounded overlapping
multi-view batches, and emits both per-batch depth evidence and one binary PLY
in the *supplied* COLMAP world. The Rust importer owns publication, TSDF and
browser products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "vestra.da3-pose-conditioned/v1"
PLY_SCHEMA = "vestra.da3-pose-conditioned-ply/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


@dataclass(frozen=True)
class Frame:
    index: int
    file_name: str
    sha256: str
    w2c: tuple[float, ...]
    camera_id: int
    intrinsics: tuple[float, ...]


def read_inputs(scene: Path, pose_hash: str) -> tuple[str, list[Frame]]:
    manifest = json.loads((scene / "manifest.json").read_text(encoding="utf-8"))
    raster_hash = manifest.get("raster_manifest_hash")
    if not isinstance(raster_hash, str) or len(raster_hash) != 64:
        fail("scene has no immutable raster manifest")
    raster = json.loads((scene / "chunks" / f"raster-{raster_hash}.json").read_text(encoding="utf-8"))
    if raster.get("schema") != "vestra.raster/v1":
        fail("unsupported raster manifest schema")
    pose = json.loads((scene / "chunks" / f"pose-{pose_hash}.json").read_text(encoding="utf-8"))
    if pose.get("schema") != "vestra.pose-solution/v1":
        fail("unsupported pose solution schema")
    if pose.get("raster_fingerprint") != raster.get("raster_fingerprint"):
        fail("pose solution does not bind this scene's raster fingerprint")
    evidence = pose.get("global_trajectory")
    if not isinstance(evidence, dict):
        fail("pose solution has no global calibrated trajectory")
    camera_models = {int(camera["camera_id"]): camera for camera in evidence.get("camera_models", [])}
    frame_camera_ids = {int(index): int(camera) for index, camera in evidence.get("frame_camera_ids", {}).items()}
    raster_frames = {int(frame["frame_index"]): frame for frame in raster.get("frames", [])}
    decoded = scene / "decoded"
    frames: list[Frame] = []
    for pose_frame in pose.get("frames", []):
        if not pose_frame.get("registered"):
            continue
        index = int(pose_frame["frame_index"])
        raster_frame = raster_frames.get(index)
        camera = camera_models.get(frame_camera_ids.get(index, -1))
        if raster_frame is None or camera is None:
            fail(f"registered pose frame {index} has no raster or camera model")
        file_name = raster_frame.get("file_name")
        expected_sha = raster_frame.get("sha256")
        if not isinstance(file_name, str) or Path(file_name).name != file_name or not isinstance(expected_sha, str):
            fail(f"invalid raster identity for frame {index}")
        image = decoded / file_name
        if not image.is_file():
            fail(f"decoded raster is missing before GPU inference: {file_name}")
        if sha256_file(image) != expected_sha:
            fail(f"decoded raster hash mismatch before GPU inference: {file_name}")
        params = camera.get("parameters", [])
        if camera.get("model") not in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE"} or len(params) < 3:
            fail(f"unsupported COLMAP camera model for frame {index}: {camera.get('model')!r}")
        if camera.get("model") == "PINHOLE":
            if len(params) < 4:
                fail(f"PINHOLE camera {camera['camera_id']} has too few parameters")
            fx, fy, cx, cy = map(float, params[:4])
        else:
            focal, cx, cy = map(float, params[:3])
            fx = fy = focal
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            fail(f"invalid COLMAP intrinsics for frame {index}")
        w2c = tuple(map(float, pose_frame["world_to_camera"]))
        if len(w2c) != 12 or not all(math.isfinite(value) for value in w2c):
            fail(f"invalid W2C for frame {index}")
        frames.append(Frame(index, file_name, expected_sha, w2c, int(camera["camera_id"]), (fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)))
    frames.sort(key=lambda frame: frame.index)
    if not frames:
        fail("pose solution contains no registered frames")
    if len({frame.index for frame in frames}) != len(frames):
        fail("pose solution contains duplicate registered frame indices")
    return str(raster["raster_fingerprint"]), frames


def batches(frames: list[Frame], batch_size: int, overlap: int) -> list[list[Frame]]:
    if len(frames) < 3:
        fail("pose-conditioned external-scale inference requires at least three registered frames")
    if batch_size < 3 or overlap < 0 or overlap >= batch_size:
        fail("batch size must be >=3 and overlap must be in [0, batch_size)")
    stride = batch_size - overlap
    result: list[list[Frame]] = []
    start = 0
    while start < len(frames):
        batch = frames[start : start + batch_size]
        if len(batch) < 3:
            # The official external-scale alignment needs a non-degenerate
            # camera set. Reuse the terminal three frames rather than running
            # a 1- or 2-view tail with silently different semantics.
            batch = frames[-max(3, min(batch_size, len(frames))) :]
        if not result or [frame.index for frame in result[-1]] != [frame.index for frame in batch]:
            result.append(batch)
        if start + batch_size >= len(frames):
            break
        start += stride
    return result


def w2c44(frame: Frame, np: Any) -> Any:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = np.asarray(frame.w2c, dtype=np.float64).reshape(3, 4)
    return matrix


def normal_and_world_points(depth: Any, rgb: Any, conf: Any, intrinsics: Any, extrinsics: Any, stride: int, confidence_threshold: float, np: Any) -> tuple[Any, Any, Any]:
    """Returns valid XYZ, normals, and RGB in the supplied W2C world."""
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth[yy, xx]
    fx, fy, cx, cy = float(intrinsics[0, 0]), float(intrinsics[1, 1]), float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (xx.astype(np.float32) - cx) * sampled_depth / fx
    y = (yy.astype(np.float32) - cy) * sampled_depth / fy
    local = np.stack((x, y, sampled_depth), axis=-1)
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    world = (local.reshape(-1, 3) - translation) @ rotation
    # Derivatives in the camera grid retain a stable normal orientation when
    # transformed by the authoritative COLMAP rotation.
    dx = np.zeros_like(local); dy = np.zeros_like(local)
    dx[:, :-1] = local[:, 1:] - local[:, :-1]; dx[:, -1] = dx[:, -2]
    dy[:-1, :] = local[1:, :] - local[:-1, :]; dy[-1, :] = dy[-2, :]
    normals = np.cross(dx, dy).reshape(-1, 3) @ rotation
    length = np.linalg.norm(normals, axis=1)
    sampled_conf = conf[yy, xx].reshape(-1)
    valid = np.isfinite(world).all(axis=1) & np.isfinite(normals).all(axis=1) & np.isfinite(sampled_depth.reshape(-1)) & (sampled_depth.reshape(-1) > 0) & np.isfinite(sampled_conf) & (sampled_conf >= confidence_threshold) & (length > 1e-8)
    normals[valid] /= length[valid, None]
    sampled_rgb = rgb[yy, xx].reshape(-1, 3)
    return world[valid], normals[valid], sampled_rgb[valid]


def batch_records(path: Path, np: Any) -> Iterable[tuple[int, Any, Any, Any, Any, Any]]:
    """Yields one frame at a time without retaining all DA3 batches in RAM."""
    with np.load(path) as data:
        required = ("depth", "conf", "rgb", "intrinsics", "extrinsics", "frame_indices")
        missing = [key for key in required if key not in data]
        if missing:
            fail(f"sidecar batch is missing arrays: {', '.join(missing)}")
        depth, rgb, conf = data["depth"], data["rgb"], data["conf"]
        intrinsics, extrinsics, indices = data["intrinsics"], data["extrinsics"], data["frame_indices"]
        if depth.ndim != 3 or conf.shape != depth.shape or rgb.shape != (*depth.shape, 3):
            fail(f"invalid depth/conf/rgb dimensions in {path.name}")
        if intrinsics.shape != (depth.shape[0], 3, 3) or extrinsics.shape != (depth.shape[0], 4, 4):
            fail(f"invalid camera dimensions in {path.name}")
        if indices.shape != (depth.shape[0],):
            fail(f"invalid frame indices in {path.name}")
        for index in range(depth.shape[0]):
            yield int(indices[index]), depth[index], rgb[index], conf[index], intrinsics[index], extrinsics[index]


def write_ply(path: Path, batch_paths: Iterable[Path], confidence_percentile: float, pixel_stride: int, np: Any) -> int:
    paths = list(batch_paths)
    confidence_parts: list[Any] = []
    for batch_path in paths:
        with np.load(batch_path) as data:
            if "conf" not in data:
                fail(f"sidecar batch is missing confidence: {batch_path.name}")
            values = data["conf"].reshape(-1)
            confidence_parts.append(values[np.isfinite(values)])
    finite_confidence = np.concatenate(confidence_parts) if confidence_parts else np.empty(0, dtype=np.float32)
    threshold = float(np.percentile(finite_confidence, confidence_percentile)) if finite_confidence.size else 0.0
    del confidence_parts, finite_confidence
    emitted_frame_indices: set[int] = set()
    point_count = 0
    for batch_path in paths:
        for frame_index, depth, rgb, conf, intrinsics, extrinsics in batch_records(batch_path, np):
            if frame_index in emitted_frame_indices:
                continue
            emitted_frame_indices.add(frame_index)
            point_count += len(
                normal_and_world_points(depth, rgb, conf, intrinsics, extrinsics, pixel_stride, threshold, np)[0]
            )
    with path.open("wb") as output:
        output.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {point_count}\n" "property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        emitted = 0
        emitted_frame_indices.clear()
        for batch_path in paths:
            for frame_index, depth, rgb, conf, intrinsics, extrinsics in batch_records(batch_path, np):
                if frame_index in emitted_frame_indices:
                    continue
                emitted_frame_indices.add(frame_index)
                xyz, normals, colors = normal_and_world_points(depth, rgb, conf, intrinsics, extrinsics, pixel_stride, threshold, np)
                for position, normal, color in zip(xyz, normals, colors):
                    output.write(struct.pack("<ffffffBBB", *map(float, position), *map(float, normal), *map(int, color.clip(0,255))))
                    emitted += 1
    if emitted != point_count:
        fail(f"PLY count mismatch: expected {point_count}, wrote {emitted}")
    return emitted


def run(args: argparse.Namespace) -> None:
    raster_fingerprint, frames = read_inputs(args.scene, args.pose_solution)
    layout = batches(frames, args.batch_size, args.overlap)
    if args.pixel_stride <= 0:
        fail("pixel stride must be positive")
    if not 0.0 <= args.confidence_percentile <= 100.0:
        fail("confidence percentile must be in [0, 100]")
    if args.output.exists():
        fail(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "raster_fingerprint": raster_fingerprint, "pose_solution_hash": args.pose_solution,
        "model_ref": args.model_ref, "model_revision": args.model_revision, "batch_size": args.batch_size,
        "overlap": args.overlap, "pixel_stride": args.pixel_stride, "process_resolution": [504, 336],
        "align_to_input_ext_scale": True, "frames": [frame.index for frame in frames],
        "batches": [{"frames": [frame.index for frame in batch]} for batch in layout],
    }
    if args.validate_only:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    import numpy as np
    from depth_anything_3.api import DepthAnything3
    import torch

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_ref, revision=args.model_revision).to(device=device).eval()
    manifest["torch_version"] = torch.__version__
    manifest["cuda_version"] = torch.version.cuda
    manifest["resolved_model_revision"] = getattr(model, "_commit_hash", None) or args.model_revision
    batch_paths: list[Path] = []
    for batch_index, batch in enumerate(layout):
        paths = [str(args.scene / "decoded" / frame.file_name) for frame in batch]
        ex = np.stack([w2c44(frame, np) for frame in batch])
        intrinsics = np.stack([np.asarray(frame.intrinsics, dtype=np.float64).reshape(3, 3) for frame in batch])
        prediction = model.inference(paths, extrinsics=ex, intrinsics=intrinsics, align_to_input_ext_scale=True, process_res=504, process_res_method="upper_bound_resize", export_dir=None)
        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = np.asarray(prediction.conf if prediction.conf is not None else np.ones_like(depth), dtype=np.float32)
        processed = np.asarray(prediction.processed_images, dtype=np.uint8)
        if depth.ndim != 3 or depth.shape[0] != len(batch) or depth.shape[1:] != (336, 504):
            fail(f"DA3 returned unsupported depth shape: {depth.shape}")
        if conf.shape != depth.shape or processed.shape != (*depth.shape, 3):
            fail(f"DA3 returned inconsistent confidence/image shapes: {conf.shape}, {processed.shape}")
        output_intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
        output_extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
        if output_intrinsics.shape != (len(batch), 3, 3) or output_extrinsics.shape != (len(batch), 4, 4):
            fail("DA3 did not return one calibrated camera per input frame")
        output = args.output / f"batch-{batch_index:04d}.npz"
        np.savez_compressed(output, depth=depth, conf=conf, rgb=processed, intrinsics=output_intrinsics, extrinsics=output_extrinsics, frame_indices=np.asarray([frame.index for frame in batch], dtype=np.int64))
        batch_paths.append(output)
        manifest["batches"][batch_index].update({"file": output.name, "sha256": sha256_file(output), "depth_shape": list(depth.shape), "confidence_present": prediction.conf is not None, "cameras": [{"frame_index": frame.index, "world_to_camera": list(frame.w2c), "intrinsics": list(frame.intrinsics)} for frame in batch]})
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ply = args.output / "world.ply"
    points = write_ply(ply, batch_paths, args.confidence_percentile, args.pixel_stride, np)
    manifest.update({"ply": {"schema": PLY_SCHEMA, "file": ply.name, "sha256": sha256_file(ply), "points": points, "confidence_percentile": args.confidence_percentile}})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose-solution", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-ref", default="depth-anything/DA3-BASE")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--confidence-percentile", type=float, default=40.0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
