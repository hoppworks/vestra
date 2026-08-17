#!/usr/bin/env python3
"""Render a COLMAP fused PLY from calibrated source cameras.

This is deliberately an independent *inspection* tool.  It never feeds
geometry back into Vestra; it renders the dense MVS output through the
immutable, globally bundle-adjusted COLMAP W2C cameras and records coverage.
That makes an attractive free-orbit point cloud insufficient evidence: a
candidate must also look coherent from the cameras that created it.

Only the standard library is used so this can run beside the pinned COLMAP
container on the Workhorse.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float
    z: float
    red: int
    green: int
    blue: int


@dataclass(frozen=True)
class Camera:
    frame_index: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    w2c: tuple[float, ...]


PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def read_header(handle: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    first = handle.readline().decode("ascii").strip()
    if first != "ply":
        raise ValueError("not a PLY file")
    encoding = ""
    count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    while True:
        line = handle.readline().decode("ascii").strip()
        if line == "end_header":
            break
        fields = line.split()
        if not fields:
            continue
        if fields[:1] == ["format"]:
            encoding = fields[1]
        elif fields[:2] == ["element", "vertex"]:
            count = int(fields[2])
            in_vertex = True
        elif fields[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and fields[:1] == ["property"]:
            if fields[1] == "list":
                raise ValueError("list-valued vertex properties are unsupported")
            properties.append((fields[2], fields[1]))
    if encoding not in {"binary_little_endian", "ascii"} or count is None:
        raise ValueError("expected binary_little_endian or ASCII PLY vertices")
    required = {"x", "y", "z", "red", "green", "blue"}
    present = {name for name, _ in properties}
    if not required <= present:
        raise ValueError(f"PLY lacks required properties: {sorted(required - present)}")
    return encoding, count, properties


def read_vertices(path: Path, maximum: int | None) -> list[Vertex]:
    with path.open("rb") as handle:
        encoding, count, properties = read_header(handle)
        stride = struct.calcsize("<" + "".join(PLY_TYPES[kind] for _, kind in properties))
        take_every = max(1, math.ceil(count / maximum)) if maximum else 1
        values = {name: index for index, (name, _) in enumerate(properties)}
        vertices: list[Vertex] = []
        if encoding == "binary_little_endian":
            unpack = struct.Struct("<" + "".join(PLY_TYPES[kind] for _, kind in properties)).unpack
            for index in range(count):
                raw = handle.read(stride)
                if len(raw) != stride:
                    raise ValueError("truncated PLY vertex payload")
                if index % take_every:
                    continue
                row = unpack(raw)
                vertices.append(Vertex(
                    float(row[values["x"]]), float(row[values["y"]]), float(row[values["z"]]),
                    int(row[values["red"]]), int(row[values["green"]]), int(row[values["blue"]]),
                ))
        else:
            for index in range(count):
                row = handle.readline().decode("ascii").split()
                if len(row) != len(properties):
                    raise ValueError("truncated ASCII PLY vertex payload")
                if index % take_every:
                    continue
                vertices.append(Vertex(
                    float(row[values["x"]]), float(row[values["y"]]), float(row[values["z"]]),
                    int(float(row[values["red"]])), int(float(row[values["green"]])), int(float(row[values["blue"]])),
                ))
    return vertices


def intrinsics(model: dict) -> tuple[float, float, float, float]:
    kind = model["model"]
    p = model["parameters"]
    if kind in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    if kind in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}:
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    raise ValueError(f"unsupported COLMAP camera model {kind!r}")


def load_cameras(path: Path, wanted: set[int]) -> list[Camera]:
    solution = json.loads(path.read_text())
    evidence = solution.get("global_trajectory")
    if not evidence:
        raise ValueError("pose solution has no calibrated global trajectory")
    models = {int(m["camera_id"]): m for m in evidence["camera_models"]}
    frame_camera = {int(k): int(v) for k, v in evidence["frame_camera_ids"].items()}
    cameras: list[Camera] = []
    for frame in solution["frames"]:
        frame_index = int(frame["frame_index"])
        if frame_index not in wanted or not frame["registered"]:
            continue
        model = models[frame_camera[frame_index]]
        fx, fy, cx, cy = intrinsics(model)
        cameras.append(Camera(
            frame_index, int(model["width"]), int(model["height"]), fx, fy, cx, cy,
            tuple(float(value) for value in frame["world_to_camera"]),
        ))
    missing = wanted - {camera.frame_index for camera in cameras}
    if missing:
        raise ValueError(f"requested frames are not registered: {sorted(missing)}")
    return cameras


def render(vertices: Iterable[Vertex], camera: Camera, width: int, height: int, output: Path) -> tuple[dict, array]:
    depth = array("f", [math.inf]) * (width * height)
    rgb = bytearray(width * height * 3)
    visible = 0
    finite = 0
    sx, sy = width / camera.width, height / camera.height
    fx, fy, cx, cy = camera.fx * sx, camera.fy * sy, camera.cx * sx, camera.cy * sy
    r = camera.w2c
    for point in vertices:
        if not (math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)):
            continue
        finite += 1
        x = r[0] * point.x + r[1] * point.y + r[2] * point.z + r[3]
        y = r[4] * point.x + r[5] * point.y + r[6] * point.z + r[7]
        z = r[8] * point.x + r[9] * point.y + r[10] * point.z + r[11]
        if not math.isfinite(z) or z <= 0:
            continue
        u, v = round(fx * x / z + cx), round(fy * y / z + cy)
        if not (0 <= u < width and 0 <= v < height):
            continue
        index = v * width + u
        if z >= depth[index]:
            continue
        if not math.isfinite(depth[index]):
            visible += 1
        depth[index] = z
        rgb[index * 3:index * 3 + 3] = bytes((point.red, point.green, point.blue))
    output.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)
    return {
        "frame_index": camera.frame_index,
        "width": width,
        "height": height,
        "visible_pixels": visible,
        "coverage": visible / (width * height),
        "finite_vertices_seen": finite,
        "output": output.name,
    }, depth


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def sampled_depth(depth: array, width: int, height: int, x: float, y: float) -> float | None:
    """Return the nearest visible MVS surface around an exact source pixel.

    Sparse COLMAP tracks and independently rasterized MVS vertices need not
    quantize to the same output pixel.  The 3x3 depth-tested search stays local
    while avoiding a false 'no coverage' result from rounding alone.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    centre_x, centre_y = round(x), round(y)
    values = [
        depth[row * width + column]
        for row in range(max(0, centre_y - 1), min(height, centre_y + 2))
        for column in range(max(0, centre_x - 1), min(width, centre_x + 2))
        if math.isfinite(depth[row * width + column])
    ]
    return min(values) if values else None


def track_depth_report(solution: dict, cameras: Iterable[Camera], depth_by_frame: dict[int, array], width: int, height: int) -> dict:
    """Compare dense MVS Z against independent sparse COLMAP landmark Z."""
    camera_by_frame = {camera.frame_index: camera for camera in cameras}
    errors: list[float] = []
    observations = 0
    covered = 0
    for track in solution.get("global_trajectory", {}).get("tracks", []):
        if float(track.get("reprojection_error_px", math.inf)) > 2.5:
            continue
        point = track.get("position")
        if not isinstance(point, list) or len(point) != 3:
            continue
        for observation in track.get("observations", []):
            frame = int(observation.get("frame_index", -1))
            camera = camera_by_frame.get(frame)
            xy = observation.get("image_xy")
            if camera is None or not isinstance(xy, list) or len(xy) != 2:
                continue
            observations += 1
            scale_x, scale_y = width / camera.width, height / camera.height
            x = (float(xy[0]) + 0.5) * scale_x - 0.5
            y = (float(xy[1]) + 0.5) * scale_y - 0.5
            measured = sampled_depth(depth_by_frame[frame], width, height, x, y)
            r = camera.w2c
            expected = r[8] * float(point[0]) + r[9] * float(point[1]) + r[10] * float(point[2]) + r[11]
            if measured is None or not math.isfinite(expected) or expected <= 0:
                continue
            covered += 1
            errors.append(abs(math.log(measured / expected)))
    return {
        "schema": "vestra.colmap-mvs-track-depth-report/v1",
        "maximum_sparse_track_reprojection_error_px": 2.5,
        "observations": observations,
        "covered_observations": covered,
        "coverage": covered / observations if observations else 0.0,
        "median_abs_log_depth_error": percentile(errors, 50),
        "p95_abs_log_depth_error": percentile(errors, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--pose-solution", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--maximum-points", type=int, default=None,
                        help="deterministic uniform stride before rendering")
    parser.add_argument("--track-report", action="store_true",
                        help="compare depth-tested MVS Z with sparse COLMAP tracks in --frames")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.maximum_points == 0:
        parser.error("dimensions and maximum-points must be positive")
    vertices = read_vertices(args.ply, args.maximum_points)
    if not vertices:
        raise SystemExit("no vertices selected from PLY")
    args.output.mkdir(parents=True, exist_ok=True)
    solution = json.loads(args.pose_solution.read_text())
    cameras = load_cameras(args.pose_solution, set(args.frames))
    rendered = [render(vertices, camera, args.width, args.height,
                       args.output / f"mvs-frame-{camera.frame_index:06d}.ppm")
                for camera in cameras]
    reports = [report for report, _ in rendered]
    summary = {
        "schema": "vestra.colmap-mvs-camera-inspection/v1",
        "ply": str(args.ply),
        "vertices_rendered": len(vertices),
        "frames": reports,
    }
    if args.track_report:
        summary["track_depth"] = track_depth_report(
            solution, cameras,
            {camera.frame_index: depth for camera, (_, depth) in zip(cameras, rendered)},
            args.width, args.height,
        )
    (args.output / "inspection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
