#!/usr/bin/env python3
"""Build a reproducible global COLMAP pose candidate for one Vestra scene.

This is intentionally an experiment runner, not a publishing shortcut.  It
consumes the immutable raster manifest already recorded in a ``.vestra``
bundle, verifies every PPM byte before COLMAP sees it, adds both sequential and
vocabulary-tree retrieval matches, runs a final global bundle adjustment, and
writes a selected-only ``images.txt`` plus provenance for
``vestra-lab pose-import-colmap``. The complete optimized model is retained beside
it for dense MVS and audit/replay.

The Rust importer and global-fusion gate remain the authority for accepting or
rejecting the resulting pose solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Iterable


SCHEMA = "vestra.colmap-global-pose-run/v1"
MAX_BRIDGE_IMAGES = 100_000


@dataclass(frozen=True)
class Settings:
    colmap: str
    container_image: str | None
    threads: int
    camera_model: str
    sequential_overlap: int
    retrieval_images: int
    vocabulary_tree_sha256: str
    pose_image_width: int | None
    source_video_sha256: str | None
    bridge_fps: float | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {rendered}\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"exit={result.returncode}\n\n")
    if result.returncode:
        raise RuntimeError(f"COLMAP command failed ({result.returncode}); see {log}")


def colmap_command(args: argparse.Namespace, colmap_args: list[str]) -> list[str]:
    """Returns a direct or pinned-container COLMAP invocation.

    Container mode binds only the immutable scene, the vocabulary-tree parent,
    and the output parent at their original absolute paths. No network is
    required after the image is present locally.
    """
    if not args.container_image:
        return [args.colmap, *colmap_args]
    roots = {
        args.scene.resolve(),
        args.output.resolve().parent,
        args.vocabulary_tree.resolve().parent,
    }
    command = [args.container_engine, "run", "--rm", "--network", "none"]
    for root in sorted(roots):
        command.extend(["--volume", f"{root}:{root}:Z"])
    return [*command, args.container_image, "colmap", *colmap_args]


def read_raster_manifest(scene: Path) -> tuple[dict, Path]:
    scene_manifest = json.loads((scene / "manifest.json").read_text(encoding="utf-8"))
    raster_hash = scene_manifest.get("raster_manifest_hash")
    if not isinstance(raster_hash, str) or len(raster_hash) != 64:
        raise ValueError("scene has no immutable raster_manifest_hash")
    raster_path = scene / "chunks" / f"raster-{raster_hash}.json"
    raster = json.loads(raster_path.read_text(encoding="utf-8"))
    # This is the durable core contract written by `RasterManifest`; it binds
    # COLMAP to the exact decoded/cropped PPM evidence rather than an ad-hoc
    # video-frame list.
    if raster.get("schema") != "vestra.raster/v1":
        raise ValueError("unsupported raster manifest schema")
    return raster, raster_path


def verify_rasters(scene: Path, raster: dict) -> list[str]:
    decoded = scene / "decoded"
    frames = raster.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("raster manifest has no frames")
    names: set[str] = set()
    for expected_index, frame in enumerate(frames):
        if frame.get("frame_index") != expected_index:
            raise ValueError("raster frame indices must be contiguous and ordered")
        name = frame.get("file_name")
        digest = frame.get("sha256")
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError(f"unsafe or duplicate raster name: {name!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid raster hash for {name!r}")
        if sha256_file(decoded / name) != digest:
            raise ValueError(f"decoded raster hash mismatch: {name}")
        names.add(name)
    return sorted(names)


def stage_colmap_images(decoded: Path, names: list[str], output: Path) -> Path:
    """Creates COLMAP's *only* image root from immutable manifest members.

    The decoded cache may retain extra candidate frames for future keyframe
    selection. COLMAP recursively discovers images, so pointing it at that
    cache would silently turn a selected-frame experiment into a different
    reconstruction. Hard-linking keeps the evidence byte-identical and cheap;
    copying is a conservative cross-filesystem fallback.
    """
    image_root = output / "images"
    image_root.mkdir()
    for name in names:
        source = decoded / name
        destination = image_root / name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return image_root


def bridge_image_name(selected_name: str, timestamp_millis: int, ordinal: int) -> str:
    """Name a bridge image immediately after its preceding selected image.

    Selected raster names are part of the durable contract and therefore are
    never renamed.  A ``~bridge`` suffix sorts after ``frame-000001.ppm`` but
    before ``frame-000002.ppm`` for the normal Vestra frame naming scheme,
    while remaining an ordinary PPM filename for COLMAP.
    """
    path = Path(selected_name)
    suffix = path.suffix or ".ppm"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return f"{stem}~bridge-{timestamp_millis:012d}-{ordinal:06d}{suffix}"


def temporal_bridge_schedule(
    frames: list[dict], names: list[str], bridge_fps: float
) -> list[dict]:
    """Return exact integer-millisecond bridge timestamps between keyframes.

    The schedule is deliberately bounded to gaps *between* selected frames;
    it never adds a prefix or suffix of the source video.  Fraction arithmetic
    makes the result deterministic for decimal FPS values and avoids a
    floating-point accumulation drift.  Timestamps are rounded up to the
    nearest source millisecond, and duplicate/endpoint timestamps are omitted.
    """
    if len(frames) != len(names):
        raise ValueError("bridge schedule requires one name per raster frame")
    if not isinstance(bridge_fps, (int, float)) or not float(bridge_fps) > 0:
        raise ValueError("bridge-fps must be positive")
    if not float(bridge_fps) < float("inf"):
        raise ValueError("bridge-fps must be finite")
    if float(bridge_fps) > 1000:
        raise ValueError("bridge-fps cannot exceed 1000 for integer-millisecond timestamps")
    if len(frames) < 2:
        return []
    timestamps = [frame.get("timestamp_millis") for frame in frames]
    if any(not isinstance(value, int) or value < 0 for value in timestamps):
        raise ValueError("raster timestamps must be non-negative integer milliseconds")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("raster timestamps must be strictly increasing")
    interval = Fraction(1000, 1) / Fraction(str(float(bridge_fps)))
    result: list[dict] = []
    ordinal = 0
    last_timestamp = -1
    for left, right, selected_name in zip(timestamps, timestamps[1:], names):
        candidate = Fraction(left, 1) + interval
        while candidate < right:
            timestamp = ceil(candidate)
            # At very high rates several ideal samples can land in one source
            # millisecond. Keep the schedule strictly interior and unique.
            if timestamp <= left:
                timestamp = left + 1
            if timestamp >= right:
                break
            if timestamp <= last_timestamp:
                candidate += interval
                continue
            if len(result) >= MAX_BRIDGE_IMAGES:
                raise ValueError(f"bridge schedule exceeds {MAX_BRIDGE_IMAGES} images")
            result.append(
                {
                    "timestamp_millis": timestamp,
                    "file_name": bridge_image_name(selected_name, timestamp, ordinal),
                    "kind": "bridge",
                    "selected_before": selected_name,
                }
            )
            ordinal += 1
            last_timestamp = timestamp
            candidate += interval
    if [entry["timestamp_millis"] for entry in result] != sorted(
        entry["timestamp_millis"] for entry in result
    ):
        raise AssertionError("bridge schedule lost chronological ordering")
    return result


def _ordered_image_entries(frames: list[dict], names: list[str], bridges: list[dict]) -> list[dict]:
    if len(frames) != len(names):
        raise ValueError("image ordering requires one name per raster frame")
    entries = [
        {"timestamp_millis": frame["timestamp_millis"], "file_name": name, "kind": "selected"}
        for frame, name in zip(frames, names)
    ]
    entries.extend(bridges)
    ordered = sorted(entries, key=lambda entry: (entry["timestamp_millis"], entry["kind"] != "selected"))
    if [entry["file_name"] for entry in ordered] != sorted(entry["file_name"] for entry in ordered):
        raise ValueError(
            "selected raster filenames must sort chronologically for COLMAP sequential matching"
        )
    return ordered


def source_extraction_schedule(frames: list[dict]) -> tuple[float, int] | None:
    """Returns a fast uniform schedule, or ``None`` for exact per-frame decode.

    High-resolution pose evidence must describe the same source instants as the
    immutable DA3 rasters. A generic ``fps`` filter is exact only when the
    manifest timestamps are uniformly spaced. Vestra's quality selector is
    intentionally irregular, so those manifests use one accurate decode per
    recorded timestamp instead of approximating a new cadence.
    """
    if len(frames) < 2:
        raise ValueError("high-resolution pose extraction requires two or more raster frames")
    timestamps = [frame.get("timestamp_millis") for frame in frames]
    if any(not isinstance(value, int) or value < 0 for value in timestamps):
        raise ValueError("raster timestamps must be non-negative integer milliseconds")
    if timestamps[0] != 0:
        raise ValueError("high-resolution pose extraction requires a zero timestamp first frame")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("raster timestamps must be strictly increasing")
    interval = timestamps[1] - timestamps[0]
    prefix_count = 2
    while prefix_count < len(timestamps) and timestamps[prefix_count] - timestamps[prefix_count - 1] == interval:
        prefix_count += 1
    if prefix_count < len(timestamps) - 1 or (
        prefix_count < len(timestamps) and timestamps[prefix_count] <= timestamps[prefix_count - 1]
    ):
        return None
    return 1000.0 / interval, prefix_count


def stage_source_resolution_images(
    args: argparse.Namespace,
    raster: dict,
    names: list[str],
    output: Path,
    bridge_fps: float | None = None,
) -> Path:
    """Decodes the source video at its locked crop, retaining pose detail.

    This is deliberately only a pose-evidence upgrade. `verify_rasters` has
    already established which timestamps/names belong to the Vestra scene, and
    this function verifies the original source digest before extracting those
    same uniform instants into a private COLMAP image root.
    """
    assert args.source_video is not None and args.pose_image_width is not None
    video = args.source_video.resolve()
    expected_sha = raster.get("source_sha256")
    if not video.is_file() or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise FileNotFoundError("source video or its immutable raster digest is unavailable")
    actual_sha = sha256_file(video)
    if actual_sha != expected_sha:
        raise ValueError("source video SHA-256 does not match the raster manifest")
    crop = raster.get("crop")
    if not isinstance(crop, dict) or any(
        not isinstance(crop.get(field), int) or crop[field] < 0
        for field in ("x", "y", "width", "height")
    ):
        raise ValueError("raster manifest has invalid crop geometry")
    if crop["width"] <= 0 or crop["height"] <= 0:
        raise ValueError("raster crop dimensions must be positive")
    if args.pose_image_width <= 0 or args.pose_image_width > crop["width"]:
        raise ValueError("pose-image-width must be positive and no wider than the locked source crop")
    image_root = output / "images"
    image_root.mkdir()
    filter_parts = [
        f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}",
    ]
    if args.pose_image_width != crop["width"]:
        filter_parts.append(f"scale={args.pose_image_width}:-2:flags=lanczos")
    log = output / "colmap.log"
    bridges = temporal_bridge_schedule(raster["frames"], names, bridge_fps) if bridge_fps else []
    if bridges:
        # A bridge run always uses exact timestamp seeks for both selected and
        # bridge images. The selected names remain byte-for-byte contract
        # names; only the bridge files are synthetic and clearly marked.
        for entry in _ordered_image_entries(raster["frames"], names, bridges):
            timestamp = entry["timestamp_millis"] / 1000.0
            run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                    "-ss", f"{timestamp:.3f}", "-vf", ",".join(filter_parts),
                    "-frames:v", "1", str(image_root / entry["file_name"]),
                ],
                log,
            )
        return image_root

    schedule = source_extraction_schedule(raster["frames"])
    if schedule is None:
        for frame, name in zip(raster["frames"], names):
            timestamp = frame["timestamp_millis"] / 1000.0
            run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                    "-ss", f"{timestamp:.3f}", "-vf", ",".join(filter_parts),
                    "-frames:v", "1", str(image_root / name),
                ],
                log,
            )
        return image_root

    rate, regular_count = schedule
    temporary_pattern = image_root / "selected-%06d.ppm"
    run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", ",".join([f"fps={rate:.12f}", *filter_parts]),
            "-frames:v", str(regular_count), str(temporary_pattern),
        ],
        log,
    )
    if regular_count < len(names):
        final_timestamp = raster["frames"][-1]["timestamp_millis"] / 1000.0
        run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                "-ss", f"{final_timestamp:.3f}", "-vf", ",".join(filter_parts),
                "-frames:v", "1", str(image_root / f"selected-{len(names):06d}.ppm"),
            ],
            log,
        )
    decoded = sorted(image_root.glob("selected-*.ppm"))
    if len(decoded) != len(names):
        raise RuntimeError(
            f"source extraction emitted {len(decoded)} frames; expected exactly {len(names)}"
        )
    for source, name in zip(decoded, names):
        source.rename(image_root / name)
    return image_root


def _model_lines(text: str) -> tuple[list[str], list[str]]:
    """Split a COLMAP text file into comments and meaningful data lines."""
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    data = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return comments, data


def _parse_image_records(images_text: str) -> list[dict]:
    lines = images_text.splitlines()
    records: list[dict] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError("COLMAP images.txt has a malformed image pose line")
        try:
            image_id = int(fields[0])
            camera_id = int(fields[8])
        except ValueError as error:
            raise ValueError("COLMAP images.txt has a non-integer image or camera ID") from error
        if index >= len(lines):
            raise ValueError("COLMAP images.txt is missing an observation line")
        observation_line = lines[index]
        index += 1
        observation_fields = observation_line.split()
        if len(observation_fields) % 3:
            raise ValueError("COLMAP images.txt has malformed 2D observations")
        observations = []
        for offset in range(0, len(observation_fields), 3):
            try:
                point_id = int(observation_fields[offset + 2])
            except ValueError as error:
                raise ValueError("COLMAP images.txt has a non-integer POINT3D_ID") from error
            observations.append(
                (observation_fields[offset], observation_fields[offset + 1], point_id)
            )
        records.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": fields[9],
                "pose_line": line,
                "observations": observations,
            }
        )
    return records


def _filter_cameras_text(cameras_text: str, camera_ids: set[int]) -> str:
    comments, data = _model_lines(cameras_text)
    kept = []
    for line in data:
        fields = line.split()
        if not fields:
            continue
        try:
            camera_id = int(fields[0])
        except ValueError as error:
            raise ValueError("COLMAP cameras.txt has a non-integer CAMERA_ID") from error
        if camera_id in camera_ids:
            kept.append(line)
    return "\n".join([*comments, *kept]) + "\n"


def filter_colmap_text_model(
    cameras_text: str,
    images_text: str,
    points3d_text: str,
    selected_names: Iterable[str],
) -> tuple[str, str, str]:
    """Remove bridge images and their observations from a COLMAP text model.

    Image observation indices are positional and are referenced by
    ``points3D.txt`` tracks. Therefore dropped point references become ``-1``
    in the selected image's observation line instead of deleting a triple and
    shifting every subsequent ``POINT2D_IDX``. Point tracks are retained only
    when their pair refers to a retained selected-image observation; bridge
    pairs are removed. Cameras not used by a selected image are removed too.
    """
    selected = set(selected_names)
    records = _parse_image_records(images_text)
    selected_records = [record for record in records if record["name"] in selected]
    selected_ids = {record["image_id"] for record in selected_records}
    selected_by_id = {record["image_id"]: record for record in selected_records}
    if len(selected_by_id) != len(selected_records):
        raise ValueError("COLMAP images.txt contains duplicate selected image IDs")

    point_rows: list[tuple[int, str, list[tuple[int, int]]]] = []
    for line in points3d_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            raise ValueError("COLMAP points3D.txt has a malformed point track")
        try:
            point_id = int(fields[0])
            track = [
                (int(fields[offset]), int(fields[offset + 1]))
                for offset in range(8, len(fields), 2)
            ]
        except ValueError as error:
            raise ValueError("COLMAP points3D.txt has a non-integer track pair") from error
        point_rows.append((point_id, " ".join(fields[:8]), track))

    observation_points = {
        (record["image_id"], index): observation[2]
        for record in selected_records
        for index, observation in enumerate(record["observations"])
        if observation[2] >= 0
    }
    kept_tracks: dict[int, set[tuple[int, int]]] = {}
    for point_id, _header, track in point_rows:
        valid = {
            (image_id, point_index)
            for image_id, point_index in track
            if image_id in selected_ids
            and observation_points.get((image_id, point_index)) == point_id
        }
        if valid:
            kept_tracks[point_id] = valid
    valid_pairs = {pair for pairs in kept_tracks.values() for pair in pairs}

    selected_comments, _ = _model_lines(images_text)
    selected_image_lines: list[str] = []
    for record in selected_records:
        observations = []
        for index, (x, y, point_id) in enumerate(record["observations"]):
            if point_id >= 0 and (record["image_id"], index) not in valid_pairs:
                point_id = -1
            observations.extend((x, y, str(point_id)))
        selected_image_lines.extend((record["pose_line"], " ".join(observations)))

    selected_point_comments, _ = _model_lines(points3d_text)
    selected_point_lines: list[str] = []
    for point_id, header, track in point_rows:
        retained = [pair for pair in track if pair in kept_tracks.get(point_id, set())]
        if retained:
            selected_point_lines.append(" ".join((header, *(str(value) for pair in retained for value in pair))))

    cameras = _filter_cameras_text(cameras_text, {record["camera_id"] for record in selected_records})
    images_output = "\n".join([*selected_comments, *selected_image_lines]) + "\n"
    points_output = "\n".join([*selected_point_comments, *selected_point_lines]) + "\n"
    return cameras, images_output, points_output


def largest_model(models_root: Path) -> Path:
    models = [candidate for candidate in models_root.iterdir() if candidate.is_dir()]
    if not models:
        raise RuntimeError("COLMAP mapper emitted no sparse models")
    def registered(model: Path) -> int:
        images = model / "images.bin"
        # images.bin starts with a little-endian u64 record count.
        try:
            return int.from_bytes(images.read_bytes()[:8], "little")
        except OSError:
            return -1
    return max(models, key=registered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="existing .vestra bundle")
    parser.add_argument("--output", type=Path, required=True, help="new, empty run directory")
    parser.add_argument("--vocabulary-tree", type=Path, required=True)
    parser.add_argument("--colmap", default="colmap", help="pinned COLMAP executable or wrapper")
    parser.add_argument(
        "--container-image",
        help="optional pre-pulled COLMAP container image; keeps the run independent of host COLMAP",
    )
    parser.add_argument("--container-engine", default="podman")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--retrieval-images", type=int, default=30)
    parser.add_argument(
        "--source-video", type=Path,
        help="optional original video; enables source-resolution camera evidence",
    )
    parser.add_argument(
        "--pose-image-width", type=int,
        help="locked-crop width for --source-video evidence; must not upscale",
    )
    parser.add_argument(
        "--bridge-fps", type=float,
        help=(
            "exact timestamp samples per second inserted only between selected "
            "rasters; requires --source-video and keeps the selected raster contract"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.sequential_overlap <= 0 or args.retrieval_images <= 0:
        raise ValueError("threads, sequential overlap, and retrieval images must be positive")
    if (args.source_video is None) != (args.pose_image_width is None):
        raise ValueError("--source-video and --pose-image-width must be supplied together")
    if args.bridge_fps is not None and args.bridge_fps <= 0:
        raise ValueError("bridge-fps must be positive")
    if args.bridge_fps is not None and args.source_video is None:
        raise ValueError("--bridge-fps requires --source-video and --pose-image-width")
    scene = args.scene.resolve()
    output = args.output.resolve()
    tree = args.vocabulary_tree.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    if not scene.joinpath("manifest.json").is_file() or not tree.is_file():
        raise FileNotFoundError("scene manifest or vocabulary tree is missing")
    raster, raster_path = read_raster_manifest(scene)
    frame_names = verify_rasters(scene, raster)
    bridge_schedule = (
        temporal_bridge_schedule(raster["frames"], frame_names, args.bridge_fps)
        if args.bridge_fps is not None
        else []
    )
    output.mkdir(parents=True)
    images = (
        stage_source_resolution_images(args, raster, frame_names, output, args.bridge_fps)
        if args.source_video is not None
        else stage_colmap_images(scene / "decoded", frame_names, output)
    )
    database = output / "database.db"
    sparse = output / "sparse"
    text = output / "sparse-text"
    selected_text = output / "selected-text"
    sparse.mkdir()
    # COLMAP 4 model_converter also requires its destination to exist.
    text.mkdir()
    selected_text.mkdir()
    log = output / "colmap.log"
    settings = Settings(
        colmap=args.colmap,
        container_image=args.container_image,
        threads=args.threads,
        camera_model="SIMPLE_RADIAL",
        sequential_overlap=args.sequential_overlap,
        retrieval_images=args.retrieval_images,
        vocabulary_tree_sha256=sha256_file(tree),
        pose_image_width=args.pose_image_width,
        source_video_sha256=sha256_file(args.source_video) if args.source_video else None,
        bridge_fps=args.bridge_fps,
    )
    common_feature = colmap_command(args, [
        "feature_extractor", "--database_path", str(database), "--image_path", str(images),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", settings.camera_model,
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", str(settings.threads),
    ])
    run(common_feature, log)
    run(colmap_command(args, [
        "sequential_matcher", "--database_path", str(database),
        "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", str(settings.threads),
        "--SequentialMatching.overlap", str(settings.sequential_overlap),
        "--SequentialMatching.quadratic_overlap", "1", "--FeatureMatching.guided_matching", "1",
    ]), log)
    # Retrieval adds only visually similar candidates; geometric verification
    # remains COLMAP's matcher/mapper responsibility. This is intentionally
    # not exhaustive matching for a long local video.
    run(colmap_command(args, [
        "vocab_tree_matcher", "--database_path", str(database),
        "--VocabTreeMatching.vocab_tree_path", str(tree),
        "--VocabTreeMatching.num_images", str(settings.retrieval_images),
        "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", str(settings.threads),
        "--FeatureMatching.guided_matching", "1",
    ]), log)
    run(colmap_command(args, [
        "mapper", "--database_path", str(database), "--image_path", str(images),
        "--output_path", str(sparse), "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.num_threads", str(settings.threads),
    ]), log)
    model = largest_model(sparse)
    ba = output / "global-ba"
    # COLMAP 4's bundle_adjuster refuses a non-existent output directory,
    # unlike mapper which creates its numbered model directories itself.
    ba.mkdir()
    run(colmap_command(args, [
        "bundle_adjuster", "--input_path", str(model), "--output_path", str(ba),
        "--BundleAdjustment.refine_focal_length", "1", "--BundleAdjustment.refine_extra_params", "1",
    ]), log)
    run(colmap_command(args, [
        "model_converter", "--input_path", str(ba), "--output_path", str(text),
        "--output_type", "TXT",
    ]), log)
    full_cameras_txt = text / "cameras.txt"
    full_images_txt = text / "images.txt"
    full_points_txt = text / "points3D.txt"
    selected_cameras, selected_images, selected_points = filter_colmap_text_model(
        full_cameras_txt.read_text(encoding="utf-8"),
        full_images_txt.read_text(encoding="utf-8"),
        full_points_txt.read_text(encoding="utf-8"),
        frame_names,
    )
    (selected_text / "cameras.txt").write_text(selected_cameras, encoding="utf-8")
    (selected_text / "images.txt").write_text(selected_images, encoding="utf-8")
    (selected_text / "points3D.txt").write_text(selected_points, encoding="utf-8")
    selected_name_set = set(frame_names)
    full_records = _parse_image_records(full_images_txt.read_text(encoding="utf-8"))
    registered = sum(1 for record in full_records if record["name"] in selected_name_set)
    all_input_names = [*frame_names, *(entry["file_name"] for entry in bridge_schedule)]
    provenance = {
        "schema": SCHEMA,
        "scene": str(scene),
        "raster_manifest": str(raster_path),
        "raster_fingerprint": raster["raster_fingerprint"],
        "input_frames": len(raster["frames"]),
        "selected_frames": len(frame_names),
        "bridge_frames": len(bridge_schedule),
        "input_images": len(all_input_names),
        "selected_images": len(frame_names),
        "bridge_images": len(bridge_schedule),
        "registered_images": registered,
        "input_image_names": all_input_names,
        "selected_image_names": frame_names,
        "bridge_image_names": [entry["file_name"] for entry in bridge_schedule],
        "counts": {
            "input": len(all_input_names),
            "selected": len(frame_names),
            "bridge": len(bridge_schedule),
            "registered": registered,
        },
        "image_root": str(images),
        "registered_frames": registered,
        "bridge": {
            "enabled": args.bridge_fps is not None,
            "fps": args.bridge_fps,
            "count": len(bridge_schedule),
            "images": bridge_schedule,
        },
        "settings": asdict(settings),
        "settings_fingerprint": hashlib.sha256(
            json.dumps(asdict(settings), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        # `images_txt` is intentionally the selected-only model consumed by
        # pose-import-colmap. The complete model remains available for dense
        # MVS and audit/replay.
        "images_txt": str(selected_text / "images.txt"),
        "selected_model_text": str(selected_text),
        "full_model_text": str(text),
        "full_model": str(ba),
        "full_images_txt": str(full_images_txt),
    }
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
