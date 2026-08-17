#!/usr/bin/env python3
"""Build a reproducible global COLMAP pose candidate for one Vestra scene.

This is intentionally an experiment runner, not a publishing shortcut.  It
consumes the immutable raster manifest already recorded in a ``.vestra``
bundle, verifies every PPM byte before COLMAP sees it, adds both sequential and
vocabulary-tree retrieval matches, runs a final global bundle adjustment, and
writes the exact ``images.txt`` plus provenance for ``vestra pose-import-colmap``.

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
from pathlib import Path
from typing import Iterable


SCHEMA = "vestra.colmap-global-pose-run/v1"


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


def source_extraction_schedule(frames: list[dict]) -> tuple[float, int]:
    """Returns `(uniform_rate, prefix_count)` without approximating timestamps.

    High-resolution pose evidence must describe the same source instants as the
    immutable DA3 rasters. A generic `fps` filter is exact only when the
    manifest timestamps are uniformly spaced. Vestra's quality selector may
    retain one final partial cadence interval at end-of-video; that final frame
    is decoded by its explicit timestamp after the regular prefix. Any other
    irregular schedule is rejected instead of silently sampling neighbours.
    """
    if len(frames) < 2:
        raise ValueError("high-resolution pose extraction requires two or more raster frames")
    timestamps = [frame.get("timestamp_millis") for frame in frames]
    if any(not isinstance(value, int) or value < 0 for value in timestamps):
        raise ValueError("raster timestamps must be non-negative integer milliseconds")
    if timestamps[0] != 0:
        raise ValueError("high-resolution pose extraction requires a zero timestamp first frame")
    interval = timestamps[1] - timestamps[0]
    if interval <= 0:
        raise ValueError("raster timestamps must be strictly increasing")
    prefix_count = 2
    while prefix_count < len(timestamps) and timestamps[prefix_count] - timestamps[prefix_count - 1] == interval:
        prefix_count += 1
    if prefix_count < len(timestamps) - 1 or (
        prefix_count < len(timestamps) and timestamps[prefix_count] <= timestamps[prefix_count - 1]
    ):
        raise ValueError(
            "high-resolution pose extraction accepts only a uniform raster prefix and one final tail frame"
        )
    return 1000.0 / interval, prefix_count


def stage_source_resolution_images(
    args: argparse.Namespace,
    raster: dict,
    names: list[str],
    output: Path,
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
    rate, regular_count = source_extraction_schedule(raster["frames"])
    image_root = output / "images"
    image_root.mkdir()
    temporary_pattern = image_root / "selected-%06d.ppm"
    filter_parts = [
        f"fps={rate:.12f}",
        f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}",
    ]
    if args.pose_image_width != crop["width"]:
        filter_parts.append(f"scale={args.pose_image_width}:-2:flags=lanczos")
    log = output / "colmap.log"
    run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", ",".join(filter_parts), "-frames:v", str(regular_count), str(temporary_pattern),
        ],
        log,
    )
    if regular_count < len(names):
        final_timestamp = raster["frames"][-1]["timestamp_millis"] / 1000.0
        run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                "-ss", f"{final_timestamp:.3f}", "-vf", ",".join(filter_parts[1:]),
                "-frames:v", "1", str(image_root / f"selected-{len(names):06d}.ppm"),
            ],
            log,
        )
    decoded = sorted(image_root.glob("selected-*.ppm"))
    if len(decoded) != len(names):
        raise RuntimeError(
            f"source extraction emitted {len(decoded)} frames; expected exactly {len(names)}"
        )
    for source, name in zip(decoded, names, strict=True):
        source.rename(image_root / name)
    return image_root


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.sequential_overlap <= 0 or args.retrieval_images <= 0:
        raise ValueError("threads, sequential overlap, and retrieval images must be positive")
    if (args.source_video is None) != (args.pose_image_width is None):
        raise ValueError("--source-video and --pose-image-width must be supplied together")
    scene = args.scene.resolve()
    output = args.output.resolve()
    tree = args.vocabulary_tree.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    if not scene.joinpath("manifest.json").is_file() or not tree.is_file():
        raise FileNotFoundError("scene manifest or vocabulary tree is missing")
    raster, raster_path = read_raster_manifest(scene)
    frame_names = verify_rasters(scene, raster)
    output.mkdir(parents=True)
    images = (
        stage_source_resolution_images(args, raster, frame_names, output)
        if args.source_video is not None
        else stage_colmap_images(scene / "decoded", frame_names, output)
    )
    database = output / "database.db"
    sparse = output / "sparse"
    text = output / "sparse-text"
    sparse.mkdir()
    # COLMAP 4 model_converter also requires its destination to exist.
    text.mkdir()
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
    images_txt = text / "images.txt"
    # COLMAP text models store exactly two physical lines per image: the pose
    # line and its 2D observations (which may be blank). Preserve blanks so
    # observation coordinates are never mistaken for extra cameras.
    model_lines = [
        line for line in images_txt.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    registered = sum(1 for line in model_lines[::2] if len(line.split()) >= 10)
    provenance = {
        "schema": SCHEMA,
        "scene": str(scene),
        "raster_manifest": str(raster_path),
        "raster_fingerprint": raster["raster_fingerprint"],
        "input_frames": len(raster["frames"]),
        "image_root": str(images),
        "registered_frames": registered,
        "settings": asdict(settings),
        "settings_fingerprint": hashlib.sha256(
            json.dumps(asdict(settings), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "images_txt": str(images_txt),
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
