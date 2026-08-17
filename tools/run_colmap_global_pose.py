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
    if raster.get("schema") != "vestra.raster-manifest/v1":
        raise ValueError("unsupported raster manifest schema")
    return raster, raster_path


def verify_rasters(scene: Path, raster: dict) -> Path:
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
    return decoded


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.sequential_overlap <= 0 or args.retrieval_images <= 0:
        raise ValueError("threads, sequential overlap, and retrieval images must be positive")
    scene = args.scene.resolve()
    output = args.output.resolve()
    tree = args.vocabulary_tree.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    if not scene.joinpath("manifest.json").is_file() or not tree.is_file():
        raise FileNotFoundError("scene manifest or vocabulary tree is missing")
    raster, raster_path = read_raster_manifest(scene)
    decoded = verify_rasters(scene, raster)
    output.mkdir(parents=True)
    database = output / "database.db"
    sparse = output / "sparse"
    text = output / "sparse-text"
    sparse.mkdir()
    log = output / "colmap.log"
    settings = Settings(
        colmap=args.colmap,
        container_image=args.container_image,
        threads=args.threads,
        camera_model="SIMPLE_RADIAL",
        sequential_overlap=args.sequential_overlap,
        retrieval_images=args.retrieval_images,
        vocabulary_tree_sha256=sha256_file(tree),
    )
    common_feature = colmap_command(args, [
        "feature_extractor", "--database_path", str(database), "--image_path", str(decoded),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", settings.camera_model,
        "--SiftExtraction.use_gpu", "0", "--SiftExtraction.num_threads", str(settings.threads),
    ])
    run(common_feature, log)
    run(colmap_command(args, [
        "sequential_matcher", "--database_path", str(database),
        "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", str(settings.threads),
        "--SequentialMatching.overlap", str(settings.sequential_overlap),
        "--SequentialMatching.quadratic_overlap", "1", "--SiftMatching.guided_matching", "1",
    ]), log)
    # Retrieval adds only visually similar candidates; geometric verification
    # remains COLMAP's matcher/mapper responsibility. This is intentionally
    # not exhaustive matching for a long local video.
    run(colmap_command(args, [
        "vocab_tree_matcher", "--database_path", str(database),
        "--VocabTreeMatching.vocab_tree_path", str(tree),
        "--VocabTreeMatching.num_images", str(settings.retrieval_images),
        "--SiftMatching.use_gpu", "0", "--SiftMatching.num_threads", str(settings.threads),
        "--SiftMatching.guided_matching", "1",
    ]), log)
    run(colmap_command(args, [
        "mapper", "--database_path", str(database), "--image_path", str(decoded),
        "--output_path", str(sparse), "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.num_threads", str(settings.threads),
    ]), log)
    model = largest_model(sparse)
    ba = output / "global-ba"
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
