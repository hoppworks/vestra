#!/usr/bin/env python3
"""Emit a Vestra global-pose solution from the official VGGT camera head.

VGGT is a *pose provider* here.  It receives only the exact PPMs named by a
Vestra raster manifest and writes OpenCV world-to-camera matrices; it never
rewrites DA3 depth evidence or publishes a world by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SCHEMA = "vestra.pose-solution/v1"
CONVENTION = "OpenCV world; W2C row-major 3x4 f64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--vggt-root", type=Path, required=True, help="pinned official VGGT checkout")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=518)
    return parser.parse_args()


def raster_manifest(scene: Path) -> tuple[dict, Path]:
    manifest = json.loads((scene / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.get("raster_manifest_hash")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("scene has no immutable raster manifest")
    path = scene / "chunks" / f"raster-{digest}.json"
    raster = json.loads(path.read_text(encoding="utf-8"))
    if raster.get("schema") != "vestra.raster/v1":
        raise ValueError("unsupported Vestra raster manifest")
    frames = raster.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("raster manifest contains no frames")
    if [frame.get("frame_index") for frame in frames] != list(range(len(frames))):
        raise ValueError("raster frames are not contiguous")
    return raster, path


def main() -> int:
    args = parse_args()
    scene = args.scene.resolve()
    model_path = args.model.resolve()
    vggt_root = args.vggt_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.resolution != 518:
        raise ValueError("official VGGT camera head is pinned to 518px")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    raster, raster_path = raster_manifest(scene)
    frames = raster["frames"]
    names: list[str] = []
    for frame in frames:
        name = frame.get("file_name")
        expected_hash = frame.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe raster name {name!r}")
        path = scene / "decoded" / name
        if sha256(path) != expected_hash:
            raise ValueError(f"raster hash mismatch: {name}")
        names.append(name)

    import torch
    import torch.nn.functional as functional
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT global pose export requires CUDA")
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    image_paths = [str(scene / "decoded" / name) for name in names]
    # The official demo intentionally normalizes each source raster to a square
    # before its fixed 518px camera-head inference.  The original rasters and
    # their crop contract remain immutable in the scene manifest.
    images, _ = load_and_preprocess_images_square(image_paths, 1024)
    images = images.to(device)
    model = VGGT()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        camera_images = functional.interpolate(
            images, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False
        )[None]
        tokens, _ = model.aggregator(camera_images)
        pose_encoding = model.camera_head(tokens)[-1]
        extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding, camera_images.shape[-2:])
    w2c = extrinsic.squeeze(0).float().cpu().numpy()
    if w2c.shape != (len(names), 3, 4):
        raise RuntimeError(f"unexpected VGGT extrinsic shape {w2c.shape!r}")

    revision = subprocess.check_output(
        ["git", "-C", str(vggt_root), "rev-parse", "HEAD"], text=True
    ).strip()
    settings = {
        "provider": "facebookresearch/vggt",
        "revision": revision,
        "model_sha256": sha256(model_path),
        "resolution": args.resolution,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
    }
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    solution = {
        "schema": SCHEMA,
        "provider": {"kind": "vggt", "version": revision, "settings_fingerprint": fingerprint},
        "raster_fingerprint": raster["raster_fingerprint"],
        "coordinate_convention": CONVENTION,
        "frames": [
            {
                "frame_index": index,
                "image_name": name,
                "registered": True,
                "world_to_camera": [float(value) for value in w2c[index].reshape(-1)],
            }
            for index, name in enumerate(names)
        ],
        "diagnostics": {"input_frames": len(names), "registered_frames": len(names), "duplicate_images": 0},
        "provenance": {"raster_manifest": str(raster_path), "settings": settings},
    }
    output.write_text(json.dumps(solution, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": len(names), "settings": settings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
