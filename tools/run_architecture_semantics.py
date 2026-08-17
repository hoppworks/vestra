#!/usr/bin/env python3
"""Create versioned indoor-semantic evidence for Vestra architecture products.

The runner is intentionally an adapter: it writes masks and provenance but
does not create geometry, fill surfaces, or choose a world product.  Geometry
remains owned and validated by vestra-core.

The default is deliberately absent.  Callers must select a checkpoint and
state its licence explicitly; many useful indoor parsing checkpoints carry
research-only data or weight terms and must not become an implicit product
dependency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SCHEMA = "vestra.architecture-semantics/v1"
CLASS_UNKNOWN = 0
CLASS_FLOOR = 1
CLASS_WALL = 2
CLASS_CEILING_OR_ROOF = 3
CLASS_DOOR_OR_OPENING = 4
CLASS_WINDOW = 5
CLASS_NON_ARCHITECTURAL = 6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-dir", type=Path, required=True,
                        help="Vestra decoded-frame cache containing frame-*.ppm files")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; receives masks.npz and manifest.json")
    parser.add_argument("--model-id", required=True,
                        help="Hugging Face image-segmentation checkpoint identifier")
    parser.add_argument("--model-license", required=True,
                        help="Licence/terms recorded verbatim in the evidence manifest")
    parser.add_argument("--revision", default="main",
                        help="Immutable model revision; use a commit hash for repeatability")
    parser.add_argument("--device", default="cuda",
                        help="Transformers device string, normally cuda on Workhorse")
    parser.add_argument("--maximum-frames", type=int, default=0,
                        help="Optional deterministic cap; zero processes every decoded frame")
    return parser.parse_args()


def frame_index(path: Path) -> int:
    match = re.fullmatch(r"frame-(\d+)\.ppm", path.name)
    if match is None:
        raise ValueError(f"unexpected decoded frame name: {path.name}")
    return int(match.group(1)) - 1


def architecture_class(label: str) -> int:
    normalized = label.lower().strip()
    if normalized in {"floor", "flooring"}:
        return CLASS_FLOOR
    if normalized in {"wall", "partition", "divider"}:
        return CLASS_WALL
    if normalized in {"ceiling", "roof", "roofing"}:
        return CLASS_CEILING_OR_ROOF
    if normalized in {"door", "doorway", "screen door"}:
        return CLASS_DOOR_OR_OPENING
    if normalized in {"window", "windowpane"}:
        return CLASS_WINDOW
    return CLASS_NON_ARCHITECTURAL


def main() -> int:
    args = arguments()
    if not args.model_license.strip():
        raise ValueError("--model-license must be non-empty")
    if args.maximum_frames < 0:
        raise ValueError("--maximum-frames cannot be negative")
    frames = sorted(args.decoded_dir.glob("frame-*.ppm"), key=frame_index)
    if args.maximum_frames:
        frames = frames[:args.maximum_frames]
    if not frames:
        raise FileNotFoundError(f"no decoded PPM frames in {args.decoded_dir}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    # Heavy dependencies intentionally stay inside main so --help works on the
    # Rust development machine without a Python ML environment.
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import model_info
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    device = torch.device(args.device)
    resolved_revision = model_info(args.model_id, revision=args.revision).sha
    processor = AutoImageProcessor.from_pretrained(args.model_id, revision=resolved_revision)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model_id, revision=resolved_revision
    ).to(device).eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    height = width = None
    classes = []
    confidences = []
    frame_indices = []
    with torch.inference_mode():
        for number, path in enumerate(frames, start=1):
            image = Image.open(path).convert("RGB")
            if width is None:
                width, height = image.size
            elif image.size != (width, height):
                raise ValueError("decoded cache has mixed raster dimensions")
            inputs = processor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )[0]
            probabilities = logits.softmax(dim=0)
            confidence, raw_class = probabilities.max(dim=0)
            raw_class = raw_class.cpu().numpy()
            mapped = np.vectorize(
                lambda value: architecture_class(id2label.get(int(value), "unknown")),
                otypes=[np.uint8],
            )(raw_class)
            classes.append(mapped)
            confidences.append((confidence.cpu().numpy() * 255.0).round().astype(np.uint8))
            frame_indices.append(frame_index(path))
            print(f"[{number}/{len(frames)}] {path.name}", file=sys.stderr, flush=True)

    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "masks.npz",
        frame_indices=np.asarray(frame_indices, dtype=np.uint32),
        classes=np.stack(classes, axis=0),
        confidences=np.stack(confidences, axis=0),
    )
    manifest = {
        "schema": SCHEMA,
        "runner": "vestra.tools.run_architecture_semantics/1",
        "model_id": args.model_id,
        "model_revision": resolved_revision,
        "model_license": args.model_license,
        "raster": {"width": width, "height": height, "frames": len(frame_indices)},
        "classes": {
            "unknown": CLASS_UNKNOWN,
            "floor": CLASS_FLOOR,
            "wall": CLASS_WALL,
            "ceiling_or_roof": CLASS_CEILING_OR_ROOF,
            "door_or_opening": CLASS_DOOR_OR_OPENING,
            "window": CLASS_WINDOW,
            "non_architectural": CLASS_NON_ARCHITECTURAL,
        },
        "arrays": "masks.npz",
        "confidence_encoding": "uint8 / 255",
        "geometry_policy": "semantic masks select observed geometry; they never create geometry",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # make invalid model/terms obvious in job logs
        print(f"architecture semantics failed: {error}", file=sys.stderr)
        raise SystemExit(1)
