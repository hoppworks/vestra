#!/usr/bin/env python3
"""Write a deterministic VPS1 fixture for the first C++ stitcher oracle test.

Every frame observes the same fronto-parallel plane from the same calibrated
camera.  It is intentionally simple: two 3-frame windows with one overlap must
produce exactly four owned frames and one valid identity Sim3 seam.  It proves
the interchange contract before a Vestra-generated fixture is introduced.
"""

import argparse
import struct
from pathlib import Path


def write_fixture(path: Path) -> None:
    frame_count, height, width = 4, 3, 4
    chunk, overlap, confidence_percentile, point_size, min_overlap = 3, 1, 0.0, 1.0, 3
    intrinsics = (2.0, 0.0, 1.5, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0)
    # W2C identity: the fixed plane lives at z=2 in every local window.
    extrinsics = (1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0)
    with path.open("wb") as output:
        output.write(b"VPS1")
        output.write(struct.pack("<IIIIII d f I", 1, frame_count, height, width,
                                 chunk, overlap, confidence_percentile,
                                 point_size, min_overlap))
        for frame in range(frame_count):
            output.write(struct.pack("<9f", *intrinsics))
            output.write(struct.pack("<12f", *extrinsics))
            output.write(struct.pack(f"<{height * width}f", *([2.0] * (height * width))))
            output.write(struct.pack(f"<{height * width}f", *([1.0] * (height * width))))
            rgb = bytearray()
            for pixel in range(height * width):
                rgb.extend(((frame * 53) % 256, (pixel * 17) % 256, 127))
            output.write(rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    write_fixture(arguments.output)
