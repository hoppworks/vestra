import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inspect_colmap_patchmatch_depth", ROOT / "tools" / "inspect_colmap_patchmatch_depth.py"
)
PATCHMATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PATCHMATCH
SPEC.loader.exec_module(PATCHMATCH)


class PatchMatchDepthTest(unittest.TestCase):
    def test_reads_colmap_header_and_reports_positive_depth_coverage(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.bin"
            values = np.array([1.0, 0.0, np.inf, 4.0], dtype="<f4")
            path.write_bytes(b"2&2&1&" + values.tobytes())
            depth = PATCHMATCH.read_colmap_array(path, np)
            report = PATCHMATCH.summarize_depth(depth, np)
        self.assertEqual(depth.shape, (2, 2, 1))
        self.assertEqual(report["valid_pixels"], 2)
        self.assertAlmostEqual(report["coverage"], 0.5)
        self.assertAlmostEqual(report["depth_median"], 2.5)

    def test_sparse_tracks_use_pixel_centre_mapping_and_global_camera_depth(self):
        import numpy as np

        depth = np.full((2, 2, 1), 4.0, dtype=np.float32)
        pose = {
            "frames": [{"frame_index": 3, "registered": True,
                        "world_to_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}],
            "global_trajectory": {"tracks": [{"position": [0, 0, 4], "observations": [
                {"frame_index": 3, "image_xy": [4.5, 4.5]},
                {"frame_index": 2, "image_xy": [4.5, 4.5]},
            ]}]},
        }
        report = PATCHMATCH.track_depth_report(pose, 3, depth, 10, 10, np)
        self.assertEqual(report["observations"], 1)
        self.assertEqual(report["covered_observations"], 1)
        self.assertAlmostEqual(report["median_abs_log_depth_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
