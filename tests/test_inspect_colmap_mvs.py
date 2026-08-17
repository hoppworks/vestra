import importlib.util
import math
import sys
import unittest
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inspect_colmap_mvs", ROOT / "tools" / "inspect_colmap_mvs.py")
MVS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MVS
SPEC.loader.exec_module(MVS)


class MvsTrackDepthReportTest(unittest.TestCase):
    def test_track_report_uses_exact_pixel_centre_resize_and_depth_gate(self):
        camera = MVS.Camera(
            frame_index=7,
            width=4,
            height=4,
            fx=1.0,
            fy=1.0,
            cx=2.0,
            cy=2.0,
            w2c=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        depth = array("f", [math.inf]) * 4
        # Original pixel (1.5, 1.5) maps to output (0.5, 0.5), whose local
        # 3x3 lookup sees this finite Z at (1, 1).
        depth[1 * 2 + 1] = 4.0
        solution = {
            "global_trajectory": {
                "tracks": [
                    {
                        "position": [0.0, 0.0, 4.0],
                        "reprojection_error_px": 0.2,
                        "observations": [{"frame_index": 7, "image_xy": [1.5, 1.5]}],
                    },
                    {
                        "position": [0.0, 0.0, 4.0],
                        "reprojection_error_px": 3.0,
                        "observations": [{"frame_index": 7, "image_xy": [1.5, 1.5]}],
                    },
                ]
            }
        }
        report = MVS.track_depth_report(solution, [camera], {7: depth}, 2, 2)
        self.assertEqual(report["observations"], 1)
        self.assertEqual(report["covered_observations"], 1)
        self.assertEqual(report["coverage"], 1.0)
        self.assertAlmostEqual(report["median_abs_log_depth_error"], 0.0)

    def test_sparse_track_outside_mvs_coverage_is_not_counted_as_match(self):
        camera = MVS.Camera(
            frame_index=3,
            width=2,
            height=2,
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
            w2c=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        )
        solution = {"global_trajectory": {"tracks": [{
            "position": [0.0, 0.0, 2.0],
            "reprojection_error_px": 0.1,
            "observations": [{"frame_index": 3, "image_xy": [0.0, 0.0]}],
        }]}}
        report = MVS.track_depth_report(solution, [camera], {3: array("f", [math.inf])}, 1, 1)
        self.assertEqual(report["observations"], 1)
        self.assertEqual(report["covered_observations"], 0)
        self.assertIsNone(report["p95_abs_log_depth_error"])


if __name__ == "__main__":
    unittest.main()
