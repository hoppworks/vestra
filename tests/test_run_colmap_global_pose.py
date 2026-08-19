import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_colmap_global_pose.py"
SPEC = importlib.util.spec_from_file_location("run_colmap_global_pose", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceExtractionScheduleTest(unittest.TestCase):
    @staticmethod
    def frames(*timestamps: int) -> list[dict]:
        return [
            {"frame_index": index, "timestamp_millis": timestamp}
            for index, timestamp in enumerate(timestamps)
        ]

    def test_uniform_prefix_and_final_tail_keep_the_fast_path(self) -> None:
        self.assertEqual(
            MODULE.source_extraction_schedule(self.frames(0, 125, 250, 375, 490)),
            (8.0, 4),
        )

    def test_quality_selected_timestamps_request_exact_per_frame_decode(self) -> None:
        self.assertIsNone(
            MODULE.source_extraction_schedule(self.frames(0, 500, 1125, 1500, 2125))
        )

    def test_non_increasing_timestamps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            MODULE.source_extraction_schedule(self.frames(0, 500, 500))


class TemporalBridgeScheduleTest(unittest.TestCase):
    def test_bridge_names_insert_between_exact_selected_names(self) -> None:
        frames = [
            {"timestamp_millis": timestamp}
            for timestamp in (0, 1200, 3000)
        ]
        names = ["frame-000001.ppm", "frame-000002.ppm", "frame-000003.ppm"]
        bridges = MODULE.temporal_bridge_schedule(frames, names, 2.0)
        self.assertEqual(
            [entry["timestamp_millis"] for entry in bridges],
            [500, 1000, 1700, 2200, 2700],
        )
        ordered = MODULE._ordered_image_entries(frames, names, bridges)
        self.assertEqual(
            [entry["file_name"] for entry in ordered],
            [
                "frame-000001.ppm",
                "frame-000001~bridge-000000000500-000000.ppm",
                "frame-000001~bridge-000000001000-000001.ppm",
                "frame-000002.ppm",
                "frame-000002~bridge-000000001700-000002.ppm",
                "frame-000002~bridge-000000002200-000003.ppm",
                "frame-000002~bridge-000000002700-000004.ppm",
                "frame-000003.ppm",
            ],
        )

    def test_bridge_schedule_is_bounded_to_selected_gaps(self) -> None:
        frames = [{"timestamp_millis": timestamp} for timestamp in (100, 400)]
        names = ["frame-000001.ppm", "frame-000002.ppm"]
        self.assertEqual(MODULE.temporal_bridge_schedule(frames, names, 2.0), [])


class ColmapTextModelFilterTest(unittest.TestCase):
    def test_removes_bridge_tracks_without_shifting_selected_point2d_indices(self) -> None:
        cameras = """# Camera list
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 SIMPLE_RADIAL 640 480 500 320 240 0
2 SIMPLE_RADIAL 640 480 500 320 240 0
"""
        images = """# Image list
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
1 1 0 0 0 0 0 0 1 frame-000001.ppm
10 20 10 30 40 -1

2 1 0 0 0 0 0 0 1 frame-000002.ppm
11 21 10

3 1 0 0 0 0 0 0 2 frame-000001~bridge-000000000500-000000.ppm
12 22 10

"""
        points = """# 3D point list
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
10 1 2 3 255 0 0 0.2 1 0 2 0 3 0
11 2 3 4 0 255 0 0.3 3 0
"""
        selected_cameras, selected_images, selected_points = MODULE.filter_colmap_text_model(
            cameras,
            images,
            points,
            ["frame-000001.ppm", "frame-000002.ppm"],
        )
        self.assertIn("1 SIMPLE_RADIAL", selected_cameras)
        self.assertNotIn("2 SIMPLE_RADIAL", selected_cameras)
        self.assertIn("frame-000001.ppm", selected_images)
        self.assertIn("frame-000002.ppm", selected_images)
        self.assertNotIn("~bridge-", selected_images)
        self.assertIn("10 1 2 3 255 0 0 0.2 1 0 2 0", selected_points)
        self.assertNotIn(" 3 0", selected_points)
        self.assertNotIn("11 2 3 4", selected_points)


if __name__ == "__main__":
    unittest.main()
