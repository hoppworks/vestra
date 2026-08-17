import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "inspect_colmap_mvs", Path(__file__).parents[1] / "tools" / "inspect_colmap_mvs.py"
)
MVS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MVS
SPEC.loader.exec_module(MVS)


class ColmapMvsInspectorTest(unittest.TestCase):
    def test_binary_ply_projects_through_registered_pinhole_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "cloud.ply"
            header = (
                "ply\nformat binary_little_endian 1.0\n"
                "element vertex 2\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            # The nearer green vertex must occlude the red vertex at the same pixel.
            ply.write_bytes(header + struct.pack("<fffBBBfffBBB", 0, 0, 2, 255, 0, 0, 0, 0, 1, 0, 255, 0))
            vertices = MVS.read_vertices(ply, None)
            self.assertEqual(len(vertices), 2)
            camera = MVS.Camera(7, 4, 4, 2, 2, 2, 2, (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))
            report = MVS.render(vertices, camera, 4, 4, root / "render.ppm")
            self.assertEqual(report["visible_pixels"], 1)
            self.assertEqual((root / "render.ppm").read_bytes()[-(4 * 4 * 3) + (2 * 4 + 2) * 3:][0:3], bytes((0, 255, 0)))

    def test_load_cameras_maps_simple_pinhole_and_rejects_unregistered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps({
                "frames": [{"frame_index": 2, "registered": True, "world_to_camera": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}],
                "global_trajectory": {
                    "frame_camera_ids": {"2": 5},
                    "camera_models": [{"camera_id": 5, "model": "SIMPLE_PINHOLE", "width": 8, "height": 6, "parameters": [4, 4, 3]}],
                },
            }))
            camera = MVS.load_cameras(path, {2})[0]
            self.assertEqual((camera.fx, camera.fy, camera.cx, camera.cy), (4, 4, 4, 3))
            with self.assertRaisesRegex(ValueError, "not registered"):
                MVS.load_cameras(path, {3})


if __name__ == "__main__":
    unittest.main()
