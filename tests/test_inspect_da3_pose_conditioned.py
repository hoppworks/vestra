import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inspect_da3_pose_conditioned", ROOT / "tools" / "inspect_da3_pose_conditioned.py"
)
INSPECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = INSPECTOR
SPEC.loader.exec_module(INSPECTOR)


class RasterMappingTest(unittest.TestCase):
    def test_source_pixel_centres_use_half_pixel_resize_mapping(self):
        self.assertAlmostEqual(
            INSPECTOR.source_pixel_to_raster_pixel(809.5, 1620.0, 504.0), 251.5
        )
        self.assertAlmostEqual(
            INSPECTOR.source_pixel_to_raster_pixel(539.5, 1080.0, 336.0), 167.5
        )


if __name__ == "__main__":
    unittest.main()
