import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "fuse_da3_patchmatch_depth", ROOT / "tools" / "fuse_da3_patchmatch_depth.py"
)
PATCHMATCH_HYBRID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PATCHMATCH_HYBRID
SPEC.loader.exec_module(PATCHMATCH_HYBRID)


class PatchMatchHybridTest(unittest.TestCase):
    def test_nearest_pixel_centre_resample_preserves_valid_depth_and_holes(self):
        import numpy as np

        source = np.array([[[1.0], [2.0]], [[3.0], [np.nan]]], dtype=np.float32)
        result = PATCHMATCH_HYBRID.resample_nearest_depth(source, 4, 4, np)
        self.assertEqual(result.shape, (4, 4))
        self.assertAlmostEqual(float(result[0, 0]), 1.0)
        self.assertAlmostEqual(float(result[2, 0]), 3.0)
        self.assertTrue(np.isnan(result[3, 3]))


if __name__ == "__main__":
    unittest.main()
