import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("fuse_da3_mvs_depth", ROOT / "tools" / "fuse_da3_mvs_depth.py")
HYBRID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = HYBRID
SPEC.loader.exec_module(HYBRID)


class MvsHybridProjectionTest(unittest.TestCase):
    def test_nearest_global_mvs_surface_wins_the_depth_buffer(self):
        import numpy as np

        # Two world positions project to the principal pixel; the nearer one
        # must be retained, while off-image and behind-camera data are ignored.
        positions = np.array([
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0],
            [20.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float32)
        k = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        depth = HYBRID.project_mvs_depth(positions, k, w2c, 3, 3, np)
        self.assertEqual(float(depth[1, 1]), 2.0)
        self.assertTrue(np.isinf(depth[0, 0]))

    def test_coarse_local_ratio_changes_only_well_observed_tiles(self):
        import numpy as np

        da3 = np.full((4, 4), 2.0, dtype=np.float32)
        mvs = np.full((4, 4), np.inf, dtype=np.float32)
        mvs[:2, :2] = 3.0
        ratio = HYBRID.coarse_local_ratio(da3, mvs, cells=2, minimum_samples=3, np=np)
        self.assertTrue(np.allclose(ratio[:2, :2], 1.5))
        self.assertTrue(np.allclose(ratio[:2, 2:], 1.0))
        self.assertTrue(np.allclose(ratio[2:, :], 1.0))


if __name__ == "__main__":
    unittest.main()
