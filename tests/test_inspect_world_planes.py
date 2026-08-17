import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("inspect_world_planes", ROOT / "tools" / "inspect_world_planes.py")
PLANES = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PLANES
SPEC.loader.exec_module(PLANES)


class WorldPlaneInspectionTest(unittest.TestCase):
    def test_recovers_a_clean_dominant_plane_and_reports_normalized_residual(self):
        import numpy as np

        x, y = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
        plane = np.stack([x.ravel(), y.ravel(), np.full(x.size, 3.0)], axis=1)
        outliers = np.array([[6.0, 5.0, -2.0], [-5.0, 3.0, 1.0], [1.0, -4.0, 7.0]])
        report = PLANES.inspect_positions(np.vstack([plane, outliers]), trials=128, threshold_fraction=0.001, np=np)

        first = report["planes"][0]
        self.assertGreater(first["sample_fraction"], 0.99)
        self.assertLess(first["residual_p95_fraction_of_diagonal"], 1e-12)
        self.assertAlmostEqual(abs(first["normal"][2]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
