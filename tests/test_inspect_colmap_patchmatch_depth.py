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


if __name__ == "__main__":
    unittest.main()
