import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_benchmark.py"


class BenchmarkRunnerTest(unittest.TestCase):
    def test_randomized_trials_preserve_raw_samples_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            profile = {
                "schema": "vestra.benchmark/v1",
                "seed": 7,
                "trials": 10,
                "warmup": 1,
                "arms": [
                    {
                        "name": "left",
                        "measurement": "stdout_samples_ms",
                        "command": [sys.executable, "-c", "print('{\\\"samples_ms\\\":[1,2,3]}')"],
                    },
                    {
                        "name": "right",
                        "measurement": "stdout_samples_ms",
                        "command": [sys.executable, "-c", "print('{\\\"samples_ms\\\":[4,5,6]}')"],
                    },
                ],
            }
            profile_path = directory / "profile.json"
            output_path = directory / "result.json"
            profile_path.write_text(json.dumps(profile))
            subprocess.run(
                [sys.executable, str(RUNNER), "--profile", str(profile_path), "--output", str(output_path)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output_path.read_text())
            self.assertEqual(result["schema"], "vestra.benchmark-result/v1")
            self.assertEqual(len(result["raw_trials"]["left"]), 10)
            self.assertEqual(result["summaries"]["left"]["mean_trial_median_ms"], 2.0)
            self.assertEqual(result["summaries"]["right"]["mean_trial_median_ms"], 5.0)
            self.assertEqual(len(result["summaries"]["left"]["ci95_ms"]), 2)


if __name__ == "__main__":
    unittest.main()
