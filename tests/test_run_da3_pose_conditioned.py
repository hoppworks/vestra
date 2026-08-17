import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_da3_pose_conditioned.py"
SPEC = importlib.util.spec_from_file_location("run_da3_pose_conditioned", RUNNER)
assert SPEC and SPEC.loader
SIDECAR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SIDECAR
SPEC.loader.exec_module(SIDECAR)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class PoseConditionedRunnerTest(unittest.TestCase):
    def write_scene(self, root: Path) -> tuple[Path, str]:
        scene = root / "scene.vestra"
        decoded = scene / "decoded"
        chunks = scene / "chunks"
        decoded.mkdir(parents=True)
        chunks.mkdir()
        frames = []
        for index in range(5):
            name = f"frame-{index:06d}.ppm"
            payload = b"P6\n2 2\n255\n" + bytes([index, 0, 0]) * 4
            (decoded / name).write_bytes(payload)
            frames.append({"frame_index": index, "file_name": name, "sha256": digest(payload)})
        raster_fingerprint = "r" * 64
        raster_hash = "a" * 64
        pose_hash = "b" * 64
        (chunks / f"raster-{raster_hash}.json").write_text(json.dumps({
            "schema": "vestra.raster/v1", "raster_fingerprint": raster_fingerprint, "frames": frames,
        }))
        pose_frames = [{
            "frame_index": index, "image_name": frames[index]["file_name"], "registered": True,
            "world_to_camera": [1, 0, 0, -index, 0, 1, 0, 0, 0, 0, 1, 0],
        } for index in range(5)]
        (chunks / f"pose-{pose_hash}.json").write_text(json.dumps({
            "schema": "vestra.pose-solution/v1", "raster_fingerprint": raster_fingerprint,
            "frames": pose_frames,
            "global_trajectory": {
                "camera_models": [{"camera_id": 1, "model": "SIMPLE_RADIAL", "width": 504, "height": 336, "parameters": [300, 252, 168, 0]}],
                "frame_camera_ids": {str(index): 1 for index in range(5)}, "tracks": [],
            },
        }))
        (scene / "manifest.json").write_text(json.dumps({"raster_manifest_hash": raster_hash}))
        return scene, pose_hash

    def test_validate_only_binds_rasters_and_writes_deterministic_overlap_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, pose_hash = self.write_scene(root)
            output = root / "artifact"
            subprocess.run([
                sys.executable, str(RUNNER), "--scene", str(scene), "--pose-solution", pose_hash,
                "--output", str(output), "--batch-size", "3", "--overlap", "1", "--pixel-stride", "3", "--validate-only",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["schema"], "vestra.da3-pose-conditioned/v1")
            self.assertEqual(manifest["pixel_stride"], 3)
            self.assertEqual([batch["frames"] for batch in manifest["batches"]], [[0, 1, 2], [2, 3, 4]])
            # The 504×336 pose camera is rescaled into the decoded 2×2 PPM
            # evidence before any GPU inference receives it.
            self.assertAlmostEqual(SIDECAR.read_inputs(scene, pose_hash)[1][0].intrinsics[0], 300 * 2 / 504)

    def test_validate_only_rejects_raster_tampering_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, pose_hash = self.write_scene(root)
            (scene / "decoded" / "frame-000001.ppm").write_bytes(b"tampered")
            result = subprocess.run([
                sys.executable, str(RUNNER), "--scene", str(scene), "--pose-solution", pose_hash,
                "--output", str(root / "artifact"), "--validate-only",
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decoded raster hash mismatch", result.stderr)

    def test_ply_emits_each_overlapped_frame_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = np.ones((2, 2, 2), dtype=np.float32)
            conf = np.ones_like(depth)
            rgb = np.full((2, 2, 2, 3), 127, dtype=np.uint8)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
            intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 1.0
            intrinsics[:, 0, 2] = intrinsics[:, 1, 2] = 0.5
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
            first, second = root / "first.npz", root / "second.npz"
            np.savez_compressed(first, depth=depth, conf=conf, rgb=rgb, intrinsics=intrinsics, extrinsics=extrinsics, frame_indices=np.array([0, 1]))
            np.savez_compressed(second, depth=depth, conf=conf, rgb=rgb, intrinsics=intrinsics, extrinsics=extrinsics, frame_indices=np.array([1, 2]))
            output = root / "world.ply"
            emitted = SIDECAR.write_ply(output, [first, second], 0.0, 1, np)
            self.assertEqual(emitted, 12)
            header = output.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
            self.assertIn("element vertex 12", header)

    def test_pose44_preserves_da3_documented_3x4_w2c(self):
        matrices = np.array([[
            [1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4],
        ]], dtype=np.float32)
        pose = SIDECAR.pose44(matrices, 1, np)
        np.testing.assert_array_equal(pose[0, :3, :], matrices[0])
        np.testing.assert_array_equal(pose[0, 3, :], np.array([0, 0, 0, 1], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
