"""Recorder regression tests using short real-physics attempts and fault fixtures."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from collect_grasp_trajs import (
    CollectedScene, EpisodeRecorder, collect_dataset, episode_metadata,
)
from random_starts import sample_valid_start
from three_block_scene import ThreeBlockScene


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grasp_recorder_test_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_short_failed_attempt_kept_with_exact_action_image_alignment(self):
        manifest = collect_dataset(self.root / "data", n_trajectories=1, seed=7, max_steps=3)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["summary"]["failed_episodes"], 1)
        folder = self.root / "data" / "episode_0000"
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["schema_version"], 2)
        self.assertFalse(metadata["displacement_check_enabled"])
        self.assertFalse(metadata["image"]["shadows_enabled"])
        self.assertEqual(metadata["safety_scope"], "guard_tilt_only")
        self.assertEqual(metadata["outcome"]["phase"], "FAILED")
        self.assertIn("step limit", metadata["outcome"]["failure_reason"])
        self.assertEqual(metadata["num_actions"], 3)
        self.assertEqual(metadata["num_states"], 4)
        with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
            self.assertEqual(data["actions"].shape, (3, 5))
            self.assertEqual(data["qpos"].shape[0], 4)
            np.testing.assert_allclose(data["time"], [0, 0.05, 0.1, 0.15], atol=1e-9)
            self.assertTrue(np.all(data["actions"][:, 3] == 0))
            self.assertEqual(data["unsafe"].dtype, np.bool_)
            self.assertFalse(data["unsafe"].any())
            self.assertFalse(data["episode_failed"].any())
            self.assertFalse(data["task_success"].any())
            self.assertTrue(np.all(data["action_phase"] == "SETTLE"))
        images = sorted((folder / "images").glob("*.png"))
        self.assertEqual([p.name for p in images], [f"{i:06d}.png" for i in range(4)])
        for path in images:
            with Image.open(path) as frame:
                self.assertEqual(frame.size, (640, 480))
                self.assertEqual(frame.mode, "RGB")

    def test_same_seed_reproduces_start_and_states(self):
        for name in ("first", "second"):
            collect_dataset(self.root / name, n_trajectories=1, seed=91, max_steps=2)
        first = self.root / "first" / "episode_0000"
        second = self.root / "second" / "episode_0000"
        with np.load(first / "trajectory.npz", allow_pickle=False) as a, np.load(second / "trajectory.npz", allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertEqual((first / "images/000000.png").read_bytes(), (second / "images/000000.png").read_bytes())

    def test_multiple_episodes_have_separate_initial_states(self):
        manifest = collect_dataset(self.root / "data", n_trajectories=2, seed=10, max_steps=1)
        starts = []
        for row in manifest["episodes"]:
            folder = self.root / "data" / row["path"]
            with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
                self.assertEqual(data["time"][0], 0)
                self.assertEqual(len(data["time"]), 2)
                starts.append(data["pinch_xyz"][0].copy())
        self.assertGreater(np.linalg.norm(starts[0] - starts[1]), 1e-4)

    def test_existing_output_is_never_overwritten(self):
        folder = self.root / "existing"
        folder.mkdir()
        sentinel = folder / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            collect_dataset(folder, n_trajectories=1)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(folder.iterdir()), [sentinel])

    def test_invalid_arguments_do_not_create_output(self):
        invalid = [
            {"n_trajectories": 0}, {"seed": -1}, {"max_steps": 0},
            {"max_start_attempts": 0}, {"speed_mps": float("nan")},
            {"phase_timeout_s": 0}, {"start_low": [0.1, 0, 0.25]},
            {"start_low": [0.33, -0.02, 0.24]}, {"camera": "bad_camera"},
        ]
        for index, kwargs in enumerate(invalid):
            with self.subTest(kwargs=kwargs):
                folder = self.root / str(index)
                with self.assertRaises(ValueError):
                    collect_dataset(folder, **kwargs)
                self.assertFalse(folder.exists())

    def test_partial_physics_exception_is_quarantined(self):
        def broken_step(env, action):
            ThreeBlockScene.step(env, action)
            raise RuntimeError("synthetic failure after physics")

        with patch.object(CollectedScene, "step", broken_step):
            with self.assertRaisesRegex(RuntimeError, "without a completed recorded action"):
                collect_dataset(self.root / "data", n_trajectories=1)
        manifest = json.loads((self.root / "data/manifest.json").read_text())
        folder = self.root / "data/episode_0000"
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(metadata["status"], "error")
        self.assertFalse((folder / "trajectory.npz").exists())
        self.assertEqual(len(list((folder / "images").glob("*.png"))), 1)

    def test_recorder_rejects_frozen_duplicate_and_nonzero_yaw(self):
        env = CollectedScene()
        try:
            start = sample_valid_start(env, np.random.default_rng(3))
            metadata = episode_metadata(env, start, 3, "front_pixels", .06, 15, 2000)
            recorder = EpisodeRecorder(self.root / "episode", env, metadata)
            transition = {"action": np.zeros(5), "physical_action": np.zeros(5),
                          "terminated": False, "truncated": False}
            with self.assertRaisesRegex(RuntimeError, "clock misalignment"):
                recorder.append(transition, "SETTLE")
            transition["action"][3] = 0.1
            with self.assertRaisesRegex(RuntimeError, "fixed yaw"):
                recorder.append(transition, "SETTLE")
            self.assertEqual(len(recorder.states), 1)
            self.assertEqual(len(recorder.actions), 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
