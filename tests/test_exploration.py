"""Focused exploration checks; synthetic faults are never demonstration data."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from collect_exploration_trajs import collect_exploration, ExplorationRecorder
from collect_grasp_trajs import episode_metadata
from exploration import ExplorationScene, ExplorationController, MODES, waypoint_reachable
from inspect_grasp_dataset import validate_dataset, validate_episode
from random_starts import validate_start


class ExplorationTests(unittest.TestCase):
    def make_env(self):
        env = ExplorationScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def test_ik_check_does_not_change_simulator_or_goal(self):
        env = self.make_env()
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        self.assertTrue(waypoint_reachable(env, [.425, 0, .19]))
        self.assertFalse(waypoint_reachable(env, [3, 0, .19]))
        np.testing.assert_array_equal(qpos, env.data.qpos)
        np.testing.assert_array_equal(qvel, env.data.qvel)
        self.assertEqual(env.data.time, 0)

    def test_all_modes_bounded_no_reset_and_valid_actions(self):
        env = self.make_env()
        for mode in MODES:
            with self.subTest(mode=mode):
                env.reset(seed=0)
                self.assertTrue(validate_start(env, env.arm_start)["valid"])
                controller = ExplorationController(env, np.random.default_rng(7), mode, max_steps=18)
                goal = env.hover_goal.copy()
                with patch.object(env, "reset", wraps=env.reset) as reset:
                    while not controller.finished:
                        before = env.data.time
                        phase = controller.step()
                        if phase is not None:
                            self.assertAlmostEqual(env.data.time - before, .05)
                            action = env.last_transition["action"]
                            self.assertTrue(np.all(np.abs(action) <= 1))
                            self.assertEqual(action[3], 0)
                        np.testing.assert_array_equal(env.hover_goal, goal)
                    reset.assert_not_called()
                self.assertLessEqual(controller.steps, 18)
                self.assertEqual(controller.report()["phase"], "DONE")
                final_qpos = env.data.qpos.copy()
                self.assertIsNone(controller.step())
                np.testing.assert_array_equal(final_qpos, env.data.qpos)

    def test_goal_cannot_change_inside_episode(self):
        env = self.make_env()
        env.step(np.array([0, 0, 0, 0, -1]))
        with self.assertRaisesRegex(ValueError, "inside a trajectory"):
            env.set_hover_goal([.425, 0, .18])

    def test_unsafe_priority_and_settling_not_immediate_freeze(self):
        env = self.make_env()
        controller = ExplorationController(env, np.random.default_rng(1), "carry", max_steps=40)
        # Artificial pose for stopping semantics, not a generated collision.
        joint = env.data.joint("object_joint_1")
        half = np.deg2rad(40) / 2
        joint.qpos[3:] = [np.cos(half), np.sin(half), 0, 0]
        env.post_step()
        self.assertEqual(env.hover_report["class_label"], 1)
        self.assertFalse(env.hover_report["hover_success"])
        phase = controller.step()
        self.assertEqual(phase, "EVENT_HOLD")
        self.assertGreater(env.data.time, 0)
        while not controller.finished:
            controller.step()
        self.assertIn("guard_hazard", controller.stop_reason)
        self.assertGreaterEqual(env.data.time, 1 - 1e-8)

    def test_disabled_shift_does_not_end_exploration(self):
        env = self.make_env()
        controller = ExplorationController(env, np.random.default_rng(1), "free", max_steps=30)
        env.data.joint("object_joint_1").qpos[0] += .06
        env.post_step()
        controller.step()
        self.assertFalse(env.task_report()["episode_failed"])
        self.assertNotEqual(controller.phase, "EVENT_HOLD")

    def test_short_recording_and_schema_three_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            manifest = collect_exploration(folder, n_trajectories=4, seed=1, max_seconds=.1)
            self.assertEqual(set(manifest["mode_schedule"]), set(MODES))
            self.assertEqual(manifest["summary"]["total_actions"], 8)
            self.assertEqual(manifest["summary"]["total_states"], 12)
            result = validate_dataset(folder)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["hover_success_episode_count"], 0)
            self.assertNotIn("success_count", result)
            meta = json.loads((folder / "episode_0000/metadata.json").read_text())
            self.assertFalse(meta["displacement_check_enabled"])
            self.assertFalse(meta["image"]["shadows_enabled"])
            self.assertEqual(meta["outcome"]["phase"], "DONE")
            self.assertEqual(meta["dataset_kind"], "guided_exploration_v1")

    def test_manifest_cannot_misidentify_exploration_as_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            collect_exploration(root, n_trajectories=1, max_seconds=.1)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest))
            result = validate_dataset(root)
            self.assertFalse(result["valid"])
            self.assertIn("schema versions disagree", " ".join(result["errors"]))

    def test_seed_reproduces_actions_states_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a", "b"):
                collect_exploration(root / name, n_trajectories=1, seed=23, mode="free", max_seconds=1)
            a, b = root / "a/episode_0000", root / "b/episode_0000"
            with np.load(a / "trajectory.npz") as first, np.load(b / "trajectory.npz") as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key], err_msg=key)
            self.assertEqual((a / "images/000020.png").read_bytes(), (b / "images/000020.png").read_bytes())

    def test_invalid_arguments_and_existing_directory_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kwargs in ({"seed": -1}, {"n_trajectories": 0}, {"max_seconds": float("nan")},
                           {"max_seconds": True}, {"mode": "bad"}, {"camera": "wrist_pixels"},
                           {"max_attempts": 0}, {"start_low": [0, 0, 0]}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    collect_exploration(root / "invalid", **kwargs)
                self.assertFalse((root / "invalid").exists())
            sentinel = root / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                collect_exploration(root, n_trajectories=1)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_physics_exception_marks_partial_data_invalid(self):
        original = ExplorationScene.step
        def broken_step(env, action):
            original(env, action)
            raise RuntimeError("injected exception after integration")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            with patch.object(ExplorationScene, "step", broken_step), self.assertRaisesRegex(RuntimeError, "injected"):
                collect_exploration(root, n_trajectories=1, max_seconds=.1)
            meta = json.loads((root / "episode_0000/metadata.json").read_text())
            self.assertEqual(meta["status"], "error")
            self.assertTrue((root / "episode_0000/partial_trajectory.npz").exists())
            self.assertFalse(validate_dataset(root)["valid"])

    def test_tampered_hover_labels_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            collect_exploration(root, n_trajectories=1, max_seconds=.1)
            episode = root / "episode_0000"
            with np.load(episode / "trajectory.npz") as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["class_label"][1] = 0
            np.savez_compressed(episode / "trajectory.npz", **arrays)
            result = validate_episode(episode)
            self.assertFalse(result["valid"])
            self.assertIn("class_label", " ".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
