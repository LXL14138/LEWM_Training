"""Aim-offset regressions using real actions, not manufactured object poses.

Small temporary recorders below omit image encoding only; their physical states,
executed actions, labels, and NPZ/metadata writer are the production code paths.
"""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import auto_grasp
import collect_grasp_trajs as collection
from auto_grasp import AutoGraspController
from scene_config import BLOCKS
from three_block_scene import ThreeBlockScene


def append_numeric_state(recorder):
    """Test-only suppression of image encoding, never a changed state/label."""
    recorder.states.append(collection.capture_state(recorder.env))


class MisgraspParameterTests(unittest.TestCase):
    def test_invalid_offsets_rejected_before_environment_reset(self):
        env = Mock()
        for offset in (0.06001, -0.06001, float("nan"), float("inf"), -float("inf")):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                AutoGraspController(env, grasp_offset_x_m=offset)
        env.reset.assert_not_called()

    def test_invalid_close_timeout_rejected_before_environment_reset(self):
        env = Mock()
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                AutoGraspController(env, close_timeout_s=timeout)
        env.reset.assert_not_called()

    def test_collector_invalid_perturbations_create_no_output(self):
        invalid = [
            {"grasp_offset_x_m": 0.061},
            {"grasp_offset_x_m": float("nan")},
            {"grasp_offset_x_m": float("inf")},
            {"close_timeout_s": 0},
            {"close_timeout_s": -1},
            {"close_timeout_s": float("nan")},
            {"close_timeout_s": float("inf")},
            {"grasp_offset_x_range": [0.01]},
            {"grasp_offset_x_range": [0.01, 0.02, 0.03]},
            {"grasp_offset_x_range": [[0.01, 0.02]]},
            {"grasp_offset_x_range": [0.02, 0.01]},
            {"grasp_offset_x_range": [0.02, 0.02]},
            {"grasp_offset_x_range": [-0.061, 0.02]},
            {"grasp_offset_x_range": [0, 0.061]},
            {"grasp_offset_x_range": [float("nan"), 0.02]},
            {"grasp_offset_x_range": [-0.02, float("inf")]},
            {"grasp_offset_x_m": 0.01, "grasp_offset_x_range": [0.02, 0.04]},
        ]
        with tempfile.TemporaryDirectory() as root, patch.object(collection, "CollectedScene") as scene:
            output = Path(root) / "must_not_exist"
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    collection.collect_dataset(output, n_trajectories=1, **kwargs)
                self.assertFalse(output.exists())
            scene.assert_not_called()

    def test_auto_cli_invalid_parameters_create_no_snapshot(self):
        with tempfile.TemporaryDirectory() as root, patch.object(auto_grasp, "ThreeBlockScene") as scene:
            output = Path(root) / "must_not_exist"
            for extra in (["--grasp-offset-x", "0.061"], ["--close-timeout", "0"]):
                argv = ["auto_grasp.py", "--headless", "--snapshot-dir", str(output), *extra]
                with self.subTest(extra=extra), patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        auto_grasp.main()
                    self.assertEqual(raised.exception.code, 2)
                self.assertFalse(output.exists())
            scene.assert_not_called()


class MisgraspPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ThreeBlockScene()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def use_historical_cube(self):
        # The old +4-cm contact/miss outcomes are geometry-specific regressions.
        blocks = copy.deepcopy(BLOCKS)
        blocks[0]["size"] = [0.04, 0.04, 0.04]
        blocks[1]["xy"] = [0.425, -0.09]
        blocks[2]["xy"] = [0.425, 0.09]
        self.env = ThreeBlockScene(blocks=blocks)
        self.env.arm_start = np.array([0.30, 0.0, 0.25])
        self.addCleanup(self.env.close)

    def run_controller(self, controller, maximum=1000):
        rows = []
        for _ in range(maximum):
            if controller.finished:
                break
            before = float(self.env.data.time)
            info = controller.step()
            self.assertAlmostEqual(float(self.env.data.time) - before,
                                   self.env.control_timestep(), places=9)
            self.assertEqual(controller.last_action[3], 0)
            self.assertTrue(np.all(np.abs(controller.last_action) <= 1))
            self.assertFalse(info["unsafe"])
            self.assertFalse(info["episode_failed"])
            rows.append({"time": float(self.env.data.time), "phase": controller.phase,
                         "task": dict(info["task"]), "action": controller.last_action.copy(),
                         "qpos": self.env.data.qpos.copy()})
        self.assertTrue(controller.finished, "Attempt exceeded bounded test steps")
        return rows

    def test_offset_changes_waypoints_only_not_initial_physics(self):
        original = AutoGraspController(self.env)
        qpos = self.env.data.qpos.copy()
        qvel = self.env.data.qvel.copy()
        ctrl = self.env.data.ctrl.copy()
        positions = np.array([self.env.data.body(f"object_{i}").xipos.copy() for i in range(3)])
        waypoints = {name: xyz.copy() for name, xyz in original.waypoints.items()}
        fixed_yaw = self.env.arm_fixed_yaw_deg
        for offset in (0.04, -0.04, 0.06, -0.06):
            with self.subTest(offset=offset):
                controller = AutoGraspController(self.env, grasp_offset_x_m=offset)
                np.testing.assert_array_equal(self.env.data.qpos, qpos)
                np.testing.assert_array_equal(self.env.data.qvel, qvel)
                np.testing.assert_array_equal(self.env.data.ctrl, ctrl)
                np.testing.assert_array_equal(
                    [self.env.data.body(f"object_{i}").xipos for i in range(3)], positions)
                self.assertEqual(self.env.arm_fixed_yaw_deg, fixed_yaw)
                np.testing.assert_array_equal(controller.waypoints["SETTLE"], waypoints["SETTLE"])
                for name in ("APPROACH", "DESCEND", "CLOSE", "LIFT", "HOLD"):
                    np.testing.assert_allclose(controller.waypoints[name] - waypoints[name],
                                               [offset, 0, 0], rtol=0, atol=1e-15)
                np.testing.assert_array_equal(controller.original_target_xyz, positions[0])
                np.testing.assert_allclose(controller.grasp_target_xyz,
                                           positions[0] + [offset, 0, 0], rtol=0, atol=1e-15)

    def test_explicit_zero_matches_default_actions_and_states(self):
        default = AutoGraspController(self.env)
        default_rows = self.run_controller(default)
        zero = AutoGraspController(self.env, grasp_offset_x_m=0.0, close_timeout_s=None)
        zero_rows = self.run_controller(zero)
        self.assertEqual(default.phase, "SUCCESS")
        self.assertEqual(zero.phase, "SUCCESS")
        self.assertEqual(len(default_rows), len(zero_rows))
        for expected, actual in zip(default_rows, zero_rows):
            self.assertEqual(expected["phase"], actual["phase"])
            np.testing.assert_array_equal(expected["action"], actual["action"])
            np.testing.assert_array_equal(expected["qpos"], actual["qpos"])

    def test_reset_reapplies_offset_without_accumulation(self):
        controller = AutoGraspController(self.env, grasp_offset_x_m=0.04)
        expected = {key: value.copy() for key, value in controller.waypoints.items()}
        for _ in range(3):
            controller.step()
            controller.reset()
            for key in expected:
                np.testing.assert_array_equal(controller.waypoints[key], expected[key])
            self.assertEqual(controller.report()["grasp_offset_x_m"], 0.04)

    def test_one_cm_offset_can_still_succeed_without_relabeling(self):
        controller = AutoGraspController(self.env, grasp_offset_x_m=0.01, close_timeout_s=3)
        rows = self.run_controller(controller)
        self.assertEqual(controller.phase, "SUCCESS", controller.report())
        self.assertTrue(rows[-1]["task"]["task_success"])
        self.assertFalse(rows[0]["task"]["task_success"])

    def test_historical_cube_four_cm_offset_real_safe_miss(self):
        self.use_historical_cube()
        controller = AutoGraspController(self.env, grasp_offset_x_m=0.04, close_timeout_s=3)
        rows = self.run_controller(controller)
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("No stable two-finger grasp", controller.failure_reason)
        self.assertFalse(any(row["task"]["grasped"] for row in rows))
        self.assertFalse(any(row["task"]["task_success"] for row in rows))
        self.assertNotIn("LIFT", controller.phase_history)
        for guard in self.env.safety_report()["guards"]:
            self.assertLess(guard["displacement_xy_m"], 1e-8)
            self.assertLess(guard["tilt_deg"], 1e-6)

    def test_historical_cube_short_close_timeout_does_not_shorten_other_phases(self):
        self.use_historical_cube()
        controller = AutoGraspController(self.env, grasp_offset_x_m=0.04, close_timeout_s=0.15)
        rows = self.run_controller(controller)
        self.assertIn("APPROACH", controller.phase_history)
        self.assertIn("DESCEND", controller.phase_history)
        self.assertIn("CLOSE", controller.phase_history)
        first_close = next(row["time"] for row in rows if row["phase"] == "CLOSE")
        close_elapsed = rows[-1]["time"] - first_close
        self.assertGreaterEqual(close_elapsed, 0.15 - 1e-8)
        self.assertLessEqual(close_elapsed, 0.20 + 1e-8)
        self.assertIn("No stable two-finger grasp", controller.failure_reason)

    def test_close_timeout_remains_bounded_by_phase_timeout(self):
        controller = AutoGraspController(self.env, phase_timeout_s=0.20, close_timeout_s=2)
        # Phase-only fixture: an open gripper far from the cube cannot grasp.
        # No object positions, detector readings or safety labels are modified.
        controller._transition("CLOSE")
        rows = self.run_controller(controller)
        self.assertGreaterEqual(rows[-1]["time"], 0.20 - 1e-8)
        self.assertLessEqual(rows[-1]["time"], 0.25 + 1e-8)
        self.assertIn("No stable two-finger grasp", controller.failure_reason)


class MisgraspCollectionTests(unittest.TestCase):
    def test_offset_rng_is_reproducible_and_independent_of_random_starts(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(collection.EpisodeRecorder, "_append_state", append_numeric_state), \
                contextlib.redirect_stdout(io.StringIO()):
            manifests = []
            for name, kwargs in (
                ("normal", {}),
                ("offset_a", {"grasp_offset_x_range": [-0.04, 0.04]}),
                ("offset_b", {"grasp_offset_x_range": [-0.04, 0.04]}),
            ):
                manifests.append(collection.collect_dataset(
                    Path(root) / name, n_trajectories=3, seed=714,
                    max_steps=1, close_timeout_s=3, **kwargs))
            normal, first, second = manifests
            offsets = []
            for index in range(3):
                baseline = normal["episodes"][index]
                row = first["episodes"][index]
                repeat = second["episodes"][index]
                self.assertEqual(baseline["sampling_seed"], row["sampling_seed"])
                self.assertEqual(baseline["requested_start_xyz"], row["requested_start_xyz"])
                self.assertEqual(row["requested_start_xyz"], repeat["requested_start_xyz"])
                self.assertEqual(row["grasp_offset_x_m"], repeat["grasp_offset_x_m"])
                self.assertGreaterEqual(row["grasp_offset_x_m"], -0.04)
                self.assertLess(row["grasp_offset_x_m"], 0.04)
                offsets.append(row["grasp_offset_x_m"])
                with (Path(root) / "offset_a" / row["path"] / "metadata.json").open() as handle:
                    metadata = json.load(handle)
                self.assertEqual(metadata["controller"]["grasp_offset_x_m"], row["grasp_offset_x_m"])
                self.assertEqual(metadata["controller"]["close_timeout_s"], 3)
                perturbation = metadata["grasp_perturbation"]
                self.assertFalse(perturbation["labels_overridden"])
                self.assertFalse(perturbation["objects_teleported"])
                np.testing.assert_allclose(
                    np.array(perturbation["grasp_target_xyz_m"]) - perturbation["original_target_xyz_m"],
                    [row["grasp_offset_x_m"], 0, 0], rtol=0, atol=1e-15)
            self.assertEqual(len(set(offsets)), 3)

    def test_final_failed_grasp_does_not_overwrite_earlier_success_labels(self):
        # Keep the actual historical cube slip case; taller targets need not
        # share that physical outcome for the same -2.5-cm aim offset.
        blocks = copy.deepcopy(BLOCKS)
        blocks[0]["size"] = [0.04, 0.04, 0.04]
        blocks[1]["xy"] = [0.425, -0.09]
        blocks[2]["xy"] = [0.425, 0.09]
        env = collection.CollectedScene(blocks=blocks)
        env.arm_start = np.array([0.30, 0.0, 0.25])
        try:
            controller = AutoGraspController(env, grasp_offset_x_m=-0.025)
            with tempfile.TemporaryDirectory() as root, \
                    patch.object(collection.EpisodeRecorder, "_append_state", append_numeric_state):
                recorder = collection.EpisodeRecorder(Path(root) / "episode", env, {})
                for _ in range(1000):
                    if controller.finished:
                        break
                    phase_before = controller.phase
                    controller.step()
                    self.assertIsNotNone(env.last_transition)
                    recorder.append(env.last_transition, phase_before)
                self.assertEqual(controller.phase, "FAILED")
                self.assertIn("Lost two-finger grasp", controller.failure_reason)
                metadata = recorder.finish(controller.report())
                with np.load(Path(root) / "episode" / "trajectory.npz", allow_pickle=False) as data:
                    self.assertTrue(data["task_success"].any())
                    self.assertFalse(data["task_success"][-1])
                    self.assertFalse(data["unsafe"].any())
                    self.assertFalse(data["episode_failed"].any())
                    self.assertEqual(len(data["time"]), len(data["actions"]) + 1)
                    self.assertTrue(np.all(data["actions"][:, 3] == 0))
                    self.assertEqual(metadata["counts"]["task_success"], int(data["task_success"].sum()))
                self.assertEqual(metadata["outcome"]["phase"], "FAILED")
                self.assertFalse(metadata["outcome"]["task"]["task_success"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
