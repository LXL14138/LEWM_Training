"""Opt-in simulation lift-probe tests, including actual offset and centred grasps.

Fault fixtures below exercise stopping rules, not demonstrations of collisions.
No data are collected, no robot/object poses are changed during real-run tests.
"""

import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from auto_grasp import AutoGraspController
from scene_config import BLOCKS
from three_block_scene import ThreeBlockScene


class ProbeLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ThreeBlockScene()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def make_controller(self, *, historical_cube=False, **kwargs):
        if historical_cube:
            # Explicit old geometry for reproducible, geometry-specific miss
            # outcomes; other tests continue to exercise today's tall target.
            blocks = copy.deepcopy(BLOCKS)
            blocks[0]["size"] = [0.04, 0.04, 0.04]
            blocks[1]["xy"] = [0.425, -0.09]
            blocks[2]["xy"] = [0.425, 0.09]
            self.env = ThreeBlockScene(blocks=blocks)
            self.env.arm_start = np.array([0.30, 0.0, 0.25])
            self.addCleanup(self.env.close)
        options = dict(grasp_offset_x_m=0.04, close_timeout_s=3.0, probe_lift=True)
        options.update(kwargs)
        return AutoGraspController(self.env, **options)

    def run_to_end(self, controller):
        for _ in range(1500):
            if controller.finished:
                return controller.report()
            controller.step()
        self.fail("Lift probe exceeded the bounded test step count")

    def test_historical_cube_four_cm_probe_lifts_without_two_pad_contact(self):
        controller = self.make_controller(historical_cube=True)
        expected_samples = []
        ungrasped_lift_steps = 0
        for _ in range(1500):
            if controller.finished:
                break
            phase_before = controller.phase
            n_before = len(controller.probe_samples)
            previous_time = float(self.env.data.time)
            controller.step()
            self.assertAlmostEqual(float(self.env.data.time) - previous_time, 0.05, places=10)
            self.assertTrue(np.isfinite(controller.last_action).all())
            self.assertTrue(np.all(np.abs(controller.last_action) <= 1.0))
            self.assertEqual(controller.last_action[3], 0.0)
            physical = self.env.unnormalize_action(controller.last_action)
            self.assertLessEqual(np.linalg.norm(physical[:3]), 0.02 + 1e-9)
            if phase_before in ("CLOSE", "PROBE_LIFT", "PROBE_HOLD"):
                self.assertEqual(physical[4], 1.0)
            if phase_before == "PROBE_LIFT" and not controller.last_info["task"]["grasped"]:
                ungrasped_lift_steps += 1
                self.assertNotEqual(controller.phase, "FAILED")
            if len(controller.probe_samples) > n_before:
                self.assertEqual(len(controller.probe_samples), n_before + 1)
                sample = controller.probe_samples[-1]
                task = controller.last_info["task"]
                expected_phase = "PROBE_LIFT" if phase_before == "CLOSE" else phase_before
                self.assertEqual(sample["phase"], expected_phase)
                self.assertEqual(sample["time_s"], float(self.env.data.time))
                self.assertEqual(sample["target_com_z_m"], float(task["target_height_m"]))
                self.assertEqual(sample["pinch_z_m"], float(self.env.data.site_xpos[self.env._pinch_site_id][2]))
                for key in ("grasped", "left_contact", "right_contact", "task_success"):
                    self.assertEqual(sample[key], bool(task[key]))
                for key in ("left_normal_force_n", "right_normal_force_n"):
                    self.assertEqual(sample[key], float(task[key]))
                self.assertEqual(sample["unsafe"], bool(controller.last_info["unsafe"]))
                self.assertEqual(sample["episode_failed"], bool(controller.last_info["episode_failed"]))
                expected_samples.append(dict(sample))
        report = controller.report()
        self.assertEqual(controller.phase, "PROBE_DONE", report)
        self.assertEqual(controller.phase_history, [
            "SETTLE", "APPROACH", "DESCEND", "CLOSE", "PROBE_LIFT", "PROBE_HOLD", "PROBE_DONE",
        ])
        self.assertGreater(ungrasped_lift_steps, 20)
        self.assertIsNone(report["failure_reason"])
        self.assertFalse(report["task"]["task_success"])
        self.assertFalse(report["safety"]["unsafe"])
        self.assertFalse(report["task"]["episode_failed"])
        self.assertFalse(report["training_data_collected"])
        probe = report["probe_lift"]
        self.assertEqual(probe["samples"], expected_samples)
        self.assertGreater(probe["final_pinch_z_m"] - probe["initial_pinch_z_m"], 0.10)
        self.assertLess(probe["peak_target_rise_m"], 0.02)
        self.assertFalse(probe["ever_dual_contact"])
        self.assertFalse(probe["ever_task_success"])
        self.assertEqual(probe["initial_target_com_z_m"], expected_samples[0]["target_com_z_m"])
        self.assertEqual(probe["final_target_com_z_m"], expected_samples[-1]["target_com_z_m"])
        self.assertEqual(probe["peak_target_com_z_m"], max(row["target_com_z_m"] for row in expected_samples))
        self.assertEqual(probe["peak_target_rise_m"], probe["peak_target_com_z_m"] - probe["initial_target_com_z_m"])
        np.testing.assert_allclose(np.diff([row["time_s"] for row in expected_samples]), 0.05, atol=1e-10)
        hold_samples = [row for row in expected_samples if row["phase"] == "PROBE_HOLD"]
        self.assertEqual(len(hold_samples), 20)

    def test_historical_cube_four_cm_fails_at_close_without_lifting(self):
        controller = self.make_controller(probe_lift=False, historical_cube=True)
        report = self.run_to_end(controller)
        self.assertEqual(report["phase"], "FAILED")
        self.assertIn("No stable two-finger grasp", report["failure_reason"])
        self.assertNotIn("PROBE_LIFT", report["phase_history"])
        self.assertNotIn("LIFT", report["phase_history"])
        self.assertFalse(report["probe_lift"]["enabled"])
        self.assertEqual(report["probe_lift"]["samples"], [])
        self.assertFalse(report["task"]["task_success"])
        self.assertFalse(report["task"]["episode_failed"])

    def test_tall_target_four_cm_offset_stops_during_blocked_descent(self):
        # At the same +4-cm offset the new taller target obstructs DESCEND.
        # A lift probe must not bypass that earlier phase's timeout or invent
        # probe measurements for an action sequence that never reached CLOSE.
        for probe_lift in (False, True):
            with self.subTest(probe_lift=probe_lift):
                controller = self.make_controller(probe_lift=probe_lift)
                report = self.run_to_end(controller)
                self.assertEqual(report["phase"], "FAILED", report)
                self.assertEqual(report["phase_history"], [
                    "SETTLE", "APPROACH", "DESCEND", "FAILED",
                ])
                self.assertIn("DESCEND timed out", report["failure_reason"])
                self.assertEqual(report["probe_lift"]["samples"], [])
                self.assertFalse(report["task"]["task_success"])
                self.assertFalse(report["task"]["episode_failed"])
                self.assertFalse(report["safety"]["unsafe"])

    def test_real_centred_probe_can_succeed_but_ends_neutrally(self):
        controller = self.make_controller(grasp_offset_x_m=0.0)
        report = self.run_to_end(controller)
        self.assertEqual(report["phase"], "PROBE_DONE", report)
        self.assertTrue(report["task"]["task_success"])
        self.assertTrue(report["task"]["grasped"])
        self.assertTrue(report["probe_lift"]["ever_dual_contact"])
        self.assertTrue(report["probe_lift"]["ever_task_success"])
        # Completion is absolute COM height, not a geometry-independent 10-cm
        # rise. The taller block starts at 5 cm, with the same 10-cm threshold.
        required_rise = (report["task"]["height_threshold_m"]
                         - report["probe_lift"]["initial_target_com_z_m"])
        self.assertGreater(report["probe_lift"]["peak_target_rise_m"], required_rise)
        self.assertFalse(report["task"]["episode_failed"])
        self.assertNotIn("SUCCESS", report["phase_history"])

    def test_probe_configuration_does_not_change_reset_physics_or_old_waypoints(self):
        normal = self.make_controller(probe_lift=False)
        qpos, qvel = self.env.data.qpos.copy(), self.env.data.qvel.copy()
        waypoints = {key: value.copy() for key, value in normal.waypoints.items()}
        probe = self.make_controller()
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.data.qvel, qvel)
        for key, value in waypoints.items():
            np.testing.assert_array_equal(probe.waypoints[key], value)
        np.testing.assert_array_equal(probe.waypoints["PROBE_LIFT"], probe.waypoints["LIFT"])
        np.testing.assert_array_equal(probe.waypoints["PROBE_HOLD"], probe.waypoints["HOLD"])

    def test_done_freezes_and_reset_clears_probe_measurements(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")  # Explicit state-machine fixture.
        controller.step()
        self.assertTrue(controller.probe_samples)
        controller._transition("PROBE_DONE")
        before = (float(self.env.data.time), self.env.data.qpos.copy(), controller.control_steps)
        with patch.object(self.env, "step", wraps=self.env.step) as step:
            for _ in range(4):
                controller.step()
            step.assert_not_called()
        self.assertTrue(controller.finished)
        self.assertEqual(float(self.env.data.time), before[0])
        np.testing.assert_array_equal(self.env.data.qpos, before[1])
        self.assertEqual(controller.control_steps, before[2])
        controller.reset()
        self.assertFalse(controller.finished)
        self.assertEqual(controller.phase, "SETTLE")
        self.assertEqual(controller.probe_samples, [])
        self.assertEqual(controller.control_steps, 0)
        self.assertTrue(controller.probe_lift)
        self.assertFalse(controller.last_info["task"]["task_success"])

    def test_invalid_boolean_rejected_before_reset(self):
        for value in (None, 0, 1, "true", [], np.bool_(True)):
            with self.subTest(value=value):
                with patch.object(self.env, "reset", wraps=self.env.reset) as reset:
                    with self.assertRaisesRegex(ValueError, "boolean"):
                        AutoGraspController(self.env, probe_lift=value)
                    reset.assert_not_called()

    def test_current_guard_hazard_stops_before_probe_action(self):
        self.addCleanup(setattr, self.env, "displacement_check_enabled", self.env.displacement_check_enabled)
        self.env.displacement_check_enabled = True
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        # Artificial guard displacement is only a stopping-rule fixture.
        self.env.data.joint("object_joint_1").qpos[0] += 0.02
        self.env.post_step()
        with patch.object(self.env, "step", wraps=self.env.step) as step:
            info = controller.step()
            step.assert_not_called()
        self.assertEqual(controller.phase, "FAILED")
        self.assertTrue(info["unsafe"])
        self.assertTrue(info["episode_failed"])
        self.assertEqual(controller.probe_samples, [])

    def test_latched_failure_stops_even_if_current_reading_safe(self):
        controller = self.make_controller()
        controller._transition("PROBE_HOLD")
        info = dict(self.env.get_step_info(), unsafe=False, episode_failed=True)
        with patch.object(self.env, "get_step_info", return_value=info):
            with patch.object(self.env, "step", wraps=self.env.step) as step:
                controller.step()
                step.assert_not_called()
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("Guard safety", controller.failure_reason)

    def test_post_action_hazard_is_recorded_and_stops(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        real_step = self.env.step

        def unsafe_reading(action):
            observation, reward, terminated, truncated, info = real_step(action)
            return observation, reward, terminated, truncated, dict(info, unsafe=True, episode_failed=True)

        with patch.object(self.env, "step", side_effect=unsafe_reading):
            controller.step()
        self.assertEqual(controller.phase, "FAILED")
        self.assertEqual(controller.control_steps, 1)
        self.assertTrue(controller.probe_samples[-1]["unsafe"])
        self.assertTrue(controller.probe_samples[-1]["episode_failed"])

    def test_finite_control_error_stops_without_fake_sample(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        with patch.object(self.env, "step", side_effect=RuntimeError("synthetic probe IK failure")):
            controller.step()
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("Robot control error", controller.failure_reason)
        self.assertEqual(controller.control_steps, 0)
        self.assertEqual(controller.probe_samples, [])

    def test_environment_termination_stops_probe(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        real_step = self.env.step

        def terminated_reading(action):
            observation, reward, _, truncated, info = real_step(action)
            return observation, reward, True, truncated, info

        with patch.object(self.env, "step", side_effect=terminated_reading):
            controller.step()
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("Environment stopped", controller.failure_reason)

    def test_probe_lift_and_hold_keep_phase_timeouts(self):
        for phase in ("PROBE_LIFT", "PROBE_HOLD"):
            with self.subTest(phase=phase):
                controller = self.make_controller(phase_timeout_s=0.01)
                controller._transition(phase)
                controller.step()
                self.assertEqual(controller.phase, "FAILED")
                self.assertIn(phase + " timed out", controller.failure_reason)
                self.assertEqual(len(controller.probe_samples), 1)

    def test_close_uses_shorter_timeout_before_probe(self):
        for close_timeout, phase_timeout, steps in ((0.04, 15.0, 1), (3.0, 0.09, 2)):
            with self.subTest(close_timeout=close_timeout, phase_timeout=phase_timeout):
                controller = self.make_controller(close_timeout_s=close_timeout, phase_timeout_s=phase_timeout)
                controller._transition("CLOSE")  # Still-open distant gripper fixture.
                for _ in range(steps - 1):
                    controller.step()
                    self.assertEqual(controller.phase, "CLOSE")
                controller.step()
                self.assertEqual(controller.phase, "PROBE_LIFT")
                self.assertFalse(controller.last_info["task"]["grasped"])
                self.assertEqual(len(controller.probe_samples), 1)

    def test_report_samples_are_copies_not_internal_state(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        controller.step()
        report = controller.report()
        report["probe_lift"]["samples"][0]["target_com_z_m"] = -123
        report["probe_lift"]["samples"].clear()
        self.assertEqual(len(controller.probe_samples), 1)
        self.assertGreater(controller.probe_samples[0]["target_com_z_m"], 0)

    def test_nonfinite_action_stops_probe_before_physics(self):
        controller = self.make_controller()
        controller._transition("PROBE_LIFT")
        with patch.object(self.env, "normalize_action", return_value=np.full(5, np.nan)):
            with patch.object(self.env, "step", wraps=self.env.step) as step:
                controller.step()
                step.assert_not_called()
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("Invalid robot state", controller.failure_reason)
        self.assertEqual(controller.probe_samples, [])


if __name__ == "__main__":
    unittest.main()
