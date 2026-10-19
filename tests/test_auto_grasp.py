"""Automatic fixed-scene grasp regression tests; no training data collection.

The repeated runs reset the identical deterministic scene. They test repeatability,
not generalization to different layouts. Explicit fault fixtures below test the
controller's response; they are not demonstrations of a robot causing a fault.
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from auto_grasp import AutoGraspController, AutoGraspInput
from three_block_scene import ThreeBlockScene


EXPECTED_PHASES = [
    "SETTLE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "HOLD", "SUCCESS",
]


class AutoGraspControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the compiled model; every controller construction resets physics.
        cls.env = ThreeBlockScene()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.controller = AutoGraspController(self.env)

    def run_until(self, phases, controller=None):
        controller = self.controller if controller is None else controller
        for _ in range(1500):
            if controller.phase in phases or controller.finished:
                return controller.last_info
            controller.step()
        self.fail("Automatic sequence exceeded the bounded test step count")

    def assert_real_success(self, controller):
        report = controller.report()
        self.assertEqual(report["phase"], "SUCCESS", report)
        self.assertEqual(report["phase_history"], EXPECTED_PHASES)
        self.assertIsNone(report["failure_reason"])
        self.assertTrue(report["task"]["task_success"])
        self.assertTrue(report["task"]["grasped"])
        self.assertTrue(report["task"]["left_contact"])
        self.assertTrue(report["task"]["right_contact"])
        self.assertGreaterEqual(report["task"]["target_height_m"], 0.10)
        self.assertGreaterEqual(report["task"]["hold_seconds"], 1.0 - 1e-9)
        self.assertFalse(report["task"]["episode_failed"])
        self.assertFalse(report["safety"]["unsafe"])
        self.assertFalse(report["world_model_loaded"])
        self.assertFalse(report["training_data_collected"])

    def test_default_sequence_real_success_and_normalized_actions(self):
        executed_phases = []
        for _ in range(1500):
            if self.controller.finished:
                break
            previous_phase = self.controller.phase
            previous_reference = self.controller.reference_xyz.copy()
            self.controller.step()
            executed_phases.append(previous_phase)
            action = self.controller.last_action
            self.assertEqual(action.shape, (5,))
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.all(np.abs(action) <= 1.0))
            self.assertEqual(action[3], 0.0)
            physical = self.env.unnormalize_action(action)
            should_close = previous_phase in ("CLOSE", "LIFT", "HOLD")
            self.assertEqual(physical[4], 1.0 if should_close else -1.0)
            self.assertLessEqual(np.linalg.norm(physical[:3]), 0.02 + 1e-9)
            # On the unobstructed successful path the independent reference
            # respects the 0.06 m/s limit, not a 3-mm actual-error clamp.
            self.assertLessEqual(
                np.linalg.norm(self.controller.reference_xyz - previous_reference),
                self.controller.speed_mps * self.env.control_timestep() + 1e-9,
            )
            if previous_phase == "CLOSE" and self.controller.phase == "LIFT":
                self.assertTrue(self.controller.last_info["task"]["grasped"])
        self.assert_real_success(self.controller)
        self.assertEqual(list(dict.fromkeys(executed_phases)), EXPECTED_PHASES[:-1])

    def test_five_identical_scene_resets_repeat_success(self):
        # Deliberately no randomized starts: this is not a success-rate estimate.
        for attempt in range(5):
            with self.subTest(identical_scene_attempt=attempt + 1):
                self.controller.reset()
                self.run_until({"SUCCESS", "FAILED"})
                self.assert_real_success(self.controller)

    def test_successful_result_freezes_physics(self):
        self.run_until({"SUCCESS", "FAILED"})
        self.assert_real_success(self.controller)
        sim_time = float(self.env.data.time)
        qpos = self.env.data.qpos.copy()
        steps = self.controller.control_steps
        with patch.object(self.env, "step", wraps=self.env.step) as step:
            for _ in range(10):
                self.controller.step()
            step.assert_not_called()
        self.assertEqual(float(self.env.data.time), sim_time)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        self.assertEqual(self.controller.control_steps, steps)

    def test_timeout_failure_and_frozen_result(self):
        controller = AutoGraspController(self.env, phase_timeout_s=0.01)
        controller.step()
        self.assertEqual(controller.phase, "FAILED")
        self.assertIn("SETTLE timed out", controller.failure_reason)
        self.assertFalse(controller.last_info["success"])
        sim_time = float(self.env.data.time)
        qpos = self.env.data.qpos.copy()
        for _ in range(5):
            controller.step()
        self.assertEqual(float(self.env.data.time), sim_time)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_unsafe_initial_fixture_stops_before_action(self):
        # Retain the old opt-in displacement fault fixture.
        self.addCleanup(setattr, self.env, "displacement_check_enabled", self.env.displacement_check_enabled)
        self.env.displacement_check_enabled = True
        # Artificial safety fixture, not a claim that the arm pushed this guard.
        self.env.data.joint("object_joint_1").qpos[0] += 0.02
        self.env.post_step()
        sim_time = float(self.env.data.time)
        qpos = self.env.data.qpos.copy()
        with patch.object(self.env, "step", wraps=self.env.step) as step:
            info = self.controller.step()
            step.assert_not_called()
        self.assertEqual(self.controller.phase, "FAILED")
        self.assertTrue(info["unsafe"])
        self.assertTrue(info["episode_failed"])
        self.assertEqual(self.controller.control_steps, 0)
        self.assertEqual(float(self.env.data.time), sim_time)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_lost_grasp_reading_during_lift_fails(self):
        self.run_until({"LIFT"})
        self.assertEqual(self.controller.phase, "LIFT")
        self.assertTrue(self.env.task_report()["grasped"])
        # Inject only a detector reading, without teleporting the target or
        # adding grasp constraints. This tests the lift failure branch.
        no_grasp = {
            **self.env.grasp_report(), "grasped": False,
            "left_contact": False, "left_normal_force_n": 0.0,
        }
        with patch.object(self.env, "grasp_report", return_value=no_grasp):
            info = self.controller.step()
        self.assertEqual(self.controller.phase, "FAILED")
        self.assertIn("Lost two-finger grasp while lifting", self.controller.failure_reason)
        self.assertFalse(info["task"]["grasped"])
        self.assertFalse(info["success"])

    def test_reset_clears_failed_attempt(self):
        self.addCleanup(setattr, self.env, "displacement_check_enabled", self.env.displacement_check_enabled)
        self.env.displacement_check_enabled = True
        self.env.data.joint("object_joint_1").qpos[0] += 0.02
        self.env.post_step()
        self.controller.step()
        self.assertTrue(self.controller.finished)
        info = self.controller.reset()
        self.assertEqual(self.controller.phase, "SETTLE")
        self.assertEqual(self.controller.phase_history, ["SETTLE"])
        self.assertIsNone(self.controller.failure_reason)
        self.assertFalse(self.controller.finished)
        self.assertEqual(self.controller.control_steps, 0)
        self.assertEqual(float(self.env.data.time), 0.0)
        self.assertFalse(info["episode_failed"])
        self.assertFalse(info["unsafe"])
        self.assertFalse(info["success"])
        self.assertEqual(info["task"]["hold_seconds"], 0.0)

    def test_close_timeout_never_advances_to_lift(self):
        # An explicit phase fixture places a still-open distant gripper in CLOSE;
        # two-finger contact cannot be assumed from issuing a close command.
        self.controller.phase_timeout_s = 0.05
        self.controller._transition("CLOSE")
        self.run_until({"FAILED"})
        self.assertEqual(self.controller.phase, "FAILED")
        self.assertIn("No stable two-finger grasp", self.controller.failure_reason)
        self.assertNotIn("LIFT", self.controller.phase_history)
        self.assertFalse(self.controller.last_info["task"]["grasped"])

    def test_invalid_parameters_rejected_before_reset(self):
        for parameter, values in (
            ("speed_mps", (0, -0.1, 0.1001, float("nan"), float("inf"))),
            ("phase_timeout_s", (0, -1, float("nan"), float("inf"))),
        ):
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    with patch.object(self.env, "reset", wraps=self.env.reset) as reset:
                        with self.assertRaises(ValueError):
                            AutoGraspController(self.env, **{parameter: value})
                        reset.assert_not_called()

    def test_finite_state_control_error_becomes_frozen_failure(self):
        sim_time = float(self.env.data.time)
        qpos = self.env.data.qpos.copy()
        # A synthetic IK/control exception, not a simulated robot collision.
        with patch.object(self.env, "step", side_effect=RuntimeError("test IK failure")) as step:
            self.controller.step()
            self.controller.step()
            step.assert_called_once()
        self.assertEqual(self.controller.phase, "FAILED")
        self.assertIn("Robot control error: test IK failure", self.controller.failure_reason)
        self.assertFalse(self.controller.last_info["success"])
        self.assertEqual(float(self.env.data.time), sim_time)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)


class AutoGraspInputTests(unittest.TestCase):
    def setUp(self):
        self.inputs = AutoGraspInput()

    def key(self, key):
        self.inputs.handle(pygame.event.Event(pygame.KEYDOWN, key=key))

    def test_space_toggles_pause(self):
        self.assertFalse(self.inputs.paused)
        self.key(pygame.K_SPACE)
        self.assertTrue(self.inputs.paused)
        self.key(pygame.K_SPACE)
        self.assertFalse(self.inputs.paused)

    def test_focus_loss_pauses_until_explicit_resume(self):
        for event_type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED):
            with self.subTest(event_type=event_type):
                self.inputs.paused = False
                self.inputs.handle(pygame.event.Event(event_type))
                self.assertTrue(self.inputs.paused)
                self.inputs.handle(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
                self.assertTrue(self.inputs.paused)
                self.key(pygame.K_SPACE)
                self.assertFalse(self.inputs.paused)

    def test_tab_changes_only_camera(self):
        self.key(pygame.K_TAB)
        self.assertEqual(self.inputs.camera_index, 1)
        self.key(pygame.K_TAB)
        self.assertEqual(self.inputs.camera_index, 0)
        self.assertFalse(self.inputs.paused)
        self.assertFalse(self.inputs.reset_requested)
        self.assertFalse(self.inputs.quit_requested)

    def test_r_requests_restart_and_unpauses(self):
        self.inputs.paused = True
        self.key(pygame.K_r)
        self.assertTrue(self.inputs.reset_requested)
        self.assertFalse(self.inputs.paused)
        self.assertFalse(self.inputs.quit_requested)

    def test_escape_or_window_close_requests_quit(self):
        self.key(pygame.K_ESCAPE)
        self.assertTrue(self.inputs.quit_requested)
        self.inputs = AutoGraspInput()
        self.inputs.handle(pygame.event.Event(pygame.QUIT))
        self.assertTrue(self.inputs.quit_requested)


if __name__ == "__main__":
    unittest.main()
