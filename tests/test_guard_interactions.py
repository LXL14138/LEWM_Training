"""Interaction geometry/stop regressions. Injected contacts are TEST fixtures."""

import unittest
from unittest.mock import patch, PropertyMock

import numpy as np

from exploration import ExplorationController, ExplorationScene, PHASES
from exploration_validation import EXPLORATION_PHASES


class GuardInteractionTests(unittest.TestCase):
    def make_controller(self, seed=0):
        env = ExplorationScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env, ExplorationController(env, np.random.default_rng(seed), "guard")

    def test_sampled_pushes_cover_directions_heights_and_three_stroke_bands(self):
        env, controller = self.make_controller()
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        directions, heights, bands = set(), set(), set()
        for _ in range(100):
            controller._guard()
            self.assertEqual(controller.phase, "GUARD_APPROACH")
            row = controller.interaction
            d = np.array(row["direction_xyz"])
            np.testing.assert_allclose(np.linalg.norm(d), 1)
            outward_y = -1 if row["guard_index"] == 1 else 1
            self.assertLessEqual(d[1] * outward_y, 0)  # never approach from inner gap
            directions.add(tuple(np.round(d, 3)))
            heights.add(round(row["requested_height_m"], 2))
            stroke = row["requested_post_contact_travel_m"]
            self.assertTrue(any(lo <= stroke <= hi for lo, hi in
                                [(0.002, 0.005), (0.008, 0.015), (0.02, 0.035)]))
            bands.add(0 if stroke < .006 else 1 if stroke < .016 else 2)
            controller._close_interaction("test_plan_only")
        self.assertGreaterEqual(len(directions), 5)
        self.assertGreaterEqual(len(heights), 4)
        self.assertEqual(bands, {0, 1, 2})
        np.testing.assert_array_equal(qpos, env.data.qpos)
        np.testing.assert_array_equal(qvel, env.data.qvel)
        self.assertEqual(env.data.time, 0)

    def test_short_stroke_not_skipped_and_stop_resets_reference(self):
        env, controller = self.make_controller()
        controller._guard()
        controller.push_stroke = .003  # less than generic 5-mm arrival tolerance
        origin = controller.actual.copy()
        controller._begin_push()
        controller._advance_phase()
        self.assertEqual(controller.phase, "GUARD_PUSH")
        measured = origin + controller.push_direction * .003
        with patch.object(ExplorationController, "actual", new_callable=PropertyMock, return_value=measured):
            controller._advance_phase()
        self.assertEqual(controller.phase, "EVENT_HOLD")
        np.testing.assert_array_equal(controller.reference, measured)
        np.testing.assert_array_equal(controller.waypoint, measured)
        self.assertEqual(controller.interactions[-1]["stop_reason"], "guard_push_distance")

    def test_no_contact_and_unreachable_descent_are_not_successful_pushes(self):
        env, controller = self.make_controller()
        controller._guard()
        controller._set("GUARD_SEEK", controller.actual)
        controller._advance_phase()
        self.assertEqual(controller.event_reason, "guard_no_contact")
        self.assertNotIn("contact_pinch_xyz", controller.interactions[-1])
        self.assertEqual(controller.interactions[-1]["robot_or_green_contact_steps"], 0)

    def test_outside_approach_accepts_measured_static_error_without_changing_grasp_tolerance(self):
        _, controller = self.make_controller()
        controller._guard()
        measured = controller.waypoint + [0, 0, -.0055]
        with patch.object(ExplorationController, "actual", new_callable=PropertyMock, return_value=measured):
            controller._advance_phase()
        self.assertEqual(controller.phase, "GUARD_LOWER")

    def test_blocked_push_has_bounded_timeout(self):
        env, controller = self.make_controller()
        controller._guard()
        controller._begin_push()
        controller.started = float(env.data.time) - 3.1
        controller._advance_phase()
        self.assertEqual(controller.event_reason, "guard_push_timeout")
        self.assertEqual(controller.phase, "EVENT_HOLD")

    def test_contact_during_lowering_starts_measured_push(self):
        env, controller = self.make_controller()
        controller._guard()
        controller._set("GUARD_LOWER", controller.actual)
        env._contact_peak[controller.guard_index - 1] = .01  # synthetic contact
        controller._advance_phase()
        self.assertEqual(controller.phase, "GUARD_PUSH")
        self.assertEqual(controller.interaction["contact_phase"], "GUARD_LOWER")

    def test_brush_geometry_is_outer_face_tangent_and_does_not_change_goal(self):
        env, controller = self.make_controller()
        controller.grip_offset = np.array([0, 0, .005])
        original_goal, qpos = env.hover_goal.copy(), env.data.qpos.copy()
        for _ in range(30):
            controller._brush()
            row = controller.interaction
            entry, end = controller.brush_entry_com, controller.brush_exit_com
            guard = env.data.body(f"object_{row['guard_index']}").xipos
            outer = -1 if row["guard_index"] == 1 else 1
            self.assertAlmostEqual((entry[1] - guard[1]) * outer, .04 + row["nominal_face_gap_m"])
            np.testing.assert_allclose((end - entry)[1:], 0)
            self.assertGreaterEqual(abs(end[0] - entry[0]), .09)
            self.assertTrue(.075 <= entry[2] <= .12)
            self.assertEqual(controller.phase, "BRUSH_LIFT")
            np.testing.assert_allclose(controller.waypoint[:2], controller.actual[:2])
            controller._close_interaction("test_plan_only")
        np.testing.assert_array_equal(env.hover_goal, original_goal)
        np.testing.assert_array_equal(env.data.qpos, qpos)

    def test_brush_loss_and_hazard_preserve_existing_stop_rules(self):
        env, controller = self.make_controller()
        controller.grip_offset = np.array([0, 0, .005])
        controller._brush()
        controller.lost_since = -.2
        controller._advance_phase()
        self.assertEqual(controller.event_reason, "lost_grasp")
        self.assertEqual(controller.interactions[-1]["stop_reason"], "lost_grasp")
        controller._brush()
        report = dict(env.task_report(), episode_failed=True)
        with patch.object(env, "task_report", return_value=report):
            controller._advance_phase()
        self.assertEqual(controller.event_reason, "guard_hazard")

    def test_measured_green_contacts_are_separate_from_robot_contacts(self):
        env, controller = self.make_controller()
        controller._guard()
        j = controller.guard_index - 1
        env._contact_peak[j], env._green_contact_peak[j] = 2, 0
        controller.steps = 1
        controller._observe_interaction()
        self.assertEqual(controller.interaction["green_contact_steps"], 0)
        env._green_contact_peak[j] = .1
        controller.steps = 2
        controller._observe_interaction()
        self.assertEqual(controller.interaction["green_contact_steps"], 1)
        self.assertEqual(controller.interaction["first_green_contact_step"], 2)
        self.assertEqual(controller.interaction["robot_or_green_contact_steps"], 2)

    def test_budget_closes_partial_interaction_and_phase_names_fit_storage(self):
        _, controller = self.make_controller()
        controller._guard()
        controller.finish("time_budget")
        self.assertIsNone(controller.interaction)
        self.assertEqual(controller.interactions[-1]["stop_reason"], "time_budget")
        self.assertEqual(PHASES - {"DONE"}, EXPLORATION_PHASES)
        self.assertTrue(all(len(p) <= 16 for p in PHASES))


if __name__ == "__main__":
    unittest.main()
