"""Pure-state tests for timed grasp success and the episode-failure latch.

No MuJoCo, renderer, learned model, or robot teleports are involved here.
The separate integration tests must verify how physical measurements are made.
"""

import math
import unittest

from task_progress import TaskProgress


class TaskProgressTests(unittest.TestCase):
    def setUp(self):
        self.progress = TaskProgress(height_m=0.10, hold_seconds=1.0)

    def update(self, time, height=0.10, grasped=True, unsafe=False):
        return self.progress.update(
            sim_time=time,
            target_height_m=height,
            grasped=grasped,
            unsafe=unsafe,
        )

    def assert_incomplete(self, report, status="IN_PROGRESS"):
        self.assertEqual(report["status"], status)
        self.assertFalse(report["task_success"])
        self.assertFalse(report["episode_failed"])

    def test_initial_report_is_incomplete(self):
        report = self.progress.report()
        self.assert_incomplete(report)
        self.assertFalse(report["grasped"])
        self.assertFalse(report["goal_conditions_met"])
        self.assertEqual(report["hold_seconds"], 0.0)
        self.assertEqual(report["required_hold_seconds"], 1.0)

    def test_first_qualified_sample_starts_at_zero(self):
        report = self.update(50.0)
        self.assert_incomplete(report, "HOLDING")
        self.assertTrue(report["grasped"])
        self.assertTrue(report["height_ok"])
        self.assertTrue(report["goal_conditions_met"])
        self.assertEqual(report["target_height_m"], 0.10)
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_success_requires_full_second_from_first_qualified_sample(self):
        self.update(10.0, height=0.02, grasped=False)
        self.update(20.0)
        report = self.update(20.999)
        self.assert_incomplete(report, "HOLDING")
        self.assertAlmostEqual(report["hold_seconds"], 0.999)
        report = self.update(21.0)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue(report["task_success"])
        self.assertFalse(report["episode_failed"])
        self.assertGreaterEqual(report["hold_seconds"], 1.0 - 1e-9)

    def test_time_threshold_tolerance(self):
        for elapsed, expected in ((1.0 - 2e-9, False), (1.0 - 0.5e-9, True), (1.0, True)):
            with self.subTest(elapsed=elapsed):
                self.progress.reset()
                self.update(0.0)
                report = self.update(elapsed)
                self.assertEqual(report["task_success"], expected)

    def test_height_threshold_tolerance(self):
        for height, expected in ((0.10 - 2e-9, False), (0.10 - 0.5e-9, True), (0.10, True), (0.12, True)):
            with self.subTest(height=height):
                self.progress.reset()
                report = self.update(0.0, height=height)
                self.assertEqual(report["height_ok"], expected)
                self.assertEqual(report["goal_conditions_met"], expected)
                self.assertFalse(report["task_success"])
                self.assertEqual(report["target_height_m"], height)

    def test_height_without_grasp_never_completes(self):
        self.update(0.0, height=0.15, grasped=False)
        report = self.update(10.0, height=0.15, grasped=False)
        self.assert_incomplete(report)
        self.assertTrue(report["height_ok"])
        self.assertFalse(report["goal_conditions_met"])
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_grasp_without_height_never_completes(self):
        self.update(0.0, height=0.02)
        report = self.update(10.0, height=0.02)
        self.assert_incomplete(report)
        self.assertTrue(report["grasped"])
        self.assertFalse(report["height_ok"])
        self.assertFalse(report["goal_conditions_met"])
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_loss_of_grasp_resets_continuous_hold(self):
        self.update(0.0)
        self.update(0.8)
        report = self.update(0.9, grasped=False)
        self.assert_incomplete(report)
        self.assertEqual(report["hold_seconds"], 0.0)
        self.update(1.0)
        self.assertFalse(self.update(1.9)["task_success"])
        self.assertTrue(self.update(2.0)["task_success"])

    def test_loss_of_height_resets_continuous_hold(self):
        self.update(0.0)
        self.update(0.8)
        report = self.update(0.9, height=0.08)
        self.assert_incomplete(report)
        self.assertEqual(report["hold_seconds"], 0.0)
        self.update(1.0)
        self.assertFalse(self.update(1.9)["task_success"])
        self.assertTrue(self.update(2.0)["task_success"])

    def test_repeated_timestamp_never_advances_hold(self):
        self.update(7.0)
        for _ in range(30):
            report = self.update(7.0)
            self.assertEqual(report["hold_seconds"], 0.0)
            self.assertFalse(report["task_success"])
        report = self.update(7.5)
        self.assertAlmostEqual(report["hold_seconds"], 0.5)
        for _ in range(30):
            self.assertAlmostEqual(self.update(7.5)["hold_seconds"], 0.5)

    def test_same_timestamp_failure_is_not_ignored(self):
        self.update(0.0)
        self.update(0.8)
        report = self.update(0.8, unsafe=True)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(report["episode_failed"])
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_same_timestamp_grasp_loss_restarts_hold(self):
        self.update(0.0)
        self.update(0.8)
        self.update(0.8, grasped=False)
        report = self.update(0.8)
        self.assertEqual(report["hold_seconds"], 0.0)
        self.assertFalse(self.update(1.0)["task_success"])
        self.assertTrue(self.update(1.8)["task_success"])

    def test_unsafe_wins_at_exact_completion_time(self):
        self.update(0.0)
        report = self.update(1.0, unsafe=True)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(report["episode_failed"])
        self.assertFalse(report["task_success"])
        self.assertEqual(report["hold_seconds"], 0.0)
        # The instantaneous grasp/height predicate is distinct from history.
        self.assertTrue(report["goal_conditions_met"])

    def test_failure_latches_when_guards_are_restored(self):
        self.update(0.0, height=0.02, grasped=False, unsafe=True)
        self.update(1.0)
        report = self.update(5.0)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(report["episode_failed"])
        self.assertFalse(report["task_success"])
        self.assertTrue(report["goal_conditions_met"])
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_success_is_not_latched_after_release(self):
        self.update(0.0)
        self.assertTrue(self.update(1.0)["task_success"])
        report = self.update(1.1, grasped=False)
        self.assert_incomplete(report)
        self.assertEqual(report["hold_seconds"], 0.0)
        self.update(1.2)
        self.assertFalse(self.update(2.1)["task_success"])
        self.assertTrue(self.update(2.2)["task_success"])

    def test_success_is_not_latched_after_lowering(self):
        self.update(0.0)
        self.assertTrue(self.update(1.0)["task_success"])
        report = self.update(1.1, height=0.09)
        self.assert_incomplete(report)
        self.assertEqual(report["hold_seconds"], 0.0)

    def test_later_unsafe_revokes_success_and_latches_failure(self):
        self.update(0.0)
        self.assertTrue(self.update(1.0)["task_success"])
        report = self.update(1.1, unsafe=True)
        self.assertEqual(report["status"], "FAILED")
        self.assertFalse(report["task_success"])
        self.assertTrue(report["episode_failed"])
        self.assertEqual(self.update(10.0)["status"], "FAILED")

    def test_reset_clears_failure_timer_and_time_history(self):
        self.update(100.0, unsafe=True)
        self.progress.reset()
        report = self.progress.report()
        self.assert_incomplete(report)
        self.assertEqual(report["hold_seconds"], 0.0)
        self.update(0.0)
        self.assertTrue(self.update(1.0)["task_success"])

    def test_reset_clears_success(self):
        self.update(0.0)
        self.update(1.0)
        self.progress.reset()
        self.assert_incomplete(self.progress.report())
        self.assertEqual(self.update(0.0)["hold_seconds"], 0.0)

    def test_report_is_a_snapshot_and_does_not_advance_time(self):
        self.update(0.0)
        self.update(0.5)
        expected = self.progress.report()
        report = self.progress.report()
        self.assertIsNot(expected, report)
        report["status"] = "CORRUPTED"
        report["hold_seconds"] = 999.0
        report["episode_failed"] = True
        report["task_success"] = True
        report["extra"] = "not internal state"
        for _ in range(10):
            self.assertEqual(self.progress.report(), expected)

    def test_update_return_value_does_not_alias_internal_state(self):
        report = self.update(0.0)
        expected = self.progress.report()
        report["status"] = "CORRUPTED"
        report["hold_seconds"] = 999.0
        self.assertEqual(self.progress.report(), expected)

    def test_custom_height_and_duration(self):
        self.progress = TaskProgress(height_m=0.15, hold_seconds=2.0)
        self.assertFalse(self.update(0.0, height=0.10)["height_ok"])
        self.update(1.0, height=0.15)
        report = self.update(2.0, height=0.15)
        self.assertEqual(report["required_hold_seconds"], 2.0)
        self.assertFalse(report["task_success"])
        self.assertTrue(self.update(3.0, height=0.15)["task_success"])

    def test_invalid_configuration_is_rejected(self):
        for name in ("height_m", "hold_seconds"):
            for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    kwargs = {"height_m": 0.10, "hold_seconds": 1.0, name: value}
                    with self.assertRaises(ValueError):
                        TaskProgress(**kwargs)

    def test_nonfinite_and_negative_times_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, -0.01):
            with self.subTest(time=value):
                self.progress.reset()
                with self.assertRaises(ValueError):
                    self.update(value)

    def test_backwards_time_is_rejected(self):
        self.update(10.0)
        with self.assertRaises(ValueError):
            self.update(9.99)

    def test_nonfinite_heights_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(height=value):
                self.progress.reset()
                with self.assertRaises(ValueError):
                    self.update(0.0, height=value)

    def test_invalid_measurement_does_not_mutate_progress(self):
        self.update(0.0)
        self.update(0.5)
        expected = self.progress.report()
        with self.assertRaises(ValueError):
            self.update(0.8, height=math.nan, unsafe=True)
        self.assertEqual(self.progress.report(), expected)
        self.assertTrue(self.update(1.0)["task_success"])


if __name__ == "__main__":
    unittest.main()
