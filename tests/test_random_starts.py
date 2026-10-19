"""Reset-only tests: no rendering, policy execution, or trajectory dataset."""

import copy
import unittest
from unittest import mock

import mujoco
import numpy as np

from random_starts import (
    DEFAULT_START_HIGH, DEFAULT_START_LOW, StartSamplingError,
    _contact_kind, sample_valid_start, validate_start,
)
from scene_config import ARM_START_XYZ, BLOCKS
from three_block_scene import ThreeBlockScene


class RandomStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = ThreeBlockScene()
        cls.addClassCleanup(cls.env.close)

    def setUp(self):
        self.env.arm_start = np.asarray(ARM_START_XYZ, dtype=float).copy()
        self.env.reset(seed=0)

    def test_default_reset_valid_and_floor_support_allowed(self):
        report = validate_start(self.env, ARM_START_XYZ)
        self.assertTrue(report["valid"], report)
        self.assertEqual(len(report["contacts"]), 12)
        self.assertTrue(all(c["kind"] == "allowed_block_floor" for c in report["contacts"]))
        self.assertFalse(report["path_reachability_guaranteed"])

    def test_lower_fixed_start_and_validation_use_same_height_limit(self):
        from random_starts import SUPPORTED_START_LOW
        from scene_config import ARM_START_MIN_Z_M
        self.assertEqual(ARM_START_MIN_Z_M, 0.16)
        self.assertEqual(SUPPORTED_START_LOW[2], ARM_START_MIN_Z_M)
        self.assertTrue(validate_start(self.env, ARM_START_XYZ)["valid"])
        too_low = np.array([0.375, 0, ARM_START_MIN_Z_M - 0.001])
        self.assertIn("requested_start_outside_supported_workspace",
                      validate_start(self.env, too_low)["reasons"])
        self.env.arm_start = too_low
        with self.assertRaisesRegex(ValueError, "outside the supported starting workspace"):
            self.env.reset(seed=0)
        # Lowering the fixed demo does not silently change data sampling bounds.
        np.testing.assert_allclose(DEFAULT_START_LOW, [0.28, -0.02, 0.24])
        np.testing.assert_allclose(DEFAULT_START_HIGH, [0.32, 0.02, 0.28])

    def test_30_samples_valid_nonidentical_and_never_advance_physics(self):
        rng = np.random.default_rng(1234)
        results = []
        expected_objects = np.concatenate([
            self.env.data.joint(f"object_joint_{i}").qpos.copy() for i in range(3)
        ])
        for _ in range(30):
            sampled = sample_valid_start(self.env, rng)
            results.append(sampled["requested_start_xyz"])
            self.assertTrue(sampled["validity"]["valid"])
            self.assertEqual(sampled["attempts"], 1)
            self.assertEqual(self.env.data.time, 0)
            self.assertFalse(self.env.task_report()["episode_failed"])
            actual_objects = np.concatenate([
                self.env.data.joint(f"object_joint_{i}").qpos.copy() for i in range(3)
            ])
            np.testing.assert_array_equal(actual_objects, expected_objects)
        starts = np.asarray(results)
        self.assertTrue(np.all(starts >= DEFAULT_START_LOW))
        self.assertTrue(np.all(starts <= DEFAULT_START_HIGH))
        self.assertTrue(np.all(np.ptp(starts, axis=0) > 0.02))

    def test_same_seed_reproduces_requested_and_actual_starts(self):
        sequences = []
        for _ in range(2):
            rng = np.random.default_rng(17)
            sequences.append([sample_valid_start(self.env, rng) for _ in range(3)])
        self.assertEqual(sequences[0], sequences[1])

    def test_sampling_never_changes_global_configuration(self):
        previous_blocks = copy.deepcopy(BLOCKS)
        previous_start = list(ARM_START_XYZ)
        sample_valid_start(self.env, np.random.default_rng(42))
        self.assertEqual(BLOCKS, previous_blocks)
        self.assertEqual(list(ARM_START_XYZ), previous_start)
        self.assertEqual(self.env.arm_fixed_yaw_deg, 90.0)

    def test_second_reset_accepts_same_sample(self):
        result = sample_valid_start(self.env, np.random.default_rng(2))
        self.env.reset(seed=0)
        report = validate_start(self.env, result["requested_start_xyz"])
        self.assertTrue(report["valid"])
        self.assertEqual(report["actual_start_xyz"], result["actual_start_xyz"])

    def test_bad_parameters_rejected_before_mutation(self):
        start = self.env.arm_start.copy()
        for kwargs in ({"max_attempts": 0}, {"max_attempts": 1.2}, {"max_attempts": True},
                       {"low": [.1, 0, .25]}, {"high": [.7, 0, .25]},
                       {"low": [float("nan"), 0, .25]}, {"low": [.4, 0, .25], "high": [.3, 0, .25]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample_valid_start(self.env, np.random.default_rng(1), **kwargs)
        with self.assertRaises(TypeError):
            sample_valid_start(self.env, 42)
        np.testing.assert_array_equal(self.env.arm_start, start)

    def test_requested_workspace_and_position_mismatch_rejected(self):
        self.assertFalse(validate_start(self.env, [.1, 0, .25])["valid"])
        self.assertFalse(validate_start(self.env, [float("nan"), 0, .25])["valid"])
        report = validate_start(self.env, [.32, 0, .25])
        self.assertIn("ik_position_error_exceeds_2mm", report["reasons"])

    def test_rotation_error_detected_with_wrong_fixed_heading(self):
        self.env.arm_fixed_yaw_deg = 0.0
        try:
            report = validate_start(self.env, ARM_START_XYZ)
            self.assertIn("ik_rotation_error_exceeds_0.03rad", report["reasons"])
        finally:
            self.env.arm_fixed_yaw_deg = 90.0

    def test_joint_limit_violation_detected(self):
        self.env.data.qpos[0] = self.env.model.jnt_range[0, 1] + .1
        mujoco.mj_forward(self.env.model, self.env.data)
        report = validate_start(self.env, ARM_START_XYZ)
        self.assertIn("joint_limit_violation", report["reasons"])
        self.assertEqual(report["joint_limit_violations"][0]["joint"], "ur5e/shoulder_pan_joint")

    def test_shifted_guard_is_rejected(self):
        self.addCleanup(setattr, self.env, "displacement_check_enabled", self.env.displacement_check_enabled)
        self.env.displacement_check_enabled = True
        self.env.data.joint("object_joint_1").qpos[0] += .011
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertIn("guards_unsafe_at_start", validate_start(self.env, ARM_START_XYZ)["reasons"])

    def test_target_touching_pad_is_rejected_even_without_advancing_physics(self):
        pad_id = next(iter(self.env._grasp_geom_ids["left"]))
        self.env.data.joint("object_joint_0").qpos[:3] = self.env.data.geom_xpos[pad_id]
        mujoco.mj_forward(self.env.model, self.env.data)
        report = validate_start(self.env, ARM_START_XYZ)
        self.assertIn("robot_block_contact", report["reasons"])
        self.assertEqual(self.env.data.time, 0)

    def test_actual_robot_contact_classification_is_not_blanket_internal_exemption(self):
        def physical_geom(body_name):
            body_id = self.env.model.body(body_name).id
            return next(i for i in range(self.env.model.ngeom)
                        if self.env.model.geom_bodyid[i] == body_id and
                        (self.env.model.geom_contype[i] or self.env.model.geom_conaffinity[i]))
        left = physical_geom("ur5e/robotiq/left_pad")
        right = physical_geom("ur5e/robotiq/right_pad")
        base = physical_geom("ur5e/robotiq/base")
        driver = physical_geom("ur5e/robotiq/left_driver")
        floor = self.env.model.geom("floor").id
        self.assertEqual(_contact_kind(self.env.model, left, right), "robot_self_collision")
        self.assertEqual(_contact_kind(self.env.model, base, driver), "allowed_gripper_linkage_internal")
        self.assertEqual(_contact_kind(self.env.model, left, floor), "robot_floor_contact")

    def test_validator_rejects_nonzero_time_without_mutating_state(self):
        self.env.data.time = .1
        old_qpos = self.env.data.qpos.copy()
        report = validate_start(self.env, ARM_START_XYZ)
        self.assertIn("not_a_reset_state_simulation_time_is_nonzero", report["reasons"])
        np.testing.assert_array_equal(self.env.data.qpos, old_qpos)
        self.assertEqual(self.env.data.time, .1)

    def test_sampling_records_rejections_and_has_hard_attempt_limit(self):
        invalid = {"valid": False, "reasons": ["synthetic_bad_ik"]}
        with mock.patch("random_starts.validate_start", return_value=invalid):
            with self.assertRaises(StartSamplingError) as caught:
                sample_valid_start(self.env, np.random.default_rng(42), max_attempts=3)
        error = caught.exception
        self.assertEqual(error.attempts, 3)
        self.assertEqual(len(error.rejected), 3)
        self.assertTrue(all(row["reasons"] == ["synthetic_bad_ik"] for row in error.rejected))
        self.assertIn("after 3 attempts", str(error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
