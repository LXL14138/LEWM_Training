"""Ground-truth tilt checks using known simulator poses, not a learned model."""

import unittest

import mujoco
import numpy as np

from three_block_scene import ThreeBlockScene


class TiltSafetyTests(unittest.TestCase):
    def setUp(self):
        self.env = ThreeBlockScene(unsafe_tilt_deg=30.0)
        self.addCleanup(self.env.close)
        self.env.reset(seed=0)

    def set_pose(self, index, angle_deg, axis="x"):
        """Place a box in a known test pose, suspended to avoid penetration.

        This teleports a pose solely for testing the predicate; it is not an
        action or a demonstration trajectory for future world-model training.
        """
        half = np.deg2rad(angle_deg) / 2
        quat = np.zeros(4)
        quat[0] = np.cos(half)  # MuJoCo uses w, x, y, z.
        quat[1 + "xyz".index(axis)] = np.sin(half)
        joint = self.env.data.joint(f"object_joint_{index}")
        joint.qpos[2] = 0.25
        joint.qpos[3:] = quat
        joint.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_default_reset_info(self):
        _, info = self.env.reset(seed=0)
        self.assertFalse(info["unsafe"])
        self.assertEqual(info["safety"]["safety_scope"], "guard_tilt_only")
        self.assertEqual(len(info["safety"]["guards"]), 2)
        self.assertEqual(info["safety"]["unsafe_tilt_deg"], 30.0)
        self.assertTrue(info["task_defined"])
        self.assertIn("当前状态：安全", self.env.safety_status_text())

    def test_known_tilts_and_threshold_boundary(self):
        cases = [
            (0, False), (20, False), (29.9999, False), (30, False),
            (30.0001, True), (40, True), (-40, True), (90, True), (180, True),
        ]
        for index in (1, 2):
            for axis in ("x", "y"):
                for angle, unsafe in cases:
                    with self.subTest(index=index, axis=axis, angle=angle):
                        self.env.reset(seed=0)
                        self.set_pose(index, angle, axis)
                        report = self.env.safety_report()
                        guard = report["guards"][index - 1]
                        self.assertAlmostEqual(guard["tilt_deg"], abs(angle), places=6)
                        self.assertEqual(guard["unsafe"], unsafe)
                        self.assertEqual(report["unsafe"], unsafe)

    def test_yaw_is_not_tilt(self):
        for angle in (45, 90, -90, 180):
            with self.subTest(yaw=angle):
                for index in (1, 2):
                    self.set_pose(index, angle, "z")
                report = self.env.safety_report()
                self.assertFalse(report["unsafe"])
                for guard in report["guards"]:
                    self.assertAlmostEqual(guard["tilt_deg"], 0, places=6)

    def test_yaw_plus_tilt_and_quaternion_sign(self):
        # q_z(70 deg) * q_x(40 deg): yaw must not change the 40-degree tilt.
        cz, sz = np.cos(np.deg2rad(35)), np.sin(np.deg2rad(35))
        cx, sx = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))
        quat = np.array([cz*cx, cz*sx, sz*sx, sz*cx])
        for sign in (1, -1):
            with self.subTest(quaternion_sign=sign):
                self.env.data.joint("object_joint_1").qpos[3:] = sign * quat
                mujoco.mj_forward(self.env.model, self.env.data)
                report = self.env.safety_report()
                self.assertAlmostEqual(report["guards"][0]["tilt_deg"], 40, places=6)
                self.assertTrue(report["unsafe"])

    def test_either_or_both_guards(self):
        for left, right in ((40, 0), (0, 40), (40, 40), (20, 20)):
            with self.subTest(left=left, right=right):
                self.set_pose(1, left)
                self.set_pose(2, right)
                report = self.env.safety_report()
                self.assertEqual([g["unsafe"] for g in report["guards"]],
                                 [left > 30, right > 30])
                self.assertEqual(report["unsafe"], left > 30 or right > 30)

    def test_green_target_is_not_a_guard(self):
        self.set_pose(0, 90)
        self.assertFalse(self.env.safety_report()["unsafe"])

    def test_current_state_not_episode_history(self):
        self.set_pose(1, 40)
        self.assertTrue(self.env.get_step_info()["unsafe"])
        self.assertIn("危险 (unsafe)", self.env.safety_status_text())
        self.set_pose(1, 0)
        self.assertFalse(self.env.get_step_info()["unsafe"])
        self.set_pose(2, 40)
        _, info = self.env.reset(seed=0)
        self.assertFalse(info["unsafe"])

    def test_post_step_pose_is_current(self):
        for preview in (False, True):
            with self.subTest(preview=preview):
                self.env.reset(seed=0)
                self.set_pose(1, 29)
                self.env.data.joint("object_joint_1").qvel[3] = 1.0
                if preview:
                    self.env.step_preview()
                    info = self.env.get_step_info()
                else:
                    _, reward, terminated, _, info = self.env.step(np.zeros(5))
                    self.assertEqual(reward, 0)
                    self.assertFalse(terminated)  # Safety does not stop the simulation.
                    self.assertFalse(info["success"])
                # Independent expected angle directly from current qpos quaternion.
                _, qx, qy, _ = self.env.data.joint("object_joint_1").qpos[3:]
                expected = np.degrees(np.arccos(np.clip(1 - 2*(qx*qx + qy*qy), -1, 1)))
                self.assertGreater(expected, 30)
                self.assertAlmostEqual(info["safety"]["guards"][0]["tilt_deg"],
                                       expected, places=7)
                self.assertTrue(info["unsafe"])

    def test_custom_threshold(self):
        self.env.close()
        self.env = ThreeBlockScene(unsafe_tilt_deg=45)
        self.addCleanup(self.env.close)
        self.env.reset(seed=0)
        self.set_pose(1, 40)
        self.assertFalse(self.env.safety_report()["unsafe"])
        self.set_pose(1, 50)
        self.assertTrue(self.env.safety_report()["unsafe"])

    def test_invalid_threshold(self):
        for threshold in (np.nan, np.inf, -1, 0, 180, 181):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "unsafe_tilt_deg"):
                    ThreeBlockScene(unsafe_tilt_deg=threshold)

    def test_nonfinite_orientation_is_not_reported_safe(self):
        self.env.data.body("object_1").xmat[0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Invalid simulator orientation"):
            self.env.safety_report()


if __name__ == "__main__":
    unittest.main(verbosity=2)
