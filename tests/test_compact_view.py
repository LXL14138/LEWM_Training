"""Current compact layout/start and fixed close-up camera regression tests."""

import itertools
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from auto_grasp import AutoGraspController
from three_block_scene import ThreeBlockScene


class CompactViewTests(unittest.TestCase):
    def make_env(self):
        env = ThreeBlockScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def test_start_is_closer_above_target_and_not_already_grasping(self):
        env = self.make_env()
        np.testing.assert_allclose(env.arm_start, [.375, 0, .16])
        actual = env.data.site_xpos[env._pinch_site_id]
        np.testing.assert_allclose(actual, env.arm_start, atol=0.002)
        target = env.data.body("object_0").xipos
        self.assertLess(np.linalg.norm(actual - target), np.linalg.norm([.30 - .425, 0, .25 - .05]))
        self.assertAlmostEqual(env.arm_start[0] - target[0], -0.05)
        self.assertAlmostEqual(env.arm_start[2] - 0.10, 0.06)
        self.assertFalse(env.task_report()["grasped"])
        self.assertFalse(env.safety_report()["unsafe"])
        self.assertFalse(env.task_report()["task_success"])

    def test_new_start_static_hold_settles_without_accumulating_drift(self):
        env = self.make_env()
        initial = env.data.site_xpos[env._pinch_site_id].copy()
        for _ in range(100):
            env.step_preview()
        settled = env.data.site_xpos[env._pinch_site_id].copy()
        # The arm has a small pose-dependent gravity/PD static error. Do not
        # mistake that for cumulative drift; test both separately. 5 mm is
        # the existing scripted controller's measured-arrival tolerance.
        self.assertLess(np.linalg.norm(settled - initial), 0.005)
        for _ in range(300):
            env.step_preview()
        self.assertLess(np.linalg.norm(env.data.site_xpos[env._pinch_site_id] - settled), 0.0005)
        self.assertFalse(env.safety_report()["unsafe"])
        for row in env.block_report():
            self.assertAlmostEqual(row["position_xyz"][2], .05, places=3)

    def test_front_camera_world_fixed_and_aims_at_configured_point(self):
        env = self.make_env()
        camera = env.model.camera("front_pixels").id
        self.assertEqual(env.model.cam_bodyid[camera], 0)
        self.assertEqual(env.model.cam_mode[camera], mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        self.assertEqual(env.model.cam_targetbodyid[camera], -1)
        np.testing.assert_allclose(env.data.cam_xpos[camera], [.70125, 0, .324])
        self.assertEqual(env.model.cam_fovy[camera], 48)
        forward = env.front_camera_look_at - env.front_camera_position
        forward /= np.linalg.norm(forward)
        np.testing.assert_allclose(-env.data.cam_xmat[camera].reshape(3, 3)[:, 2], forward, atol=1e-12)
        position, rotation = env.data.cam_xpos[camera].copy(), env.data.cam_xmat[camera].copy()
        controller = AutoGraspController(env)
        for _ in range(80):
            controller.step()
            np.testing.assert_array_equal(env.data.cam_xpos[camera], position)
            np.testing.assert_array_equal(env.data.cam_xmat[camera], rotation)

    def test_all_pillars_fit_close_view_through_normal_grasp(self):
        env = self.make_env()
        controller = AutoGraspController(env)
        camera = env.model.camera("front_pixels").id
        signs = np.array(list(itertools.product((-1, 1), repeat=3)))
        focal_y = .5 * 480 / np.tan(np.deg2rad(env.model.cam_fovy[camera] / 2))
        for step in range(601):
            for index in range(3):
                geom = env.model.geom(f"object_{index}").id
                corners = (signs * env.model.geom_size[geom]) @ env.data.geom_xmat[geom].reshape(3, 3).T
                corners += env.data.geom_xpos[geom]
                camera_xyz = (corners - env.data.cam_xpos[camera]) @ env.data.cam_xmat[camera].reshape(3, 3)
                depth = -camera_xyz[:, 2]
                self.assertTrue(np.all(depth > 0))
                pixels = np.column_stack((320 + focal_y * camera_xyz[:, 0] / depth,
                                           240 - focal_y * camera_xyz[:, 1] / depth))
                self.assertTrue(np.all(pixels > 0), (step, index, pixels))
                self.assertTrue(np.all(pixels < [640, 480]), (step, index, pixels))
                if step == 0:
                    self.assertGreater(np.ptp(pixels[:, 1]), 100)
            if controller.finished:
                break
            controller.step()
        else:
            self.fail("Grasp exceeded test control-step bound")
        self.assertEqual(controller.phase, "SUCCESS")

    def test_camera_change_only_preserves_dynamics_and_side_image(self):
        close = self.make_env()
        with patch("three_block_scene.FRONT_CAMERA_POSITION", [1.053, -.014, .639]), \
                patch("three_block_scene.FRONT_CAMERA_LOOK_AT", [.425, 0, .132]):
            wide = self.make_env()
        for field in ("body_mass", "body_inertia", "geom_size", "geom_pos", "geom_friction",
                      "geom_contype", "geom_conaffinity", "actuator_gainprm"):
            np.testing.assert_array_equal(getattr(close.model, field), getattr(wide.model, field))
        np.testing.assert_array_equal(close.render(camera="side_pixels"), wide.render(camera="side_pixels"))
        controllers = [AutoGraspController(env) for env in (close, wide)]
        for _ in range(601):
            np.testing.assert_array_equal(close.data.qpos, wide.data.qpos)
            np.testing.assert_array_equal(close.data.qvel, wide.data.qvel)
            self.assertEqual(close.task_report(), wide.task_report())
            self.assertEqual(controllers[0].phase, controllers[1].phase)
            if controllers[0].finished:
                break
            for controller in controllers:
                controller.step()
            np.testing.assert_array_equal(controllers[0].last_action, controllers[1].last_action)
        else:
            self.fail("Grasp exceeded test control-step bound")
        self.assertEqual(controllers[0].phase, "SUCCESS")

    def test_invalid_camera_settings_rejected(self):
        cases = [("FRONT_CAMERA_POSITION", [0, 0]),
                 ("FRONT_CAMERA_LOOK_AT", [float("nan"), 0, 0]),
                 ("FRONT_CAMERA_LOOK_AT", [.70125, 0, .324]),
                 ("FRONT_CAMERA_LOOK_AT", [.70125, 0, 0]),
                 ("FRONT_CAMERA_FOVY_DEG", 0), ("FRONT_CAMERA_FOVY_DEG", float("inf"))]
        for field, value in cases:
            with self.subTest(field=field, value=value), patch(f"three_block_scene.{field}", value):
                with self.assertRaises(ValueError):
                    ThreeBlockScene()


if __name__ == "__main__":
    unittest.main()
