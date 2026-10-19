"""Small simulator checks; no trained model or GUI required."""

import copy
import unittest

import numpy as np

from scene_config import BLOCKS
from three_block_scene import ThreeBlockScene


class SceneTests(unittest.TestCase):
    def make_env(self, blocks=None):
        env = ThreeBlockScene(blocks=blocks)
        self.addCleanup(env.close)
        return env

    def check_blocks(self, env, expected):
        report = env.block_report()
        self.assertEqual(len(report), 3)
        for actual, block in zip(report, expected):
            self.assertTrue(actual["free_joint"])
            self.assertTrue(actual["collision_enabled"])
            np.testing.assert_allclose(actual["size_xyz"], block["size"])
            np.testing.assert_allclose(actual["rgba"][:3], block["color"], atol=1e-7)
            self.assertAlmostEqual(actual["mass_kg"], 1240 * np.prod(block["size"]))
            np.testing.assert_allclose(actual["position_xyz"],
                                       [*block["xy"], block["size"][2] / 2], atol=1e-8)

    def test_default_geometry_and_flags(self):
        env = self.make_env()
        _, info = env.reset(seed=0)
        self.check_blocks(env, BLOCKS)
        self.assertFalse(info["preview_only"])
        self.assertTrue(info["task_defined"])
        self.assertFalse(env._success)
        self.assertEqual(env.compute_reward(), 0)
        physical = {env.model.geom(f"object_{i}").id for i in range(3)}
        for contact in env.data.contact:
            self.assertFalse(contact.geom1 in physical and contact.geom2 in physical)
        for i in range(3):
            self.assertEqual(env.model.geom(f"target_object_{i}").rgba[3], 0)

    def test_default_blocks_are_equal_upright_rectangular_prisms(self):
        env = self.make_env()
        env.reset(seed=0)
        for row in env.block_report():
            np.testing.assert_allclose(row["size_xyz"], [0.04, 0.04, 0.10])
            self.assertGreater(row["size_xyz"][2], row["size_xyz"][0])

    def test_equal_blocks_start_with_bases_on_table(self):
        env = self.make_env()
        env.reset(seed=0)
        for row in env.block_report():
            self.assertAlmostEqual(row["position_xyz"][2], 0.05)
            self.assertAlmostEqual(row["position_xyz"][2] - row["size_xyz"][2] / 2, 0.0)

    def test_default_guards_are_symmetric_five_cm_from_target(self):
        env = self.make_env()
        env.reset(seed=0)
        positions = np.asarray([row["position_xyz"] for row in env.block_report()])
        np.testing.assert_allclose(positions[:, :2], [
            [0.425, 0.0], [0.425, -0.05], [0.425, 0.05],
        ], atol=1e-12)
        np.testing.assert_allclose((positions[1] + positions[2]) / 2, positions[0], atol=1e-12)

    def test_target_to_guard_surface_gaps_are_one_cm(self):
        env = self.make_env()
        env.reset(seed=0)
        target, *guards = env.block_report()
        for guard in guards:
            center_distance = abs(guard["position_xyz"][1] - target["position_xyz"][1])
            surface_gap = center_distance - (guard["size_xyz"][1] + target["size_xyz"][1]) / 2
            self.assertAlmostEqual(surface_gap, 0.01)
        # The span between the red inner faces is 6 cm;
        # the 4-cm green block occupies its center.
        red_inner_span = (guards[1]["position_xyz"][1] - guards[0]["position_xyz"][1]
                          - (guards[0]["size_xyz"][1] + guards[1]["size_xyz"][1]) / 2)
        self.assertAlmostEqual(red_inner_span, 0.06)

    def test_reset_uses_closer_guard_positions_as_safe_displacement_baseline(self):
        env = ThreeBlockScene(displacement_check_enabled=True)
        self.addCleanup(env.close)
        for seed in (0, 42):
            _, info = env.reset(seed=seed)
            self.assertFalse(info["unsafe"])
            self.assertFalse(info["episode_failed"])
            self.assertFalse(info["success"])
            safety = env.safety_report()
            self.assertEqual(safety["unsafe_displacement_m"], 0.01)
            for guard, y in zip(safety["guards"], (-0.05, 0.05)):
                np.testing.assert_allclose(guard["initial_com_xy_m"], [0.425, y], atol=1e-12)
                np.testing.assert_allclose(guard["com_xy_m"], [0.425, y], atol=1e-12)
                self.assertAlmostEqual(guard["displacement_xy_m"], 0.0)
                self.assertFalse(guard["unsafe"])
            # Explicit sensor/reset fixture, not a simulated robot push:
            # after a genuine threshold exceedance, reset must restore both
            # the new placement and its displacement baseline.
            env.data.joint("object_joint_1").qpos[0] += 0.012
            env.post_step()
            self.assertTrue(env.task_report()["episode_failed"])

    def test_target_mass_follows_new_geometry_without_density_override(self):
        env = self.make_env()
        env.reset(seed=0)
        # Explicit historical 4-cm cube fixture, independent of today's default.
        previous_blocks = copy.deepcopy(BLOCKS)
        previous_blocks[0]["size"] = [0.04, 0.04, 0.04]
        previous_blocks[1]["xy"] = [0.425, -0.09]
        previous_blocks[2]["xy"] = [0.425, 0.09]
        previous = self.make_env(previous_blocks)
        previous.arm_start = np.array([0.30, 0.0, 0.25])
        previous.reset(seed=0)
        target_mass = env.model.body("object_0").mass
        self.assertAlmostEqual(target_mass, 1240 * 0.04 * 0.04 * 0.10)
        self.assertAlmostEqual(target_mass / previous.model.body("object_0").mass, 2.5)
        for index in (1, 2):
            self.assertAlmostEqual(target_mass, env.model.body(f"object_{index}").mass)

    def test_taller_target_is_not_completed_just_by_existing(self):
        env = self.make_env()
        _, info = env.reset(seed=0)
        self.assertAlmostEqual(info["task"]["target_height_m"], 0.05)
        self.assertFalse(info["task"]["height_ok"])
        self.assertFalse(info["task"]["grasped"])
        self.assertFalse(info["task"]["task_success"])
        self.assertEqual(info["task"]["hold_seconds"], 0.0)
        self.assertFalse(info["task"]["episode_failed"])
        self.assertFalse(info["unsafe"])

    def test_physics_and_reset(self):
        env = self.make_env()
        env.reset(seed=123)
        initial = env.data.qpos.copy()
        initial_positions = np.asarray([r["position_xyz"] for r in env.block_report()])
        for _ in range(40):
            _, reward, terminated, _, info = env.step(np.zeros(5))
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertTrue(info["task_defined"])
        self.assertTrue(np.isfinite(env.data.qpos).all())
        self.assertTrue(np.isfinite(env.data.qvel).all())
        final_positions = np.asarray([r["position_xyz"] for r in env.block_report()])
        np.testing.assert_allclose(final_positions, initial_positions, atol=0.001)
        env.reset(seed=456)
        np.testing.assert_allclose(env.data.qpos, initial, atol=1e-10)
        self.check_blocks(env, BLOCKS)

    def test_explicit_historical_cube_and_tall_guards(self):
        original_blocks = copy.deepcopy(BLOCKS)
        blocks = copy.deepcopy(BLOCKS)
        blocks[0]["size"] = [0.04, 0.04, 0.04]
        blocks[1]["xy"] = [0.425, -0.09]
        blocks[2]["xy"] = [0.425, 0.09]
        for block in blocks[1:]:
            block["size"] = [0.04, 0.04, 0.10]
        env = self.make_env(blocks)
        env.arm_start = np.array([0.30, 0.0, 0.25])
        env.reset(seed=0)
        self.check_blocks(env, blocks)
        for i, block in enumerate(blocks):
            x, y, z = block["size"]
            mass = 1240 * x * y * z
            expected_inertia = mass / 12 * np.array([y*y + z*z, x*x + z*z, x*x + y*y])
            # MuJoCo may sort principal axes; compare their eigenvalues.
            np.testing.assert_allclose(np.sort(env.model.body(f"object_{i}").inertia),
                                       np.sort(expected_inertia))
        for _ in range(40):
            env.step_preview()
        for row, block in zip(env.block_report(), blocks):
            np.testing.assert_allclose(row["position_xyz"],
                                       [*block["xy"], block["size"][2] / 2], atol=0.001)
        self.assertEqual(BLOCKS, original_blocks)

    def test_historical_start_preview_holds_arm(self):
        env = self.make_env()
        env.arm_start = np.array([0.30, 0.0, 0.25])
        env.reset(seed=0)
        initial = env.data.site_xpos[env._pinch_site_id].copy()
        for _ in range(300):
            env.step_preview()
        self.assertLess(np.linalg.norm(env.data.site_xpos[env._pinch_site_id] - initial), 0.003)
        for row, block in zip(env.block_report(), BLOCKS):
                np.testing.assert_allclose(row["position_xyz"],
                                       [*block["xy"], block["size"][2] / 2], atol=0.001)

    def test_reject_overlap(self):
        blocks = copy.deepcopy(BLOCKS)
        blocks[1]["xy"] = blocks[0]["xy"][:]
        with self.assertRaisesRegex(ValueError, "overlap"):
            ThreeBlockScene(blocks)

    def test_reject_invalid_size(self):
        blocks = copy.deepcopy(BLOCKS)
        blocks[1]["size"][2] = -0.1
        with self.assertRaisesRegex(ValueError, "positive"):
            ThreeBlockScene(blocks)

    def test_reject_outside_workspace(self):
        blocks = copy.deepcopy(BLOCKS)
        blocks[1]["xy"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "workspace"):
            ThreeBlockScene(blocks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
