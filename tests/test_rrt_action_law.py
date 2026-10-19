"""rrt_action_law: the RRT pilot's action law (anti-windup PI edges, holds, clipping, nominal actions).

* LawTests: numpy only (mock plants built from the module's own calibrated NominalModel, a wall, a dragging push);
  run in either venv.
* SceneTests: the real MuJoCo scene (simulation venv, GPU render env not needed: nothing is rendered). They build
  their start poses from env.reset() and the carry controller (no dataset is read), check the scene constants, the
  free-space speed and the nominal-vs-executed mismatch against the stated tolerance, and a real blocked gripper
  (closed fingers pressed onto the green block, which rests on the floor) with and without anti-windup.
"""
import json
import unittest

import numpy as np

import random_branches_plan as plan
import rrt_action_law as law

try:
    import mujoco  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

# The wm_full_001 normalization values (typed here so the numpy tests need no data file; SceneTests read the real file).
NORMALIZATION = {"mean": [0.040703438085425696, -0.012399661743676687, 0.05471802006470693, 0.3239098870401013],
                 "scale": [0.050561455154413196, 0.047370118599586605, 0.07757630042049778, 0.9460881747008939]}
CFG = law.LawConfig(normalization=NORMALIZATION)
MM = 1e-3


class WallPlant(law.ModelPlant):
    """Model plant with a rigid wall: x can never exceed ``wall_x`` (a fixed obstacle)."""

    def __init__(self, model, start, wall_x):
        super().__init__(model, start)
        self.wall_x = wall_x

    def step(self, action):
        super().step(action)
        if self.position[0] > self.wall_x:
            self.position[0] = self.wall_x
            self.velocity[0] = 0.0


class PushedPlant(law.ModelPlant):
    """Model plant whose pinch is carried +push per step along y (e.g. by a toppling guard), whatever the action."""

    def __init__(self, model, start, push):
        super().__init__(model, start)
        self.push = push

    def step(self, action):
        super().step(action)
        self.position[1] += self.push


def unprotected_pi(plant, path, grip=1.0):
    """AnchoredHold / ReferenceTracker law without anti-windup (errors and integral clipped to +-5 cm only)."""
    integral, actions = np.zeros(3), []
    for ref in path:
        error = np.clip(ref - plant.pinch(), -0.05, 0.05)
        integral = np.clip(integral + error, -0.05, 0.05)
        a = law.normalize_env(np.array([*np.clip(error + integral, -0.05, 0.05), 0.0, 0.0]))
        a[4] = grip
        plant.step(a)
        actions.append(a)
    return np.asarray(actions)


class LawTests(unittest.TestCase):
    def test_constants_match_the_data_controllers(self):
        np.testing.assert_array_equal(law.BOX_LOW, plan.REF_BOX_LOW)
        np.testing.assert_array_equal(law.BOX_HIGH, plan.REF_BOX_HIGH)
        np.testing.assert_array_equal(law.WORKSPACE_LOW, plan.WORKSPACE_LOW)
        np.testing.assert_array_equal(law.WORKSPACE_HIGH, plan.WORKSPACE_HIGH)
        self.assertEqual((law.DIRECTION_STD, law.GUARD_ANGLE_DEG, law.CONTROL_DT), (plan.DIRECTION_STD, plan.GUARD_ANGLE_DEG, plan.CONTROL_DT))
        self.assertAlmostEqual(law.STEP_M, 0.001, places=15)
        self.assertEqual((law.EDGE_STEPS, law.CHECK_HOLD_STEPS, law.CHECK_STEPS), (8, 12, 20))
        self.assertEqual((law.GAIN_P, law.GAIN_I), (1.0, 1.0))
        actions = np.random.default_rng(0).uniform(-1, 1, (50, 5))
        np.testing.assert_array_equal(law.wm_normalize(actions, NORMALIZATION), plan.normalize_wm_actions(actions, NORMALIZATION))

    def test_wm_bounds_keep_the_float32_recipe_inside_b_training_range(self):
        low, high = CFG.env_bounds()
        z = law.wm_normalize(np.array([[*low, 0, 1.0], [*high, 0, -1.0]]), NORMALIZATION)
        self.assertTrue(np.all(z[0, :3] >= law.WM_RANGE_LOW) and np.all(z[1, :3] <= law.WM_RANGE_HIGH))
        self.assertTrue(np.all(z[0, :3] <= law.WM_RANGE_LOW + 2e-4) and np.all(z[1, :3] >= law.WM_RANGE_HIGH - 2e-4))   # tight
        # physical meaning: dx in [-15.5, +10.0] mm, dy in [-17.1, +11.8] mm, dz in [-18.6, +13.1] mm
        np.testing.assert_allclose(1000 * law.unnormalize_env(np.array([*low, 0, 0]))[:3], [-15.48, -17.13, -18.64], atol=0.01)
        np.testing.assert_allclose(1000 * law.unnormalize_env(np.array([*high, 0, 0]))[:3], [10.0, 11.79, 13.09], atol=0.01)
        for u in np.random.default_rng(1).uniform(-0.08, 0.08, (200, 3)):
            a, flags = law.env_action(u, 1.0, CFG)
            w = law.wm_normalize(a, NORMALIZATION)[:3]
            self.assertTrue(np.all(w >= law.WM_RANGE_LOW) and np.all(w <= law.WM_RANGE_HIGH))
            self.assertEqual((a[3], a[4]), (0.0, 1.0))
            np.testing.assert_array_equal(flags["wm_clipped"], (a[:3] == low) | (a[:3] == high))
        a, flags = law.env_action([0.001, -0.002, 0.0005], -1.0, CFG)
        np.testing.assert_allclose(a[:3], [0.02, -0.04, 0.01])                     # small actions pass unchanged
        self.assertFalse(flags["wm_clipped"].any())

    def test_free_space_speed_is_one_mm_per_step(self):
        model = law.NOMINAL_MODELS["grasped"]
        start = np.array([0.40, -0.01, 0.18])
        for d in ([1, 0, 0], [0, -1, 0], [0, 0, 1], [1, 1, -0.5]):
            d = law.unit(d)
            r = law.run_sequence(law.ModelPlant(model, start), [law.edge(d)] * 4 + [law.hold(12)], 1.0, CFG)
            ref_step = np.linalg.norm(np.diff(np.vstack([start, r["reference_xyz"][:32]]), axis=0), axis=1)
            np.testing.assert_allclose(ref_step, 0.001, atol=1e-12)                 # the reference advances exactly 1 mm per step
            speed = (r["pinch_xyz"][32] - r["pinch_xyz"][8]) @ d / 24               # edges 2-4 continue the first
            self.assertAlmostEqual(speed / MM, 1.0, delta=0.03)
            self.assertAlmostEqual((r["pinch_xyz"][-1] - start) @ d / MM, 32.0, delta=1.5)   # 4 edges end ~32 mm away
            self.assertEqual(r["segment_continued"].tolist(), [False, True, True, True, False])
            for key in ("reference_frozen", "reference_dragged", "box_clipped", "range_clipped"):
                self.assertFalse(np.any(r[key]), key)
            self.assertFalse(np.any(r["wm_clipped"]))
            self.assertLess(r["lag_m"].max(), law.LAG_MARGIN_M / 2)

    def test_blocked_gripper_keeps_actions_bounded(self):
        model = law.NOMINAL_MODELS["empty"]
        start = np.array([0.40, 0.0, 0.18])
        segments = [law.edge([1, 0, 0])] * 3 + [law.hold(8), law.edge([-1, 0, 0])]
        r = law.run_sequence(WallPlant(model, start, start[0] + 0.002), segments, -1.0, CFG)
        self.assertTrue(r["reference_frozen"].any())
        self.assertLessEqual(r["pinch_xyz"][:, 0].max(), start[0] + 0.002 + 1e-15)
        self.assertTrue(np.all(np.abs(r["integral_m"]) <= law.INTEGRAL_MAX_M + 1e-15))
        self.assertTrue(np.all(np.abs(r["u_m"]) <= law.LEAD_MAX_M + law.INTEGRAL_MAX_M + 1e-12))
        lead = np.linalg.norm(r["reference_xyz"] - r["pinch_xyz"][:-1], axis=1)
        self.assertTrue(np.all(lead <= law.LEAD_MAX_M + 1e-12))
        w = law.wm_normalize(r["actions"], NORMALIZATION)[:, :3]
        self.assertTrue(np.all(w >= law.WM_RANGE_LOW) and np.all(w <= law.WM_RANGE_HIGH))
        blocked_u = r["u_m"][8:24, 0]
        self.assertLess(blocked_u.max(), 0.0185)                                   # frozen lead (~6 mm) + integral clamp (12 mm)
        # the hold after the blocked edges starts fresh at the pinch: its first action is exactly zero (stops pushing)
        self.assertEqual(r["segment_index"][24], 3)
        np.testing.assert_array_equal(r["u_m"][24], np.zeros(3))
        self.assertEqual(r["segment_continued"].tolist(), [False] * 5)                # blocked: every edge restarts at the pinch
        self.assertTrue(r["reference_frozen"][:8].any())                          # the first edge was already blocked
        # without anti-windup the same push saturates the +-5 cm action range (far outside B's training range)
        path = start + np.outer(np.arange(1, 25), [0.001, 0, 0])
        raw = unprotected_pi(WallPlant(model, start, start[0] + 0.002), path, -1.0)
        self.assertEqual(raw[:, 0].max(), 1.0)
        self.assertGreater(law.wm_normalize(raw, NORMALIZATION)[:, 0].max(), 18.0)

    def test_pushed_gripper_drags_the_reference(self):
        model = law.NOMINAL_MODELS["grasped"]
        start = np.array([0.42, -0.02, 0.12])
        r = law.run_sequence(PushedPlant(model, start, 0.012), [law.hold(30)], 1.0, CFG)      # carried away at 24 cm/s
        self.assertTrue(r["reference_dragged"].any())
        lead = np.linalg.norm(r["reference_xyz"] - r["pinch_xyz"][:-1], axis=1)
        self.assertTrue(np.all(lead <= law.LEAD_MAX_M + 1e-12))
        self.assertTrue(np.all(np.abs(r["u_m"]) <= law.LEAD_MAX_M + law.INTEGRAL_MAX_M + 1e-12))
        self.assertTrue(np.all(np.abs(r["integral_m"]) <= law.INTEGRAL_MAX_M + 1e-15))

    def test_reversal_unwinds_the_integral(self):
        model = law.NOMINAL_MODELS["grasped"]
        start = np.array([0.40, 0.0, 0.18])
        r = law.run_sequence(law.ModelPlant(model, start), [law.edge([1, 0, 0]), law.edge([-1, 0, 0]), law.edge([-1, 0, 0]), law.hold(12)], 1.0, CFG)
        self.assertEqual(r["segment_continued"].tolist(), [False, True, True, False])
        self.assertGreater(r["integral_m"][7, 0], 0)
        self.assertLess(r["integral_m"][23, 0], 0)                                  # unwound and reversed
        self.assertLess(r["lag_m"].max(), law.LAG_MARGIN_M)
        self.assertAlmostEqual(r["pinch_xyz"][-1, 0], start[0] - 0.008, delta=0.0015)

    def test_integral_may_unwind_while_frozen(self):
        t = law.SegmentTracker("edge", [0.4, 0, 0.18], direction=[1, 0, 0], grip=1.0, config=CFG,
                               start_reference=[0.4, 0, 0.18], start_integral=[0.004, 0, 0])
        t.action([0.392, 0, 0.18])                                                  # lag 8 mm behind: frozen, error along +x
        self.assertTrue(t.log["reference_frozen"][0])
        self.assertAlmostEqual(t.integral[0], 0.004)                                # same sign as the error: held
        t2 = law.SegmentTracker("edge", [0.4, 0, 0.18], direction=[1, 0, 0], grip=1.0, config=CFG,
                                start_reference=[0.4, 0, 0.18], start_integral=[0.004, 0, 0])
        t2.action([0.408, 0, 0.18])                                                 # 8 mm AHEAD: error -x shrinks the integral
        self.assertTrue(t2.log["reference_frozen"][0])
        self.assertAlmostEqual(t2.integral[0], 0.004 - 0.008)

    def test_workspace_box_and_minimum_height(self):
        r = law.run_sequence(law.ModelPlant(law.NOMINAL_MODELS["empty"], [0.4985, 0.0, 0.15]), [law.edge([1, 0, 0])], -1.0, CFG)
        self.assertAlmostEqual(r["reference_xyz"][:, 0].max(), law.BOX_HIGH[0])
        self.assertTrue(r["box_clipped"].any())
        r = law.run_sequence(law.ModelPlant(law.NOMINAL_MODELS["empty"], [0.40, 0.0, 0.030]), [law.edge([0, 0, -1])], -1.0, CFG)
        self.assertAlmostEqual(r["reference_xyz"][:, 2].min(), law.BOX_LOW[2])      # minimum height 2.5 cm
        # outside the box: never jumps, never moves further out; moving back in is allowed
        ref, clipped = law.box_step([0.52, 0.0, 0.15], [0.001, 0.0005, 0.0])
        np.testing.assert_allclose(ref, [0.52, 0.0005, 0.15])
        self.assertTrue(clipped)
        ref, clipped = law.box_step([0.52, 0.0, 0.15], [-0.001, 0.0, 0.0])
        np.testing.assert_allclose(ref, [0.519, 0.0, 0.15])
        self.assertFalse(clipped)
        low, high = CFG.box()
        np.testing.assert_array_equal(low, np.maximum(law.BOX_LOW, law.WORKSPACE_LOW))
        np.testing.assert_array_equal(high, np.minimum(law.BOX_HIGH, law.WORKSPACE_HIGH))

    def test_segments_runner_and_chaining_across_runners(self):
        segs = [law.edge([0, 1, 0]), law.edge([1, 0, 0]), law.hold(5), law.edge([0, 0, 1])]
        self.assertEqual(law.sequence_steps(segs, CFG), 29)
        model = law.NOMINAL_MODELS["grasped"]
        start = np.array([0.41, 0.0, 0.17])
        whole = law.run_sequence(law.ModelPlant(model, start), segs, 1.0, CFG)
        self.assertEqual(whole["actions"].shape, (29, 5))
        self.assertTrue(np.all(whole["actions"][:, 3] == 0) and np.all(whole["actions"][:, 4] == 1.0))
        self.assertEqual(whole["segment_index"].tolist(), [0] * 8 + [1] * 8 + [2] * 5 + [3] * 8)
        np.testing.assert_array_equal(whole["u_m"][16], np.zeros(3))                # hold starts at the pinch, zero integral
        np.testing.assert_array_equal(whole["reference_xyz"][16:21], np.tile(whole["pinch_xyz"][16], (5, 1)))
        np.testing.assert_allclose(whole["reference_xyz"][21], whole["pinch_xyz"][21] + [0, 0, 0.001])   # fresh edge after a hold
        # a planner that builds one runner per decision and passes carry_state() reproduces the single runner
        plant = law.ModelPlant(model, start)
        first = law.run_sequence(plant, segs[:1], 1.0, CFG)
        second = law.run_sequence(plant, segs[1:], 1.0, CFG, previous=first["carry_state"])
        np.testing.assert_array_equal(np.vstack([first["actions"], second["actions"]]), whole["actions"])
        # nominal actions of the second decision = what the executed runner outputs on the model plant
        nominal = law.nominal_actions(segs[1:], first["pinch_xyz"][-1], 1.0, CFG, "grasped",
                                      prev_velocity=first["pinch_xyz"][-1] - first["pinch_xyz"][-2], previous=first["carry_state"])
        np.testing.assert_allclose(nominal, second["actions"], atol=1e-12)
        runner = law.LawRunner([law.hold(1)], 1.0, CFG)
        runner.action(start)
        with self.assertRaises(RuntimeError):
            runner.action(start)

    def test_nominal_edge_then_hold(self):
        start = np.array([0.42, 0.01, 0.17])
        a = law.nominal_edge_then_hold([1, 0.5, 0], start, 1.0, CFG)
        self.assertEqual(a.shape, (20, 5))
        np.testing.assert_array_equal(a, law.nominal_edge_then_hold([2, 1, 0], start, 1.0, CFG))   # direction is normalised
        np.testing.assert_array_equal(a, law.nominal_actions([law.edge([1, 0.5, 0]), law.hold(12)], start, 1.0, CFG, "grasped"))
        self.assertFalse(np.array_equal(a, law.nominal_edge_then_hold([1, 0.5, 0], start, 1.0, CFG, "empty")))
        v = law.nominal_prev_velocity([1, 0, 0], CFG)
        self.assertAlmostEqual(v[0], 0.001, delta=0.0003)
        np.testing.assert_array_equal(law.nominal_prev_velocity(None, CFG), np.zeros(3))
        w = law.wm_normalize(a, NORMALIZATION)
        self.assertTrue(np.all(w[:, :3] >= law.WM_RANGE_LOW) and np.all(w[:, :3] <= law.WM_RANGE_HIGH))
        m = law.action_mismatch(a, a, NORMALIZATION)
        self.assertEqual((m["mm"].max(), m["wm"].shape), (0.0, (20, 3)))

    def test_fit_recovers_a_synthetic_plant(self):
        rng = np.random.default_rng(3)
        truth = law.NominalModel(gain=tuple(map(tuple, 0.4 * np.eye(3) + 0.03 * rng.normal(size=(3, 3)))),
                                 memory=tuple(map(tuple, 0.08 * rng.normal(size=(3, 3)))), bias=(-4e-4, 2e-4, -1.1e-3), source="truth")
        runs = []
        for _ in range(12):
            v0 = rng.normal(0, 3e-4, 3)
            segs = [law.edge(law.random_direction(rng)) for _ in range(3)] + [law.hold(12)]
            r = law.run_sequence(law.ModelPlant(truth, [0.42, 0.0, 0.17], v0), segs, 1.0, CFG)
            runs.append({"actions": r["actions"], "pinch_xyz": r["pinch_xyz"], "v0": v0, "segments": segs})
        fitted, rms = law.fit_nominal_model(runs, "fit")
        np.testing.assert_allclose(np.asarray(fitted.gain), np.asarray(truth.gain), atol=1e-8)
        np.testing.assert_allclose(np.asarray(fitted.memory), np.asarray(truth.memory), atol=1e-8)
        np.testing.assert_allclose(fitted.bias, truth.bias, atol=1e-10)
        self.assertLess(rms.max(), 1e-12)
        for r in runs:
            nominal = law.nominal_actions(r["segments"], r["pinch_xyz"][0], 1.0, CFG, fitted, prev_velocity=r["v0"])
            self.assertLess(law.action_mismatch(nominal, r["actions"], NORMALIZATION)["mm"].max(), 1e-6)

    def test_module_constants_are_the_calibration_run(self):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "runs" / "rrt_action_law_calib_002" / "summary.json"
        if not path.is_file():
            self.skipTest("calibration run not present")
        models = json.loads(path.read_text(encoding="utf-8"))["models"]
        for name, model in law.NOMINAL_MODELS.items():
            for key in ("gain", "memory", "bias"):
                np.testing.assert_array_equal(np.asarray(getattr(model, key)), np.asarray(models[name][key]), err_msg=f"{name}.{key}")
            self.assertEqual(model.source, f"runs/rrt_action_law_calib_002:{name}")

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            law.unit([0, 0, 0])
        with self.assertRaises(ValueError):
            law.SegmentTracker("edge", [0.4, 0, 0.2], config=CFG)                   # no direction
        with self.assertRaises(ValueError):
            law.SegmentTracker("hold", [0.4, 0, 0.2], direction=[1, 0, 0], config=CFG)
        with self.assertRaises(ValueError):
            law.SegmentTracker("edge", [0.4, 0, 0.2], direction=[1, 0, 0], grip=1.5, config=CFG)
        with self.assertRaises(ValueError):
            law.SegmentTracker("edge", [0.4, 0, 0.2], direction=[1, 0, 0], config=CFG, start_reference=[0.4, 0, 0.2], start_integral=[0.02, 0, 0])
        with self.assertRaises(ValueError):
            law.SegmentTracker("edge", [0.4, 0, 0.2], direction=[1, 0, 0])            # config required
        with self.assertRaises(ValueError):
            law.LawConfig(normalization=NORMALIZATION, lead_max_m=0.005)           # < margin + one step
        with self.assertRaises(ValueError):
            law.LawConfig(normalization={"mean": [0, 0, 0], "scale": [1, 1, 1, 1]})
        with self.assertRaises(ValueError):
            law.hold(0)
        with self.assertRaises(ValueError):
            law.validate_segments([("jump", 3)])
        with self.assertRaises(ValueError):
            law.resolve_model("heavy")
        t = law.SegmentTracker("hold", [0.4, 0, 0.2], steps=1, config=CFG)
        t.action([0.4, 0, 0.2])
        with self.assertRaises(RuntimeError):
            t.action([0.4, 0, 0.2])
        self.assertEqual(CFG.to_json()["format"], law.LAW_FORMAT)


@unittest.skipUnless(HAVE_SIM, "needs MuJoCo (simulation venv)")
class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import calibrate_rrt_action_law as cal
        from exploration import ExplorationScene
        cls.cal = cal
        index = Path(__file__).resolve().parents[1] / "data" / "wm_full_001" / "index.json"
        cls.normalization = law.load_wm_normalization(index) if index.is_file() else NORMALIZATION
        cls.config = law.LawConfig(normalization=cls.normalization)
        cls.env = ExplorationScene()
        starts = {s["name"]: s for s in cal.build_starts(cls.env, cls.config)}
        cls.starts = {k: starts[k] for k in ("empty_reset", "grasped_goal")}

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_scene_constants(self):
        import bench_restore_check as brc
        env = self.env
        np.testing.assert_array_equal(env.action_high, law.ACTION_RANGE)
        np.testing.assert_array_equal(env.action_low, -law.ACTION_RANGE)
        np.testing.assert_array_equal(np.asarray(env._workspace_bounds[0]), law.WORKSPACE_LOW)
        np.testing.assert_array_equal(np.asarray(env._workspace_bounds[1]), law.WORKSPACE_HIGH)
        self.assertEqual((brc.HOLD_GAIN_P, brc.HOLD_GAIN_I, brc.ACTION_XYZ_LIMIT_M), (law.GAIN_P, law.GAIN_I, law.ACTION_XYZ_LIMIT_M))
        self.assertAlmostEqual(env.control_timestep(), law.CONTROL_DT)
        for x in np.random.default_rng(2).uniform(-0.06, 0.06, (50, 5)):
            np.testing.assert_array_equal(law.normalize_env(x), env.normalize_action(x))
            np.testing.assert_array_equal(law.unnormalize_env(x / 0.06), env.unnormalize_action(x / 0.06))

    def run_from(self, name, segments):
        from bench_restore_check import restore_state
        s = self.starts[name]
        restore_state(self.env, s["state"], s["goal"])
        plant = self.cal.ScenePlant(self.env)
        grasped = bool(self.env.task_report()["grasped"])
        r = law.run_sequence(plant, segments, s["grip"], self.config)
        return r, plant.contact, grasped, bool(self.env.task_report()["grasped"])

    def test_free_space_speed_and_nominal_mismatch(self):
        rng = np.random.default_rng(777)                                            # not a calibration seed
        for name, model in (("empty_reset", "empty"), ("grasped_goal", "grasped")):
            s = self.starts[name]
            for d in ([-1, 0, 0], [0, 1, 0], [0, 0, 1]):
                d = law.unit(d)
                r, contact, g0, g1 = self.run_from(name, [law.edge(d)] * 3 + [law.hold(12)])
                self.assertEqual((contact, g0, g1), (0.0, s["grip"] > 0, s["grip"] > 0))
                speed = (r["pinch_xyz"][24] - r["pinch_xyz"][8]) @ d / 16
                self.assertAlmostEqual(speed / MM, 1.0, delta=0.05, msg=f"{name} {d}")
            d_all = []
            for _ in range(4):
                segs = [law.edge(law.random_direction(rng)) for _ in range(3)] + [law.hold(12)]
                r, contact, g0, g1 = self.run_from(name, segs)
                self.assertEqual((contact, g0 == g1), (0.0, True))
                for key in ("reference_frozen", "integral_clamped", "reference_dragged", "range_clipped"):
                    self.assertFalse(np.any(r[key]), f"{name}: {key} in free space")
                nominal = law.nominal_actions(segs, r["pinch_xyz"][0], s["grip"], self.config, model, prev_velocity=s["v0"])
                d_all.append(law.action_mismatch(nominal, r["actions"], self.normalization)["mm"])
            d_all = np.concatenate(d_all)
            self.assertLessEqual(np.median(d_all), self.cal.TOLERANCE_MM["median"], name)
            self.assertLessEqual(d_all.max(), self.cal.TOLERANCE_MM["max"], name)

    def test_blocked_on_the_green_block(self):
        """Closed empty gripper lowered onto the green block (resting on the floor): a real fixed obstacle."""
        from bench_restore_check import restore_state, pinch_xyz
        from collect_random_branches import ReferenceTracker
        env = self.env
        env.reset(seed=0)
        plant = self.cal.ScenePlant(env)
        law.run_sequence(plant, [law.hold(4)] + [law.edge([1, 0, 0])] * 6 + [law.hold(20)], 1.0, self.config)
        state, goal = self.cal.get_state(env), env.hover_goal.copy()
        start = pinch_xyz(env)
        r = law.run_sequence(plant, [law.edge([0, 0, -1])] * 12 + [law.hold(12)], 1.0, self.config)
        self.assertTrue(r["reference_frozen"].any())                               # the gripper was blocked
        self.assertGreater(r["pinch_xyz"][:, 2].min(), 0.10)                        # stopped on the block top (10 cm)
        self.assertTrue(np.all(np.abs(r["integral_m"]) <= law.INTEGRAL_MAX_M + 1e-15))
        w = law.wm_normalize(r["actions"], self.normalization)[:, :3]
        self.assertTrue(np.all(w >= law.WM_RANGE_LOW) and np.all(w <= law.WM_RANGE_HIGH))
        self.assertLess(max(g["tilt_deg"] for g in env.safety_report()["guards"]), 1.0)
        # the same reference path without anti-windup (collect_random_branches.ReferenceTracker) saturates
        restore_state(env, state, goal)
        path = start + np.outer(np.arange(1, 97), [0, 0, -0.001])
        path[:, 2] = np.maximum(path[:, 2], law.BOX_LOW[2])
        tracker = ReferenceTracker(np.vstack([path, np.tile(path[-1], (12, 1))]), 1.0, 0.05)
        raw = []
        for _ in range(108):
            a = tracker.action(env)
            env.step(a)
            raw.append(a)
        raw = np.asarray(raw)
        self.assertEqual(np.abs(raw[:, 2]).max(), 1.0)
        self.assertGreater(np.abs(law.wm_normalize(raw, self.normalization)[:, :3]).max(), 10.0)


if __name__ == "__main__":
    unittest.main()
