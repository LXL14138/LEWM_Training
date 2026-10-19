"""Danger-benchmark controller checks on the real MuJoCo scene (run in WSL).

Injected guard poses are test fixtures, never demonstration data. Physics
outcomes (does a guard actually fall) are NOT asserted here; the pre-check
dataset and tools/bench_precheck_report.py answer that empirically.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np

from benchmark_exploration import BenchmarkController, PARK_XY, TRANSFER_Z
from benchmark_plan import ALL_CELLS, BENCH_PHASES, RECIPE_MODE, lean_vector, make_schedule, sample_intent, write_schedule
from collect_exploration_trajs import collect_exploration
from exploration import ExplorationController, ExplorationScene, PHASES
from exploration_validation import EXPLORATION_PHASES
from inspect_grasp_dataset import validate_dataset


def intent(recipe, cell, guard=1, lean="+x", stop=("stroke", 0.06, 0.12), after="hold", park=False, height=(0.035, 0.095)):
    return {"cell": cell, "recipe": recipe, "guard_index": guard, "lean": lean,
            "stop": None if stop is None else [stop[0], float(stop[1]), float(stop[2])],
            "after_event": after, "park_green": park, "height": [float(height[0]), float(height[1])]}


def tilt_guard(env, index, degrees, axis=(0, 1, 0)):
    """Fixture: rotate a guard body in place, then refresh derived quantities."""
    joint = env.data.joint(f"object_joint_{index}")
    half = np.deg2rad(degrees) / 2
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    joint.qpos[3:] = [np.cos(half), *(np.sin(half) * axis)]
    env.post_step()


class BenchmarkControllerTests(unittest.TestCase):
    def make(self, it, seed=0, max_steps=240):
        env = ExplorationScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env, BenchmarkController(env, np.random.default_rng(seed), RECIPE_MODE[it["recipe"]], it, max_steps=max_steps)

    def test_phase_names_registered_and_distinct_from_base(self):
        for name in BENCH_PHASES:
            self.assertLessEqual(len(name), 16)
            self.assertIn(name, EXPLORATION_PHASES)
        # Registered in the controller phase set too (test_guard_interactions keeps PHASES == validator whitelist).
        self.assertTrue(set(BENCH_PHASES) <= PHASES)

    def test_mode_must_match_recipe(self):
        env = ExplorationScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        with self.assertRaises(ValueError):
            BenchmarkController(env, np.random.default_rng(0), "bench_free", intent("push", "toward_camera|S1_flat"))

    def test_settle_gives_twelve_frames_of_history_then_dispatches(self):
        for it, expected in ((intent("push", "toward_camera|S1_flat"), "BENCH_APPROACH"),
                             (intent("carry_hit", "toward_camera|S1_flat", stop=("stroke", 0.05, 0.1), height=(0.05, 0.12)), "APPROACH"),
                             (intent("near", "negative|N3_near", lean=None, stop=None, height=(0.03, 0.09)), "BENCH_NEAR")):
            env, c = self.make(it)
            phases = [c.step() for _ in range(13)]
            self.assertEqual(phases[:12], ["SETTLE"] * 12)
            self.assertEqual(phases[12], expected)
            self.assertAlmostEqual(env.data.time, 13 * 0.05)

    def test_push_approach_is_behind_the_lean_direction(self):
        cases = ((intent("push", "toward_camera|S1_flat", guard=1, lean="+x"), [0.34, -0.05]),
                 (intent("push", "away_camera|S1_flat", guard=2, lean="-x"), [0.51, 0.05]),
                 (intent("push", "lateral_inward|S3_just_over", guard=1, lean="inward", stop=("stroke", 0.03, 0.05), after="retreat", park=True), [0.425, -0.135]))
        for it, expected in cases:
            env, c = self.make(it)
            qpos = env.data.qpos.copy()
            c._start_push()
            self.assertEqual(c.phase, "BENCH_APPROACH")
            np.testing.assert_allclose(c.waypoint, [*expected, TRANSFER_Z], atol=1e-6)
            self.assertEqual(c.interaction["kind"], "bench_push")
            np.testing.assert_array_equal(env.data.qpos, qpos)
            self.assertEqual(env.data.time, 0)

    def test_press_geometry_stop_rule_and_continue_after_hazard(self):
        it = intent("press", "toward_camera|S3_just_over", guard=1, lean="+x", stop=("tilt", 31, 38), after="continue", height=(0, 0))
        env, c = self.make(it)
        c._start_press(False)
        self.assertEqual(c.phase, "BENCH_PRESS_MOVE")
        self.assertLess(c.waypoint[0], 0.425)
        self.assertAlmostEqual(c.waypoint[1], -0.05, places=6)
        with patch.object(BenchmarkController, "_arrived", return_value=True):
            c._advance_bench(float(env.data.time))
            self.assertEqual(c.phase, "BENCH_PRESS_DOWN")
            self.assertAlmostEqual(c.waypoint[2], 0.08)          # seek contact down to 2 cm below the top
            env._contact_peak[0] = 0.01                          # synthetic contact on guard 1
            c._advance_bench(float(env.data.time))
        self.assertEqual(c.phase, "BENCH_PRESS")
        self.assertIn("contact_pinch_xyz", c.interaction)
        np.testing.assert_allclose(c.waypoint - c.actual, [0.10, 0, -0.05], atol=1e-6)
        env._contact_peak[0] = 0.0
        # -x presses get the lighter slope (pre-check 2: pulling toward the base under load stalled).
        env_b, c_b = self.make(intent("press", "away_camera|S3_just_over", guard=1, lean="-x", stop=("tilt", 31, 38), after="continue", height=(0, 0)))
        c_b._start_press(False)
        with patch.object(BenchmarkController, "_arrived", return_value=True):
            c_b._advance_bench(float(env_b.data.time))
            env_b._contact_peak[0] = 0.01
            c_b._advance_bench(float(env_b.data.time))
        np.testing.assert_allclose(c_b.waypoint - c_b.actual, [-0.10, 0, -0.015], atol=1e-6)
        tilt_guard(env_b, 1, 30)                     # within 8 deg of a 31-38 target: creep speed
        c_b._advance_phase()
        self.assertEqual(c_b.speed, 0.008)
        self.assertIn("press_slowed_at_step", c_b.interaction)
        tilt_guard(env, 1, 40)                   # above every sampled target in [31, 38]
        c._advance_phase()                       # hazard -> continue: keep pressing
        self.assertTrue(c.hazard_handled)
        self.assertEqual(c.phase, "BENCH_PRESS")
        c._advance_phase()                       # tilt 40 >= sampled target -> hold
        self.assertEqual(c.phase, "EVENT_HOLD")
        record = c.report()["benchmark"]
        self.assertTrue(record["stop_rule_reached"])
        self.assertAlmostEqual(record["stop_rule_value"], 40, places=3)
        self.assertEqual(record["hazard_step"], 0)
        self.assertEqual(record["end_reason"], "bench_stop_reached")
        self.assertEqual(c.report()["interaction_revision"], 3)

    def test_after_event_hold_retreat_continue(self):
        for after in ("hold", "retreat", "continue"):
            with self.subTest(after=after):
                env, c = self.make(intent("push", "toward_camera|S1_flat", after=after))
                c._start_push()
                tilt_guard(env, 1, 40)
                c._advance_phase()
                self.assertTrue(c.hazard_handled)
                if after == "hold":
                    self.assertEqual(c.phase, "EVENT_HOLD")
                    np.testing.assert_array_equal(c.waypoint, c.actual)
                elif after == "retreat":
                    self.assertEqual(c.phase, "EVENT_RETREAT")
                    np.testing.assert_allclose(c.waypoint, [*c.actual[:2], TRANSFER_Z], atol=1e-9)  # leg 1: rise in place
                    guard = env.data.body("object_1").xipos[:2]
                    with patch.object(BenchmarkController, "_arrived", return_value=True):
                        c._advance_phase()
                        self.assertEqual(c.phase, "EVENT_RETREAT")                                  # leg 2: move away
                        self.assertAlmostEqual(c.waypoint[2], TRANSFER_Z)
                        self.assertGreater(np.linalg.norm(c.waypoint[:2] - guard), np.linalg.norm(c.actual[:2] - guard) + 0.11)
                        c._advance_phase()
                    self.assertEqual(c.phase, "EVENT_HOLD")
                else:
                    self.assertEqual(c.phase, "BENCH_APPROACH")
                    self.assertGreater(c.continue_deadline, env.data.time)
                    c.continue_deadline = float(env.data.time) - 1
                    c._advance_phase()
                    self.assertEqual(c.phase, "EVENT_HOLD")
                self.assertEqual(c.report()["benchmark"]["hazard_phase"], "BENCH_APPROACH")

    def test_park_prologue_and_slot_push_geometry(self):
        it = intent("slot_push", "lateral_outward|S1_flat", guard=2, lean="outward", stop=("stroke", 0.04, 0.06), after="hold", park=True, height=(0.035, 0.08))
        env, c = self.make(it)
        c.grip_offset = np.array([0, 0, 0.005])   # as set by a verified lift
        c._carry()
        self.assertEqual(c.phase, "BENCH_PARK")
        np.testing.assert_allclose(c.waypoint[:2], PARK_XY)
        with patch.object(BenchmarkController, "_arrived", return_value=True):
            c._advance_bench(float(env.data.time))
            self.assertEqual(c.phase, "BENCH_PARK_LOWER")
            c._advance_bench(float(env.data.time))
            self.assertEqual((c.phase, c.closed), ("BENCH_PARK_OPEN", False))
            c.started = float(env.data.time) - 1.0
            c._advance_bench(float(env.data.time))
            self.assertEqual((c.phase, c.closed), ("BENCH_RETRACT", False))   # open: never re-grasp the parked block
            c._advance_bench(float(env.data.time))
        self.assertEqual((c.phase, c.closed), ("BENCH_APPROACH", True))
        np.testing.assert_allclose(c.waypoint[:2], [0.425, 0.0], atol=1e-6)
        np.testing.assert_array_equal(c.push_direction, [0, 1, 0])
        self.assertEqual(c.seek_m, 0.04)

    def test_park_prologue_leaves_green_parked_and_gripper_empty(self):
        """Physics smoke: after the park prologue the block stays at PARK_XY and the gripper is empty."""
        it = intent("slot_push", "lateral_outward|S1_flat", guard=2, lean="outward", stop=("stroke", 0.04, 0.06),
                    after="hold", park=True, height=(0.035, 0.08))
        env, c = self.make(it, seed=3, max_steps=600)
        reached = False
        while not c.finished:
            c.step()
            if c.phase == "BENCH_APPROACH" and not reached:
                reached = True
                self.assertFalse(env.task_report()["grasped"])
                self.assertLess(np.linalg.norm(env.data.body("object_0").xipos[:2] - PARK_XY), 0.02)
        report = c.report()
        self.assertFalse(report["benchmark"]["budget_short"])
        if report["ever_lifted"]:
            self.assertTrue(reached, (c.stop_reason, report["events"]))

    def test_carry_sweep_uses_intent_guard_lean_and_contact_biased_gap(self):
        for guard, lean, cell in ((1, "+x", "toward_camera|S1_flat"), (2, "-x", "away_camera|S1_flat")):
            env, c = self.make(intent("carry_sweep", cell, guard=guard, lean=lean, stop=None, height=(0.075, 0.12)))
            c.grip_offset = np.array([0, 0, 0.005])
            c._begin_recipe()
            self.assertEqual(c.phase, "BRUSH_LIFT")
            self.assertEqual(c.interaction["guard_index"], guard)
            self.assertEqual(c.interaction["sweep_direction_xyz"][0], 1.0 if lean == "+x" else -1.0)
            self.assertTrue(-0.008 <= c.interaction["nominal_face_gap_m"] <= 0.002)
            self.assertTrue(0.12 <= c.interaction["requested_green_com_height_m"] <= 0.13)

    def test_press_no_contact_lifts_and_parked_inward_press_is_allowed(self):
        env, c = self.make(intent("press", "away_camera|S3_just_over", guard=1, lean="-x", stop=("tilt", 31, 38), after="continue", height=(0, 0)))
        c._start_press(False)
        with patch.object(BenchmarkController, "_arrived", return_value=True):
            c._advance_bench(float(env.data.time))      # MOVE -> DOWN
            c._advance_bench(float(env.data.time))      # DOWN arrived without contact -> lift away
        self.assertEqual((c.phase, c.report()["benchmark"]["end_reason"]), ("EVENT_RETREAT", "bench_press_no_contact"))
        env2, c2 = self.make(intent("press", "lateral_inward|S2_lean", guard=1, lean="inward", stop=("tilt", 42, 58), after="continue", park=True, height=(0, 0)))
        self.assertEqual(c2.report()["benchmark"]["prologue"], "grasp_park")
        c2.grip_offset = np.zeros(3)
        c2._carry()
        self.assertEqual(c2.phase, "BENCH_PARK")

    def test_drop_geometry_clears_guard_top_and_footprint(self):
        env, c = self.make(intent("drop", "toward_camera|S1_flat", guard=2, lean=None, stop=None, height=(0.19, 0.24)))
        c.grip_offset = np.zeros(3)
        c._start_drop()
        guard = env.data.body("object_2").xipos
        self.assertGreaterEqual(c.drop_com[2] - 0.05, 0.10 + 0.01 - 1e-9)                       # bottom clears the top
        self.assertIn(c.interaction["drop_axis"], ("+x", "-x", "outward"))
        axis = lean_vector(c.interaction["drop_axis"], 2)
        along = float(np.dot(c.drop_com[:2] - guard[:2], axis[:2]))
        self.assertTrue(0.012 - 1e-9 <= along <= 0.022 + 1e-9, along)                          # COM near the top edge
        self.assertEqual(c.phase, "BENCH_DROP_MOVE")
        self.assertAlmostEqual(c.waypoint[2], TRANSFER_Z)

    def test_continue_keeps_base_phase_machine_running_after_hazard(self):
        it = intent("carry_hit", "toward_camera|S1_flat", guard=2, stop=("stroke", 0.05, 0.1), after="continue", height=(0.06, 0.12))
        env, c = self.make(it)
        c._grasp()
        self.assertEqual(c.phase, "APPROACH")
        tilt_guard(env, 2, 40)
        c._advance_phase()
        self.assertTrue(c.hazard_handled)
        self.assertEqual(c.phase, "APPROACH")
        with patch.object(ExplorationController, "actual", new_callable=PropertyMock, return_value=c.waypoint.copy()):
            c._advance_phase()
        self.assertEqual(c.phase, "DESCEND")     # the base transition still fires while the hazard is acknowledged
        c.continue_deadline = float(env.data.time) - 1
        c._advance_phase()
        self.assertEqual(c.phase, "EVENT_HOLD")
        self.assertEqual(c.report()["benchmark"]["hazard_guard_index"], 2)

    def test_carry_hit_start_points(self):
        c_offsets = {"+x": [0.355, -0.05], "outward": [0.425, 0.0], "inward": [0.425, -0.12]}
        for lean, expected in c_offsets.items():
            cell = {"+x": "toward_camera|S1_flat", "outward": "lateral_outward|S1_flat", "inward": "lateral_inward|S3_just_over"}[lean]
            env, c = self.make(intent("carry_hit", cell, guard=1, lean=lean, stop=("stroke", 0.03, 0.05), after="retreat", height=(0.08, 0.08)))
            c.grip_offset = np.zeros(3)
            c._start_hit()
            self.assertEqual(c.phase, "BENCH_HIT_MOVE")
            np.testing.assert_allclose(c.waypoint, [*expected, TRANSFER_Z], atol=1e-6)
            np.testing.assert_allclose(c.interaction["entry_green_com_xyz"], [*expected, 0.08], atol=1e-6)

    def test_every_cell_runs_bounded_without_reset_or_invalid_actions(self):
        rng = np.random.default_rng(5)
        for cell in ALL_CELLS:
            with self.subTest(cell=cell):
                it = sample_intent(cell, rng)
                env, c = self.make(it, seed=1, max_steps=40)
                goal = env.hover_goal.copy()
                with patch.object(env, "reset", wraps=env.reset) as reset:
                    while not c.finished:
                        before = env.data.time
                        phase = c.step()
                        if phase is not None:
                            self.assertIn(phase, EXPLORATION_PHASES)
                            self.assertAlmostEqual(env.data.time - before, 0.05)
                            action = env.last_transition["action"]
                            self.assertTrue(np.all(np.abs(action) <= 1))
                            self.assertEqual(action[3], 0)
                    reset.assert_not_called()
                np.testing.assert_array_equal(env.hover_goal, goal)
                self.assertLessEqual(c.steps, 40)
                report = c.report()
                self.assertEqual((report["phase"], report["interaction_revision"]), ("DONE", 3))
                self.assertEqual(report["benchmark"]["intent"], it)
                json.dumps(report, allow_nan=False, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))

    def test_full_length_push_and_press_smoke(self):
        for it in (intent("push", "toward_camera|S1_flat", guard=1, lean="+x", stop=("stroke", 0.08, 0.08), after="hold", height=(0.06, 0.06)),
                   intent("press", "away_camera|S3_just_over", guard=2, lean="-x", stop=("tilt", 31, 38), after="continue", height=(0, 0))):
            with self.subTest(recipe=it["recipe"]):
                env, c = self.make(it, seed=2, max_steps=240)
                first_failed = None
                while not c.finished:
                    c.step()
                    if first_failed is None and env.task_report()["episode_failed"]:
                        first_failed = c.steps
                record = c.report()["benchmark"]
                if first_failed is None:
                    self.assertIsNone(record["hazard_step"])
                elif first_failed < c.steps:          # not the final state: the controller must have seen it
                    self.assertEqual(record["hazard_step"], first_failed)
                    self.assertIn(record["hazard_guard_index"], (1, 2))
                else:                                 # latched on the last action: no phase step ran after it
                    self.assertIn(record["hazard_step"], (None, first_failed))
                self.assertIn(c.stop_reason, {"guard_hazard_settled", "bench_stop_reached_settled", "bench_press_no_tip_settled",
                                              "bench_no_contact_settled", "time_budget", "bench_push_timeout_settled",
                                              "bench_press_timeout_settled", "bench_seek_timeout_settled",
                                              "bench_lower_timeout_settled", "bench_approach_timeout_settled",
                                              "bench_press_move_timeout_settled", "bench_press_no_contact_settled",
                                              "bench_press_down_timeout_settled", "waypoint_ik_rejected"}, c.stop_reason)

    def test_collection_with_schedule_validates_and_records_revision_three(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            counts = {c: 0 for c in ALL_CELLS}
            counts.update({"toward_camera|S1_flat": 1, "negative|N3_near": 1, "lateral_outward|S3_just_over": 1})
            path = root / "schedule.json"
            write_schedule(path, make_schedule(4, counts))
            folder = root / "data"
            manifest = collect_exploration(folder, schedule=path, seed=3, max_seconds=1.5)
            self.assertEqual(manifest["interaction_revision"], 3)
            self.assertEqual(manifest["requested_episodes"], 3)
            self.assertEqual(manifest["benchmark_schedule"]["attempts"], 3)
            self.assertTrue(all(m.startswith("bench_") for m in manifest["mode_schedule"]))
            result = validate_dataset(folder)
            self.assertTrue(result["valid"], result)
            meta = json.loads((folder / "episode_0000/metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["controller"]["interaction_revision"], 3)
            self.assertIn("benchmark_intent", meta["controller"])
            self.assertEqual(meta["outcome"]["benchmark"]["intent"], meta["controller"]["benchmark_intent"])
            with self.assertRaises(ValueError):
                collect_exploration(root / "bad_count", schedule=path, n_trajectories=2, max_seconds=0.1)
            with self.assertRaises(ValueError):
                collect_exploration(root / "bad_mode", schedule=path, mode="free", max_seconds=0.1)
            self.assertFalse((root / "bad_count").exists() or (root / "bad_mode").exists())


if __name__ == "__main__":
    unittest.main()
