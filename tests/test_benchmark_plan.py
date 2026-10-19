"""numpy-only checks for the danger-benchmark plan (cells, intents, schedules). No simulator."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_plan import (
    ALL_CELLS, BENCH_MODES, BENCH_PHASES, BEST_EFFORT_CELLS, CELL_OPTIONS, HARD_CELLS, NEGATIVE_CELLS,
    PHASE_NAME_LIMIT, RECIPES, RECIPE_MODE, TIP_ANGLE_DEG, default_counts, lean_vector, make_schedule,
    read_schedule, sample_intent, stop_reached, validate_intent, write_schedule,
)

BASE_PHASES = {"SETTLE", "FREE", "APPROACH", "DESCEND", "CLOSE", "LIFT", "RETRACT", "CARRY", "HOVER",
               "GUARD_APPROACH", "GUARD_LOWER", "GUARD_PUSH", "GUARD_SEEK", "BRUSH_APPROACH", "BRUSH_LOWER",
               "BRUSH_SWEEP", "BRUSH_LIFT", "CARRY_GUARD", "RELEASE", "EVENT_HOLD", "DONE"}


class PlanTests(unittest.TestCase):
    def test_phase_names_fit_the_u16_recorder_and_do_not_shadow_base_phases(self):
        self.assertEqual(len(set(BENCH_PHASES)), len(BENCH_PHASES))
        for name in BENCH_PHASES:
            self.assertLessEqual(len(name), PHASE_NAME_LIMIT, name)
            self.assertTrue(name.startswith(("BENCH_", "EVENT_")), name)
        self.assertFalse(set(BENCH_PHASES) & BASE_PHASES)

    def test_cells_recipes_and_modes_are_consistent(self):
        self.assertEqual(len(HARD_CELLS), 10)
        self.assertEqual(len(BEST_EFFORT_CELLS), 2)
        self.assertEqual(len(NEGATIVE_CELLS), 4)
        self.assertEqual(set(CELL_OPTIONS), set(ALL_CELLS))
        self.assertEqual(set(RECIPE_MODE), set(RECIPES))
        self.assertTrue(all(m.startswith("bench_") for m in BENCH_MODES))
        for cell, options in CELL_OPTIONS.items():
            self.assertGreater(sum(o[0] for o in options), 0, cell)
            for option in options:
                self.assertIn(option[1], RECIPES)
        self.assertAlmostEqual(TIP_ANGLE_DEG, np.degrees(np.arctan(0.4)), places=9)
        counts = default_counts("precheck")
        self.assertEqual(sum(counts.values()), 40)
        self.assertEqual(sum(default_counts("full").values()), 1200)
        v2 = default_counts("full_v2")
        self.assertEqual(set(v2), set(ALL_CELLS))
        self.assertEqual(sum(v2.values()), 960)
        self.assertTrue(all(v2[c] >= 45 for c in HARD_CELLS))
        with self.assertRaises(ValueError):
            default_counts("huge")

    def test_lean_vectors_follow_guard_side(self):
        np.testing.assert_array_equal(lean_vector("+x", 1), [1, 0, 0])
        np.testing.assert_array_equal(lean_vector("-x", 2), [-1, 0, 0])
        np.testing.assert_array_equal(lean_vector("outward", 1), [0, -1, 0])
        np.testing.assert_array_equal(lean_vector("outward", 2), [0, 1, 0])
        np.testing.assert_array_equal(lean_vector("inward", 1), [0, 1, 0])
        np.testing.assert_array_equal(lean_vector("inward", 2), [0, -1, 0])
        with self.assertRaises(ValueError):
            lean_vector("up", 1)
        with self.assertRaises(ValueError):
            lean_vector("+x", 3)

    def test_stop_rules(self):
        self.assertTrue(stop_reached(("stroke", 0.03, 0.03), 0.0296, 5.0))
        self.assertFalse(stop_reached(("stroke", 0.03, 0.03), 0.02, 80.0))
        self.assertTrue(stop_reached(("tilt", 31.0, 31.0), 0.0, 31.0))
        self.assertFalse(stop_reached(("tilt", 31.0, 31.0), 1.0, 30.9))
        with self.assertRaises(ValueError):
            stop_reached(("time", 1, 1), 0, 0)

    def test_every_cell_samples_valid_intents_that_match_physics_rules(self):
        rng = np.random.default_rng(3)
        seen_recipes, seen_after = set(), set()
        for cell in ALL_CELLS:
            for _ in range(40):
                intent = sample_intent(cell, rng)
                self.assertEqual(intent["cell"], cell)
                seen_recipes.add(intent["recipe"])
                seen_after.add(intent["after_event"])
                direction = cell.split("|")[0]
                if direction != "negative" and intent["recipe"] not in ("drop", "free", "carry_sweep"):
                    expected = {"toward_camera": "+x", "away_camera": "-x", "lateral_outward": "outward",
                                "lateral_inward": "inward"}[direction]
                    self.assertEqual(intent["lean"], expected, (cell, intent))
                if cell.endswith("S2_lean") or cell.endswith("S3_just_over"):
                    if intent["recipe"] in ("press", "carry_press"):
                        self.assertEqual(intent["stop"][0], "tilt")
                        self.assertGreater(intent["stop"][1], 30)
                if intent["recipe"] == "rock":
                    self.assertTrue(20.0 <= intent["stop"][1] <= intent["stop"][2] < 30.0)  # released below the hazard
                    self.assertEqual(intent["after_event"], "retreat")
                if intent["recipe"] == "slot_push":
                    self.assertIn(intent["after_event"], ("continue", "retreat"))
                self.assertNotEqual(intent["recipe"], "drop")
                json.dumps(intent, allow_nan=False)
        self.assertEqual(seen_recipes, set(RECIPES) - {"drop"})   # drop stays implemented but is not scheduled
        self.assertEqual(seen_after, {"hold", "retreat", "continue"})

    def test_invalid_intents_rejected(self):
        good = sample_intent("toward_camera|S1_flat", np.random.default_rng(0))
        bad_cases = [dict(good, cell="nowhere"), dict(good, guard_index=3), dict(good, guard_index=1.0),
                     dict(good, lean="up"), dict(good, after_event="stop"), dict(good, park_green="no"),
                     dict(good, height=[0.3, 0.4]), dict(good, extra=1),
                     {**good, "recipe": "press", "stop": ["stroke", 0.01, 0.02], "lean": "+x"},
                     {**good, "recipe": "slot_push", "stop": ["stroke", 0.01, 0.02], "lean": "+x", "park_green": True},
                     {**good, "recipe": "push", "stop": ["stroke", 0.01, 0.02], "lean": "+x", "park_green": True},
                     {**good, "recipe": "press", "stop": ["tilt", 40, 90], "lean": "+x"},
                     {**good, "recipe": "push", "stop": ["stroke", 0.1, 0.2], "lean": "+x"},
                     {**good, "recipe": "press", "stop": ["tilt", 31, 38], "lean": "inward"},
                     {**good, "recipe": "press", "stop": ["tilt", 31, 38], "lean": "+x", "park_green": True},
                     {**good, "recipe": "carry_press", "stop": ["tilt", 31, 38], "lean": "inward", "park_green": True}]
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_intent(bad)
        with self.assertRaises(ValueError):
            validate_intent("not a dict")

    def test_schedule_is_deterministic_counted_shuffled_and_tamper_evident(self):
        counts = {**{c: 0 for c in ALL_CELLS}, "toward_camera|S1_flat": 4, "negative|N3_near": 2, "lateral_inward|S3_just_over": 3}
        a, b = make_schedule(11, counts), make_schedule(11, counts)
        self.assertEqual(a, b)
        self.assertNotEqual([e["intent"] for e in a["entries"]], [e["intent"] for e in make_schedule(12, counts)["entries"]])
        self.assertEqual(len(a["entries"]), 9)
        self.assertEqual([e["id"] for e in a["entries"]], list(range(9)))
        cells = [e["intent"]["cell"] for e in a["entries"]]
        self.assertEqual(cells.count("toward_camera|S1_flat"), 4)
        self.assertNotEqual(cells, sorted(cells))  # shuffled, not grouped by cell
        for entry in a["entries"]:
            self.assertEqual(entry["mode"], RECIPE_MODE[entry["intent"]["recipe"]])
        self.assertEqual(a["interaction_revision"], 3)
        with self.assertRaises(ValueError):
            make_schedule(1, {"nowhere": 1})
        with self.assertRaises(ValueError):
            make_schedule(1, {c: 0 for c in ALL_CELLS})
        with self.assertRaises(ValueError):
            make_schedule(-1, counts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schedule.json"
            digest = write_schedule(path, a)
            loaded = read_schedule(path)
            self.assertEqual(loaded["sha256"], digest)
            self.assertEqual([e["intent"] for e in loaded["entries"]], [e["intent"] for e in a["entries"]])
            with self.assertRaises(FileExistsError):
                write_schedule(path, a)
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["entries"][0]["intent"]["guard_index"] = 3 - tampered["entries"][0]["intent"]["guard_index"]
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_schedule(path)

    def test_full_schedule_covers_every_cell_with_requested_counts(self):
        schedule = make_schedule(20260930, default_counts("full"))
        self.assertEqual(len(schedule["entries"]), 1200)
        for cell, n in default_counts("full").items():
            self.assertEqual(sum(e["intent"]["cell"] == cell for e in schedule["entries"]), n, cell)
        guards = [e["intent"]["guard_index"] for e in schedule["entries"]]
        self.assertGreater(min(guards.count(1), guards.count(2)), 500)


if __name__ == "__main__":
    unittest.main()
