"""rrt_pilot_yields on synthetic rrt_branches_v1 anchors (numpy only; any venv, also the Windows numpy python).

Every fixture is written into a temporary directory; no dataset is read.
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

import rrt_action_law as law
import rrt_pilot_yields as ry
from wm_training.danger_strata import sha256

NORMALIZATION = {"names": ["dx", "dy", "dz", "dgripper"],
                 "mean": [0.040703438085425696, -0.012399661743676687, 0.05471802006470693, 0.3239098870401013],
                 "scale": [0.050561455154413196, 0.047370118599586605, 0.07757630042049778, 0.9460881747008939]}
FAMILIES = ("recorded", "hold", "planner", "planner", "planner", "planner", "guard", "edge_hold")
GOAL = np.array([0.43, 0.0, 0.17])


def ramp(start, end, at, n=33):
    """Worst-tilt profile: start until step `at`, then `end` (both guards: [value, 0])."""
    w = np.full(n, float(start))
    w[at:] = end
    return np.stack([w, np.zeros(n)], -1)


EDGES = {"hold": 0, "planner": 3, "guard": 3, "edge_hold": 1, "push": 3, "push4": 4, "push_hold": 1, "push2_hold": 2}


def law_arrays(families, edge_mm=1.0):
    """segment_index (R, 32), segment_kind (R, 4), pinch_xyz (R, 33, 3) as collect_rrt_branches writes them: every family
    = n edges of 8 steps then one hold; the pinch moves ``edge_mm`` per edge step along +x and rests during holds."""
    R = len(families)
    idx, kind = np.full((R, 32), -1, np.int64), np.full((R, 4), "", dtype="U4")
    pinch = np.tile(np.array([0.40, 0.0, 0.15]), (R, 33, 1))
    for r, fam in enumerate(families):
        if fam == "recorded":
            continue
        n = EDGES[fam]
        for s in range(n):
            idx[r, 8 * s:8 * s + 8], kind[r, s] = s, "edge"
        if n < 4:
            idx[r, 8 * n:], kind[r, n] = n, "hold"
        step = np.where(idx[r] < n, edge_mm / 1000.0, 0.0)
        pinch[r, 1:, 0] += np.cumsum(step)
    return idx, kind, pinch


def make_anchor(anchor_id, stratum, profiles, set_name="full_train", grasped=False, com=None, speed=None, families=FAMILIES):
    """profiles: len(families) (33, 2) tilt arrays in ``families`` order (default FAMILIES)."""
    R, S = len(families), 33
    tilt = np.stack(profiles)
    idx, kind, pinch = law_arrays(families)
    z = {"families": np.array(families), "stratum": np.array(stratum), "set": np.array(set_name), "anchor_id": np.int64(anchor_id),
         "segment_index": idx, "segment_kind": kind, "pinch_xyz": pinch,
         "guard_tilt_deg": tilt, "grasped": np.full((R, S), grasped), "target_speed_mps": np.zeros((R, S)) if speed is None else speed,
         "object_com_xyz": np.tile(np.array([[0.30, 0.0, 0.05], [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]]), (R, S, 1, 1)) if com is None else com,
         "goal_xyz": GOAL.copy(), "hover_conditions_met": np.zeros((R, S), bool),
         "robot_env_contact": np.zeros((R, S), bool), "robot_green_contact": np.zeros((R, S), bool), "green_env_contact": np.zeros((R, S), bool),
         "actions": np.zeros((R, 32, 5)), "nominal_actions": np.zeros((R, 32, 5)), "nominal_open_loop_actions": np.zeros((R, 32, 5)),
         "tracking_error_m": np.full((R, 32), 0.001), "source_guard_tilt_deg": tilt[0].copy(), "anchor_velocity_xyz": np.array([0.0005, 0.0, 0.0])}
    z["nominal_actions"][0] = np.nan
    z["nominal_open_loop_actions"][0] = np.nan
    z["wm_action"] = law.wm_normalize(z["actions"], NORMALIZATION)
    for k in ("reference_frozen", "reference_dragged", "integral_clamped", "box_clipped", "range_clipped"):
        z[k] = np.zeros((R, 32), bool)
    z["wm_clipped"] = np.zeros((R, 32, 3), bool)
    return z


def tipping_anchor(anchor_id=0, stratum="F", set_name="full_train"):
    """Fresh anchor: recorded calm (2 deg); hold calm; planner 0 topples at step 12 (h16+), planner 1 reaches 12 deg at
    step 5, planners 2-3 calm (1 deg), guard topples at step 25 (only h32), edge_hold reaches 31 deg at step 20."""
    return make_anchor(anchor_id, stratum, [ramp(0, 2, 3), ramp(0, 1, 1), ramp(0, 40, 12), ramp(0, 12, 5), ramp(0, 1, 1), ramp(0, 1, 1),
                                            ramp(0, 45, 25), ramp(0, 31, 20)], set_name)


class YieldLogicTests(unittest.TestCase):
    def test_anchor_record_counts(self):
        a = ry.anchor_record(tipping_anchor(), NORMALIZATION)
        self.assertTrue(a["fresh"])
        p = a["per_h"]
        self.assertEqual([p[h]["law_topples"] for h in ry.HORIZONS], [0, 1, 2, 2, 3])       # h8, h16, h20, h24, h32
        self.assertEqual([p[h]["action_caused"] for h in ry.HORIZONS], [0, 1, 2, 2, 3])     # the recorded sibling stays at 2 deg
        self.assertEqual(p[8]["law_reach10"], 1)
        self.assertEqual(p[20]["law_rise10"], 3)
        self.assertEqual(p[20]["law_calm"], 4)                                              # hold, two planners at 1 deg, guard still 0
        self.assertFalse(p[8]["contrastive_strict"])
        self.assertTrue(p[16]["contrastive_strict"] and p[16]["contrastive_loose"] and p[16]["contrastive_strict_law"])
        m20 = [2, 1, 40, 12, 1, 1, 0, 31]                                                   # max over steps 1..20 per rollout
        self.assertEqual(a["c3_pairs"], sum(1 for i in range(8) for j in range(i + 1, 8) if abs(m20[i] - m20[j]) >= 10))

    def test_recorded_topple_is_not_action_caused(self):
        profiles = [ramp(12, 35, 15)] + [ramp(12, 40, 10)] + [ramp(12, 11, 1)] * 6          # T anchor at 12 deg
        a = ry.anchor_record(make_anchor(1, "T", profiles), NORMALIZATION)
        self.assertFalse(a["fresh"])
        self.assertEqual((a["per_h"][8]["action_caused"], a["per_h"][8]["law_topples"]), (0, 0))
        self.assertEqual((a["per_h"][16]["law_topples"], a["per_h"][16]["action_caused"]), (1, 0))  # the recorded one topples at step 15
        self.assertTrue(a["per_h"][16]["recorded_topples"])
        self.assertFalse(a["per_h"][32]["contrastive_strict"])                              # nothing stays below 10 deg (anchor 12)
        self.assertTrue(a["per_h"][32]["contrastive_loose"])

    def test_near_goal_frames_and_law_health(self):
        R, S = 8, 33
        com = np.tile(np.array([[0.30, 0.0, 0.05], [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]]), (R, S, 1, 1))
        d = (np.arange(S) + 0.5) * 0.001                                                    # 0.5 mm + 1 mm per step (no ties at the bounds)
        com[:, :, 0] = GOAL + d[None, :, None] * np.array([1.0, 0, 0])
        speed = np.full((R, S), 0.015)
        speed[:, 20:] = 0.03
        z = make_anchor(2, "G", [ramp(0, 0, 1)] * 8, grasped=True, com=com, speed=speed)
        z["actions"][3, 5, 0] = 1.0                                                          # dx = +5 cm: outside B's WM range
        z["wm_action"] = law.wm_normalize(z["actions"], NORMALIZATION)
        z["reference_frozen"][4, :3] = True
        z["robot_env_contact"][5, 7] = True
        rows = ry.rollout_rows(z, NORMALIZATION)
        # steps 1..32: d = k + 0.5 mm; boundary_slow needs 8..22 mm (steps 8..21) and speed <= 2 cm/s (steps < 20) -> steps 8..19 = 12
        self.assertEqual(rows[1]["frames"]["boundary_slow"], 12)
        # speed_boundary: 1..2.5 cm/s (steps 1..19) and d <= 15 mm (steps 1..14) -> 14
        self.assertEqual(rows[1]["frames"]["speed_boundary"], 14)
        self.assertEqual((rows[3]["wm_steps"], rows[3]["wm_steps_in_range"]), (32, 31))
        self.assertEqual(rows[4]["flag_steps"]["reference_frozen"], 3)
        self.assertFalse(rows[5]["free_space"])
        self.assertTrue(rows[1]["free_space"])                                              # held block, no contact at all
        anchors = [ry.anchor_record(z, NORMALIZATION)]
        h = ry.law_health(anchors)
        self.assertEqual((h["wm_range"]["law_steps"], h["wm_range"]["law_steps_in_range"]), (7 * 32, 7 * 32 - 1))
        self.assertEqual((h["free_space"]["rollouts"], h["contact"]["rollouts"]), (6, 1))
        self.assertAlmostEqual(h["free_space"]["flags_rollout_fraction"]["reference_frozen"], 1 / 6)
        self.assertGreater(h["all"]["nominal_mismatch_step_mm"]["max"], 40)                  # the +5 cm step is a 50 mm mismatch

    def test_bootstrap_and_tables(self):
        self.assertEqual(ry.bootstrap_mean([2, 2, 2]), {"mean": 2.0, "lo": 2.0, "hi": 2.0, "anchors": 3})
        b = ry.bootstrap_mean([0, 1, 2, 3], seed=5)
        self.assertEqual(b, ry.bootstrap_mean([0, 1, 2, 3], seed=5))
        self.assertLessEqual(b["lo"], b["mean"])
        self.assertLessEqual(b["mean"], b["hi"])
        anchors = [ry.anchor_record(tipping_anchor(i, "F", s), NORMALIZATION) for i, s in enumerate(["full_train", "bench_train"])]
        t = ry.family_table(anchors)
        self.assertEqual(t["F"]["planner"]["rollouts"], 8)
        self.assertAlmostEqual(t["F"]["planner"]["topple_h16"], 0.25)
        self.assertAlmostEqual(t["all"]["guard"]["topple_h32"], 1.0)
        self.assertAlmostEqual(t["F"]["edge_hold"]["action_caused_h20"], 1.0)                 # recorded stays at 2 deg
        self.assertAlmostEqual(t["F"]["edge_hold"]["fresh_topple_h16"], 0.0)
        self.assertAlmostEqual(t["F"]["planner"]["fresh_rise10_h8"], 0.25)                  # planner 1 reaches 12 deg at step 5
        y = ry.anchor_yields(anchors)
        self.assertEqual(y["by_stratum"]["F"]["per_horizon"]["h20"]["fresh_topples_per_anchor"]["mean"], 2.0)
        self.assertEqual(set(y["by_stratum_set"]), {"F/full_train", "F/bench_train"})
        anchors[0]["tag"], anchors[1]["tag"] = "d1", "d2"
        y = ry.anchor_yields(anchors)
        self.assertEqual(set(y["by_dataset_stratum"]), {"d1/F", "d2/F"})
        grouped = ry.family_table(anchors, group=lambda a: f"{a['tag']}/{a['stratum']}")
        self.assertEqual(set(grouped), {"all", "d1/F", "d2/F"})
        self.assertEqual(grouped["d1/F"]["planner"]["rollouts"], 4)

    def test_decision_points(self):
        # recorded calm; r1 topples at step 5 (d0 positive, fallen before d8 -> d8 invalid); r2 at 10 deg at step 8 then
        # topples at step 25 (d0 not positive, d8 positive, not fresh); r3 upright until 8, topples at 27 (d8 fresh); r4-7 calm
        profiles = [ramp(0, 2, 3), ramp(0, 40, 5), ramp(0, 10, 5), ramp(0, 0, 1), ramp(0, 1, 1), ramp(0, 1, 1), ramp(0, 1, 1), ramp(0, 1, 1)]
        profiles[2][25:, 0] = 45.0
        profiles[3][27:, 0] = 50.0
        worst = np.stack(profiles).max(-1)
        pools = ry.decision_point_pools(worst, [False] + [True] * 7)
        self.assertEqual(pools[0], {"valid": 7, "fresh_valid": 7, "positive": 1, "action_caused": 1, "fresh": 1, "calm": 5})
        self.assertEqual(pools[8], {"valid": 6, "fresh_valid": 5, "positive": 2, "action_caused": 2, "fresh": 1, "calm": 4})   # r2 at 10 deg
        worst_rec = worst.copy()
        worst_rec[0, 26:] = 35.0                                                             # recorded topples at step 26
        pools = ry.decision_point_pools(worst_rec, [False] + [True] * 7)
        self.assertEqual((pools[0]["action_caused"], pools[8]["action_caused"]), (1, 0))      # within 1..20 fine, 1..28 not
        a = ry.anchor_record(make_anchor(3, "F", profiles), NORMALIZATION)
        self.assertEqual(a["dp"][8]["positive"], 2)
        y = ry.anchor_yields([a])
        self.assertEqual(y["all"]["decision_points"]["d0_plus_d8"]["positive"]["mean"], 3.0)
        self.assertEqual(y["all"]["decision_points"]["anchors_with_action_caused"], 1)
        self.assertEqual(y["all"]["decision_points"]["totals"]["fresh_valid"], 12)

    def test_mismatch_breakdown(self):
        z = tipping_anchor()
        z["actions"][2, 3, 0] = 0.02                                                          # 1 mm in the first segment
        z["actions"][3, 12, 1] = -0.04                                                        # 2 mm later
        a = ry.anchor_record(z, NORMALIZATION)
        self.assertAlmostEqual(a["anchor_speed_mps"], 0.01)                                   # 0.5 mm per 0.05 s
        b = ry.mismatch_breakdown([r for r in a["rows"] if r["law"]])
        self.assertEqual((b["slow_anchor"]["rollouts"], b["fast_anchor"]["rollouts"]), (7, 0))
        self.assertAlmostEqual(b["slow_anchor"]["steps_1_8"]["max"], 1.0)
        self.assertAlmostEqual(b["slow_anchor"]["steps_9_32"]["max"], 2.0)
        self.assertEqual(b["slow_anchor"]["steps_1_8"]["steps"], 7 * 8)

    def test_planner_shape(self):
        fams = ("recorded", "hold", "edge_hold", "push_hold", "push2_hold", "planner", "push4", "guard")
        idx, kind, _ = law_arrays(fams)
        self.assertEqual(ry.planner_shape(idx, kind, 0).tolist(), [False, False, True, True, False, False, False, False])
        self.assertEqual(ry.planner_shape(idx, kind, 8).tolist(), [False, False, False, False, True, False, False, False])

    def test_instances_subgroups_and_distinct(self):
        """push2_hold (rollout 4) topples at step 20: an instance at d = 0 (window 1..20) AND at d = 8 (window 9..28, the
        planner's edge + hold shape); it is the only grasped rollout. 2 instances, 1 distinct rollout."""
        fams = ("recorded", "hold", "edge_hold", "push_hold", "push2_hold", "planner", "push4", "guard")
        prof = [ramp(0, 2, 3)] + [ramp(0, 1, 1) for _ in range(7)]
        prof[4] = ramp(0, 40, 20)
        z = make_anchor(7, "F", prof, families=fams)
        z["grasped"][4] = True
        a = ry.anchor_record(z, NORMALIZATION)
        self.assertEqual((a["dp"][0]["action_caused"], a["dp"][8]["action_caused"]), (1, 1))
        self.assertEqual(a["distinct"]["action_caused_rollouts"], 1)
        self.assertEqual(a["distinct"]["positive_rollouts"], 1)
        self.assertEqual((a["dp_sub"]["planner_shape"][0]["action_caused"], a["dp_sub"]["planner_shape"][8]["action_caused"]), (0, 1))
        self.assertEqual(a["dp_sub"]["planner_shape"][0]["valid"], 2)                     # edge_hold, push_hold at d = 0
        self.assertEqual(sum(a["dp_sub"]["grasped"][d]["valid"] for d in (0, 8)), 2)
        self.assertEqual(sum(a["dp_sub"]["grasped_planner_shape"][d]["action_caused"] for d in (0, 8)), 1)
        self.assertEqual(a["dp"], ry.decision_point_pools(np.asarray(z["guard_tilt_deg"]).max(-1), [False] + [True] * 7))
        fam_of = {i["family"] for i in a["instances"]}
        self.assertEqual(fam_of, set(fams[1:]))
        y = ry.anchor_yields([a, ry.anchor_record(tipping_anchor(8), NORMALIZATION)])
        dp = y["all"]["decision_points"]
        self.assertEqual(dp["distinct_rollouts"]["action_caused_rollouts"]["total"], 1 + 3)   # tipping_anchor: planner 0, guard, edge_hold
        self.assertEqual(dp["totals"]["action_caused"], 2 + 5)                                  # planner 0 and edge_hold at d0 and d8, guard at d8
        self.assertEqual(dp["by_family_totals"]["push2_hold"]["action_caused"], 2)
        listed = {(r["anchor_id"], r["instances"], r["distinct_rollouts"]) for r in y["action_caused_anchors"]}
        self.assertIn((7, 2, 1), listed)
        table = ry.anchor_table([a])
        self.assertEqual(table[0]["families"], list(fams))
        self.assertEqual(len(table[0]["instances"]), len(a["instances"]))
        self.assertEqual(table[0]["rollouts"][4]["max"]["20"], 40.0)

    def test_executed_speed(self):
        idx, kind, pinch = law_arrays(("recorded", "push2_hold"), edge_mm=1.0)
        s = ry.executed_speed(pinch[1], idx[1], kind[1])
        np.testing.assert_allclose(s["speed_mm"][:16], 1.0, atol=1e-9)
        np.testing.assert_allclose(s["speed_mm"][16:], 0.0, atol=1e-12)
        self.assertEqual([seg for seg, _ in s["edge_average_mm"]], [0, 1])
        np.testing.assert_allclose([v for _, v in s["edge_average_mm"]], 1.0)
        with self.assertRaises(ValueError):
            ry.executed_speed(pinch[0], idx[0], kind[0])                                 # the recorded rollout has no segments
        slow = pinch[1].copy()
        slow[9:17, 0] = slow[8, 0] + 0.0005 * np.arange(1, 9)                              # second edge at 0.5 mm per step
        slow[17:, 0] = slow[16, 0]
        row = {"step_kind": kind[1][idx[1]], "step_segment": idx[1], **ry.executed_speed(slow, idx[1], kind[1])}
        st = ry.speed_stats([row])
        self.assertAlmostEqual(st["edge_steps_first_segment"]["mean"], 1.0)
        self.assertAlmostEqual(st["edge_steps_later_segments"]["max"], 0.5)
        self.assertEqual(st["edge_average_later_in_band"], 0.0)                            # 0.5 mm per step is outside 0.8-1.2
        self.assertEqual(st["hold_steps_later_segments"]["max"], 0.0)

    def test_wrong_shapes_are_refused(self):
        z = tipping_anchor()
        z["families"] = np.array(("hold",) + FAMILIES[1:])
        with self.assertRaises(ValueError):
            ry.rollout_rows(z, NORMALIZATION)


class YieldCliTests(unittest.TestCase):
    def write_dataset(self, root):
        data, report = root / "data", root / "report"
        data.mkdir()
        report.mkdir()
        rows = []
        for i, (stratum, z) in enumerate([("F", tipping_anchor(0)), ("T", make_anchor(1, "T", [ramp(12, 35, 15)] + [ramp(12, 11, 1)] * 7))]):
            path = data / f"anchor_{i:04d}.npz"
            np.savez_compressed(path, **z)
            rows.append({"id": i, "stratum": stratum, "file": path.name, "sha256": sha256(path)})
        (data / "manifest.json").write_text(json.dumps({"format": "rrt_branches_v1", "status": "complete", "anchors": rows,
                                                        "wm_action": {"normalization": NORMALIZATION}}), encoding="utf-8")
        (report / "summary.json").write_text(json.dumps({
            "manifest_sha256": sha256(data / "manifest.json"), "sanity": {"anchors": 2}, "counts": {"rollouts_by_family": {"recorded": 2, "hold": 14}},
            "workers": 3, "wall_seconds": 12.0, "seconds_per_rollout": 2.0, "anchor_seconds_median": 16.0, "bytes": 8_000_000,
            "bytes_per_anchor": 4_000_000, "bytes_per_rollout": 500_000, "memory": {"sampled": {"peak_pss_sum_kb": 5 * 1024**2}}}), encoding="utf-8")
        return data, report

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report = self.write_dataset(root)
            out = root / "yields"
            args = ["--dataset", str(data), "--report", str(report), "--output", str(out)]
            with redirect_stdout(io.StringIO()):
                summary = ry.main(args)
            for name in ("run.json", "summary.json", "REPORT_ZH.md"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual((summary["anchors"], summary["rollouts"]), (2, 16))
            self.assertEqual(set(summary["family_table_by_dataset"]), {"all", "data/F", "data/T"})
            self.assertEqual(summary["cost"][0]["wall_seconds_per_anchor"], 6.0)
            self.assertEqual([t["anchor_id"] for t in summary["anchor_table"]], [0, 1])
            self.assertEqual(summary["law_health"]["free_space"]["executed_speed_mm_per_step"]["edge_average_later_in_band"], 1.0)
            text = (out / "REPORT_ZH.md").read_text(encoding="utf-8")
            for needle in ("## 2c.", "## 2d.", "## 2e.", "执行的夹爪速度"):
                self.assertIn(needle, text)
            self.assertEqual([r["anchor_id"] for r in summary["yields"]["action_caused_anchors"]], [0])
            # two datasets: grouped by dataset in the report
            data2, report2 = root / "data2", root / "report2"
            data2.mkdir()
            report2.mkdir()
            for name in ("anchor_0000.npz",):
                (data2 / name).write_bytes((data / name).read_bytes())
            m = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            m["anchors"] = m["anchors"][:1]
            (data2 / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            s2 = json.loads((report / "summary.json").read_text(encoding="utf-8"))
            s2["manifest_sha256"] = sha256(data2 / "manifest.json")
            (report2 / "summary.json").write_text(json.dumps(s2), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                both = ry.main(["--dataset", str(data), "--report", str(report), "--dataset", str(data2), "--report", str(report2),
                                "--output", str(root / "yields_both")])
            self.assertEqual(both["anchors"], 3)
            self.assertIn("data2/F", both["yields"]["by_dataset_stratum"])
            self.assertIn("data2/F", (root / "yields_both" / "REPORT_ZH.md").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["cost"][0]["peak_pss_sum_gib"], 5.0)
            with self.assertRaises(FileExistsError):
                ry.main(args)
            with (data / "anchor_0001.npz").open("ab") as f:                              # tampered npz -> FAILED.json, no summary
                f.write(b"x")
            bad = root / "yields_bad"
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                ry.main(["--dataset", str(data), "--report", str(report), "--output", str(bad)])
            self.assertTrue((bad / "FAILED.json").is_file())
            self.assertFalse((bad / "summary.json").exists())

    def test_report_must_match_the_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, report = self.write_dataset(root)
            s = json.loads((report / "summary.json").read_text(encoding="utf-8"))
            s["manifest_sha256"] = "0" * 64
            (report / "summary.json").write_text(json.dumps(s), encoding="utf-8")
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                ry.main(["--dataset", str(data), "--report", str(report), "--output", str(root / "y")])


if __name__ == "__main__":
    unittest.main()
