"""RRT-law branch collection: stage-1 logic and CLI on synthetic caches, stage 2 on the real scene.

Every fixture is synthetic (written into a temporary directory); no dataset, cache or benchmark episode is read.
* PlanLogicTests: numpy only, any venv.
* PlanCliTests: need h5py (run with ~/lewm-train/.venv). The fixture's val / test / bench_dev episodes have NO raw
  files, so any attempt to open them fails the test.
* SimulationTests: need MuJoCo + the GPU render environment (run with ~/lewm-manip/.venv in WSL). A real-scene
  episode is recorded with the data-collection carry controller, a plan is built from it with the stage-1 functions,
  and stage 2 runs end to end with 2 workers (tiny smoke test: 2 anchors x 8 rollouts).
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

import rrt_action_law as law
import rrt_branches_plan as rp
from wm_training.danger_strata import sha256

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False
try:
    import collect_rrt_branches as crr
    from exploration import ExplorationScene
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

NORMALIZATION = {"names": ["dx", "dy", "dz", "dgripper"],
                 "mean": [0.040703438085425696, -0.012399661743676687, 0.05471802006470693, 0.3239098870401013],
                 "std": [0.050561455154413196, 0.047370118599586605, 0.07757630042049778, 0.9460881747008939],
                 "scale": [0.050561455154413196, 0.047370118599586605, 0.07757630042049778, 0.9460881747008939]}
CFG = law.LawConfig(normalization=NORMALIZATION)
RENDER_ENV = {"MUJOCO_GL": "glfw", "GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
              "LIBGL_ALWAYS_SOFTWARE": None, "LD_LIBRARY_PATH": "/usr/lib/wsl/lib"}


def synthetic_episode(n=220):
    """0-19 free | 20-59 F (gripper low beside guard 1) | 60-99 T (guard 1 at 12 deg, gripper near it, high) |
    100-139 G (grasped at the goal, gripper closing) | 140-179 M (grasped, high, away from the goal) | 180+ free."""
    com = np.zeros((n, 3, 3))
    com[:, 0], com[:, 1], com[:, 2] = [0.425, 0.0, 0.05], [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]
    a = {"time": 0.05 * np.arange(n), "pinch_xyz": np.tile([0.375, 0.0, 0.16], (n, 1)), "object_com_xyz": com,
         "grasped": np.zeros(n, bool), "hover_error_m": np.full(n, 0.2), "guard_tilt_deg": np.zeros((n, 2)),
         "unsafe": np.zeros(n, bool), "class_label": np.full(n, 2, np.int64), "actions": np.zeros((n - 1, 5)),
         "hover_goal_xyz": np.tile([0.43, 0.0, 0.17], (n, 1))}
    a["actions"][:, 4] = -1.0
    a["pinch_xyz"][20:60] = [0.425, -0.10, 0.06]
    a["pinch_xyz"][60:100] = [0.425, -0.10, 0.14]
    a["guard_tilt_deg"][60:100, 0] = 12.0
    a["grasped"][100:180] = True
    a["actions"][99:179, 4] = 1.0
    a["object_com_xyz"][100:140, 0] = [0.43, 0.0, 0.17]
    a["pinch_xyz"][100:140] = [0.43, 0.0, 0.18]
    a["hover_error_m"][100:140] = 0.005
    a["class_label"][100:140] = 0
    a["object_com_xyz"][140:180, 0] = [0.40, 0.0, 0.21]
    a["pinch_xyz"][140:180] = [0.40, 0.0, 0.22]
    a["hover_error_m"][140:180] = 0.06
    return a


class PlanLogicTests(unittest.TestCase):
    def test_strata_rules_and_priorities(self):
        a = synthetic_episode()
        labels = rp.classify_rrt_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])
        self.assertEqual(set(labels[:20]), {""})                                   # green at rest 5 cm from both guards is not "near"
        self.assertEqual(set(labels[20:60]), {"F"})
        self.assertEqual(set(labels[60:100]), {"T"})                               # T has no height gate
        self.assertEqual(set(labels[100:140]), {"G"})
        self.assertEqual(set(labels[140:180]), {"M"})
        one = {k: v[20:21].copy() for k, v in a.items() if k != "actions"}
        one["guard_tilt_deg"][0] = [0.0, 7.0]                                      # the tilted guard is the far one
        self.assertEqual(rp.classify_rrt_frames(one["pinch_xyz"], one["object_com_xyz"], one["grasped"], one["hover_error_m"],
                                                one["guard_tilt_deg"])[0], "")
        one["guard_tilt_deg"][0] = [7.0, 0.0]
        self.assertEqual(rp.classify_rrt_frames(one["pinch_xyz"], one["object_com_xyz"], one["grasped"], one["hover_error_m"],
                                                one["guard_tilt_deg"])[0], "T")
        low = {k: v[140:141].copy() for k, v in a.items() if k != "actions"}
        low["object_com_xyz"][0, 0] = [0.425, 0.0, 0.12]                          # carried green low between the guards: F, not M
        self.assertEqual(rp.classify_rrt_frames(low["pinch_xyz"], low["object_com_xyz"], low["grasped"], low["hover_error_m"],
                                                low["guard_tilt_deg"])[0], "F")
        with self.assertRaises(ValueError):
            rp.classify_rrt_frames(a["pinch_xyz"][:5], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])

    def test_admissible_frames(self):
        a = synthetic_episode()
        labels = rp.classify_rrt_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])
        a["actions"][119, 4] = -1.0                                                 # frame 120: the command in force is "open"
        ok = rp.admissible_frames(220, a["guard_tilt_deg"], a["actions"], labels)
        self.assertFalse(ok[120])
        self.assertTrue(ok[121] and ok[100])
        self.assertFalse(ok[:12].any())
        self.assertFalse(ok[220 - 32:].any())                                       # a + 32 <= last frame
        self.assertTrue(ok[220 - 33])
        a["guard_tilt_deg"][70, 0] = 26.0
        self.assertFalse(rp.admissible_frames(220, a["guard_tilt_deg"], a["actions"], labels)[70])

    def test_select_anchors_two_sources(self):
        episodes = {("full_train", 11): synthetic_episode(), ("bench_train", 100002): synthetic_episode()}
        quotas = {"T": 2, "F": 2, "G": 2, "M": 4}
        anchors, available = rp.select_anchors(episodes, quotas, seed=5)
        self.assertEqual(anchors, rp.select_anchors(episodes, quotas, seed=5)[0])
        counts = {s: sum(x["stratum"] == s for x in anchors) for s in rp.STRATA}
        self.assertEqual((counts["T"], counts["F"], counts["G"]), (2, 2, 2))
        self.assertLessEqual(counts["M"], 4)
        self.assertEqual(available["G"]["by_source"], {"full_train": 40, "bench_train": 40})       # frames 100-139
        for key, arrays in episodes.items():
            frames = sorted(x["frame"] for x in anchors if (x["source"], x["episode_id"]) == key)
            self.assertTrue(all(b - a >= rp.MIN_GAP_FRAMES for a, b in zip(frames, frames[1:])))
            labels = rp.classify_rrt_frames(arrays["pinch_xyz"], arrays["object_com_xyz"], arrays["grasped"], arrays["hover_error_m"],
                                            arrays["guard_tilt_deg"])
            for x in anchors:
                if (x["source"], x["episode_id"]) == key:
                    self.assertEqual(labels[x["frame"]], x["stratum"])

    def test_rollout_specs(self):
        a = synthetic_episode()
        pinch, com = a["pinch_xyz"][30], a["object_com_xyz"][30]
        fams = rp.DEFAULT_FAMILY_SETS["T"]
        specs = rp.draw_rollouts(123, fams, pinch, com, None, CFG)
        self.assertEqual(specs, rp.draw_rollouts(123, fams, pinch, com, None, CFG))
        self.assertNotEqual(specs, rp.draw_rollouts(124, fams, pinch, com, None, CFG))
        self.assertIsNone(specs[0]["segments"])
        self.assertEqual(specs[1]["segments"], [["hold", 32]])
        for s in specs[1:]:
            segs = rp.segments_from_json(s["segments"])
            self.assertEqual(law.sequence_steps(segs, CFG), 32)
        planner = [s for s in specs if s["family"] == "planner"]
        self.assertTrue(all([k for k, _ in s["segments"]] == ["edge"] * 3 + ["hold"] and s["segments"][-1][1] == 8 for s in planner))
        edge_hold = next(s for s in specs if s["family"] == "edge_hold")
        self.assertEqual([k for k, _ in edge_hold["segments"]], ["edge", "hold"])
        self.assertEqual(edge_hold["segments"][1][1], 24)
        for s in (x for x in specs if x["family"] == "guard"):
            first = np.asarray(s["segments"][0][1])
            self.assertEqual((first[2], s["guard_index"]), (0.0, 1))
            toward = (com[1, :2] - pinch[:2]) / np.linalg.norm(com[1, :2] - pinch[:2])
            self.assertLessEqual(np.degrees(np.arccos(np.clip(first[:2] @ toward, -1, 1))), rp.law.GUARD_ANGLE_DEG + 1e-9)
        with self.assertRaises(ValueError):
            rp.segments_from_json([["edge", [1, 0, 0]], ["hold", 3]])

    def test_push_families(self):
        a = synthetic_episode()
        pinch, com = a["pinch_xyz"][30], a["object_com_xyz"][30]                   # gripper low beside guard 1 (y = -0.05)
        specs = rp.draw_rollouts(99, ("recorded", "push", "push4", "push_hold"), pinch, com, None, CFG)
        self.assertEqual(specs, rp.draw_rollouts(99, ("recorded", "push", "push4", "push_hold"), pinch, com, None, CFG))
        toward = (com[1, :2] - pinch[:2]) / np.linalg.norm(com[1, :2] - pinch[:2])
        for spec, edges, hold in ((specs[1], 3, 8), (specs[2], 4, None), (specs[3], 1, 24)):
            kinds = [k for k, _ in spec["segments"]]
            self.assertEqual(kinds, ["edge"] * edges + ([] if hold is None else ["hold"]))
            if hold is not None:
                self.assertEqual(spec["segments"][-1][1], hold)
            self.assertEqual(law.sequence_steps(rp.segments_from_json(spec["segments"]), CFG), 32)
            self.assertEqual((spec["guard_index"], len(spec["push_angles_deg"])), (1, edges))
            for (kind, d), angle in zip(spec["segments"][:edges], spec["push_angles_deg"]):
                d = np.asarray(d)
                self.assertEqual(d[2], 0.0)
                off = np.degrees(np.arccos(np.clip(d[:2] @ toward, -1, 1)))
                self.assertLessEqual(off, law.GUARD_ANGLE_DEG + 1e-9)
                self.assertAlmostEqual(off, abs(angle), places=6)
        self.assertEqual(len(set(tuple(s[1]) for s in specs[2]["segments"])), 4)   # independent jitter per edge

    def test_push2_hold_and_appended_families(self):
        """push2_hold = 2 guard-directed edges + hold 16, so its window from step 8 is one edge + hold 12 (the planner's
        check shape); appending a family to a set leaves every earlier rollout's spec unchanged (rng per rollout)."""
        a = synthetic_episode()
        pinch, com = a["pinch_xyz"][30], a["object_com_xyz"][30]
        base = ("recorded", "hold", "push", "push", "push4", "push4", "push_hold", "guard")
        specs = rp.draw_rollouts(4242, base + ("push2_hold",), pinch, com, None, CFG)
        self.assertEqual(specs[:8], rp.draw_rollouts(4242, base, pinch, com, None, CFG))
        p2 = specs[8]
        self.assertEqual([k for k, _ in p2["segments"]], ["edge", "edge", "hold"])
        self.assertEqual(p2["segments"][2][1], 16)
        self.assertEqual(law.sequence_steps(rp.segments_from_json(p2["segments"]), CFG), 32)
        toward = (com[1, :2] - pinch[:2]) / np.linalg.norm(com[1, :2] - pinch[:2])
        for (_, d), angle in zip(p2["segments"][:2], p2["push_angles_deg"]):
            d = np.asarray(d)
            self.assertEqual(d[2], 0.0)
            self.assertAlmostEqual(np.degrees(np.arccos(np.clip(d[:2] @ toward, -1, 1))), abs(angle), places=6)
            self.assertLessEqual(abs(angle), law.GUARD_ANGLE_DEG)
        self.assertIn("push2_hold", rp.FAMILIES)
        self.assertEqual(rp.parse_family_sets([f"T={','.join(base + ('push2_hold',))}"])["T"][-1], "push2_hold")

    def test_goal_image_strata(self):
        """--goal-image-strata G: G anchors only from episodes with a linked goal image (achieved COM); T and F picks
        (filled before G) are unchanged; default None keeps the unrestricted picks bit for bit."""
        episodes = {("full_train", 11): synthetic_episode(), ("full_train", 12): synthetic_episode(), ("bench_train", 100002): synthetic_episode()}
        quotas = {"T": 2, "F": 2, "G": 6, "M": 2}
        plain, _ = rp.select_anchors(episodes, quotas, seed=9)
        self.assertEqual(plain, rp.select_anchors(episodes, quotas, seed=9, stratum_episodes=None)[0])
        self.assertEqual(rp.parse_goal_image_strata(None), ())
        self.assertEqual(rp.parse_goal_image_strata("G, G,M"), ("G", "M"))
        with self.assertRaises(ValueError):
            rp.parse_goal_image_strata("G,X")
        links = {("wm_full_001", 11): {"goal_image_id": 5, "goal_xyz": [0.431, 0.0, 0.17]},
                 ("wm_full_001", 12): {"goal_image_id": 6, "goal_xyz": None},                  # no achieved COM: not usable
                 ("wm_train_bench_001", 7): {"goal_image_id": 7, "goal_xyz": [0.43, 0.0, 0.17]}}
        restrict = rp.goal_image_episodes(episodes, links, ("G",))
        self.assertEqual(restrict, {"G": {("full_train", 11)}})
        picked, available = rp.select_anchors(episodes, quotas, seed=9, stratum_episodes=restrict)
        g = [x for x in picked if x["stratum"] == "G"]
        self.assertTrue(g and all((x["source"], x["episode_id"]) == ("full_train", 11) for x in g))
        self.assertLessEqual(len(g), 2)                                                   # frames 100-139, >= 33 apart
        self.assertEqual(available["G"]["episodes"], 1)
        self.assertEqual([x for x in picked if x["stratum"] in "TF"], [x for x in plain if x["stratum"] in "TF"])
        with self.assertRaises(ValueError):
            rp.select_anchors(episodes, quotas, seed=9, stratum_episodes={"X": set()})
        cap = rp.anchor_capacity(episodes, 9, stratum_episodes=restrict)
        self.assertEqual(cap["options"]["goal_image_strata"], ["G"])
        self.assertEqual(cap["alone"]["G"]["by_source"]["bench_train"], 0)
        self.assertTrue(1 <= cap["alone"]["G"]["anchors"] <= 2)
        self.assertEqual(rp.anchor_capacity(episodes, 9)["options"]["goal_image_strata"], [])
        arrays = episodes[("full_train", 12)]
        source = {"source": "full_train", "episode_id": 12}
        anchor = {"source": "full_train", "episode_id": 12, "stratum": "G", "frame": 110}
        with self.assertRaises(RuntimeError):
            rp.make_anchor_record(0, anchor, 1, source, arrays, rp.DEFAULT_FAMILY_SETS["G"], links[("wm_full_001", 12)], CFG, goal_image_strata=("G",))
        rec = rp.make_anchor_record(0, anchor, 1, source, arrays, rp.DEFAULT_FAMILY_SETS["G"], links[("wm_full_001", 11)], CFG, goal_image_strata=("G",))
        self.assertEqual((rec["goal"]["goal_source"], rec["goal"]["goal_xyz"]), ("goal_image_achieved_com", [0.431, 0.0, 0.17]))

    def test_f_radius_and_recorded_filter(self):
        a = synthetic_episode()
        one = {k: v[20:21].copy() for k, v in a.items() if k != "actions"}           # pinch 5 cm from guard 1, low: F at 6 cm
        args = (one["pinch_xyz"], one["object_com_xyz"], one["grasped"], one["hover_error_m"], one["guard_tilt_deg"])
        self.assertEqual(rp.classify_rrt_frames(*args)[0], "F")
        self.assertEqual(rp.classify_rrt_frames(*args, f_near_radius_m=0.045)[0], "")
        g = {k: v[140:141].copy() for k, v in a.items() if k != "actions"}           # grasped, high, far from the goal: M
        g["pinch_xyz"][0] = [0.425, -0.10, 0.10]                                    # 5 cm from guard 1, low: F (6 cm)
        g["object_com_xyz"][0, 0] = [0.425, -0.10, 0.09]
        gargs = (g["pinch_xyz"], g["object_com_xyz"], g["grasped"], g["hover_error_m"], g["guard_tilt_deg"])
        self.assertEqual(rp.classify_rrt_frames(*gargs)[0], "F")
        self.assertEqual(rp.classify_rrt_frames(*gargs, f_near_radius_m=0.045)[0], "")  # not F, and never M (M keeps 6 cm)
        for bad in (0.0, 0.07, float("nan")):
            with self.assertRaises(ValueError):
                rp.classify_rrt_frames(*args, f_near_radius_m=bad)
        labels = rp.classify_rrt_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])
        base = rp.admissible_frames(220, a["guard_tilt_deg"], a["actions"], labels)
        self.assertTrue(np.array_equal(base, rp.admissible_frames(220, a["guard_tilt_deg"], a["actions"], labels, None)))
        tilt = a["guard_tilt_deg"].copy()
        tilt[75, 1] = 20.0                                                           # a recorded rise at frame 75 only
        ok = rp.admissible_frames(220, tilt, a["actions"], labels, recorded_max_tilt_h20=15.0)
        self.assertFalse(ok[55:75].any())                                            # 75 lies within a+1..a+20 for a = 55..74
        self.assertTrue(ok[54] and ok[75] and ok[76] and ok[40])
        self.assertFalse(rp.admissible_frames(220, tilt, a["actions"], labels, recorded_max_tilt_h20=11.0)[80])   # 12 deg ahead of a T frame
        self.assertTrue(rp.admissible_frames(220, tilt, a["actions"], labels, recorded_max_tilt_h20=11.0)[30])    # frames 31..50 upright
        anchors, _ = rp.select_anchors({("full_train", 1): a}, {"T": 0, "F": 50, "G": 0, "M": 0}, 3, f_near_radius_m=0.045)
        self.assertEqual(anchors, [])                                                # no F frame within 4.5 cm in the fixture

    def test_family_set_b(self):
        sets_b, fractions = rp.parse_family_sets_b(["T=recorded,hold,push,push4"], "T=0.4")
        self.assertEqual((sets_b, fractions), ({"T": ("recorded", "hold", "push", "push4")}, {"T": 0.4}))
        self.assertEqual(rp.parse_family_sets_b(None, None), ({}, {}))
        for texts, frac in ((["T=recorded,hold"], None), (["T=recorded,hold"], "F=0.5"), (["T=hold,recorded"], "T=0.5"),
                            (["T=recorded,hold"], "T=1.5"), (["M=recorded,approach"], "M=0.5"), (["T=recorded", "T=recorded,hold"], "T=0.5")):
            with self.assertRaises(ValueError):
                rp.parse_family_sets_b(texts, frac)
        sets_a = rp.DEFAULT_FAMILY_SETS
        picks = [rp.family_set_for(seed, "T", sets_a, sets_b, fractions) for seed in range(2000)]
        share = sum(p[0] == "B" for p in picks) / len(picks)
        self.assertAlmostEqual(share, 0.4, delta=0.04)
        self.assertEqual(picks, [rp.family_set_for(seed, "T", sets_a, sets_b, fractions) for seed in range(2000)])   # deterministic
        self.assertTrue(all(p[1] == (sets_b["T"] if p[0] == "B" else sets_a["T"]) for p in picks))
        self.assertEqual(rp.family_set_for(5, "F", sets_a, sets_b, fractions), ("A", sets_a["F"]))                 # F has no set B
        self.assertEqual(rp.family_set_for(5, "T", sets_a, sets_b, {"T": 0.0})[0], "A")

    def test_approach_lands_in_the_shell(self):
        goal = np.array([0.43, 0.0, 0.17])
        green = goal + [0.025, 0.004, -0.006]
        pinch = green + [0.0, 0.0, 0.012]
        com = np.array([green, [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]])
        specs = rp.draw_rollouts(7, ("recorded", "approach", "approach", "approach"), pinch, com, goal, CFG)
        for s in specs[1:]:
            ap = s["approach"]
            self.assertTrue(ap["feasible"])
            self.assertTrue(rp.APPROACH_SHELL_M[0] <= ap["predicted_goal_distance_m"] <= rp.APPROACH_SHELL_M[1])
            plant = law.ModelPlant(law.resolve_model("grasped"), pinch)
            law.run_sequence(plant, rp.segments_from_json(s["segments"]), 1.0, CFG)
            np.testing.assert_allclose(green + plant.position - pinch, ap["predicted_end_xyz"], atol=1e-12)
            self.assertEqual(sum(k == "edge" for k, _ in s["segments"]), ap["edges"])
        far = rp.approach_spec(np.random.default_rng(0), pinch, goal + [0.10, 0, 0], goal, CFG)
        self.assertFalse(far["feasible"])
        self.assertEqual(far["edges"], 3)
        with self.assertRaises(ValueError):
            rp.draw_rollouts(7, ("recorded", "approach"), pinch, com, None, CFG)

    def test_option_parsing_and_estimate(self):
        sets = rp.parse_family_sets(["G=recorded,hold,approach,approach"])
        self.assertEqual(sets["G"], ("recorded", "hold", "approach", "approach"))
        self.assertEqual(sets["T"], rp.DEFAULT_FAMILY_SETS["T"])
        for bad in (["X=recorded"], ["T=hold,recorded"], ["T=recorded,approach"], ["G=recorded,fly"]):
            with self.assertRaises(ValueError):
                rp.parse_family_sets(bad)
        self.assertEqual(rp.parse_quotas("T=10,G=3", 0.5), {"T": 5, "F": 750, "G": 2, "M": 375})
        with self.assertRaises(ValueError):
            rp.parse_quotas(None, 0.0)
        e = rp.cost_estimate(100, 800, 3)
        self.assertEqual((e["rollouts"], e["frames"], e["bytes"]), (800, 800 * 33, 800 * rp.BYTES_PER_ROLLOUT))
        self.assertAlmostEqual(e["wall_hours"]["3"] * 3, e["wall_hours"]["1"])
        self.assertAlmostEqual(e["memory_gib_at_workers"], rp.MAIN_PSS_GIB + 3 * rp.WORKER_PSS_GIB)

    def test_goal_image_manifest_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(rp.load_goal_images(None)[0], {})
            self.assertEqual(rp.load_goal_images(root / "missing")[0], {})
            d = root / "g1"
            d.mkdir()
            (d / "manifest.json").write_text(json.dumps({"goals": [
                {"goal_id": 3, "episode_id": 7, "split": "train", "achieved_green_com_xyz": [0.43, 0.0, 0.171]},
                {"id": 4, "source": {"cache": "/x/wm_full_001", "split": "val", "episode_id": 9}},
                {"goal_image_id": 5, "status": "failed", "episode_id": 8},
                {"image_id": 6, "source": {"cache": "wm_train_bench_001", "cache_episode_id": 100003}}]}), encoding="utf-8")
            links, info = rp.load_goal_images(d)
            self.assertEqual(links, {("wm_full_001", 7): {"goal_image_id": 3, "goal_xyz": [0.43, 0.0, 0.171]},
                                     ("wm_train_bench_001", 100003): {"goal_image_id": 6, "goal_xyz": None}})
            self.assertEqual(info["skipped"], {"split val": 1, "status": 1})
            for bad in ({"goals": [{"goal_id": 1, "id": 2, "episode_id": 3}]}, {"goals": [{"goal_id": 1, "episode_id": 3}, {"goal_id": 2, "episode_id": 3}]},
                        {"things": []}, {"format": "goal_images_v2", "status": "running", "images": []},
                        {"format": "something_else", "status": "complete", "images": []}):
                (d / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    rp.load_goal_images(d)
            # the real goal_images_v2 layout (data/goal_images_002/manifest.json 'images' rows): val / train / planner sets,
            # status 'accepted', achieved_green_xyz, planner goals without a source episode (id -1, cache None)
            v2 = {"format": "goal_images_v2", "status": "complete", "images": [
                {"id": 0, "set": "val", "name": "val_ep0027_v0", "source_episode_id": 27, "status": "accepted", "split": "val",
                 "cache": "wm_full_001", "achieved_green_xyz": [0.44, -0.03, 0.16], "aim_raised": False},
                {"id": 500, "set": "train", "name": "train_ep0001_v0", "source_episode_id": 1, "status": "accepted", "split": "train",
                 "cache": "wm_full_001", "achieved_green_xyz": [0.4586, 0.0145, 0.1848], "aim_raised": False},
                {"id": 501, "set": "train", "name": "train_ep0004_v0", "source_episode_id": 4, "status": "rejected", "split": "train",
                 "cache": "wm_full_001", "achieved_green_xyz": None},
                {"id": 2000, "set": "planner", "name": "planner_0000_v0", "source_episode_id": -1, "status": "accepted", "split": None,
                 "cache": None, "achieved_green_xyz": [0.42, 0.0, 0.17]},
                {"id": 2001, "set": "planner", "name": "planner_0001_v0", "source_episode_id": -1, "status": "accepted", "split": None,
                 "cache": None, "achieved_green_xyz": [0.43, 0.0, 0.17]}]}
            (d / "manifest.json").write_text(json.dumps(v2), encoding="utf-8")
            links, info = rp.load_goal_images(d)
            self.assertEqual(links, {("wm_full_001", 1): {"goal_image_id": 500, "goal_xyz": [0.4586, 0.0145, 0.1848]}})
            self.assertEqual(info["skipped"], {"split val": 1, "status": 1, "no source episode": 2})


# ----- stage-1 CLI on synthetic caches -------------------------------------------------------------------------------
def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def write_raw_episode(folder, arrays):
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    write_json(folder / "metadata.json", {"status": "complete", "num_states": len(arrays["time"])})


def write_h5(path, n, npz_sha, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((n, 224, 224, 3), np.uint8)
    pixels[:, :, :, 0] = value
    pixels[:, 0, 0, 1] = np.arange(n) % 256
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.attrs["complete"] = True
        f.attrs["source_npz_sha256"] = npz_sha


@unittest.skipUnless(HAVE_H5, "stage-1 CLI needs h5py (training venv)")
class PlanCliTests(unittest.TestCase):
    def build_fixture(self, root):
        raw = root / "exploration_raw"
        write_json(raw / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        train_rows = []
        for eid in (11, 12):
            folder = raw / f"episode_{eid:04d}"
            write_raw_episode(folder, synthetic_episode())
            npz_sha = sha256(folder / "trajectory.npz")
            write_h5(root / "wm_full_001" / "episodes" / f"episode_{eid:04d}.h5", 220, npz_sha, eid)
            train_rows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "num_states": 220, "source_npz_sha256": npz_sha,
                               "source_metadata_sha256": sha256(folder / "metadata.json")})
        held = [{"id": 21, "path": "episodes/episode_0021.h5", "num_states": 220}]         # NO raw files: never opened
        write_json(root / "wm_full_001" / "index.json", {"status": "complete", "source_root": str(raw), "source_manifest_sha256": sha256(raw / "manifest.json"),
                                                         "normalization": NORMALIZATION, "source_action_columns": [0, 1, 2, 4],
                                                         "episodes": {"train": train_rows, "val": held, "test": [{"id": 31, "path": "x", "num_states": 1}]}})
        write_json(root / "classifier" / "run.json", {"dev_episode_ids": [21], "report_episode_ids": [22]})
        bench = root / "danger_train_001"
        write_json(bench / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        episodes, rows, strata = [], [], {"cell": {"dev": [], "test": []}}
        for eid in (0, 1, 2):
            if eid < 2:
                folder = bench / f"episode_{eid:04d}"
                write_raw_episode(folder, synthetic_episode())
                npz_sha, meta_sha = sha256(folder / "trajectory.npz"), sha256(folder / "metadata.json")
                write_h5(root / "wm_train_bench_001" / "episodes" / f"episode_{eid:04d}.h5", 220, npz_sha, 50 + eid)
            else:
                npz_sha, meta_sha = "0" * 64, "0" * 64                                      # bench_dev row: NO raw files
            episodes.append({"dataset": 0, "episode_id": eid, "path": f"episode_{eid:04d}", "stratum": "cell", "frames": 220})
            strata["cell"]["test" if eid < 2 else "dev"].append({"dataset": 0, "episode_id": eid})
            rows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "num_states": 220, "source_dataset": 0, "source_id": eid,
                         "source_npz_sha256": npz_sha, "source_metadata_sha256": meta_sha})
        split = root / "split" / "benchmark_split.json"
        write_json(split, {"format": "danger_bench_split_v1", "datasets": [{"index": 0, "path": str(bench), "manifest_sha256": sha256(bench / "manifest.json")}],
                           "strata": strata, "episodes": episodes})
        write_json(root / "wm_train_bench_001" / "index.json", {"status": "complete", "eval_cache": True, "split_sha256": sha256(split),
                                                                "source_manifest_sha256": {str(bench): sha256(bench / "manifest.json")},
                                                                "normalization": NORMALIZATION, "bench_split": {"bench_dev": [2], "bench_test": [0, 1]},
                                                                "episodes": {"val": rows}})
        write_json(root / "ft" / "run.json", {"finetune": {"data": {"bench_rows_used_for_training": "bench_test"}},
                                              "bench_cache": str(root / "wm_train_bench_001"),
                                              "bench_cache_index_sha256": sha256(root / "wm_train_bench_001" / "index.json"),
                                              "cache_index_sha256": sha256(root / "wm_full_001" / "index.json")})
        write_json(root / "goal_images" / "manifest.json", {"goals": [{"goal_id": 0, "episode_id": 11, "split": "train",
                                                                       "achieved_green_com_xyz": [0.431, 0.001, 0.171]}]})
        return ["--full-cache", str(root / "wm_full_001"), "--bench-cache", str(root / "wm_train_bench_001"), "--bench-split", str(split),
                "--ft-run", str(root / "ft" / "run.json"), "--classifier-run", str(root / "classifier" / "run.json"),
                "--goal-images", str(root / "goal_images"), "--quota", "T=2,F=2,G=4,M=2", "--seed", "3"]

    def test_plan_end_to_end_on_synthetic_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)
            out = root / "plan_out"
            with redirect_stdout(io.StringIO()):
                summary = rp.main(args + ["--output", str(out)])
            for name in ("run.json", "anchors.json", "cache_pixels.npz", "summary.json", "REPORT_ZH.md"):
                self.assertTrue((out / name).is_file(), name)
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            anchors = doc["anchors"]
            self.assertEqual(summary["counts"]["by_stratum"], {"T": 2, "F": 2, "G": 4, "M": 2})
            self.assertEqual({(x["set"], x["episode_id"]) for x in anchors} - {("full_train", 11), ("full_train", 12), ("bench_train", 0), ("bench_train", 1)}, set())
            self.assertEqual(summary["scanned"], {"full_train": 2, "bench_train": 2})
            with np.load(out / "cache_pixels.npz") as npz:
                for x in anchors:
                    block = npz[f"a{x['id']:04d}"]
                    value = x["episode_id"] if x["set"] == "full_train" else 50 + x["episode_id"]
                    self.assertTrue(np.all(block[:, 5, 5, 0] == value))
                    np.testing.assert_array_equal(block[:, 0, 0, 1], np.arange(x["frame"] - 10, x["frame"] + 1))
            for x in anchors:
                self.assertEqual(len(x["sequences"]), 8)
                self.assertEqual(x["grip_command"], 1.0 if x["stratum"] in ("G", "M") else -1.0)
                if x["stratum"] == "G":
                    self.assertTrue(any(s["family"] == "approach" for s in x["sequences"]))
                    linked = (x["set"], x["episode_id"]) == ("full_train", 11)
                    self.assertEqual(x["goal"]["goal_image_id"], 0 if linked else -1)
                    self.assertEqual(x["goal"]["goal_xyz"], [0.431, 0.001, 0.171] if linked else [0.43, 0.0, 0.17])
            self.assertIn("wall_hours", summary["estimate"])
            self.assertEqual(summary["files_sha256"]["anchors.json"], sha256(out / "anchors.json"))
            with self.assertRaises(FileExistsError):
                rp.main(args + ["--output", str(out)])
            # a raw file that no longer matches the cache index is refused (FAILED.json, no summary)
            with (root / "exploration_raw" / "episode_0012" / "metadata.json").open("a", encoding="utf-8") as f:
                f.write(" ")
            bad = root / "plan_bad"
            with self.assertRaises(ValueError):
                rp.main(args + ["--output", str(bad)])
            self.assertTrue((bad / "FAILED.json").is_file())
            self.assertFalse((bad / "summary.json").exists())

    def test_family_set_b_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)
            out = root / "plan_b"
            with redirect_stdout(io.StringIO()):
                summary = rp.main(args + ["--output", str(out), "--families-b", "T=recorded,hold,push,push4", "--families-b-fraction", "T=1.0"])
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["families_b"], {"T": {"families": ["recorded", "hold", "push", "push4"], "fraction": 1.0}})
            for x in doc["anchors"]:
                self.assertEqual(x["family_set"], "B" if x["stratum"] == "T" else "A")
                if x["stratum"] == "T":
                    self.assertEqual([s["family"] for s in x["sequences"]], ["recorded", "hold", "push", "push4"])
            self.assertEqual(summary["counts"]["by_stratum_family_set"]["T/B"], 2)
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                rp.main(args + ["--output", str(root / "plan_b_bad"), "--families-b", "T=recorded,hold"])      # fraction missing
            self.assertFalse((root / "plan_b_bad").exists())                                                   # refused before any output

    def test_goal_image_strata_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)                           # only full_train episode 11 has a goal image
            out = root / "plan_gi"
            with redirect_stdout(io.StringIO()):
                summary = rp.main(args + ["--output", str(out), "--goal-image-strata", "G"])
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            g = [x for x in doc["anchors"] if x["stratum"] == "G"]
            self.assertTrue(1 <= len(g) <= 2)                         # quota 4, but one linked episode (G frames 100-139)
            for x in g:
                self.assertEqual((x["set"], x["episode_id"], x["goal"]["goal_image_id"], x["goal"]["goal_source"]),
                                 ("full_train", 11, 0, "goal_image_achieved_com"))
            self.assertEqual(summary["counts"]["goal_images_linked_G"], len(g))
            self.assertEqual(doc["rules"]["goal_image_strata"], ["G"])
            cap = root / "capacity_gi"
            with redirect_stdout(io.StringIO()):
                c = rp.main(args + ["--output", str(cap), "--capacity-only", "--goal-image-strata", "G"])
            self.assertEqual(c["capacity"]["options"]["goal_image_strata"], ["G"])
            self.assertEqual(c["capacity"]["alone"]["G"]["by_source"]["bench_train"], 0)
            no_goal = [a for a in args]
            i = no_goal.index("--goal-images")
            del no_goal[i:i + 2]
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                rp.main(no_goal + ["--output", str(root / "plan_gi_bad"), "--goal-image-strata", "G"])
            self.assertFalse((root / "plan_gi_bad").exists())         # refused before any output

    def test_capacity_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)
            out = root / "capacity"
            with redirect_stdout(io.StringIO()):
                summary = rp.main(args + ["--output", str(out), "--capacity-only"])
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["REPORT_ZH.md", "run.json", "summary.json"])   # no plan files
            self.assertEqual(summary["format"], rp.CAPACITY_FORMAT)
            self.assertEqual(json.loads((out / "run.json").read_text(encoding="utf-8"))["format"], rp.CAPACITY_FORMAT)
            cap = summary["capacity"]
            for s in rp.STRATA:
                # 4 synthetic episodes, every stratum spans 40 frames: 1 or 2 anchors (>= 33 frames apart) per episode
                self.assertTrue(4 <= cap["alone"][s]["anchors"] <= 8, (s, cap["alone"][s]))
                self.assertLessEqual(cap["joint"][s]["anchors"], cap["alone"][s]["anchors"] + 4)
                self.assertEqual(sum(cap["joint"][s]["by_source"].values()), cap["joint"][s]["anchors"])
            self.assertEqual(cap["joint"]["T"], cap["alone"]["T"])                              # T is first in priority
            self.assertEqual(summary["scanned"], {"full_train": 2, "bench_train": 2})

    def test_goal_image_of_a_held_out_episode_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)
            write_json(root / "goal_images" / "manifest.json", {"format": "goal_images_v2", "status": "complete", "images": [
                {"id": 7, "source_episode_id": 21, "split": "train", "cache": "wm_full_001", "status": "accepted",
                 "achieved_green_xyz": [0.43, 0.0, 0.17]}]})                      # 21 is a VAL id of the fixture
            out = root / "plan_goal_bad"
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                rp.main(args + ["--output", str(out)])
            self.assertTrue((out / "FAILED.json").is_file())
            self.assertFalse((out / "summary.json").exists())

    def test_training_side_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.build_fixture(root)
            write_json(root / "classifier" / "run.json", {"dev_episode_ids": [11], "report_episode_ids": []})   # a train id in the val half
            with self.assertRaises(ValueError):
                rp.full_train_sources(root / "wm_full_001" / "index.json", root / "classifier" / "run.json")
            ft = json.loads((root / "ft" / "run.json").read_text(encoding="utf-8"))
            ft["finetune"]["data"]["bench_rows_used_for_training"] = "bench_dev"
            write_json(root / "ft" / "run.json", ft)
            with self.assertRaises(ValueError):
                rp.bench_train_sources(root / "wm_train_bench_001" / "index.json", root / "split" / "benchmark_split.json", root / "ft" / "run.json")
            ft["finetune"]["data"]["bench_rows_used_for_training"] = "bench_test"
            write_json(root / "ft" / "run.json", ft)
            rows, _, info = rp.bench_train_sources(root / "wm_train_bench_001" / "index.json", root / "split" / "benchmark_split.json", root / "ft" / "run.json")
            self.assertEqual(([r["episode_id"] for r in rows], info["refused_rows"]), ([0, 1], 1))       # the bench_dev row is refused
            split = json.loads((root / "split" / "benchmark_split.json").read_text(encoding="utf-8"))
            split["datasets"][0]["path"] = str(root / "danger_bench_001")                                   # benchmark data is never a source
            write_json(root / "split" / "benchmark_split.json", split)
            with self.assertRaises(ValueError):
                rp.bench_train_sources(root / "wm_train_bench_001" / "index.json", root / "split" / "benchmark_split.json", root / "ft" / "run.json")


# ----- stage 2 on the real scene ---------------------------------------------------------------------------------------
def record_carry_episode(folder, env, steps, seed):
    """A real-scene exploration-format episode recorded while the data-collection carry controller works."""
    from collect_exploration_trajs import ExplorationRecorder
    from collect_grasp_trajs import episode_metadata
    from exploration import ExplorationController
    env.reset(seed=0)
    controller = ExplorationController(env, np.random.default_rng(seed), "carry", max_steps=10_000)
    meta = episode_metadata(env, {"kind": "unit_test_fixture"}, seed, "front_pixels", 0.0, 0.0, steps)
    meta.update(hover_goal_xyz=env.hover_goal.tolist())
    recorder = ExplorationRecorder(folder, env, meta, "front_pixels")
    tail = None
    for _ in range(steps):
        phase = controller.phase
        env.last_transition = None
        if not controller.finished:
            controller.step()
        if env.last_transition is None:                                  # the controller finished early: hold still
            from bench_restore_check import AnchoredHold, pinch_xyz
            tail = tail or AnchoredHold(pinch_xyz(env), 1.0 if env.task_report()["grasped"] else -1.0)
            env.step(tail.action(env))
            phase = "TEST_TAIL_HOLD"
        recorder.append(env.last_transition, phase)
    recorder.finish({"phase": "DONE"})


@unittest.skipUnless(HAVE_SIM, "stage 2 needs the simulation venv")
class SimulationTests(unittest.TestCase):
    def make_plan(self, root):
        from PIL import Image
        import collect_random_branches as crb
        env = ExplorationScene()
        self.addCleanup(env.close)
        data = root / "raw"
        data.mkdir()
        write_json(data / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        folder = data / "episode_0000"
        record_carry_episode(folder, env, 250, seed=3)
        arrays = rp.load_episode_arrays(folder / "trajectory.npz")
        labels = rp.classify_rrt_frames(arrays["pinch_xyz"], arrays["object_com_xyz"], arrays["grasped"], arrays["hover_error_m"], arrays["guard_tilt_deg"])
        ok = rp.admissible_frames(len(labels), arrays["guard_tilt_deg"], arrays["actions"], labels)
        picks = {}
        for stratum in ("F", "G", "M"):
            frames = [f for f in np.flatnonzero(ok & (labels == stratum)) if all(abs(f - p) >= rp.MIN_GAP_FRAMES for p in picks.values())]
            if frames:
                picks[stratum] = int(frames[len(frames) // 2])
        self.assertIn("G", picks, f"the carry fixture produced no G frame: {sorted(set(labels))}")
        source = {"source": "full_train", "episode_id": 0, "cache_episode_id": 0, "episode_dir": str(folder), "npz_sha256": sha256(folder / "trajectory.npz"),
                  "metadata_sha256": sha256(folder / "metadata.json"), "cache_h5": "unused", "frames": len(labels)}
        letterbox, _ = crb.import_letterbox()
        records, pixels = [], {}
        chosen = [("G", picks["G"])] + [(s, f) for s, f in picks.items() if s != "G"][:1]
        push_set = ("recorded", "hold", "push", "push4", "push_hold", "push2_hold", "guard")   # the second anchor exercises the push families
        for i, (stratum, frame) in enumerate(chosen):
            anchor = {"source": "full_train", "episode_id": 0, "stratum": stratum, "frame": frame}
            families = rp.DEFAULT_FAMILY_SETS[stratum] if i == 0 else push_set
            records.append(rp.make_anchor_record(i, anchor, 1000 + i, source, arrays, families, None, CFG))
            pixels[f"a{i:04d}"] = np.stack([letterbox(Image.open(folder / "images" / f"{f:06d}.png"), 224) for f in range(frame - 10, frame + 1)])
        out = root / "plan"
        out.mkdir()
        write_json(out / "anchors.json", {"format": rp.PLAN_FORMAT, "normalization": NORMALIZATION, "prefix": 10, "horizon": 32,
                                          "families": {k: list(v) for k, v in rp.DEFAULT_FAMILY_SETS.items()}, "anchors": records})
        np.savez_compressed(out / "cache_pixels.npz", **pixels)
        write_json(out / "summary.json", {"format": rp.PLAN_FORMAT, "files_sha256": {"anchors.json": sha256(out / "anchors.json"),
                                                                                     "cache_pixels.npz": sha256(out / "cache_pixels.npz")}})
        return out, folder, records

    def test_end_to_end_two_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_dir, folder, records = self.make_plan(root)
            with redirect_stdout(io.StringIO()) as printed:
                estimate = crr.main(["--plan", str(plan_dir), "--plan-only", "--workers", "2"])
            self.assertEqual((estimate["anchors"], estimate["rollouts"]), (2, 15))
            self.assertIn("wall_hours", printed.getvalue())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["plan", "raw"])            # --plan-only wrote nothing
            out, report = root / "branches", root / "report"
            args = ["--plan", str(plan_dir), "--output", str(out), "--report", str(report), "--workers", "2"]
            with redirect_stdout(io.StringIO()):
                summary = crr.main(args)
            s = summary["sanity"]
            self.assertEqual((s["anchors"], s["prefix_replay_exact_anchors"], s["recorded_exact_anchors"]), (2, 2, 2))
            self.assertEqual(s["prefix_diagnostics_match_source_anchors"], 2)
            self.assertLessEqual(s["prefix_restored_frame_max_abs_diff"], 1e-9)
            self.assertEqual((s["anchor_frame_equals_cache_fraction"], s["prefix_frames_equal_cache"]), (1.0, "22/22"))
            self.assertEqual((s["prefix_frames_raw_png_identical"], s["anchor_images_equal_across_rollouts_anchors"]), ("22/22", 2))
            for name in ("run.json", "summary.json", "REPORT_ZH.md"):
                self.assertTrue((report / name).is_file())
            self.assertTrue(summary["memory"]["sampled"]["active"])
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["format"]), ("complete", "rrt_branches_v1"))
            with np.load(folder / "trajectory.npz") as npz:
                source = {k: np.asarray(npz[k]) for k in ("actions", "guard_tilt_deg", "pinch_xyz", "object_com_xyz", "class_label", "unsafe")}
            low, high = CFG.env_bounds()
            for row, rec in zip(manifest["anchors"], records):
                a = row["frame"]
                R = len(rec["families"])
                with np.load(out / row["file"]) as npz:
                    z = {k: np.asarray(npz[k]) for k in npz.files}
                self.assertEqual(sha256(out / row["file"]), row["sha256"])
                self.assertEqual(z["pixels"].shape, (R, 33, 224, 224, 3))
                self.assertEqual((z["actions"].shape, z["nominal_actions"].shape, z["wm_action"].shape), ((R, 32, 5), (R, 32, 5), (R, 32, 4)))
                self.assertEqual((z["segment_kind"].shape, z["segment_direction"].shape), ((R, 4), (R, 4, 3)))
                # prefix diagnostics (frames a-10..a): stepped frames bit-exact, the restored frame within 1e-9, index 10 = anchor state
                self.assertEqual(z["prefix_guard_tilt_deg"].shape, (11, 2))
                for key in ("guard_tilt_deg", "pinch_xyz", "object_com_xyz"):
                    np.testing.assert_array_equal(z[f"prefix_{key}"][1:], source[key][a - 9:a + 1])
                    np.testing.assert_allclose(z[f"prefix_{key}"][0], source[key][a - 10], atol=1e-9, rtol=0)
                    np.testing.assert_array_equal(z[f"prefix_{key}"][10], z[key][0, 0])
                np.testing.assert_array_equal(z["prefix_class_label"], source["class_label"][a - 10:a + 1].astype(np.int8))
                np.testing.assert_array_equal(z["prefix_unsafe"], source["unsafe"][a - 10:a + 1])
                self.assertEqual(z["prefix_robot_env_contact"].shape, (11,))
                for r, spec in enumerate(rec["sequences"]):
                    if spec["family"] in rp.PUSH_EDGES:
                        n = rp.PUSH_EDGES[spec["family"]]
                        np.testing.assert_array_equal(z["segment_kind"][r, :n], ["edge"] * n)
                        self.assertTrue(np.all(z["segment_direction"][r, :n, 2] == 0.0))                    # horizontal pushes
                self.assertEqual(z["families"].tolist(), list(rec["families"]))
                np.testing.assert_array_equal(z["actions"][0], source["actions"][a:a + 32])
                np.testing.assert_array_equal(z["guard_tilt_deg"][0], source["guard_tilt_deg"][a:a + 33])
                self.assertTrue(np.isnan(z["nominal_actions"][0]).all() and np.isfinite(z["nominal_actions"][1:]).all())
                self.assertTrue(np.isfinite(z["nominal_open_loop_actions"][1:]).all())
                law_actions = z["actions"][1:]
                self.assertTrue(np.all(law_actions[..., 3] == 0) and np.all(law_actions[..., 4] == rec["grip_command"]))
                self.assertTrue(np.all(law_actions[..., :3] >= low) and np.all(law_actions[..., :3] <= high))
                np.testing.assert_array_equal(z["wm_action"], law.wm_normalize(z["actions"], NORMALIZATION))
                self.assertTrue(np.all(z["segment_index"][0] == -1) and np.all(z["segment_index"][1:] >= 0))
                for r, spec in enumerate(rec["sequences"][1:], start=1):
                    kinds = [k for k, _ in spec["segments"]]
                    self.assertEqual(z["segment_kind"][r, :len(kinds)].tolist(), kinds)
                    if spec["family"] == "hold":
                        drift = np.linalg.norm(z["pinch_xyz"][r] - z["pinch_xyz"][r, 0], axis=1).max()
                        self.assertLess(drift, 0.004)
                outcomes = row["rollouts"]
                self.assertEqual(z["robot_env_contact"].shape, (R, 33))
                for r, o in enumerate(outcomes):
                    held = bool(z["grasped"][r, 0])
                    touch = z["robot_env_contact"][r, 1:].any() or (z["green_env_contact"][r, 1:].any() if held else z["robot_green_contact"][r, 1:].any())
                    self.assertEqual(o["free_space"], not touch)
                    if held:
                        self.assertTrue(z["robot_green_contact"][r, 0])                          # the held block touches the fingers
                    if r > 0 and o["free_space"]:
                        self.assertLess(o["nominal_mismatch_max_mm"], 2.5, o)                    # free space: nominal ~ executed
                    if o["family"] == "approach":
                        self.assertLess(abs(o["green_goal_distance_final_m"] - o["approach"]["predicted_goal_distance_m"]), 0.006)
            with self.assertRaises(FileExistsError):
                crr.main(args)

    def test_tampered_source_writes_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_dir, folder, _ = self.make_plan(root)
            with (folder / "metadata.json").open("a", encoding="utf-8") as f:
                f.write(" ")
            out, report = root / "branches", root / "report"
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                crr.main(["--plan", str(plan_dir), "--output", str(out), "--report", str(report), "--workers", "1", "--anchor-ids", "0"])
            self.assertTrue((report / "FAILED.json").is_file())
            self.assertFalse((report / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
