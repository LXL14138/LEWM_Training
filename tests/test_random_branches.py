"""Random-action branch study: pure planning logic, the stage-1 CLI on synthetic caches, and stage 2 on the real scene.

Every fixture here is synthetic (written into a temporary directory); no dataset, cache or benchmark episode is read.
* PlanLogicTests: numpy only, any venv (the bit-exact comparison with evaluate_bench_counterfactual.normalize_actions
  needs torch + h5py: it runs in ~/lewm-train/.venv and is skipped elsewhere).
* PlanCliTests: need h5py (run with ~/lewm-train/.venv).
* SimulationTests: need MuJoCo + the GPU render environment (run with ~/lewm-manip/.venv in WSL).
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import random_branches_plan as plan
from wm_training.danger_strata import sha256

try:
    import h5py
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False
try:
    import collect_random_branches as crb
    from exploration import ExplorationScene
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

NORMALIZATION = {"names": ["dx", "dy", "dz", "dgripper"], "mean": [0.04, -0.01, 0.05, 0.3], "std": [0.05, 0.05, 0.08, 0.9],
                 "scale": [0.05, 0.05, 0.08, 0.9]}
RENDER_ENV = {"MUJOCO_GL": "glfw", "GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA",
              "LIBGL_ALWAYS_SOFTWARE": None, "LD_LIBRARY_PATH": "/usr/lib/wsl/lib"}


def base_arrays(n):
    com = np.zeros((n, 3, 3))
    com[:, 0], com[:, 1], com[:, 2] = [0.425, 0.0, 0.05], [0.425, -0.05, 0.05], [0.425, 0.05, 0.05]
    return {"time": 0.05 * np.arange(n), "pinch_xyz": np.tile([0.375, 0.0, 0.16], (n, 1)), "object_com_xyz": com,
            "grasped": np.zeros(n, bool), "hover_error_m": np.full(n, 0.2), "guard_tilt_deg": np.zeros((n, 2)),
            "unsafe": np.zeros(n, bool), "class_label": np.full(n, 2, np.int64)}


def exploration_arrays():
    """270 frames: 0-59 free, 60-119 gripper low beside guard 1, 120-179 carrying high, 180-239 near the goal, 240-269 free."""
    a = base_arrays(270)
    a["pinch_xyz"][60:120] = [0.425, -0.10, 0.06]
    a["grasped"][120:240] = True
    a["object_com_xyz"][120:180, 0] = [0.43, 0.0, 0.20]
    a["hover_error_m"][120:180] = 0.1
    a["object_com_xyz"][180:240, 0] = [0.43, 0.0, 0.17]
    a["hover_error_m"][180:240] = 0.01
    a["class_label"][180:240] = 0
    return a


def fall_arrays(n=100, onset=70):
    a = base_arrays(n)
    ramp = np.clip((np.arange(n) - 50) * 31.0 / (onset - 50), 0, None)
    a["guard_tilt_deg"][:, 0] = ramp
    a["unsafe"] = ramp > 30
    a["class_label"][a["unsafe"]] = 1
    a["pinch_xyz"][40:] = [0.425, -0.10, 0.06]
    return a


def negative_arrays(n=100):
    a = base_arrays(n)
    a["pinch_xyz"][40:61] = [0.47, 0.05, 0.08]
    return a


class PlanLogicTests(unittest.TestCase):
    def test_classify_frames_priorities(self):
        a = exploration_arrays()
        labels = plan.classify_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])
        self.assertEqual(set(labels[:60]), {"free"})                   # green resting 5 cm from both guards is NOT near
        self.assertEqual(set(labels[60:120]), {"near_guard"})
        self.assertEqual(set(labels[120:180]), {"carrying"})           # green 20 cm high: above the guards
        self.assertEqual(set(labels[180:240]), {"near_goal"})
        self.assertEqual(set(labels[240:]), {"free"})
        low = {k: v[120:121].copy() for k, v in a.items()}
        low["object_com_xyz"][0, 0] = [0.425, 0.0, 0.12]               # carried green block low between the guards
        self.assertEqual(plan.classify_frames(low["pinch_xyz"], low["object_com_xyz"], low["grasped"], low["hover_error_m"],
                                              low["guard_tilt_deg"])[0], "near_guard")
        tilted = {k: v[60:61].copy() for k, v in a.items()}
        tilted["guard_tilt_deg"][0] = [21.0, 0.0]
        self.assertEqual(plan.classify_frames(tilted["pinch_xyz"], tilted["object_com_xyz"], tilted["grasped"], tilted["hover_error_m"],
                                              tilted["guard_tilt_deg"])[0], "free")
        with self.assertRaises(ValueError):
            plan.classify_frames(a["pinch_xyz"][:5], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])

    def test_admissible_frames_and_spacing(self):
        tilt = np.zeros((60, 2))
        tilt[40, 1] = 26.0
        ok = plan.admissible_frames(60, tilt)
        self.assertEqual(np.flatnonzero(ok).tolist(), [f for f in range(12, 36) if f != 40])
        rng = np.random.default_rng(0)
        self.assertIsNone(plan.pick_spaced(rng, [10, 20, 30], [20]))
        self.assertEqual(plan.pick_spaced(rng, [10, 20, 50], [20]), 50)

    def test_select_set_v_quotas_eligibility_and_determinism(self):
        episodes = {7: exploration_arrays(), 9: exploration_arrays()}
        quotas = {"near_guard": 3, "carrying": 2, "near_goal": 5, "free": 1}
        anchors, available = plan.select_set_v(episodes, quotas, seed=3)
        self.assertEqual(anchors, plan.select_set_v(episodes, quotas, seed=3)[0])
        counts = {s: sum(x["stratum"] == s for x in anchors) for s in quotas}
        self.assertEqual(counts["free"], 1)
        self.assertEqual(counts["carrying"], 2)
        self.assertLessEqual(counts["near_guard"], 3)
        self.assertEqual(available["near_goal"]["episodes"], 2)
        for x in anchors:
            a = episodes[x["episode_id"]]
            label = plan.classify_frames(a["pinch_xyz"], a["object_com_xyz"], a["grasped"], a["hover_error_m"], a["guard_tilt_deg"])[x["frame"]]
            self.assertEqual(label, x["stratum"])
            self.assertTrue(12 <= x["frame"] <= 270 - 1 - 24)
        for eid in episodes:
            frames = sorted(x["frame"] for x in anchors if x["episode_id"] == eid)
            self.assertTrue(all(b - a >= plan.MIN_GAP_FRAMES for a, b in zip(frames, frames[1:])))

    def test_fall_and_negative_anchor_rules(self):
        a = fall_arrays()
        for seed in range(20):
            k, frame, draws = plan.fall_anchor(np.random.default_rng(seed), 70, 100, a["guard_tilt_deg"])
            self.assertTrue(5 <= k <= 30 and frame == 70 - k and draws >= 1)
            self.assertLessEqual(a["guard_tilt_deg"][frame].max(), 25.0)
        self.assertIsNone(plan.fall_anchor(np.random.default_rng(0), 70, 60, a["guard_tilt_deg"][:60]))   # a = 70 - k > 35 = 59 - 24
        n = negative_arrays()
        frame = plan.negative_anchor(np.random.default_rng(1), n)
        self.assertTrue(40 <= frame <= 60)
        with self.assertRaises(ValueError):
            plan.negative_anchor(np.random.default_rng(1), fall_arrays())

    def test_sequences_directions_and_reference_path(self):
        pinch, com = np.array([0.40, -0.02, 0.08]), exploration_arrays()["object_com_xyz"][0]
        specs = plan.draw_sequences(123, pinch, com)
        self.assertEqual([s["family"] for s in specs], list(plan.FAMILIES))
        self.assertEqual(specs, plan.draw_sequences(123, pinch, com))
        self.assertNotEqual(specs, plan.draw_sequences(124, pinch, com))
        for s in specs[:2]:
            self.assertIsNone(s["directions"])
        for s in specs[2:]:
            d = np.asarray(s["directions"])
            np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
            self.assertTrue(set(s["speeds_mps"]) <= set(plan.SPEEDS_MPS))
        for s in specs[6:]:
            first = np.asarray(s["directions"][0])
            self.assertEqual(first[2], 0.0)
            self.assertEqual(s["guard_index"], 1)                        # guard 1 at y = -0.05 is nearer
            toward = (com[1, :2] - pinch[:2]) / np.linalg.norm(com[1, :2] - pinch[:2])
            angle = np.degrees(np.arccos(np.clip(first[:2] @ toward, -1, 1)))
            self.assertLessEqual(angle, plan.GUARD_ANGLE_DEG + 1e-9)
            self.assertAlmostEqual(angle, abs(s["guard_angle_deg"]), places=6)
        low, high = plan.reference_box(pinch)
        path = plan.reference_path(pinch, [[1, 0, 0], [0, 1, 0], [0, 0, -1]], [0.05, 0.02, 0.035], low, high)
        self.assertEqual(path.shape, (24, 3))
        np.testing.assert_allclose(path[0], pinch + [0.0025, 0, 0])
        np.testing.assert_allclose(path[7], pinch + [0.02, 0, 0])
        np.testing.assert_allclose(path[15], pinch + [0.02, 0.008, 0])
        np.testing.assert_allclose(path[23], pinch + [0.02, 0.008, -0.014])
        clipped = plan.reference_path([0.49, 0.0, 0.03], [[1, 0, 0], [0, 0, -1], [0, 0, -1]], [0.05] * 3,
                                      *plan.reference_box([0.49, 0.0, 0.03]))
        self.assertAlmostEqual(clipped[:, 0].max(), 0.50)
        self.assertAlmostEqual(clipped[:, 2].min(), 0.025)
        outside_low, outside_high = plan.reference_box([0.30, 0.0, 0.30])            # widened to contain the anchor
        self.assertTrue(np.all(outside_low <= [0.30, 0.0, 0.30]) and np.all(outside_high >= [0.30, 0.0, 0.30]))
        with self.assertRaises(ValueError):
            plan.reference_path(pinch, [[2, 0, 0], [0, 1, 0], [0, 0, 1]], [0.05] * 3, low, high)

    def test_wm_action_float32_recipe_is_bit_exact(self):
        rng = np.random.default_rng(4)
        actions = rng.uniform(-1, 1, (8, 24, 5))
        out = plan.normalize_wm_actions(actions, NORMALIZATION)
        mean, scale = np.asarray(NORMALIZATION["mean"], np.float32), np.asarray(NORMALIZATION["scale"], np.float32)
        expected = ((actions.astype(np.float32)[..., [0, 1, 2, 4]] - mean) / scale).astype(np.float32)
        self.assertEqual((out.dtype, out.shape), (np.float32, (8, 24, 4)))
        np.testing.assert_array_equal(out, expected)                                  # bit-exact, not allclose
        np.testing.assert_array_equal(plan.normalize_wm_actions(actions[3], NORMALIZATION), expected[3])
        for bad in ({"mean": [0, 0, 0], "scale": [1, 1, 1, 1]}, {"mean": [0, 0, 0, 0], "scale": [1, 0, 1, 1]}):
            with self.assertRaises(ValueError):
                plan.normalize_wm_actions(actions, bad)
        with self.assertRaises(ValueError):
            plan.normalize_wm_actions(actions[..., :4], NORMALIZATION)

    def test_wm_action_equals_the_evaluator_function(self):
        try:
            from wm_training.evaluate_bench_counterfactual import normalize_actions
        except ImportError as error:                                                  # needs torch: training venv only
            self.skipTest(f"evaluate_bench_counterfactual not importable here ({error})")
        rng = np.random.default_rng(5)
        actions = np.concatenate([rng.uniform(-1, 1, (200, 5)), np.array([[1.0, -1.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])])
        mean, scale = (np.asarray(NORMALIZATION[k], np.float32) for k in ("mean", "scale"))
        np.testing.assert_array_equal(plan.normalize_wm_actions(actions, NORMALIZATION), normalize_actions(actions, mean, scale))

    def test_tracking_saturation_and_rise_fields(self):
        pinch = np.zeros((25, 3))
        pinch[:, 2] = 0.12
        reference = np.zeros((24, 3))
        reference[:, 2] = 0.12
        reference[:, 0] = np.linspace(0.001, 0.03, 24)                                # the pinch never moves: final error 3 cm
        reference[20:, 2] = 0.08                                                      # the reference dips below min(9 cm, 12 cm)
        err = plan.tracking_error(pinch, reference)
        self.assertEqual(err.shape, (24,))
        self.assertAlmostEqual(err[0], 0.001)
        self.assertTrue(np.isnan(plan.tracking_error(pinch, np.full((24, 3), np.nan))).all())
        with self.assertRaises(ValueError):
            plan.tracking_error(pinch, reference[:23])
        actions = np.zeros((24, 5))
        actions[5, 0], actions[6, 2], actions[7, 1], actions[8, 4] = 1.0, -1.0, 0.999999, 1.0   # gripper at +-1 is not saturation
        self.assertEqual(plan.saturated_steps(actions).tolist(), [k in (5, 6) for k in range(24)])
        tilt = np.zeros((25, 2))
        tilt[:, 0] = 12.0                                                             # already tilted at the anchor
        tilt[4, 0] = 20.0
        tilt[12, 1] = 40.0
        unsafe = tilt.max(1) > 30
        wm = np.zeros((24, 4), np.float32)
        wm[3, 1] = -19.5
        o = plan.rollout_outcome(tilt, unsafe, np.full(25, 2), pinch, np.zeros((25, 2)), np.zeros((25, 2)), reference=reference,
                                 actions=actions, wm_action=wm)
        self.assertAlmostEqual(o["tracking_error_final_m"], np.hypot(0.03, 0.04))
        self.assertGreaterEqual(o["tracking_error_max_m"], o["tracking_error_final_m"])
        self.assertEqual((o["saturated_steps"], o["wm_action_absmax"], o["reference_below_free_floor"]), (2, 19.5, True))
        self.assertEqual((o["rise_h8"], o["rise_h16"]), (8.0, 28.0))
        self.assertEqual((o["anchor_pinch_z_m"], o["pinch_min_z_m"]), (0.12, 0.12))
        recorded = plan.rollout_outcome(tilt, unsafe, np.full(25, 2), pinch, np.zeros((25, 2)), np.zeros((25, 2)),
                                        reference=np.full((24, 3), np.nan), actions=actions)
        self.assertIsNone(recorded["tracking_error_final_m"])
        self.assertIsNone(recorded["reference_below_free_floor"])
        self.assertIsNone(recorded["wm_action_absmax"])
        low_anchor = pinch.copy()
        low_anchor[:, 2] = 0.05                                                       # anchor at 5 cm: the floor is the anchor height
        hold = plan.rollout_outcome(tilt, unsafe, np.full(25, 2), low_anchor, np.zeros((25, 2)), np.zeros((25, 2)),
                                    reference=np.tile(low_anchor[0], (24, 1)))
        self.assertEqual((hold["tracking_error_final_m"], hold["reference_below_free_floor"]), (0.0, False))

    def test_criteria_pools_split_by_anchor_tilt_and_saturation(self):
        def row(family, group, anchor, m8, m16, saturated=0, final=None):
            return {"family": family, "group": group, "set": group[0], "anchor_tilt_deg": anchor, "max_tilt_h8": m8, "max_tilt_h16": m16,
                    "max_tilt_h24": m16, "rise_h8": m8 - anchor, "rise_h16": m16 - anchor, "rise_h24": m16 - anchor, "saturated_steps": saturated,
                    "tracking_error_final_m": final, "tracking_error_max_m": final, "wm_action_absmax": 19.0 if saturated else 2.0,
                    "reference_below_free_floor": None if final is None else False, "class0_frames": 0, "contact_steps": 1,
                    "pinch_disp_max_m": 0.02, "pinch_disp_final_m": 0.01, "pinch_min_z_m": 0.1}
        rows = [row("random", "B/fall", 15.0, 20.0, 35.0, final=0.001),               # persistence positive (anchor >= 10)
                row("random", "V/free", 2.0, 31.0, 32.0, saturated=3, final=0.06),    # new positive, saturated, tracking lost
                row("random", "V/free", 1.0, 2.0, 3.0, final=0.004),                  # S1 negative
                row("random", "B/negative", 0.0, 4.0, 12.0, final=0.001),             # S2: h8 negative, h16 positive
                row("recorded", "B/fall", 15.0, 20.0, 35.0)]
        pools = plan.criteria_pools(rows)
        self.assertEqual(set(pools), {"random", "recorded"})
        s1 = pools["random"]["S1_h16"]
        self.assertEqual((s1["positives"]["all"], s1["positives"]["anchor_tilt_lt10"], s1["positives"]["anchor_tilt_ge10"]), (2, 1, 1))
        self.assertEqual((s1["positives"]["no_saturated_action"], s1["positives"]["anchor_tilt_lt10_no_saturated_action"]), (1, 0))
        self.assertEqual(s1["positives"]["by_group"], {"B/fall": 1, "V/free": 1})
        self.assertEqual(s1["negatives"]["all"], 1)
        self.assertEqual((pools["random"]["S2_h8"]["positives"]["all"], pools["random"]["S2_h8"]["negatives"]["all"]), (2, 2))
        self.assertEqual((pools["random"]["S2_h16"]["positives"]["all"], pools["random"]["S2_h16"]["positives"]["anchor_tilt_lt10"]), (3, 2))
        agg = plan.aggregate_outcomes(rows)
        b = agg["overall"]["random"]
        self.assertEqual((b["anchor_tilt_ge10"], b["anchor_tilt_lt10"]["rollouts"], b["anchor_tilt_lt10"]["over30_h16"]), (1, 3, 1))
        self.assertEqual((b["rise10_h16"], b["below5_h16"], b["with_saturated_action"]), (3, 1, 1))
        self.assertEqual((b["tracking"]["final_gt2cm"], b["tracking"]["final_gt5cm"], b["tracking"]["final_gt2cm_with_saturated_action"]), (1, 1, 1))
        self.assertEqual((b["wm_action_absmax"]["gt10"], b["wm_action_absmax"]["gt5"]), (1, 1))
        self.assertIsNone(agg["overall"]["recorded"]["tracking"])

    def test_wm_action_normalization_and_aggregation(self):
        actions = np.array([[0.1, 0.2, 0.3, 0.0, 1.0], [-0.1, 0.0, 0.05, 0.0, -1.0]])
        out = plan.normalize_wm_actions(actions, NORMALIZATION)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out[0], (np.array([0.1, 0.2, 0.3, 1.0]) - NORMALIZATION["mean"]) / NORMALIZATION["scale"], rtol=1e-6)
        tilt = np.zeros((25, 2))
        tilt[10:, 1] = 35.0
        unsafe = tilt.max(1) > 30
        labels = np.full(25, 2)
        labels[3] = 0
        pinch = np.zeros((25, 3))
        pinch[:, 0] = np.linspace(0, 0.03, 25)
        peaks = np.zeros((25, 2))
        peaks[5, 0] = 1.0
        o = plan.rollout_outcome(tilt, unsafe, labels, pinch, peaks, np.zeros((25, 2)))
        self.assertEqual((o["first_unsafe_step"], o["class0_frames"], o["contact_steps"]), (10, 1, 1))
        self.assertEqual((o["max_tilt_h8"], o["max_tilt_h16"]), (0.0, 35.0))
        self.assertAlmostEqual(o["pinch_disp_final_m"], 0.03)
        rows = [{**o, "family": "random", "set": "V", "group": "V/free"}, {**o, "family": "hold", "set": "B", "group": "B/fall", "max_tilt_h24": 3.0}]
        agg = plan.aggregate_outcomes(rows)
        self.assertEqual(agg["overall"]["random"]["over30_h16"], 1)
        self.assertEqual(agg["overall"]["random"]["over30_h8"], 0)
        self.assertEqual(agg["by_group"]["B/fall"]["hold"]["below5_h24"], 1)
        self.assertIsNone(agg["by_set"]["V"]["hold"])
        order = plan.round_robin([{"g": g, "i": i} for i, g in enumerate("aaabbc")], lambda x: x["g"], seed=0)
        self.assertEqual(len(order), 6)
        self.assertEqual(sorted(x["g"] for x in order[:3]), ["a", "b", "c"])


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
    pixels[:, 0, 0, 1] = np.arange(n) % 256                        # frame index stamped into one pixel
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.attrs["complete"] = True
        f.attrs["source_npz_sha256"] = npz_sha


@unittest.skipUnless(HAVE_H5, "stage-1 CLI needs h5py (training venv)")
class PlanCliTests(unittest.TestCase):
    def build_fixture(self, root):
        raw = root / "raw"
        write_json(raw / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        val_rows = []
        for eid in (3, 4, 5, 6):
            folder = raw / f"episode_{eid:04d}"
            write_raw_episode(folder, exploration_arrays())
            npz_sha = sha256(folder / "trajectory.npz")
            write_h5(root / "full" / "episodes" / f"episode_{eid:04d}.h5", 270, npz_sha, eid)
            val_rows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "source_npz_sha256": npz_sha,
                             "source_metadata_sha256": sha256(folder / "metadata.json")})
        write_json(root / "full" / "index.json", {"status": "complete", "source_root": str(raw), "source_manifest_sha256": sha256(raw / "manifest.json"),
                                                  "normalization": NORMALIZATION, "source_action_columns": [0, 1, 2, 4],
                                                  "episodes": {"train": [], "val": val_rows, "test": [{"id": 99, "path": "episodes/episode_0099.h5"}]}})
        write_json(root / "classifier" / "run.json", {"dev_episode_ids": [3], "report_episode_ids": [4, 5, 6]})
        bench = root / "bench_raw"
        write_json(bench / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        episodes, bench_rows, strata = [], [], {}
        kinds = [("toward_camera|S1_flat", fall_arrays(), 70), ("away_camera|S2_lean", fall_arrays(), 70),
                 ("negative|N3_near", negative_arrays(), None), ("negative|N1_touch", negative_arrays(), None)]
        for eid, (stratum, arrays, onset) in enumerate(kinds):
            folder = bench / f"episode_{eid:04d}"
            write_raw_episode(folder, arrays)
            npz_sha = sha256(folder / "trajectory.npz")
            write_h5(root / "bench_cache" / "episodes" / f"episode_{eid:04d}.h5", 100, npz_sha, 50 + eid)
            episodes.append({"dataset": 0, "episode_id": eid, "path": folder.name, "stratum": stratum, "frames": 100, "onset_frame": onset})
            strata.setdefault(stratum, {"dev": [], "test": []})["test"].append({"dataset": 0, "episode_id": eid})
            bench_rows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "source_dataset": 0, "source_id": eid,
                               "source_npz_sha256": npz_sha, "source_metadata_sha256": sha256(folder / "metadata.json")})
        split = root / "split" / "benchmark_split.json"
        write_json(split, {"format": "danger_bench_split_v1", "datasets": [{"index": 0, "path": str(bench), "manifest_sha256": sha256(bench / "manifest.json")}],
                           "strata": strata, "episodes": episodes})
        write_json(root / "bench_cache" / "index.json", {"status": "complete", "eval_cache": True, "split_sha256": sha256(split),
                                                         "normalization": NORMALIZATION, "bench_split": {"bench_dev": [], "bench_test": [0, 1, 2, 3]},
                                                         "episodes": {"val": bench_rows}})
        return ["--full-cache", str(root / "full"), "--bench-cache", str(root / "bench_cache"), "--classifier-run", str(root / "classifier" / "run.json"),
                "--bench-split", str(split), "--b-falls", "2", "--b-negatives", "2", "--seed", "5", "--v-scale", "0.075"]

    def test_plan_end_to_end_on_synthetic_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.build_fixture(root)
            out = root / "plan_out"
            summary = plan.main(args + ["--output", str(out)])
            for name in ("run.json", "anchors.json", "cache_pixels.npz", "summary.json", "REPORT_ZH.md"):
                self.assertTrue((out / name).is_file(), name)
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            anchors = doc["anchors"]
            self.assertEqual(summary["counts"]["B"], {"B/fall": 2, "B/negative": 2})
            self.assertEqual(summary["counts"]["V"], {"near_guard": 3, "carrying": 3, "near_goal": 3, "free": 2})   # quotas 40/40/40/30 x 0.075
            self.assertEqual({x["episode_id"] for x in anchors if x["set"] == "V"}, {4, 5, 6})                      # dev episode 3 never used
            self.assertEqual(doc["normalization"]["scale"], NORMALIZATION["scale"])
            with np.load(out / "cache_pixels.npz") as npz:
                for x in anchors:
                    block = npz[f"a{x['id']:04d}"]
                    self.assertEqual(block.shape, (11, 224, 224, 3))
                    value = x["source"]["cache_episode_id"] if x["set"] == "V" else 50 + x["source"]["episode_id"]
                    self.assertTrue(np.all(block[:, 5, 5, 0] == value))
                    np.testing.assert_array_equal(block[:, 0, 0, 1], np.arange(x["frame"] - 10, x["frame"] + 1))
                    self.assertEqual(x["stratum_by_rule"], x["stratum"]) if x["set"] == "V" else None
                    if x["group"] == "B/fall":
                        self.assertEqual(x["frame"], 70 - x["lead_k"])
            self.assertEqual(summary["files_sha256"]["anchors.json"], sha256(out / "anchors.json"))
            with self.assertRaises(FileExistsError):
                plan.main(args + ["--output", str(out)])
            # a raw file that no longer matches the cache is refused (FAILED.json, no summary)
            with (root / "raw" / "episode_0005" / "metadata.json").open("a", encoding="utf-8") as f:
                f.write(" ")
            bad = root / "plan_bad"
            with self.assertRaises(ValueError):
                plan.main(args + ["--output", str(bad)])
            self.assertTrue((bad / "FAILED.json").is_file())
            self.assertFalse((bad / "summary.json").exists())


def write_scene_episode(folder, env, steps, seed):
    """A real-scene exploration-format episode (PNG images, trajectory.npz, metadata.json) from random valid actions."""
    from collect_exploration_trajs import ExplorationRecorder
    from collect_grasp_trajs import episode_metadata
    rng = np.random.default_rng(seed)
    env.reset(seed=0)
    env.set_hover_goal([0.43, 0.0, 0.17])
    meta = episode_metadata(env, {"kind": "unit_test_fixture"}, seed, "front_pixels", 0.0, 0.0, steps)
    meta.update(hover_goal_xyz=env.hover_goal.tolist())
    recorder = ExplorationRecorder(folder, env, meta, "front_pixels")
    for _ in range(steps):
        action = rng.uniform(-0.3, 0.3, 5)
        action[3], action[4] = 0.0, float(rng.choice([-1.0, 1.0]))
        env.last_transition = None
        env.step(action)
        recorder.append(env.last_transition, "FREE")
    recorder.finish({"phase": "DONE"})


@unittest.skipUnless(HAVE_SIM, "stage 2 needs the simulation venv")
class SimulationTests(unittest.TestCase):
    def make_plan(self, root, frames=(15, 18)):
        env = ExplorationScene()
        self.addCleanup(env.close)
        data = root / "raw"
        data.mkdir()
        write_json(data / "manifest.json", {"execution": {"render_environment": RENDER_ENV}})
        folder = data / "episode_0000"
        write_scene_episode(folder, env, 45, seed=1)
        arrays = plan.load_npz_fields(folder / "trajectory.npz")
        letterbox, _ = crb.import_letterbox()
        from PIL import Image
        source = {"kind": "unit_test", "dataset_dir": str(data), "dataset_manifest_sha256": sha256(data / "manifest.json"), "episode_id": 0,
                  "global_id": None, "episode_dir": str(folder), "npz_sha256": sha256(folder / "trajectory.npz"),
                  "metadata_sha256": sha256(folder / "metadata.json"), "frames": 46, "cache_episode_id": 0}
        records, pixels = [], {}
        for i, frame in enumerate(frames):
            anchor = {"set": "V", "group": "V/free", "stratum": "free", "episode_id": 0, "frame": frame, "lead_k": None, "k_draws": None}
            records.append(plan.make_anchor_record(i, anchor, 1000 + i, source, arrays))
            pixels[f"a{i:04d}"] = np.stack([letterbox(Image.open(folder / "images" / f"{f:06d}.png"), 224) for f in range(frame - 10, frame + 1)])
        out = root / "plan"
        out.mkdir()
        write_json(out / "anchors.json", {"format": plan.PLAN_FORMAT, "normalization": NORMALIZATION, "prefix": 10, "horizon": 24,
                                          "families": list(plan.FAMILIES), "anchors": records})
        np.savez_compressed(out / "cache_pixels.npz", **pixels)
        write_json(out / "summary.json", {"format": plan.PLAN_FORMAT, "files_sha256": {"anchors.json": sha256(out / "anchors.json"),
                                                                                       "cache_pixels.npz": sha256(out / "cache_pixels.npz")}})
        return out, folder

    def test_letterbox_is_the_cache_function(self):
        letterbox, how = crb.import_letterbox()
        self.assertEqual(letterbox.__module__, "wm_training.prepare_data")
        self.assertIn(how, ("direct_import", "h5py_placeholder_during_import"))
        import sys
        self.assertNotIn("h5py", sys.modules) if how != "direct_import" else None
        from PIL import Image
        image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8))
        out = letterbox(image, 224)
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertTrue(np.all(out[:28] == [124, 116, 104]))              # ImageNet-mean padding band above the 224x168 image

    def test_select_anchors(self):
        anchors = [{"id": i, "group": g} for i, g in enumerate("aaaabbc")]
        self.assertEqual([x["id"] for x in crb.select_anchors(anchors)], list(range(7)))
        self.assertEqual([x["id"] for x in crb.select_anchors(anchors, [6, 0])], [0, 6])
        self.assertEqual([x["id"] for x in crb.select_anchors(anchors, keep_fraction=0.5)], [0, 1, 4, 6])
        with self.assertRaises(ValueError):
            crb.select_anchors(anchors, [9])

    def test_end_to_end_two_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_dir, folder = self.make_plan(root)
            out, report = root / "branches", root / "report"
            args = ["--plan", str(plan_dir), "--output", str(out), "--report", str(report), "--workers", "2"]
            summary = crb.main(args)
            s = summary["sanity"]
            self.assertEqual((s["anchors"], s["prefix_replay_exact_anchors"], s["recorded_exact_anchors"]), (2, 2, 2))
            self.assertEqual((s["anchor_frame_equals_cache_fraction"], s["anchor_frame_cache_max_abs_diff"]), (1.0, 0))
            self.assertEqual(s["prefix_frames_equal_cache"], "22/22")
            self.assertEqual((s["anchor_raw_png_identical_fraction"], s["anchor_images_equal_across_rollouts_anchors"]), (1.0, 2))
            self.assertEqual((s["prefix_frames_raw_png_identical"], s["prefix_frames_raw_png_max_abs_diff"]), ("22/22", 0))
            for name in ("run.json", "summary.json", "REPORT_ZH.md"):
                self.assertTrue((report / name).is_file())
            mem = summary["memory"]
            self.assertEqual(len(mem["worker_peak_rss_kb"]), 2)                         # two worker processes, measured inside each
            self.assertTrue(all(kb > 100_000 for kb in mem["worker_peak_rss_kb"].values()))
            self.assertEqual(mem["sum_of_process_peaks_kb"], mem["main_peak_rss_kb"] + sum(mem["worker_peak_rss_kb"].values()))
            sampled = mem["sampled"]
            self.assertTrue(sampled["active"] and sampled["samples"] >= 2)
            self.assertGreaterEqual(sampled["max_processes"], 3)                          # main + 2 workers were seen
            self.assertTrue(0 < sampled["peak_pss_sum_kb"] <= sampled["peak_rss_sum_kb"])
            self.assertEqual(set(summary["criteria_pools"]), {"recorded", "hold", "random", "guard"})
            pools = summary["criteria_pools"]["random"]["S2_h8"]
            self.assertEqual(pools["positives"]["all"] + pools["negatives"]["all"], 8)       # 2 anchors x 4 random rollouts
            self.assertIsNone(summary["outcomes"]["overall"]["recorded"]["tracking"])
            self.assertEqual(summary["outcomes"]["overall"]["random"]["tracking"]["rollouts"], 8)
            report_text = (report / "REPORT_ZH.md").read_text(encoding="utf-8")
            for heading in ("## 3. S1 / S2", "## 4.", "## 5."):
                self.assertIn(heading, report_text)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["format"]), ("complete", "random_branches_v2"))
            self.assertIn("tracking_error_m", manifest["diagnostic_arrays"])
            mean32, scale32 = (np.asarray(NORMALIZATION[k], np.float32) for k in ("mean", "scale"))
            with np.load(folder / "trajectory.npz") as npz:
                source = {k: np.asarray(npz[k]) for k in ("actions", "guard_tilt_deg", "class_label", "pinch_xyz")}
            for row in manifest["anchors"]:
                a = row["frame"]
                self.assertEqual(sha256(out / row["file"]), row["sha256"])
                with np.load(out / row["file"]) as npz:
                    z = {k: np.asarray(npz[k]) for k in npz.files}
                self.assertEqual(z["pixels"].shape, (8, 25, 224, 224, 3))
                self.assertEqual((z["actions"].shape, z["wm_action"].shape, z["physics_state"].shape), ((8, 24, 5), (8, 24, 4), (8, 25, 343)))
                self.assertEqual(z["prefix_pixels"].shape, (11, 224, 224, 3))
                self.assertEqual(z["families"].tolist(), list(plan.FAMILIES))
                np.testing.assert_array_equal(z["prefix_pixels"][10], z["pixels"][0, 0])
                for r in range(8):
                    np.testing.assert_array_equal(z["pixels"][r, 0], z["pixels"][0, 0])
                    np.testing.assert_array_equal(z["pinch_xyz"][r, 0], source["pinch_xyz"][a])
                np.testing.assert_array_equal(z["actions"][0], source["actions"][a:a + 24])
                np.testing.assert_array_equal(z["guard_tilt_deg"][0], source["guard_tilt_deg"][a:a + 25])
                np.testing.assert_array_equal(z["class_label"][0], source["class_label"][a:a + 25])
                np.testing.assert_array_equal(z["prefix_actions"], source["actions"][a - 10:a])
                np.testing.assert_array_equal(z["wm_action"], ((z["actions"].astype(np.float32)[..., [0, 1, 2, 4]] - mean32) / scale32).astype(np.float32))
                np.testing.assert_array_equal(z["prefix_wm_action"],
                                              ((z["prefix_actions"].astype(np.float32)[..., [0, 1, 2, 4]] - mean32) / scale32).astype(np.float32))
                self.assertEqual((z["tracking_error_m"].shape, z["action_saturated"].shape), ((8, 24), (8, 24)))
                self.assertTrue(np.isnan(z["tracking_error_m"][0]).all() and np.isfinite(z["tracking_error_m"][1:]).all())
                np.testing.assert_array_equal(z["tracking_error_m"][1:], np.linalg.norm(z["pinch_xyz"][1:, 1:] - z["reference_xyz"][1:], axis=2))
                np.testing.assert_array_equal(z["action_saturated"], (np.abs(z["actions"][..., :3]) >= 1 - 1e-12).any(-1))
                outcome = row["rollouts"]
                for r in range(1, 8):
                    self.assertEqual(outcome[r]["tracking_error_final_m"], float(z["tracking_error_m"][r, -1]))
                    self.assertEqual(outcome[r]["saturated_steps"], int(z["action_saturated"][r].sum()))
                grip = source["actions"][a][4]
                self.assertTrue(np.all(z["actions"][1:, :, 4] == grip) and np.all(z["actions"][:, :, 3] == 0))
                self.assertTrue(np.all(np.abs(z["actions"]) <= 1))
                hold_drift = np.linalg.norm(z["pinch_xyz"][1] - z["pinch_xyz"][1, 0], axis=1).max()
                self.assertLess(hold_drift, 0.003)
                for r in range(2, 8):
                    ref = z["reference_xyz"][r]
                    self.assertLess(np.linalg.norm(z["pinch_xyz"][r, -1] - ref[-1]), 0.02)
                    step = np.linalg.norm(np.diff(np.vstack([z["pinch_xyz"][r, 0], ref]), axis=0), axis=1)
                    self.assertTrue(np.all(step <= 0.05 * 0.05 + 1e-12))
                self.assertTrue(np.isnan(z["reference_xyz"][0]).all())
            with self.assertRaises(FileExistsError):
                crb.main(args)

    def test_saturation_flag_on_executed_actions(self):
        """A reference 20 cm away drives the PI tracker onto the +-5 cm bound; the executed action is flagged."""
        from bench_restore_check import pinch_xyz
        env = ExplorationScene()
        self.addCleanup(env.close)
        env.reset(seed=0)
        start = pinch_xyz(env)
        tracker = crb.ReferenceTracker(np.tile(start + [0.2, 0.0, 0.0], (6, 1)), grip=-1.0, dt=0.05)
        executed = []
        for _ in range(6):
            env.last_transition = None
            env.step(tracker.action(env))
            executed.append(np.asarray(env.last_transition["action"], float))
        executed = np.asarray(executed)
        self.assertGreaterEqual(executed[0, 0], 1.0 - 1e-12)
        self.assertTrue(plan.saturated_steps(executed).all())
        small = crb.ReferenceTracker(np.tile(pinch_xyz(env), (3, 1)), grip=-1.0, dt=0.05)
        env.last_transition = None
        env.step(small.action(env))
        self.assertFalse(plan.saturated_steps(np.asarray(env.last_transition["action"], float)))

    def test_memory_sampler_sees_child_processes(self):
        import subprocess
        import sys
        import time
        sampler = crb.MemorySampler(interval=0.05)
        if not sampler.active:
            self.skipTest("no /proc smaps_rollup here")
        child = subprocess.Popen([sys.executable, "-c", "import time; b = b'x' * (80 * 1024 * 1024); time.sleep(1.5)"])
        self.addCleanup(child.wait)
        time.sleep(0.5)
        sampler.start()
        time.sleep(0.4)
        result = sampler.stop()
        self.assertGreaterEqual(result["samples"], 2)
        self.assertGreaterEqual(result["max_processes"], 2)                              # this process + the child
        self.assertGreater(result["peak_pss_sum_kb"], 80 * 1024)                          # the child's 80 MB buffer is counted
        self.assertLessEqual(result["peak_pss_sum_kb"], result["peak_rss_sum_kb"])
        self.assertGreater(result["min_mem_available_kb"], 0)
        self.assertEqual(sampler.stop(), result)                                          # stopping twice is harmless

    def test_memory_and_worst_tracking_summaries(self):
        import os
        rows = [{"id": 1, "group": "B/fall", "worker_pid": 11, "worker_peak_rss_kb": 500,
                 "rollouts": [{"rollout": 0, "family": "recorded", "tracking_error_final_m": None},
                              {"rollout": 2, "family": "random", "tracking_error_final_m": 0.06, "tracking_error_max_m": 0.07, "saturated_steps": 4,
                               "contact_steps": 9, "anchor_tilt_deg": 12.0, "max_tilt_h24": 80.0}]},
                {"id": 2, "group": "V/free", "worker_pid": 11, "worker_peak_rss_kb": 700,
                 "rollouts": [{"rollout": 3, "family": "random", "tracking_error_final_m": 0.001, "tracking_error_max_m": 0.002, "saturated_steps": 0,
                               "contact_steps": 0, "anchor_tilt_deg": 0.5, "max_tilt_h24": 1.0}]},
                {"id": 3, "group": "V/free", "worker_pid": 12, "worker_peak_rss_kb": 600, "rollouts": []}]
        mem = crb.memory_summary(rows)
        self.assertEqual(mem["worker_peak_rss_kb"], {"11": 700, "12": 600})
        self.assertEqual((mem["worker_peak_rss_kb_max"], mem["main_pid"]), (700, os.getpid()))
        self.assertEqual(mem["sum_of_process_peaks_kb"], mem["main_peak_rss_kb"] + 1300)
        single = crb.memory_summary([{**rows[0], "worker_pid": os.getpid()}])       # workers == 1: the main process is not counted twice
        self.assertEqual((single["worker_peak_rss_kb"], single["sum_of_process_peaks_kb"]), ({}, single["main_peak_rss_kb"]))
        worst = crb.worst_tracking(rows)
        self.assertEqual([(w["anchor"], w["rollout"]) for w in worst], [(1, 2), (2, 3)])
        self.assertAlmostEqual(worst[0]["final_error_cm"], 6.0)

    def test_tampered_source_writes_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_dir, folder = self.make_plan(root, frames=(15,))
            with (folder / "metadata.json").open("a", encoding="utf-8") as f:
                f.write(" ")
            out, report = root / "branches", root / "report"
            with self.assertRaises(ValueError):
                crb.main(["--plan", str(plan_dir), "--output", str(out), "--report", str(report), "--workers", "1"])
            self.assertTrue((report / "FAILED.json").is_file())
            self.assertFalse((report / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
