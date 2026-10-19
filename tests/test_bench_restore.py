"""Restore-from-frame checks on the real MuJoCo scene (run in WSL with the GPU render environment).

Synthetic mini-episodes written here are test fixtures, never demonstration data. The
smoke test reads ONE real benchmark event from data/danger_bench_001 through the split
file and is skipped when that data is absent.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bench_restore_check as brc
from collect_grasp_trajs import capture_state
from exploration import ExplorationScene

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "danger_bench_split_001" / "benchmark_split.json"
SYNTHETIC_FIELDS = ("time", "qpos", "qvel", "physics_state", "object_com_xyz", "object_quat_wxyz",
                    "pinch_xyz", "guard_tilt_deg", "unsafe")


def write_synthetic_episode(folder, env, steps, rng):
    """Reset, step random valid actions, capture every frame; write metadata.json + trajectory.npz."""
    env.reset(seed=0)
    goal = env.hover_goal.copy()
    states, actions = [capture_state(env)], []
    for _ in range(steps):
        action = rng.uniform(-0.3, 0.3, 5)
        action[3] = 0.0
        action[4] = float(rng.choice([-1.0, 1.0]))
        env.step(action)
        actions.append(np.asarray(env.last_transition["action"], dtype=float).copy())
        states.append(capture_state(env))
    arrays = {k: np.asarray([s[k] for s in states]) for k in SYNTHETIC_FIELDS}
    arrays["unsafe"] = arrays["unsafe"].astype(bool)
    arrays["hover_goal_xyz"] = np.tile(goal, (len(states), 1))
    arrays["actions"] = np.asarray(actions, dtype=float).reshape(-1, 5)
    folder = Path(folder)
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    (folder / "metadata.json").write_text(json.dumps({
        "status": "complete", "hover_goal_xyz": goal.tolist(), "control_dt": env.control_timestep(),
        "physics_state_spec": brc.STATE_SPEC, "num_states": len(states), "image": {"camera": "front_pixels"}}), encoding="utf-8")
    return arrays


class RestoreTests(unittest.TestCase):
    def make_env(self):
        env = ExplorationScene()
        self.addCleanup(env.close)
        return env

    def test_restore_mid_episode_reproduces_state_and_replay(self):
        env = self.make_env()
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp) / "episode_0000"
            arrays = write_synthetic_episode(episode, env, steps=6, rng=np.random.default_rng(0))
            env.reset(seed=0)                       # a second reset; restore resets again internally
            brc.restore(env, episode, 3)
            np.testing.assert_allclose(env.data.qpos, arrays["qpos"][3], atol=1e-9, rtol=0)
            np.testing.assert_allclose(env.data.qvel, arrays["qvel"][3], atol=1e-9, rtol=0)
            self.assertAlmostEqual(env.data.time, arrays["time"][3], places=12)
            safety = env.safety_report()
            np.testing.assert_allclose([g["tilt_deg"] for g in safety["guards"]], arrays["guard_tilt_deg"][3], atol=1e-9, rtol=0)
            self.assertEqual(safety["unsafe"], bool(arrays["unsafe"][3]))
            np.testing.assert_array_equal(env.hover_goal, arrays["hover_goal_xyz"][3])
            # Replaying the recorded actions from the restored frame reproduces the later captured frames.
            for frame in range(3, 6):
                env.step(arrays["actions"][frame])
                np.testing.assert_allclose(env.data.qpos, arrays["qpos"][frame + 1], atol=1e-9, rtol=0)
                np.testing.assert_allclose(env.data.qvel, arrays["qvel"][frame + 1], atol=1e-9, rtol=0)
                com = np.array([env.data.body(f"object_{i}").xipos for i in range(3)])
                np.testing.assert_allclose(com, arrays["object_com_xyz"][frame + 1], atol=1e-9, rtol=0)
            with self.assertRaises(ValueError):
                brc.restore(env, episode, 7)
            with self.assertRaises(ValueError):
                brc.restore_state(env, arrays["physics_state"][0][:-1], arrays["hover_goal_xyz"][0])

    def test_time_advances_one_control_step_after_restore(self):
        env = self.make_env()
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp) / "episode_0000"
            arrays = write_synthetic_episode(episode, env, steps=4, rng=np.random.default_rng(1))
            brc.restore(env, episode, 2)
            before = float(env.data.time)
            self.assertAlmostEqual(before, arrays["time"][2], places=12)
            result = env.step(brc.hold_action(env))
            self.assertEqual(len(result), 5)
            self.assertAlmostEqual(env.data.time - before, 0.05, places=9)
            env.task_report()                       # the tracker accepted the restored, forward-moving clock
            self.assertGreater(env.data.time, 0)

    def test_anchored_hold_keeps_pinch_within_2mm_and_zero_action_drifts(self):
        env = self.make_env()
        env.reset(seed=0)
        zero = brc.hold_action(env)
        np.testing.assert_array_equal(zero, np.zeros(5))
        self.assertEqual(zero[3], 0)
        start = brc.pinch_xyz(env)
        hold = brc.AnchoredHold(start, grip=0.0, dt=env.control_timestep())
        drift = []
        for _ in range(10):
            env.step(hold.action(env))
            drift.append(np.linalg.norm(brc.pinch_xyz(env) - start))
        self.assertLessEqual(max(drift), 0.002, [round(1000 * d, 2) for d in drift])
        # Measured fact behind the design: the open-loop zero action re-anchors on the sagging pinch.
        env.reset(seed=0)
        start = brc.pinch_xyz(env)
        for _ in range(10):
            env.step(zero)
        self.assertGreater(np.linalg.norm(brc.pinch_xyz(env) - start), 0.005)

    def test_rollout_records_every_executed_action(self):
        env = self.make_env()
        env.reset(seed=0)
        anchor = brc.pinch_xyz(env)
        controller = brc.AnchoredHold(anchor, grip=-1.0, velocity_xy=[-0.04, 0.0], move_steps=3, dt=env.control_timestep())
        result = brc.rollout(env, controller, 5)
        self.assertEqual(result["actions"].shape, (5, 5))
        self.assertEqual(result["tilt"].shape, (6, 2))
        self.assertTrue(np.all(np.abs(result["actions"]) <= 1) and np.all(result["actions"][:, 3] == 0))
        self.assertTrue(np.all(result["actions"][:, 4] == -1))
        self.assertLess(brc.pinch_xyz(env)[0], anchor[0])            # moved toward -x
        self.assertFalse(result["reached_unsafe"])


class PureLogicTests(unittest.TestCase):
    def test_parse_leads(self):
        self.assertEqual(brc.parse_leads("1,2,...,20"), list(range(1, 21)))
        self.assertEqual(brc.parse_leads("1-5"), [1, 2, 3, 4, 5])
        self.assertEqual(brc.parse_leads("5,1,3"), [1, 3, 5])
        with self.assertRaises(ValueError):
            brc.parse_leads("0,1")

    def test_select_events_round_robin_and_listed(self):
        events = [{"dataset": 0, "episode_id": i, "stratum": s, "onset_frame": 50, "frames": 100}
                  for i, s in enumerate(["a|S1_flat"] * 3 + ["b|S2_lean"] * 3 + ["c|transient"] * 2 + ["negative|N1_touch"])]
        chosen = brc.select_events(events, 4, seed=1)
        strata = [e["stratum"] for e in chosen]
        self.assertEqual(sorted(strata), ["a|S1_flat", "a|S1_flat", "b|S2_lean", "b|S2_lean"])   # transient/negative excluded
        self.assertTrue(all(strata[i] != strata[i + 1] for i in range(3)))                    # round-robin over strata
        self.assertEqual(len({e["episode_id"] for e in chosen}), 4)
        self.assertEqual(brc.select_events(events, 4, seed=1), chosen)            # deterministic
        listed = brc.select_events(events, 2, seed=0, strata=["c|"], order="listed")
        self.assertEqual([e["episode_id"] for e in listed], [6, 7])
        with self.assertRaises(ValueError):
            brc.select_events(events, 1, seed=0, strata=["zzz"])
        self.assertEqual(len(brc.select_events(events, 99, seed=0)), 6)

    def test_retreat_direction(self):
        np.testing.assert_allclose(brc.retreat_direction([0.5, 0.0, 0.1], [0.425, 0.0, 0.05]), [1.0, 0.0])
        upright = [1.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(brc.retreat_direction([0.425, 0.0, 0.2], [0.425, 0.0, 0.05], upright), [-1.0, 0.0])


@unittest.skipUnless(SPLIT.is_file(), "danger benchmark split not available")
class RealEventSmokeTests(unittest.TestCase):
    def test_no_return_search_on_first_listed_event(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "no_return"
            summary = brc.main(["--split", str(SPLIT), "--output", str(output), "--mode", "no_return", "--events", "1",
                                "--leads", "1,5", "--hold-steps", "20", "--strata", "toward_camera|S1_flat",
                                "--order", "listed", "--seed", "0"])
            for name in ("summary.json", "records.jsonl", "rollouts.npz", "REPORT_ZH.md"):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((output / "FAILED.json").exists())
            split = brc.load_split(SPLIT)
            first = next(e for e in split["events"] if e["stratum"].startswith("toward_camera|S1_flat"))
            self.assertEqual(summary["selected"]["no_return"][0]["episode_id"], first["episode_id"])
            records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual([e["lead"] for e in record["leads"]], [1, 5])
            for entry in record["leads"]:
                for variant in ("hold", "retreat"):
                    self.assertTrue(np.isfinite(entry[variant]["max_tilt_deg"]))
                    self.assertTrue(np.isfinite(entry[variant]["final_tilt_deg"]).all())
                    self.assertIn(entry[variant]["reached_unsafe"], (True, False))
            with np.load(output / "rollouts.npz", allow_pickle=False) as npz:
                self.assertEqual(npz["actions"].shape, (4, 20, 5))
                self.assertEqual(npz["tilt"].shape, (4, 21, 2))
                self.assertTrue(np.isfinite(npz["tilt"]).all())
                self.assertEqual(sorted(set(npz["variant"].tolist())), ["hold", "retreat"])
            self.assertEqual(summary["no_return"]["events"], 1)
            with self.assertRaises(FileExistsError):
                brc.main(["--split", str(SPLIT), "--output", str(output), "--mode", "no_return"])

    def test_failed_json_when_no_event_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bad"
            with self.assertRaises(ValueError):
                brc.main(["--split", str(SPLIT), "--output", str(output), "--mode", "no_return", "--strata", "no_such_stratum"])
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
