"""Counterfactual branch generation: pure planning checks plus ONE real training event on the MuJoCo scene (WSL).

The smoke test reads one fall event of the TRAINING split (data/danger_train_split_001 -> danger_train_*;
never a benchmark dataset) and is skipped when that data is absent. Generated episodes are test output in a
temporary directory, never demonstration data.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import collect_counterfactual_branches as ccb
import counterfactual_plan as plan
import counterfactual_split
from exploration import PHASES
from exploration_validation import EXPLORATION_PHASES
from inspect_grasp_dataset import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "danger_train_split_001" / "benchmark_split.json"


def fake_events(*onsets):
    return [{"dataset": 0, "episode_id": i, "path": f"episode_{i:04d}", "episode_dir": f"/x/episode_{i:04d}",
             "stratum": "toward_camera|S1_flat", "part": "test", "onset_frame": onset, "frames": onset + 40}
            for i, onset in enumerate(onsets)]


class PlanTests(unittest.TestCase):
    def test_phases_registered_in_controller_set_and_validator_whitelist(self):
        for name in plan.CF_PHASES:
            self.assertLessEqual(len(name), 16)
            self.assertIn(name, PHASES)
            self.assertIn(name, EXPLORATION_PHASES)
        self.assertEqual(plan.parse_branches("push_back,hold"), ["hold", "push_back"])
        with self.assertRaises(ValueError):
            plan.parse_branches("hold,hold")
        with self.assertRaises(ValueError):
            plan.parse_branches("stop")

    def test_plan_jobs_skips_short_leads_and_is_deterministic(self):
        events = fake_events(30, 5)                         # onset 5: lead 2 -> restore 3 ok, lead 4 -> 1 < 3 skipped
        jobs, skipped = plan.plan_jobs(events, [2, 4], ["hold", "push_back"], 20, 10, 20, 0.04)
        self.assertEqual(len(jobs), 2 * 2 * 2 - 2)
        self.assertEqual(skipped, [{"dataset": 0, "episode_id": 1, "lead": 4, "onset_frame": 5, "reason": "restore frame 1 < 3"}])
        self.assertEqual([j["id"] for j in jobs], list(range(6)))
        self.assertEqual([(j["lead"], j["branch"]) for j in jobs[:4]], [(2, "hold"), (2, "push_back"), (4, "hold"), (4, "push_back")])
        self.assertEqual(jobs[4]["prefix"], 3)                                          # clipped to the restore frame
        self.assertEqual(jobs[0]["prefix"], 10)
        self.assertEqual(plan.jobs_digest(jobs), plan.jobs_digest(plan.plan_jobs(events, [2, 4], ["hold", "push_back"], 20, 10, 20, 0.04)[0]))
        self.assertNotEqual(plan.jobs_digest(jobs), plan.jobs_digest(plan.plan_jobs(events, [2, 4], ["hold"], 20, 10, 20, 0.04)[0]))
        with self.assertRaises(ValueError):
            plan.plan_jobs(fake_events(4), [2, 4], ["hold"], 20, 10, 20, 0.04)          # every pair skipped
        with self.assertRaises(ValueError):
            plan.plan_jobs(events, [2], ["hold"], 20, 2, 20, 0.04)                      # prefix below MIN_PREFIX
        self.assertIsNone(plan.restore_frame_for(4, 2))
        self.assertEqual(plan.restore_frame_for(5, 2), 3)

    def test_push_back_direction_against_lean_with_onset_fallback(self):
        d, source = plan.push_back_direction([0.3, 0.0], [0.5, 0.0])
        np.testing.assert_allclose(d, [-1.0, 0.0])
        self.assertEqual(source, "restore")
        d, source = plan.push_back_direction([0.0, 0.01], [0.0, -0.5])            # upright at the restore frame
        np.testing.assert_allclose(d, [0.0, 1.0])
        self.assertEqual(source, "onset")
        with self.assertRaises(ValueError):
            plan.push_back_direction([0.0, 0.0], [0.0, 0.0])

    def test_record_validation(self):
        record = {"source_dataset": 0, "source_dataset_path": "/x", "source_episode_id": 1, "source_path": "episode_0001",
                  "source_manifest_sha256": "a" * 64, "source_npz_sha256": "b" * 64, "source_stratum": "toward_camera|S1_flat",
                  "source_part": "test", "source_frames": 100, "onset_frame": 50, "lead": 4, "restore_frame": 46, "prefix_frames": 10,
                  "branch": "retreat", "phase": "CF_RETREAT", "hold_steps": 20, "move_steps": 5, "move_speed_mps": 0.04, "grip_policy": "keep",
                  "active_steps": 5, "grip_command": 1.0, "direction_xy": [0.6, 0.8], "lean_source": "guard_com", "worst_guard_index": 1,
                  "replay_max_qpos_diff": 0.0, "replay_exact": True, "max_tilt_deg": 31.0, "reached_unsafe": True, "first_unsafe_step": 7,
                  "final_tilt_deg": [31.0, 0.0], "pinch_drift_max_mm": 1.5, "time_rebased_to_zero": True}
        plan.validate_branch_record(record)
        self.assertEqual(plan.expected_phases(record), ["CF_RETREAT"] * 5 + ["CF_HOLD"] * 15)
        bad = [{"restore_frame": 45}, {"active_steps": 20}, {"first_unsafe_step": None}, {"direction_xy": [1.0, 1.0]},
               {"branch": "hold", "phase": "CF_HOLD"}, {"prefix_frames": 2}, {"time_rebased_to_zero": False}]
        for change in bad:
            with self.assertRaises(ValueError, msg=change):
                plan.validate_branch_record({**record, **change})
        with self.assertRaises(ValueError):
            plan.validate_branch_record(record, {k: record[k] for k in plan.PLAN_KEYS} | {"lead": 5})
        hold = {**record, "branch": "hold", "phase": "CF_HOLD", "active_steps": 20, "direction_xy": None, "lean_source": None}
        plan.validate_branch_record(hold)
        cont = {**record, "branch": "continue", "phase": "CF_CONTINUE", "active_steps": 20, "direction_xy": None, "lean_source": None}
        plan.validate_branch_record(cont)                               # 100 - 1 - 46 = 53 remaining actions -> capped at 20
        with self.assertRaises(ValueError):
            plan.validate_branch_record({**cont, "source_frames": 60})   # only 13 remaining actions


@unittest.skipUnless(SPLIT.is_file(), "training split not available")
class RealEventSmokeTests(unittest.TestCase):
    def test_two_leads_two_branches_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "cf_smoke"
            args = ["--split", str(SPLIT), "--output", str(out), "--part", "test", "--events", "1", "--order", "listed",
                    "--leads", "4,8", "--branches", "hold,push_back", "--hold-steps", "20", "--prefix", "5", "--move-steps", "5",
                    "--workers", "1", "--seed", "0"]
            report = ccb.main(args + ["--plan-only"])
            self.assertFalse(out.exists())
            self.assertEqual((report["jobs"], report["frames"]), (4, 4 * 26))
            manifest = ccb.main(args)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["episodes"]), 4)
            self.assertEqual(manifest["interaction_revision"], 4)
            record = manifest["counterfactual_branches"]
            self.assertEqual((record["format"], record["leads"], record["branches"]), ("counterfactual_branches_v1", [4, 8], ["hold", "push_back"]))
            self.assertEqual(len(record["selection"]["events"]), 1)
            self.assertTrue(all("danger_train" in d["path"] for d in record["source_datasets"]))
            jobs = json.loads((out / "branch_jobs.json").read_text(encoding="utf-8"))
            self.assertEqual((jobs["sha256"], len(jobs["jobs"])), (record["sha256"], 4))
            self.assertEqual(manifest["summary"]["inexact_replays"], 0)
            # The written dataset passes the exploration validator (format, label consistency, phases, images).
            check = validate_dataset(out)
            self.assertTrue(check["valid"], check["errors"])
            self.assertEqual(check["num_episodes"], 4)
            event = record["selection"]["events"][0]
            source_dir = Path(record["source_datasets"][event["dataset"]]["path"]) / f"episode_{event['episode_id']:04d}"
            with np.load(source_dir / "trajectory.npz", allow_pickle=False) as npz:
                source = {k: np.asarray(npz[k]) for k in ("qpos", "qvel", "action_phase", "time", "unsafe")}
            source_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
            for row in manifest["episodes"]:
                episode = out / row["path"]
                meta = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
                cf = meta["outcome"]["counterfactual"]
                self.assertEqual(meta["controller"]["type"], "counterfactual_branch_simulator_truth")
                self.assertEqual(meta["controller"]["counterfactual"], {k: cf[k] for k in plan.PLAN_KEYS})
                self.assertEqual(ccb.scene_keys(meta), ccb.scene_keys(source_meta))
                self.assertEqual((meta["num_states"], meta["num_actions"]), (26, 25))
                self.assertEqual(cf["restore_frame"], event["onset_frame"] - cf["lead"])
                self.assertEqual(cf["prefix_frames"], 5)
                self.assertTrue(cf["replay_exact"])
                self.assertEqual(cf["replay_max_qpos_diff"], 0.0)
                pngs = sorted((episode / "images").glob("*.png"))
                self.assertEqual(len(pngs), 26)
                with np.load(episode / "trajectory.npz", allow_pickle=False) as npz:
                    arrays = {k: np.asarray(npz[k]) for k in npz.files}
                phases = arrays["action_phase"].tolist()
                start = cf["restore_frame"] - 5
                self.assertEqual(phases[:5], source["action_phase"][start:start + 5].tolist())      # recorded prefix keeps its labels
                if row["branch"] == "hold":
                    self.assertEqual(phases[5:], ["CF_HOLD"] * 20)
                    self.assertIsNone(cf["direction_xy"])
                else:
                    self.assertEqual(phases[5:], ["CF_PUSHBACK"] * 5 + ["CF_HOLD"] * 15)
                    self.assertAlmostEqual(np.linalg.norm(cf["direction_xy"]), 1.0, places=9)
                    self.assertIn(cf["lean_source"], ("restore", "onset"))
                self.assertTrue(np.all(np.abs(arrays["actions"]) <= 1) and np.all(arrays["actions"][:, 3] == 0))
                self.assertEqual(arrays["time"][0], 0.0)
                self.assertAlmostEqual(arrays["time"][-1], 25 * 0.05, places=9)
                # Branch point state = the recorded restore frame, bit for bit; prefix frames are the recorded ones.
                np.testing.assert_array_equal(arrays["qpos"][5], source["qpos"][cf["restore_frame"]])
                np.testing.assert_array_equal(arrays["qpos"][:6], source["qpos"][start:start + 6])
                self.assertFalse(arrays["unsafe"][:6].any())
                self.assertEqual(cf["reached_unsafe"], bool(arrays["unsafe"][5:].any()))
                self.assertEqual(arrays["actions"][5:, 4].tolist(), [cf["grip_command"]] * 20)
                # The validator rejects a tampered phase pattern.
                from exploration_validation import validate_exploration
                tampered = dict(arrays)
                tampered["action_phase"] = np.array(phases[:5] + ["CF_CONTINUE"] + phases[6:], dtype="U16")
                with self.assertRaises(ValueError):
                    validate_exploration(meta, tampered, meta["control_dt"])
            with self.assertRaises(FileExistsError):
                ccb.main(args)
            # Split helper: every branch goes to the source part (test); strata are measured outcomes.
            split_dir = Path(temp) / "split"
            counterfactual_split.main(["--dataset", str(out), "--output", str(split_dir)])
            split = json.loads((split_dir / "benchmark_split.json").read_text(encoding="utf-8"))
            self.assertEqual(split["format"], "danger_bench_split_v1")
            members = [(e["dataset"], e["episode_id"]) for s in split["strata"].values() for e in s["test"]]
            self.assertEqual(sorted(members), [(0, i) for i in range(4)])
            self.assertFalse(any(s["dev"] for s in split["strata"].values()))
            self.assertEqual(split["datasets"][0]["schedule_sha256"], record["sha256"])
            self.assertEqual({e["branch"] for e in split["episodes"]}, {"hold", "push_back"})
            for name in split["strata"]:
                self.assertTrue(name.startswith("no_event|") or name.split("|")[1] in ("S1_flat", "S2_lean", "S3_just_over", "transient"), name)
            self.assertIn("越过 30°", (split_dir / "REPORT_ZH.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                counterfactual_split.main(["--dataset", str(out), "--output", str(split_dir)])

    def test_benchmark_datasets_are_refused_as_sources(self):
        bench = ROOT / "data" / "danger_bench_split_001" / "benchmark_split.json"
        if not bench.is_file():
            self.skipTest("benchmark split not available")
        with self.assertRaises(ValueError):
            ccb.load_split_events(bench, "test")


if __name__ == "__main__":
    unittest.main()
