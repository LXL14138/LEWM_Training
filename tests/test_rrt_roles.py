"""Stage-2 roles of rrt_branches_plan.py (--role train / dev / confirm) and collect_rrt_branches.py (--role).

Every fixture is synthetic (temporary directory; a synthetic stage-2 plan.json whose sha256 is patched in as the frozen
one) except RealPlanTests, which only READS the frozen runs/rrt_data_plan_005/plan.json and checks that the role settings
parsed from it are the plan's numbers. No dataset, cache or benchmark episode is read.
* RoleLogicTests / RealPlanTests: numpy only (any venv).
* RolePlanCliTests: need h5py (training venv).
* RoleCollectorTests: need MuJoCo + the render environment (simulation venv); real-scene collections of a dev plan and of a
  role-less plan without / with --role train (default-mode key sets, bit-identical arrays).
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import rrt_branches_plan as rp
from tests import test_rrt_branches as trb
from wm_training.danger_strata import sha256

REPO = Path(__file__).resolve().parents[1]
FROZEN_PLAN = REPO / "runs" / "rrt_data_plan_005" / "plan.json"
SET_A = ["recorded", "hold", "planner", "planner", "planner", "planner", "guard", "edge_hold"]
SET_B2 = ["recorded", "hold", "push", "push", "push4", "push4", "push_hold", "guard", "push2_hold"]
QUIET = redirect_stdout(io.StringIO())
# key sets of the files written before the roles existed (runs/rrt_branches_plan_pilot_004/run.json, runs/rrt_branches_pilot_003/
# run.json, data/rrt_branches_pilot_003/manifest.json): the default mode must still write exactly these
OLD_PLAN_RUN_KEYS = ["format", "args", "input_sha256", "implementation_sha256", "python", "numpy", "rules", "test_episodes_opened",
                     "evaluation_data_opened", "fitted_on_evaluation_data"]
OLD_PLAN_ARGS = ["output", "full_cache", "bench_cache", "bench_split", "ft_run", "classifier_run", "goal_images", "seed", "quota", "scale",
                 "families", "max_source_episodes", "workers_for_estimate", "families_b", "families_b_fraction", "f_near_radius_m",
                 "recorded_max_tilt_h20", "goal_image_strata", "capacity_only"]
OLD_COLLECT_RUN_KEYS = ["format", "args", "plan", "output", "report", "input_sha256", "source_note", "implementation_sha256", "anchors", "workers",
                        "render_environment", "render_environment_matches_source", "state_spec", "law", "nominal_models", "versions", "estimate",
                        "world_model_loaded", "test_episodes_opened", "evaluation_data_opened"]
OLD_COLLECT_ARGS = ["plan", "output", "report", "workers", "anchor_ids", "keep_fraction", "plan_only"]
OLD_MANIFEST_KEYS = ["format", "status", "plan", "plan_files_sha256", "prefix", "horizon", "image_size", "camera", "law", "nominal_models", "alignment",
                     "prefix_alignment", "wm_action", "arrays", "render_environment", "render_environment_matches_source", "versions", "anchors",
                     "summary"]
try:
    import h5py  # noqa: F401
    HAVE_H5 = True
except ImportError:
    HAVE_H5 = False


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def synthetic_stage2_plan(path, goal_images, m=1, mb=2):
    """A stage-2 plan with the frozen plan's structure, small numbers and the fixture's goal-image path."""
    stoch = SET_B2[2:]
    command = (f"rrt_branches_plan.py --output runs/rrt_branches_plan_001 --seed 3 --quota T=2,F=2,G=2,M=2 --f-near-radius-m 0.06 "
               f"--goal-images {goal_images} --goal-image-strata G --families T={','.join(SET_A)} --families-b T={','.join(SET_B2)} "
               f"--families-b-fraction T=0.5 --families F={','.join(SET_B2)} --families G={','.join(SET_A)} --families M={','.join(SET_A)}")
    fam = {"T": {"A": SET_A, "B2": SET_B2, "share_B2": 0.5}, "F": SET_B2, "G": SET_A, "M": SET_A}
    doc = {"format": "rrt_data_plan_v2", "frozen": True,
           "training_collection": {"dataset": "data/rrt_branches_001", "plan_dir": "runs/rrt_branches_plan_001", "report_dir": "runs/rrt_branches_001",
                                   "seed": 3, "families": fam, "command": command},
           "branch_dev_set": {"dataset": "data/rrt_branches_dev_001", "plan_dir": "runs/rrt_branches_plan_dev_001", "report_dir": "runs/rrt_branches_dev_001",
                              "seed": 5, "anchors": {"T": 2, "F": 2, "G": 2, "M": 2}, "families": fam,
                              "sources": {"wm_full_001 val DEV half": "classifier dev_episode_ids (1 episodes; raw exploration)",
                                          "danger_train bench_dev rows": "wm_train_bench_001 bench_split.bench_dev (1 rows; raw danger_train_001)"}},
           "confirmation_set": {"dataset": "data/random_branches_003", "plan_dir": "runs/random_branches_plan_003", "report_dir": "runs/random_branches_003",
                                "seed": 7, "quotas": {"val_report": {"T": "all available, at most 5", "F": "all available, at most 5", "G": 2, "M": 2},
                                                      "bench_test": {"T": 2, "F": 2}},
                                "multiplicity": {"m": m, "mb": mb},
                                "families": {"val_report T": SET_B2[:2] + stoch * m, "val_report F": SET_B2[:2] + stoch * m, "val_report G / M": SET_A,
                                             "bench_test T": fam["T"], "bench_test F": SET_B2[:2] + stoch * mb},
                                "sources": {"val_report": "wm_full_001 val REPORT half (classifier report_episode_ids, 1 episodes)",
                                            "bench_test": "danger benchmark bench_test (split; cache; raw; 2 episodes = 1 fall + 1 negatives)"}}}
    write_json(path, doc)
    return doc


class RoleLogicTests(unittest.TestCase):
    def test_stage2_names(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            data.mkdir()
            self.assertFalse(rp.check_stage2_name(data / "rrt_branches_dev_001", "data/rrt_branches_dev_001")["next_number_rule_applied"])
            with self.assertRaises(ValueError):
                rp.check_stage2_name(data / "rrt_branches_dev_002", "data/rrt_branches_dev_001")       # 001 is still free
            (data / "rrt_branches_dev_001").mkdir()
            self.assertTrue(rp.check_stage2_name(data / "rrt_branches_dev_002", "data/rrt_branches_dev_001")["next_number_rule_applied"])
            for bad in (Path(temp) / "runs" / "rrt_branches_dev_001", data / "rrt_branches_dev_1", data / "rrt_branches_001", data / "x_001"):
                with self.assertRaises(ValueError):
                    rp.check_stage2_name(bad, "data/rrt_branches_dev_001")
            (data / "random_branches_001").mkdir()
            with self.assertRaises(ValueError):                                                           # below the planned number
                rp.check_stage2_name(data / "random_branches_001", "data/random_branches_003")

    def test_frozen_plan_sha_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / "plan.json"
            synthetic_stage2_plan(plan, "gi")
            with self.assertRaises(ValueError):
                rp.load_stage2_plan(plan)
            with patch.object(rp, "FROZEN_STAGE2_PLAN_SHA256", sha256(plan)):
                doc, rec = rp.load_stage2_plan(plan)
            self.assertEqual(rec["sha256"], sha256(plan))

    def test_role_spec_cross_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            doc = synthetic_stage2_plan(Path(temp) / "plan.json", "gi", m=2, mb=3)
        c = rp.role_spec(doc, "confirm")
        self.assertEqual(c["quotas_by_source"], {"val_report": {"T": 5, "F": 5, "G": 2, "M": 2}, "bench_test": {"T": 2, "F": 2, "G": 0, "M": 0}})
        self.assertEqual(len(c["families_by_source"]["val_report"]["T"]["families"]), 2 + 7 * 2)
        self.assertEqual(len(c["families_by_source"]["bench_test"]["F"]["families"]), 2 + 7 * 3)
        self.assertEqual(c["expected"], {"val_report": 1, "bench_test_falls": 1, "bench_test_negatives": 1})
        d = rp.role_spec(doc, "dev")
        self.assertEqual((d["seed"], d["quotas"], d["fractions"], d["expected"]), (5, {"T": 2, "F": 2, "G": 2, "M": 2}, {"T": 0.5},
                                                                                    {"val_dev": 1, "bench_dev": 1}))
        bad = json.loads(json.dumps(doc))
        bad["confirmation_set"]["multiplicity"]["m"] = 3                                                # lists no longer match m
        with self.assertRaises(ValueError):
            rp.role_spec(bad, "confirm")
        bad = json.loads(json.dumps(doc))
        bad["branch_dev_set"]["families"]["F"] = SET_A                                                  # 'same families' broken
        with self.assertRaises(ValueError):
            rp.role_spec(bad, "dev")

    def test_role_families_and_anchor_side(self):
        with tempfile.TemporaryDirectory() as temp:
            doc = synthetic_stage2_plan(Path(temp) / "plan.json", "gi")
        c = rp.role_spec(doc, "confirm")
        self.assertEqual(rp.role_families(c, {"source": "val_report", "stratum": "T"}, 1)[0], "B2x1")
        sets = [rp.role_families(c, {"source": "bench_test", "stratum": "T"}, s)[0] for s in range(400)]
        self.assertTrue(0.4 < sets.count("B") / 400 < 0.6)
        self.assertEqual(sets, [rp.family_set_for(s, "T", {"T": tuple(SET_A)}, {"T": tuple(SET_B2)}, {"T": 0.5})[0] for s in range(400)])

        def anchor(source, folder):
            return {"id": 0, "set": source, "source": {"source": source, "episode_dir": f"/x/{folder}/episode_0001"}}
        self.assertTrue(rp.check_anchor_side("confirm", anchor("bench_test", "danger_bench_002")))
        self.assertTrue(rp.check_anchor_side("dev", anchor("val_dev", "exploration_5000_001")))
        for role, a in (("confirm", anchor("bench_test", "danger_train_001")), ("dev", anchor("bench_dev", "danger_bench_001")),
                        ("train", anchor("val_report", "exploration_5000_001")), ("confirm", anchor("val_report", "danger_bench_001")),
                        ("dev", anchor("full_train", "exploration_5000_001"))):
            with self.assertRaises(ValueError):
                rp.check_anchor_side(role, a)

    def test_sources_argument_never_changes_a_pick(self):
        ep = trb.synthetic_episode()
        episodes = {("full_train", 1): ep, ("bench_train", 2): ep}
        a1, av1 = rp.select_anchors(episodes, {"T": 2, "F": 2, "G": 2, "M": 2}, 11)
        a2, av2 = rp.select_anchors(episodes, {"T": 2, "F": 2, "G": 2, "M": 2}, 11, sources=("full_train", "bench_train", "x"))
        self.assertEqual(a1, a2)
        self.assertEqual(av2["T"]["by_source"]["x"], 0)
        self.assertEqual({k: v for k, v in av2["T"]["by_source"].items() if k != "x"}, av1["T"]["by_source"])

    def test_val_goal_image_links(self):
        with tempfile.TemporaryDirectory() as temp:
            images = [{"id": 1, "set": "val", "split": "val", "cache": "wm_full_001", "variant": 0, "tag": "dev", "source_episode_id": 21,
                       "status": "accepted", "name": "val_ep0021_v0", "achieved_green_xyz": [0.43, 0.0, 0.17]},
                      {"id": 2, "set": "val", "split": "val", "cache": "wm_full_001", "variant": 1, "tag": "dev", "source_episode_id": 21,
                       "status": "accepted", "name": "val_ep0021_v1", "achieved_green_xyz": [0.431, 0.0, 0.17]},
                      {"id": 3, "set": "val", "split": "val", "cache": "wm_full_001", "variant": 0, "tag": "report", "source_episode_id": 22,
                       "status": "accepted", "name": "val_ep0022_v0", "achieved_green_xyz": [0.44, 0.0, 0.17]},
                      {"id": 4, "set": "train", "split": "train", "cache": "wm_full_001", "variant": 0, "tag": "class0", "source_episode_id": 11,
                       "status": "accepted", "name": "train_ep0011_v0", "achieved_green_xyz": [0.44, 0.0, 0.17]}]
            write_json(Path(temp) / "gi" / "manifest.json", {"format": "goal_images_v2", "status": "complete", "images": images})
            halves = {"dev_episode_ids": [21], "report_episode_ids": [22]}
            links, info = rp.load_val_goal_images(Path(temp) / "gi", "dev", halves)
            self.assertEqual(links, {("wm_full_001", 21): {"goal_image_id": 1, "goal_xyz": [0.43, 0.0, 0.17], "goal_image_set": "val",
                                                           "goal_image_name": "val_ep0021_v0", "goal_image_tag": "dev", "goal_image_variant": 0}})
            self.assertEqual(info["skipped"], {"other variant": 1, "tag report": 1, "set train": 1})
            self.assertEqual(list(rp.load_val_goal_images(Path(temp) / "gi", "report", halves)[0]), [("wm_full_001", 22)])
            images[0]["tag"] = "report"                                                                  # tag disagrees with the halves
            write_json(Path(temp) / "gi" / "manifest.json", {"format": "goal_images_v2", "status": "complete", "images": images})
            with self.assertRaises(ValueError):
                rp.load_val_goal_images(Path(temp) / "gi", "dev", halves)


@unittest.skipUnless(FROZEN_PLAN.is_file(), "the frozen stage-2 plan is not in this checkout")
class RealPlanTests(unittest.TestCase):
    def test_role_settings_are_the_frozen_plans_numbers(self):
        doc, rec = rp.load_stage2_plan(FROZEN_PLAN)                                  # sha256 must be the frozen one
        self.assertEqual(rec["sha256"], "102c069d84ec089df4888b30d7b2f7760dba4e48cb8a4c16ece17753239f194f")
        d, c = rp.role_spec(doc, "dev"), rp.role_spec(doc, "confirm")
        self.assertEqual((d["seed"], d["quotas"], d["fractions"], d["f_near_radius_m"], d["goal_image_strata"]),
                         (20261002, {"T": 100, "F": 80, "G": 60, "M": 60}, {"T": 0.47}, 0.045, ("G",)))
        self.assertEqual(d["expected"], {"val_dev": 125, "bench_dev": 480})
        self.assertEqual(d["names"], {"dataset": "data/rrt_branches_dev_001", "plan_dir": "runs/rrt_branches_plan_dev_001", "report_dir": "runs/rrt_branches_dev_001"})
        self.assertEqual((c["seed"], c["multiplicity"]), (20261003, {"m": 3, "mb": 5}))
        self.assertEqual(c["quotas_by_source"], {"val_report": {"T": 70, "F": 100, "G": 60, "M": 60}, "bench_test": {"T": 300, "F": 100, "G": 0, "M": 0}})
        self.assertEqual(c["expected"], {"val_report": 125, "bench_test_falls": 509, "bench_test_negatives": 138})
        fb = c["families_by_source"]
        self.assertEqual([len(fb["val_report"][s]["families"]) for s in "TFGM"], [23, 23, 8, 8])
        self.assertEqual(len(fb["bench_test"]["F"]["families"]), 37)
        self.assertEqual(list(fb["val_report"]["T"]["families"]), doc["confirmation_set"]["families"]["val_report T"])
        self.assertEqual(list(fb["bench_test"]["F"]["families"]), doc["confirmation_set"]["families"]["bench_test F"])
        self.assertEqual((fb["bench_test"]["T"]["fraction"], list(fb["bench_test"]["T"]["B"])), (0.47, doc["training_collection"]["families"]["T"]["B2"]))
        self.assertEqual(c["names"]["dataset"], "data/random_branches_003")
        fr = rp.frozen_training_args(doc)
        self.assertEqual((fr.seed, fr.quota, fr.families_b_fraction), (20261001, "T=2380,F=900,G=1310,M=750", "T=0.47"))


@unittest.skipUnless(HAVE_H5, "the stage-1 CLI needs h5py (training venv)")
class RolePlanCliTests(unittest.TestCase):
    def build(self, root):
        raw = root / "exploration_5000_001"
        trb.write_json(raw / "manifest.json", {"execution": {"render_environment": trb.RENDER_ENV}})
        rows = {"train": [], "val": []}
        for part, ids in (("train", (11, 12)), ("val", (21, 22))):
            for eid in ids:
                folder = raw / f"episode_{eid:04d}"
                trb.write_raw_episode(folder, trb.synthetic_episode())
                npz_sha = sha256(folder / "trajectory.npz")
                if part == "val":                                            # distinct raw files for the val halves
                    with (folder / "metadata.json").open("a", encoding="utf-8") as f:
                        f.write(" " * eid)
                    arrays = dict(np.load(folder / "trajectory.npz"))
                    arrays["time"] = arrays["time"] + 1e-6 * eid
                    (folder / "trajectory.npz").unlink()
                    np.savez_compressed(folder / "trajectory.npz", **arrays)
                    npz_sha = sha256(folder / "trajectory.npz")
                trb.write_h5(root / "wm_full_001" / "episodes" / f"episode_{eid:04d}.h5", 220, npz_sha, eid)
                rows[part].append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "num_states": 220, "source_npz_sha256": npz_sha,
                                   "source_metadata_sha256": sha256(folder / "metadata.json")})
        full_index = {"status": "complete", "source_root": str(raw), "source_manifest_sha256": sha256(raw / "manifest.json"),
                      "normalization": trb.NORMALIZATION, "source_action_columns": [0, 1, 2, 4],
                      "episodes": {**rows, "test": [{"id": 31, "path": "x", "num_states": 1, "source_npz_sha256": "3" * 64}]}}
        trb.write_json(root / "wm_full_001" / "index.json", full_index)
        trb.write_json(root / "classifier" / "run.json", {"dev_episode_ids": [21], "report_episode_ids": [22]})
        # danger_train (B's bench cache): 0, 1 training rows, 2 the bench_dev row
        bench = root / "danger_train_001"
        trb.write_json(bench / "manifest.json", {"execution": {"render_environment": trb.RENDER_ENV}})
        episodes, brows, strata = [], [], {"cell": {"dev": [], "test": []}}
        for eid in (0, 1, 2):
            folder = bench / f"episode_{eid:04d}"
            arrays = trb.synthetic_episode()
            arrays["time"] = arrays["time"] + 1e-5 * (eid + 1)
            trb.write_raw_episode(folder, arrays)
            npz_sha = sha256(folder / "trajectory.npz")
            trb.write_h5(root / "wm_train_bench_001" / "episodes" / f"episode_{eid:04d}.h5", 220, npz_sha, 50 + eid)
            episodes.append({"dataset": 0, "episode_id": eid, "path": f"episode_{eid:04d}", "stratum": "cell", "frames": 220})
            strata["cell"]["test" if eid < 2 else "dev"].append({"dataset": 0, "episode_id": eid})
            brows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "num_states": 220, "source_dataset": 0, "source_id": eid,
                          "source_npz_sha256": npz_sha, "source_metadata_sha256": sha256(folder / "metadata.json")})
        split = root / "danger_train_split_001" / "benchmark_split.json"
        trb.write_json(split, {"format": "danger_bench_split_v1", "datasets": [{"index": 0, "path": str(bench), "manifest_sha256": sha256(bench / "manifest.json")}],
                               "strata": strata, "episodes": episodes})
        trb.write_json(root / "wm_train_bench_001" / "index.json", {"status": "complete", "eval_cache": True, "split_sha256": sha256(split),
                                                                    "source_manifest_sha256": {}, "normalization": trb.NORMALIZATION,
                                                                    "bench_split": {"bench_dev": [2], "bench_test": [0, 1]}, "episodes": {"val": brows}})
        trb.write_json(root / "ft" / "run.json", {"finetune": {"data": {"bench_rows_used_for_training": "bench_test"}},
                                                  "bench_cache": str(root / "wm_train_bench_001"),
                                                  "bench_cache_index_sha256": sha256(root / "wm_train_bench_001" / "index.json"),
                                                  "cache_index_sha256": sha256(root / "wm_full_001" / "index.json")})
        # the real benchmark: 0 fall (test), 1 N1 negative (test), 2 fall (bench_dev), 3 unintended negative (test, outside the pool)
        dbench = root / "danger_bench_001"
        trb.write_json(dbench / "manifest.json", {"execution": {"render_environment": trb.RENDER_ENV}})
        eps, drows = [], []
        dstrata = {"toward|S1_flat": {"dev": [{"dataset": 0, "episode_id": 2}], "test": [{"dataset": 0, "episode_id": 0}]},
                   "negative|N1_touch": {"dev": [], "test": [{"dataset": 0, "episode_id": 1}]},
                   "negative|unintended": {"dev": [], "test": [{"dataset": 0, "episode_id": 3}]}}
        for eid, stratum, onset in ((0, "toward|S1_flat", 150), (1, "negative|N1_touch", None), (2, "toward|S1_flat", 150), (3, "negative|unintended", None)):
            folder = dbench / f"episode_{eid:04d}"
            arrays = trb.synthetic_episode()
            arrays["time"] = arrays["time"] + 1e-4 * (eid + 1)
            trb.write_raw_episode(folder, arrays)
            npz_sha = sha256(folder / "trajectory.npz")
            trb.write_h5(root / "wm_bench_001" / "episodes" / f"episode_{eid:04d}.h5", 220, npz_sha, 90 + eid)
            eps.append({"dataset": 0, "episode_id": eid, "path": f"episode_{eid:04d}", "stratum": stratum, "frames": 220, "onset_frame": onset})
            drows.append({"id": eid, "path": f"episodes/episode_{eid:04d}.h5", "num_states": 220, "source_dataset": 0, "source_id": eid,
                          "source_npz_sha256": npz_sha, "source_metadata_sha256": sha256(folder / "metadata.json")})
        dsplit = root / "danger_bench_split_001" / "benchmark_split.json"
        trb.write_json(dsplit, {"format": "danger_bench_split_v1", "datasets": [{"index": 0, "path": str(dbench), "manifest_sha256": sha256(dbench / "manifest.json")}],
                                "strata": dstrata, "episodes": eps})
        trb.write_json(root / "wm_bench_001" / "index.json", {"status": "complete", "eval_cache": True, "split_sha256": sha256(dsplit),
                                                              "source_manifest_sha256": {}, "normalization": trb.NORMALIZATION,
                                                              "bench_split": {"bench_dev": [2], "bench_test": [0, 1, 3]}, "episodes": {"val": drows}})
        images = [{"id": 10 + i, "set": "val", "split": "val", "cache": "wm_full_001", "variant": v, "tag": tag, "source_episode_id": eid, "status": "accepted",
                   "name": f"val_ep{eid:04d}_v{v}", "achieved_green_xyz": [0.431 + 0.001 * v, 0.001, 0.171]}
                  for i, (eid, tag, v) in enumerate(((21, "dev", 0), (21, "dev", 1), (22, "report", 0), (22, "report", 1)))]
        images.append({"id": 30, "set": "train", "split": "train", "cache": "wm_full_001", "variant": 0, "tag": "class0", "source_episode_id": 11,
                       "status": "accepted", "name": "train_ep0011_v0", "achieved_green_xyz": [0.431, 0.001, 0.171]})
        trb.write_json(root / "goal_images" / "manifest.json", {"format": "goal_images_v2", "status": "complete", "images": images})
        plan = root / "runs" / "rrt_data_plan_005" / "plan.json"
        synthetic_stage2_plan(plan, str(root / "goal_images"))
        common = ["--full-cache", str(root / "wm_full_001"), "--bench-cache", str(root / "wm_train_bench_001"), "--bench-split", str(split),
                  "--ft-run", str(root / "ft" / "run.json"), "--classifier-run", str(root / "classifier" / "run.json"),
                  "--eval-bench-cache", str(root / "wm_bench_001"), "--eval-bench-split", str(dsplit), "--goal-images", str(root / "goal_images")]
        return plan, common

    def run_plan(self, plan, args):
        with patch.object(rp, "FROZEN_STAGE2_PLAN_SHA256", sha256(plan)), redirect_stdout(io.StringIO()):
            return rp.main(args)

    def test_dev_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, common = self.build(root)
            out = root / "runs" / "rrt_branches_plan_dev_001"
            summary = self.run_plan(plan, common + ["--role", "dev", "--stage2-plan", str(plan), "--output", str(out)])
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            self.assertEqual((doc["role"], doc["stage2_plan"]["sha256"], doc["stage2_plan"]["dataset"]), ("dev", sha256(plan), "data/rrt_branches_dev_001"))
            self.assertEqual(summary["scanned"], {"val_dev": 1, "bench_dev": 1})
            self.assertEqual({a["set"] for a in doc["anchors"]}, {"val_dev", "bench_dev"})
            self.assertEqual({(a["set"], a["episode_id"]) for a in doc["anchors"]} - {("val_dev", 21), ("bench_dev", 2)}, set())
            for a in doc["anchors"]:
                rp.check_anchor_side("dev", a)
                if a["stratum"] == "G":
                    self.assertEqual((a["set"], a["goal"]["goal_image_tag"], a["goal"]["goal_image_variant"], a["goal"]["goal_source"]),
                                     ("val_dev", "dev", 0, "goal_image_achieved_com"))
                fams = [s["family"] for s in a["sequences"]]
                self.assertEqual(fams, SET_B2 if (a["stratum"] == "F" or a["family_set"] == "B") else SET_A)
            with np.load(out / "cache_pixels.npz") as npz:
                for a in doc["anchors"]:
                    value = 21 if a["set"] == "val_dev" else 52                                         # h5 pixel marker of the source cache
                    self.assertTrue(np.all(npz[f"a{a['id']:04d}"][:, 5, 5, 0] == value))
            self.assertEqual(summary["guard"]["shared_with_other_pools"], 0)
            run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run["role"], run["selection_data_opened"], run["evaluation_data_opened"]), ("dev", True, False))

    def test_confirm_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, common = self.build(root)
            (root / "runs" / "random_branches_plan_001").mkdir()
            out = root / "runs" / "random_branches_plan_003"
            summary = self.run_plan(plan, common + ["--role", "confirm", "--stage2-plan", str(plan), "--output", str(out)])
            doc = json.loads((out / "anchors.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["role"], "confirm")
            self.assertEqual(summary["scanned"], {"val_report": 1, "bench_test": 2})                      # unintended + bench_dev refused
            self.assertEqual(summary["sources"]["bench_test"]["bench_test_not_in_pool"], {"negative|unintended": 1})
            self.assertEqual({a["episode_id"] for a in doc["anchors"] if a["set"] == "bench_test"} - {0, 1}, set())
            for a in doc["anchors"]:
                rp.check_anchor_side("confirm", a)
                fams = [s["family"] for s in a["sequences"]]
                if a["set"] == "val_report" and a["stratum"] in "TF":
                    self.assertEqual((fams, a["family_set"]), (SET_B2[:2] + SET_B2[2:] * 1, "B2x1"))
                if a["set"] == "bench_test" and a["stratum"] == "F":
                    self.assertEqual((len(fams), a["family_set"]), (16, "B2x2"))
                if a["stratum"] in "GM":
                    self.assertEqual((a["set"], fams), ("val_report", SET_A))
                if a["stratum"] == "G":
                    self.assertEqual((a["goal"]["goal_image_tag"], a["goal"]["goal_image_name"]), ("report", "val_ep0022_v0"))
            by = summary["counts"]["by_group"]
            self.assertTrue(all(by.get(f"bench_test/{s}", 0) == 0 for s in "GM"))
            self.assertLessEqual(by.get("bench_test/T", 0), 2)
            with np.load(out / "cache_pixels.npz") as npz:
                for a in doc["anchors"]:
                    if a["set"] == "bench_test":
                        self.assertTrue(np.all(npz[f"a{a['id']:04d}"][:, 5, 5, 0] == 90 + a["episode_id"]))   # pixels from the benchmark cache

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, common = self.build(root)
            dev = common + ["--role", "dev", "--stage2-plan", str(plan)]
            for extra, name in ((["--seed", "9"], "rrt_branches_plan_dev_001"), (["--quota", "T=1"], "rrt_branches_plan_dev_001"),
                                ([], "rrt_branches_plan_dev_002"), ([], "wrong_name_001")):
                with self.assertRaises(ValueError):
                    self.run_plan(plan, dev + extra + ["--output", str(root / "runs" / name)])
                self.assertFalse((root / "runs" / name).exists())
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                          # not the frozen plan
                rp.main(dev + ["--output", str(root / "runs" / "rrt_branches_plan_dev_001")])
            with self.assertRaises(ValueError):                                                         # dev without a plan
                self.run_plan(plan, common + ["--role", "dev", "--output", str(root / "runs" / "rrt_branches_plan_dev_001")])
            # a start from the wrong side: the val-dev raw episode also listed as a TRAIN row -> hash guard, FAILED.json
            index = json.loads((root / "wm_full_001" / "index.json").read_text(encoding="utf-8"))
            index["episodes"]["train"].append({**index["episodes"]["val"][0], "id": 13})
            trb.write_json(root / "wm_full_001" / "index.json", index)
            ft = json.loads((root / "ft" / "run.json").read_text(encoding="utf-8"))
            ft["cache_index_sha256"] = sha256(root / "wm_full_001" / "index.json")
            trb.write_json(root / "ft" / "run.json", ft)
            out = root / "runs" / "rrt_branches_plan_dev_001"
            with self.assertRaises(ValueError):
                self.run_plan(plan, dev + ["--output", str(out)])
            self.assertTrue((out / "FAILED.json").is_file())
            self.assertFalse((out / "summary.json").exists())

    def test_role_train_is_the_default_plus_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, common = self.build(root)
            doc = json.loads(plan.read_text(encoding="utf-8"))
            frozen = rp.frozen_training_args(doc)
            base = ["--full-cache", str(root / "wm_full_001"), "--bench-cache", str(root / "wm_train_bench_001"),
                    "--bench-split", str(root / "danger_train_split_001" / "benchmark_split.json"), "--ft-run", str(root / "ft" / "run.json"),
                    "--classifier-run", str(root / "classifier" / "run.json"), "--goal-images", str(frozen.goal_images),
                    "--seed", str(frozen.seed), "--quota", frozen.quota, "--f-near-radius-m", str(frozen.f_near_radius_m),
                    "--goal-image-strata", frozen.goal_image_strata, "--families-b-fraction", frozen.families_b_fraction,
                    *[x for f in frozen.families for x in ("--families", f)], *[x for f in frozen.families_b for x in ("--families-b", f)]]
            with patch.object(rp, "FROZEN_STAGE2_PLAN_SHA256", sha256(plan)):
                with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                    # the frozen paths are not these
                    rp.main(base + ["--role", "train", "--stage2-plan", str(plan), "--output", str(root / "runs" / "rrt_branches_plan_001")])
            default_out = root / "plain"
            with redirect_stdout(io.StringIO()):
                rp.main(base + ["--output", str(default_out)])
            # the default mode writes the run.json key set of the plans made before the roles existed (no stage-2 option)
            run = json.loads((default_out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(list(run), OLD_PLAN_RUN_KEYS)
            self.assertEqual(list(run["args"]), OLD_PLAN_ARGS)
            self.assertFalse({"role", "stage2_plan"} & set(json.loads((default_out / "anchors.json").read_text(encoding="utf-8"))))
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                        # stage-2 inputs belong to dev / confirm
                rp.main(base + ["--eval-bench-cache", str(root / "wm_bench_001"), "--output", str(root / "plain_x")])
            self.assertFalse((root / "plain_x").exists())
            # rewrite the plan's command with the fixture's paths so that --role train reproduces it exactly
            words = doc["training_collection"]["command"].split()
            doc["training_collection"]["command"] = " ".join(words + ["--full-cache", str(root / "wm_full_001"), "--bench-cache", str(root / "wm_train_bench_001"),
                                                                      "--bench-split", str(root / "danger_train_split_001" / "benchmark_split.json"),
                                                                      "--ft-run", str(root / "ft" / "run.json"), "--classifier-run", str(root / "classifier" / "run.json")])
            write_json(plan, doc)
            role_out = root / "runs" / "rrt_branches_plan_001"
            summary = self.run_plan(plan, base + ["--role", "train", "--stage2-plan", str(plan), "--output", str(role_out)])
            a, b = (json.loads((d / "anchors.json").read_text(encoding="utf-8")) for d in (default_out, role_out))
            self.assertEqual(a["anchors"], b["anchors"])
            self.assertEqual({k: v for k, v in b.items() if k not in ("role", "stage2_plan")}, a)
            self.assertEqual(sha256(default_out / "cache_pixels.npz"), sha256(role_out / "cache_pixels.npz"))
            self.assertEqual((summary["role"], b["stage2_plan"]["section"]), ("train", "training_collection"))
            with self.assertRaises(ValueError):                                                         # any other seed is refused
                self.run_plan(plan, base + ["--seed", "4", "--role", "train", "--stage2-plan", str(plan), "--output", str(root / "runs" / "rrt_branches_plan_002")])


try:
    import collect_rrt_branches as crr
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False


@unittest.skipUnless(HAVE_SIM, "the collector needs the simulation venv")
class RoleCollectorTests(unittest.TestCase):
    def make_dev_plan(self, root, plan_json):
        plan_dir, folder, records = trb.SimulationTests.make_plan(self, root)                    # one real carry episode, 2 anchors
        target = root / "runs" / "rrt_branches_plan_dev_001"
        target.parent.mkdir(parents=True)
        shutil.move(str(plan_dir), str(target))
        doc = json.loads((target / "anchors.json").read_text(encoding="utf-8"))
        for a in doc["anchors"]:
            a["set"], a["source"]["source"], a["group"] = "val_dev", "val_dev", f"val_dev/{a['stratum']}"
        doc["role"] = "dev"
        doc["stage2_plan"] = {"path": str(plan_json), "sha256": sha256(plan_json), "role": "dev", "section": "branch_dev_set", "seed": 5,
                              "dataset": "data/rrt_branches_dev_001", "plan_dir": "runs/rrt_branches_plan_dev_001", "report_dir": "runs/rrt_branches_dev_001"}
        write_json(target / "anchors.json", doc)
        write_json(target / "summary.json", {"format": rp.PLAN_FORMAT, "files_sha256": {"anchors.json": sha256(target / "anchors.json"),
                                                                                       "cache_pixels.npz": sha256(target / "cache_pixels.npz")}})
        return target

    def test_dev_plan_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_json = root / "plan.json"
            synthetic_stage2_plan(plan_json, "gi")
            plan_dir = self.make_dev_plan(root, plan_json)
            out, report = root / "data" / "rrt_branches_dev_001", root / "runs" / "rrt_branches_dev_001"
            args = ["--plan", str(plan_dir), "--role", "dev", "--output", str(out), "--report", str(report), "--workers", "1"]
            with patch.object(rp, "FROZEN_STAGE2_PLAN_SHA256", sha256(plan_json)):
                with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                      # a dev plan needs --role dev
                    crr.main(["--plan", str(plan_dir), "--output", str(out), "--report", str(report)])
                with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                      # collected whole
                    crr.main(args + ["--anchor-ids", "0"])
                with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                      # fixed names
                    crr.main(["--plan", str(plan_dir), "--role", "dev", "--output", str(root / "data" / "dev_x"), "--report", str(report)])
                self.assertFalse(out.exists() or report.exists())
                with redirect_stdout(io.StringIO()):
                    summary = crr.main(args)
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):                          # plan.json no longer the frozen one
                crr.main(["--plan", str(plan_dir), "--role", "dev", "--output", str(root / "data" / "rrt_branches_dev_002"),
                          "--report", str(root / "runs" / "rrt_branches_dev_002")])
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["role"], manifest["stage2_plan"]["sha256"]), ("dev", sha256(plan_json)))
            self.assertEqual(manifest["names"]["output"]["planned"], "data/rrt_branches_dev_001")
            for row in manifest["anchors"]:
                dp = row["decision_points"]
                self.assertEqual(set(dp["by_d"]), {"d0", "d8"})
                with np.load(out / row["file"]) as z:
                    self.assertEqual(str(z["set"]), "val_dev")
                    fams = [str(f) for f in z["families"]]
                    self.assertEqual(dp["by_d"]["d0"]["valid"], sum(f != "recorded" for f in fams))   # every law rollout is valid at d = 0
            self.assertEqual(summary["decision_point_pools"]["anchors"], 2)
            run = json.loads((report / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((run["role"], run["evaluation_data_opened"]), ("dev", False))
            self.assertIn("branch-dev", (report / "REPORT_ZH.md").read_text(encoding="utf-8"))

    def test_default_mode_keeps_the_old_key_sets(self):
        """No --role on a role-less plan: manifest.json / run.json have exactly the key sets of the collector before the roles
        existed; --role train on the same plan adds only role / stage2_plan / names and gives bit-identical npz arrays."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_dir, _, _ = trb.SimulationTests.make_plan(self, root)                             # one real carry episode, 2 anchors
            outs = {}
            for tag, extra in (("default", []), ("train", ["--role", "train"])):
                out, report = root / f"branches_{tag}", root / f"report_{tag}"
                with redirect_stdout(io.StringIO()):
                    crr.main(["--plan", str(plan_dir), "--output", str(out), "--report", str(report), "--workers", "1", "--anchor-ids", "1", *extra])
                outs[tag] = (out, report)
            out, report = outs["default"]
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            run = json.loads((report / "run.json").read_text(encoding="utf-8"))
            self.assertEqual((list(manifest), list(run), list(run["args"])), (OLD_MANIFEST_KEYS, OLD_COLLECT_RUN_KEYS, OLD_COLLECT_ARGS))
            self.assertNotIn("decision_points", manifest["anchors"][0])
            out_t, report_t = outs["train"]
            manifest_t = json.loads((out_t / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([k for k in manifest_t if k not in ("role", "stage2_plan", "names")], OLD_MANIFEST_KEYS)
            self.assertEqual((manifest_t["role"], manifest_t["stage2_plan"], manifest_t["names"]), ("train", None, None))
            self.assertEqual(json.loads((report_t / "run.json").read_text(encoding="utf-8"))["args"]["role"], "train")
            row, row_t = manifest["anchors"][0], manifest_t["anchors"][0]
            with np.load(out / row["file"]) as z, np.load(out_t / row_t["file"]) as zt:
                self.assertEqual(sorted(z.files), sorted(zt.files))
                for k in z.files:
                    np.testing.assert_array_equal(z[k], zt[k], err_msg=k)

    def test_role_resolution(self):
        self.assertEqual((crr.effective_role({}, None), crr.effective_role({"role": "train"}, None), crr.effective_role({}, "train")),
                         ("train", "train", "train"))
        for doc, role in (({"role": "dev"}, None), ({"role": "confirm"}, None), ({"role": "confirm"}, "train"), ({}, "dev")):
            with self.assertRaises(ValueError):
                crr.effective_role(doc, role)
        self.assertFalse(crr.records_stage2(None, {"anchors": []}))                                 # the default mode
        self.assertTrue(crr.records_stage2("train", {}) and crr.records_stage2(None, {"role": "train"})
                        and crr.records_stage2(None, {"stage2_plan": {"sha256": "x"}}))

    def test_load_plan_refuses_other_sides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            anchor = {"id": 0, "set": "bench_test", "source": {"source": "bench_test", "episode_dir": "/x/danger_bench_001/episode_0000"}}
            doc = {"format": rp.PLAN_FORMAT, "prefix": rp.PREFIX, "horizon": rp.HORIZON, "normalization": trb.NORMALIZATION, "role": "confirm",
                   "stage2_plan": {"sha256": "x"}, "anchors": [anchor]}
            write_json(plan / "anchors.json", doc)
            np.savez_compressed(plan / "cache_pixels.npz", a0000=np.zeros(1))
            write_json(plan / "summary.json", {"format": rp.PLAN_FORMAT, "files_sha256": {"anchors.json": sha256(plan / "anchors.json"),
                                                                                         "cache_pixels.npz": sha256(plan / "cache_pixels.npz")}})
            self.assertEqual(crr.load_plan(plan, "confirm")[0]["role"], "confirm")
            for role in ("train", "dev", None):                                                          # None: no --role given
                with self.assertRaises(ValueError):
                    crr.load_plan(plan, role)
            doc["anchors"][0]["source"]["episode_dir"] = "/x/danger_train_001/episode_0000"               # a training episode as a confirmation start
            write_json(plan / "anchors.json", doc)
            write_json(plan / "summary.json", {"format": rp.PLAN_FORMAT, "files_sha256": {"anchors.json": sha256(plan / "anchors.json"),
                                                                                         "cache_pixels.npz": sha256(plan / "cache_pixels.npz")}})
            with self.assertRaises(ValueError):
                crr.load_plan(plan, "confirm")


if __name__ == "__main__":
    unittest.main()
