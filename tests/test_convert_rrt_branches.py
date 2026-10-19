"""convert_rrt_branches.py on synthetic rrt_branches_v1 datasets and synthetic reference caches (TRAINING venv: h5py, torch).

No real dataset, cache or benchmark episode is read. Run in WSL from the repository root:
  ~/lewm-train/.venv/bin/python -m unittest tests.test_convert_rrt_branches -v
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import h5py
    import torch  # noqa: F401
    import convert_rrt_branches as cv
    import rrt_branches_plan as rp
    from tests import rrt_synthetic as syn
    HAVE_TRAINING_VENV = True
except ImportError:
    HAVE_TRAINING_VENV = False

SIZE = 8
QUIET = contextlib.redirect_stdout(io.StringIO())
SCENE = {"safety_scope": "guard_tilt_only", "displacement_check_enabled": False, "unsafe_tilt_deg": 30.0, "image": {"width": 640, "height": 480}}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def fake_sha(tag):
    return (str(tag) * 64)[:64]


class Fixture:
    """Reference cache (wm_full_001-like), B's bench cache, hold-out cache, classifier halves and the bench split, all under
    <root>/data (the data root = parent of the reference cache); check conversions go to <root>/<name>, outside it."""

    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / "data"
        self.raw = self.data / "exploration_5000_001"
        self.bench_raw = self.data / "danger_train_001"
        rows = {"train": [{"id": i, "path": f"episodes/episode_{i:04d}.h5", "num_states": 400, "num_actions": 399,
                           "source_npz_sha256": fake_sha(f"t{i}"), "source_metadata_sha256": fake_sha("m")} for i in (11, 12)],
                "val": [{"id": i, "path": f"episodes/episode_{i:04d}.h5", "num_states": 400, "num_actions": 399,
                         "source_npz_sha256": fake_sha(f"v{i}"), "source_metadata_sha256": fake_sha("m")} for i in (21, 22)],
                "test": [{"id": 31, "path": "x", "num_states": 400, "num_actions": 399, "source_npz_sha256": fake_sha("x31"), "source_metadata_sha256": fake_sha("m")}]}
        self.ref = self.data / "wm_full_001"
        write_json(self.ref / "index.json", {
            "format": "three_blocks_lewm_v1", "status": "complete", "image_size": SIZE, "resize": "LANCZOS letterbox",
            "pixel_mean": [0.485, 0.456, 0.406], "pixel_std": [0.229, 0.224, 0.225], "source_action_columns": [0, 1, 2, 4],
            "action_names": ["dx", "dy", "dz", "dgripper"], "frameskip": 1, "control_dt": 0.05, "scene": SCENE,
            "raw_label_semantics": {"0": "hover", "1": "unsafe", "2": "other"}, "source_root": str(self.raw),
            "split_ids": {"train": [11, 12], "val": [21, 22], "test": [31]}, "selected_ids": {"train": [11, 12], "val": [21, 22]},
            "episodes": rows, "normalization": {**syn.NORM, "num_actions": 10, "constant_channels": []}})
        self.split = self.data / "danger_train_split_001" / "benchmark_split.json"
        write_json(self.split, {"format": "danger_bench_split_v1", "datasets": [{"index": 0, "path": str(self.bench_raw)}]})
        self.bench = self.data / "wm_train_bench_001"
        write_json(self.bench / "index.json", {
            "status": "complete", "eval_cache": True, "split_sha256": syn.sha(self.split), "source_roots": [str(self.bench_raw)],
            "bench_split": {"bench_test": [0, 1], "bench_dev": [2]}, "normalization": syn.NORM,
            "episodes": {"train": [], "val": [{"id": i, "source_npz_sha256": fake_sha(f"b{i}")} for i in (0, 1, 2)], "test": []}})
        self.holdout = self.data / "wm_bench_001"
        write_json(self.holdout / "index.json", {"source_roots": [str(self.data / "danger_bench_001")], "episodes": {
            "train": [], "val": [{"id": 5, "source_npz_sha256": fake_sha("h5")}], "test": []}})
        self.classifier = self.root / "classifier" / "run.json"
        write_json(self.classifier, {"dev_episode_ids": [21], "report_episode_ids": [22]})

    def spec(self, aid, source, eid, stratum="T", families=("recorded", "hold", "push", "push2_hold"), kinds=("calm", "calm", "rise", "late"), **kw):
        raw = self.raw if source in ("full_train", "val_dev", "val_report") else self.bench_raw
        tag = {"full_train": "t", "val_dev": "v", "val_report": "v", "bench_train": "b", "bench_dev": "b", "bench_test": "b"}[source]
        return {"id": aid, "set": source, "stratum": stratum, "families": list(families), "kinds": list(kinds), "anchor_tilt": kw.pop("anchor_tilt", 8.0),
                "hover_goal_xyz": [0.43, 0.0, 0.17], "source_frame": 50, "cache_episode_id": eid,
                "cache": "wm_full_001" if source in ("full_train", "val_dev", "val_report") else "wm_train_bench_001",
                "episode_dir": str(raw / f"episode_{eid:04d}"), "npz_sha256": fake_sha(f"{tag}{eid}"), **kw}

    def args(self, dataset, report, output, role="train", *extra, check=True):
        """CLI arguments; check=True adds --check-only (every synthetic dataset outside the plan's names is a check)."""
        return ["--role", role, "--dataset", str(dataset), "--report", str(report), "--output", str(output), "--reference-cache", str(self.ref),
                "--bench-cache", str(self.bench), "--holdout-cache", str(self.holdout), "--bench-split", str(self.split),
                "--classifier-run", str(self.classifier), "--workers", "1", *extra, *(["--check-only"] if check else [])]


@unittest.skipUnless(HAVE_TRAINING_VENV, "needs the training venv (h5py, torch)")
class ConverterTests(unittest.TestCase):
    def test_goal_labels_reproduce_the_scene_rule(self):
        hover = np.array([0.43, 0.0, 0.17])
        green = hover + np.array([[0.014, 0, 0], [0.016, 0, 0], [0.0, 0.0, 0.0], [0.001, 0, 0]])
        st = syn.states(np.array([2.0, 2.0, 35.0, 2.0]), np.array([True, True, True, False]), green, np.array([0.01, 0.01, 0.01, 0.01]), hover)
        err, cond, cls = cv.goal_labels(st["object_com_xyz"][:, 0], st["target_speed_mps"], st["grasped"], st["unsafe"], hover)
        np.testing.assert_array_equal(err, st["hover_error_m"])
        np.testing.assert_array_equal(cond, st["hover_conditions_met"])
        np.testing.assert_array_equal(cls, st["class_label"])
        self.assertEqual(cls.tolist(), [0, 2, 1, 2])                        # 1.4 cm in, 1.6 cm out, unsafe first, not grasped
        _, _, moved = cv.goal_labels(st["object_com_xyz"][:, 0], st["target_speed_mps"], st["grasped"], st["unsafe"], hover + [0.016, 0, 0])
        self.assertEqual(moved.tolist(), [0, 0, 1, 2])                      # relative to the new goal; class 1 unchanged

    def test_label_goal_rule(self):
        self.assertEqual(cv.label_goal_rule("G", "goal_image_achieved_com"), "goal_image_achieved_com")
        for stratum in "TFM":                                    # plan.json '(G)' / PLAN_ZH.md: T / F / M keep the hover goal
            self.assertEqual(cv.label_goal_rule(stratum, "goal_image_achieved_com"), "hover_goal")
        self.assertEqual(cv.label_goal_rule("G", "hover_goal"), "hover_goal")
        with self.assertRaises(ValueError):
            cv.label_goal_rule("G", "somewhere")

    def test_train_role_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(temp)
            goal = [0.4485, 0.0, 0.17]
            specs = [fx.spec(0, "full_train", 11),
                     fx.spec(1, "full_train", 12, stratum="G", families=("recorded", "hold", "planner", "edge_hold"), kinds=("calm",) * 4, anchor_tilt=0.0,
                             goal_source="goal_image_achieved_com", goal_xyz=goal, goal_image_id=7),
                     fx.spec(2, "bench_train", 1, families=("recorded", "push", "push4"), kinds=("rise", "rise", "calm"), anchor_tilt=12.0),
                     # a T anchor of an episode with a linked goal image (the planner gives it goal_source goal_image_achieved_com)
                     fx.spec(3, "full_train", 11, families=("recorded", "hold", "push", "push_hold"), kinds=("calm",) * 4, anchor_tilt=6.0,
                             goal_source="goal_image_achieved_com", goal_xyz=goal, goal_image_id=7)]
            dataset, report, manifest = syn.write_dataset(temp, "rrt_branches_pilot_x", specs, SIZE)
            out = Path(temp) / "cache_train"
            with QUIET:
                s = cv.main(fx.args(dataset, report, out))
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            ids = [r["id"] for r in index["episodes"]["val"]]
            self.assertEqual(ids, [1, 2, 3, 17, 18, 19, 33, 34, 49, 50, 51])       # anchor_id * 16 + rollout, 'recorded' skipped
            self.assertEqual(index["bench_split"], {"bench_test": ids, "bench_dev": []})
            self.assertEqual((index["eval_cache"], index["eval_cache_revision"], index["split_ids"]["val"]), (True, 2, ids))
            self.assertEqual(index["source_provenance"], {str(dataset.resolve()): "rrt_branches"})
            self.assertEqual(sorted(index["upstream_source_roots"]), sorted([str(fx.raw), str(fx.bench_raw)]))
            self.assertEqual(index["upstream_split_files"], [str(fx.split.resolve())])
            self.assertTrue(index["normalization"]["copied_from_reference"])
            self.assertEqual((index["check_only"], index["rrt_conversion"]["check_only"]), (True, True))     # a pilot: check only
            self.assertTrue(index["training_use"].startswith("FORBIDDEN"))
            self.assertEqual({r["source_part"] for r in index["episodes"]["val"]}, {"test"})
            self.assertEqual({r["prefix_frames"] for r in index["episodes"]["val"]}, {2})
            self.assertEqual(len({r["source_npz_sha256"] for r in index["episodes"]["val"]}), len(ids))
            row = index["episodes"]["val"][0]
            self.assertEqual(row["source_npz_sha256"], cv.rollout_source_sha256(manifest["anchors"][0]["sha256"], 1))
            self.assertEqual(row["source_metadata_sha256"], cv.manifest_row_sha256(manifest["anchors"][0]))
            # plan.json conversion.index: episode_sources[id] names the anchor FILE and its sha256 (and the upstream raw episode)
            src = index["episode_sources"]["1"]
            self.assertEqual((src["anchor_file"], src["anchor_npz_sha256"]), ("anchor_0000.npz", manifest["anchors"][0]["sha256"]))
            self.assertEqual((row["anchor_file"], row["upstream_npz_sha256"]), ("anchor_0000.npz", fake_sha("t11")))
            self.assertEqual(src["upstream_source"]["npz_sha256"], fake_sha("t11"))
            self.assertEqual({index["episode_sources"][str(i)]["label_goal_source"] for i in (17, 49)}, {"goal_image_achieved_com", "hover_goal"})
            with np.load(dataset / "anchor_0000.npz") as z0, h5py.File(out / "episodes" / "episode_0000003.h5", "r") as f:
                self.assertEqual(f["pixels"].shape, (35, SIZE, SIZE, 3))
                self.assertEqual((f["pixels"].chunks, f["pixels"].compression), ((1, SIZE, SIZE, 3), "lzf"))
                self.assertEqual((f["action"].shape, f["action"].dtype), ((34, 4), np.float32))
                np.testing.assert_array_equal(f["pixels"][:3], z0["prefix_pixels"][8:11])
                np.testing.assert_array_equal(f["pixels"][3:], z0["pixels"][3, 1:])
                np.testing.assert_array_equal(f["pixels"][2], z0["pixels"][3, 0])
                np.testing.assert_array_equal(f["action"][:2], z0["prefix_actions"][8:10][:, [0, 1, 2, 4]].astype(np.float32))
                np.testing.assert_array_equal(f["action"][2:], z0["actions"][3][:, [0, 1, 2, 4]].astype(np.float32))
                d = f["diagnostics"]
                np.testing.assert_array_equal(d["guard_tilt_deg"][:3], z0["prefix_guard_tilt_deg"][8:11])
                np.testing.assert_array_equal(d["guard_tilt_deg"][3:], z0["guard_tilt_deg"][3, 1:])
                np.testing.assert_array_equal(d["class_label"][...], d["recorded_class_label"][...])     # hover_goal: bit for bit
                np.testing.assert_array_equal(d["step_from_anchor"][...], np.arange(-2, 33))
                self.assertEqual(f.attrs["episode_id"], 3)
                self.assertTrue(f.attrs["complete"])
            with h5py.File(out / "episodes" / "episode_0000017.h5", "r") as f:                   # G anchor: relabelled against goal_xyz
                d = f["diagnostics"]
                np.testing.assert_array_equal(d["hover_goal_xyz"][0], goal)
                np.testing.assert_array_equal(d["source_hover_goal_xyz"][0], [0.43, 0.0, 0.17])
                self.assertFalse(np.array_equal(d["class_label"][...], d["recorded_class_label"][...]))
                self.assertEqual(json.loads(f.attrs["metadata_json"])["label_goal_source"], "goal_image_achieved_com")
            would_change = 0
            for eid in (49, 50, 51):                              # T anchor with a linked goal image: hover goal, recorded labels
                with h5py.File(out / "episodes" / f"episode_{eid:07d}.h5", "r") as f:
                    d = f["diagnostics"]
                    np.testing.assert_array_equal(d["hover_goal_xyz"][0], [0.43, 0.0, 0.17])
                    for k in ("class_label", "hover_error_m", "hover_conditions_met"):
                        np.testing.assert_array_equal(d[k][...], d[f"recorded_{k}"][...])
                    meta = json.loads(f.attrs["metadata_json"])
                    self.assertEqual((meta["goal_source"], meta["label_goal_source"], meta["goal_xyz"]), ("goal_image_achieved_com", "hover_goal", goal))
                    moved = cv.goal_labels(d["object_com_xyz"][:, 0], d["target_speed_mps"], d["grasped"], d["unsafe"], np.array(goal))[2]
                    would_change += int((moved != d["class_label"][...]).sum())
            self.assertGreater(would_change, 0)                   # the goal point WOULD have changed these labels (the test bites)
            self.assertEqual(index["episodes"]["val"][ids.index(49)]["relabelled_class_frames"], 0)
            self.assertGreater(s["relabelled_frames"], 0)
            self.assertEqual(s["goal_sources"], {"hover_goal": 2, "goal_image_achieved_com": 2})
            self.assertEqual(s["label_goal_sources"], {"hover_goal": 3, "goal_image_achieved_com": 1})
            self.assertEqual((s["linked_goal_image_non_g_anchors"]["anchors"], s["linked_goal_image_non_g_anchors"]["by_stratum"]), (1, {"T": 1}))
            self.assertAlmostEqual(s["linked_goal_image_non_g_anchors"]["goal_offset_m_max"], 0.0185)
            v = s["verification"]
            self.assertEqual(v["post_conversion_audit"], cv.AUDIT_SKIPPED)         # release: the training modules are not shipped
            self.assertEqual((v["upstream_raw_audit"]["rows"], v["upstream_raw_audit"]["distinct_upstream_episodes"]), (11, 3))
            for name in ("run.json", "summary.json", "REPORT_ZH.md", "progress.json"):
                self.assertTrue((out / name).is_file(), name)
            # release: the TiltWindows alignment check and the extra_cache_guard assertions (finetune_dataset / finetune_tilt) are not shipped
            with self.assertRaises(FileExistsError), QUIET:
                cv.main(fx.args(dataset, report, out))

    def test_dev_role_and_role_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(temp)
            specs = [fx.spec(0, "val_dev", 21), fx.spec(3, "bench_dev", 2, families=("recorded", "hold", "push"), kinds=("calm", "calm", "rise"))]
            dataset, report, _ = syn.write_dataset(temp, "dev_pilot", specs, SIZE, role="dev")
            with self.assertRaises(ValueError), QUIET:
                cv.main(fx.args(dataset, report, Path(temp) / "wrong_role", "train"))
            self.assertFalse((Path(temp) / "wrong_role").exists())
            out = Path(temp) / "cache_dev"
            with QUIET:
                s = cv.main(fx.args(dataset, report, out, "dev"))
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            ids = [r["id"] for r in index["episodes"]["val"]]
            self.assertEqual(index["bench_split"], {"bench_dev": ids, "bench_test": []})
            self.assertEqual({r["source_part"] for r in index["episodes"]["val"]}, {"dev"})
            self.assertEqual(s["verification"]["post_conversion_audit"], cv.AUDIT_SKIPPED)   # release: the training modules are not shipped

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(temp)
            base = [fx.spec(0, "full_train", 11)]

            def refused(name, specs, error=ValueError, role=None, extra=(), output="out", **kw):
                dataset, report, _ = syn.write_dataset(temp, name, specs, SIZE, role=role, **kw)
                out = Path(temp) / f"{name}_{output}"
                with self.assertRaises(error), QUIET:
                    cv.main(fx.args(dataset, report, out, *extra))
                return out
            self.assertFalse(refused("confirm_role", base, role="confirm").exists())             # the confirmation set is never converted
            self.assertFalse(refused("plan_record", base, stage2_plan={"sha256": "x" * 64}).exists())  # collected from a plan not given here
            self.assertFalse(refused("random_branches_003", base).exists())
            self.assertFalse(refused("report_start", [fx.spec(0, "val_report", 22)]).exists())
            self.assertFalse(refused("val_as_train", [{**fx.spec(0, "full_train", 21), "npz_sha256": fake_sha("v21")}]).exists())
            self.assertFalse(refused("dev_in_train", [fx.spec(0, "bench_train", 2)]).exists())    # a bench_dev row as a training start
            self.assertFalse(refused("wrong_sha", [{**fx.spec(0, "full_train", 11), "npz_sha256": fake_sha("zz")}]).exists())
            self.assertFalse(refused("bench_folder", [{**fx.spec(0, "bench_train", 1), "episode_dir": str(Path(temp) / "danger_bench_001" / "episode_0001")}]).exists())
            holdout = json.loads((fx.holdout / "index.json").read_text(encoding="utf-8"))
            write_json(fx.holdout / "index.json", {**holdout, "episodes": {"val": [{"id": 5, "source_npz_sha256": fake_sha("t11")}]}})
            self.assertFalse(refused("holdout_start", base).exists())                            # a start that is a benchmark episode
            write_json(fx.holdout / "index.json", holdout)
            dataset, report, _ = syn.write_dataset(temp, "pilot_ok", base, SIZE)
            for output, check in ((Path(temp) / "scratch" / "wm_rrt_branches_001", True),         # plan names are reserved
                                  (Path(temp) / "scratch" / "pilot_cache", False),                 # a pilot needs --check-only (never_converted)
                                  (fx.data / "pilot_cache", True)):                                # a check never goes into the data root
                with self.assertRaises(ValueError), QUIET:
                    cv.main(fx.args(dataset, report, output, check=check))
                self.assertFalse(output.exists(), output)
            # a recorded label that the recomputation does not reproduce -> FAILED.json, no usable index
            dataset, report, manifest = syn.write_dataset(temp, "tampered", base, SIZE)
            path = dataset / "anchor_0000.npz"
            with np.load(path) as z:
                arrays = {k: np.asarray(z[k]) for k in z.files}
            arrays["hover_error_m"][1, 5] += 1e-12
            path.unlink()
            np.savez_compressed(path, **arrays)
            manifest["anchors"][0]["sha256"] = syn.sha(path)
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (report / "summary.json").write_text(json.dumps({"manifest_sha256": syn.sha(dataset / "manifest.json")}), encoding="utf-8")
            out = Path(temp) / "tampered_out"
            with self.assertRaises(ValueError), QUIET:
                cv.main(fx.args(dataset, report, out))
            self.assertTrue((out / "FAILED.json").is_file())
            self.assertFalse((out / "index.json").exists() or (out / "summary.json").exists())
            # prefix diagnostics missing (the rrt_branches_pilot_001 / _002 format) -> refused
            dataset, report, manifest = syn.write_dataset(temp, "old_format", base, SIZE)
            with np.load(dataset / "anchor_0000.npz") as z:
                arrays = {k: np.asarray(z[k]) for k in z.files if not k.startswith("prefix_guard")}
            (dataset / "anchor_0000.npz").unlink()
            np.savez_compressed(dataset / "anchor_0000.npz", **arrays)
            manifest["anchors"][0]["sha256"] = syn.sha(dataset / "anchor_0000.npz")
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (report / "summary.json").write_text(json.dumps({"manifest_sha256": syn.sha(dataset / "manifest.json")}), encoding="utf-8")
            with self.assertRaises(ValueError), QUIET:
                cv.main(fx.args(dataset, report, Path(temp) / "old_out"))

    def test_plan_dataset_names(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(temp)
            plan = Path(temp) / "runs" / "rrt_data_plan_005" / "plan.json"
            write_json(plan, {"format": "rrt_data_plan_v2", "frozen": True,
                              "training_collection": {"dataset": "data/rrt_branches_001"}, "branch_dev_set": {"dataset": "data/rrt_branches_dev_001"},
                              "conversion": {"outputs": {"train": "data/wm_rrt_branches_001 (from data/rrt_branches_001; role train)",
                                                         "dev": "data/wm_rrt_dev_001 (from data/rrt_branches_dev_001; role dev)"}}})
            data = Path(temp) / "data"
            dataset, report, _ = syn.write_dataset(data, "rrt_branches_001", [fx.spec(0, "full_train", 11)], SIZE)
            with patch.object(rp, "FROZEN_STAGE2_PLAN_SHA256", syn.sha(plan)):
                with self.assertRaises(ValueError), QUIET:                           # the plan's dataset needs the plan
                    cv.main(fx.args(dataset, report, data / "wm_rrt_branches_001", check=False))
                with self.assertRaises(ValueError), QUIET:                           # next number only when _001 is taken
                    cv.main(fx.args(dataset, report, data / "wm_rrt_branches_002", "train", "--stage2-plan", str(plan), check=False))
                with self.assertRaises(ValueError), QUIET:                           # converted whole
                    cv.main(fx.args(dataset, report, data / "wm_rrt_branches_001", "train", "--stage2-plan", str(plan), "--anchor-ids", "0", check=False))
                with self.assertRaises(ValueError), QUIET:                           # never a check-only conversion
                    cv.main(fx.args(dataset, report, data / "wm_rrt_branches_001", "train", "--stage2-plan", str(plan)))
                self.assertFalse((data / "wm_rrt_branches_001").exists())
                with QUIET:
                    s = cv.main(fx.args(dataset, report, data / "wm_rrt_branches_001", "train", "--stage2-plan", str(plan), check=False))
            self.assertTrue(s["plan_rule"]["plan_dataset"])
            self.assertEqual(s["plan_rule"]["stage2_plan"]["sha256"], syn.sha(plan))
            index = json.loads((data / "wm_rrt_branches_001" / "index.json").read_text(encoding="utf-8"))
            self.assertEqual((index["check_only"], index["training_use"]), (False, "finetune_tilt --extra-cache (training rows)"))
            self.assertEqual(s["starts"], {"full_train": 1})                          # no bench start, and still ...
            self.assertEqual(index["upstream_split_files"], [str(fx.split.resolve())])  # ... the split file is listed (plan.json)

    def test_upstream_audit_of_the_written_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fx = Fixture(temp)
            ref, bench, hold = (json.loads((p / "index.json").read_text(encoding="utf-8")) for p in (fx.ref, fx.bench, fx.holdout))
            classifier = json.loads(fx.classifier.read_text(encoding="utf-8"))

            def index(role_digests):
                rows = [{"id": i, "upstream_npz_sha256": d} for i, d in enumerate(role_digests)]
                return {"episodes": {"val": rows}, "episode_sources": {str(i): {"upstream_source": {"npz_sha256": d}} for i, d in enumerate(role_digests)}}
            ok = cv.upstream_audit(index([fake_sha("t11"), fake_sha("t11"), fake_sha("b1")]), "train", ref, bench, hold, classifier)
            self.assertEqual((ok["rows"], ok["distinct_upstream_episodes"]), (3, 2))
            self.assertEqual(cv.upstream_audit(index([fake_sha("v21"), fake_sha("b2")]), "dev", ref, bench, hold, classifier)["rows"], 2)
            for role, digest in (("train", fake_sha("v21")), ("train", fake_sha("b2")), ("train", fake_sha("h5")), ("train", fake_sha("x31")),
                                 ("dev", fake_sha("v22")), ("dev", fake_sha("t11")), ("train", fake_sha("zz"))):
                with self.assertRaises(ValueError, msg=(role, digest)):
                    cv.upstream_audit(index([digest]), role, ref, bench, hold, classifier)
            broken = index([fake_sha("t11")])
            broken["episodes"]["val"][0]["upstream_npz_sha256"] = fake_sha("t12")          # row and episode_sources disagree
            with self.assertRaises(ValueError):
                cv.upstream_audit(broken, "train", ref, bench, hold, classifier)


if __name__ == "__main__":
    unittest.main()
