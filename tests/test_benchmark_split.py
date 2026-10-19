"""numpy-only tests of benchmark_measure / benchmark_split / tools.bench_cell_gallery on synthetic datasets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import bench_cell_gallery as gallery  # noqa: E402
import benchmark_split as split  # noqa: E402
from benchmark_measure import load_benchmark_episode, measured_cell, split_stratum  # noqa: E402
from benchmark_plan import HARD_CELLS, sample_intent  # noqa: E402
from tests.test_bench_precheck_report import write_episode  # noqa: E402
from wm_training.danger_strata import episode_features  # noqa: E402
from wm_training.tests.test_danger_strata import make_episode, profile  # noqa: E402


def build_dataset(root, episodes, seed=1):
    rows = []
    for eid, (arrays, intent, hazard) in enumerate(episodes):
        rows.append(write_episode(root, eid, arrays, intent, hazard))
        images = root / f"episode_{eid:04d}" / "images"
        images.mkdir()
        for frame in range(len(arrays["unsafe"])):
            (images / f"{frame:06d}.png").write_bytes(b"PNG" + bytes([frame % 256]))
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 3, "status": "complete", "interaction_revision": 3, "requested_episodes": len(rows),
        "seed": seed, "benchmark_schedule": {"seed": 7, "sha256": "ab" * 32, "counts": {}, "attempts": len(rows)},
        "episodes": rows}), encoding="utf-8")


class MeasureTests(unittest.TestCase):
    def test_measured_cell_and_split_stratum(self):
        n = 60
        flat = episode_features(make_episode(profile(n, 20, 90)), 0.05)
        lean = episode_features(make_episode(profile(n, 20, 45)), 0.05)
        just = episode_features(make_episode(profile(n, 20, 33), left_axis=(1, 0, 0)), 0.05)
        brief = np.zeros(n)
        brief[20:23] = 45
        transient = episode_features(make_episode(brief), 0.05)
        none = episode_features(make_episode(np.zeros(n)), 0.05)
        self.assertEqual(measured_cell(flat)[0], "toward_camera|S1_flat")
        self.assertEqual(measured_cell(lean)[0], "toward_camera|S2_lean")
        self.assertEqual(measured_cell(just)[0], "lateral_outward|S3_just_over")
        self.assertEqual(measured_cell(transient)[0], "toward_camera|transient")
        cell, detail = measured_cell(none)
        self.assertEqual((cell, detail["negative_flags"]), ("no_event", []))
        rng = np.random.default_rng(0)
        self.assertEqual(split_stratum("no_event", sample_intent("negative|N3_near", rng)), "negative|N3_near")
        self.assertEqual(split_stratum("no_event", sample_intent("toward_camera|S1_flat", rng)), "negative|unintended")
        self.assertEqual(split_stratum("away_camera|S1_flat", sample_intent("negative|N2_rock", rng)), "away_camera|S1_flat")


class SplitTests(unittest.TestCase):
    def test_split_is_stratified_seeded_and_reports_quota(self):
        rng = np.random.default_rng(3)
        n = 40
        episodes = []
        for _ in range(9):
            episodes.append((make_episode(profile(n, 15, 90)), sample_intent("toward_camera|S1_flat", rng), 15))
        for _ in range(3):
            episodes.append((make_episode(profile(n, 15, 45), left_axis=(1, 0, 0)), sample_intent("lateral_outward|S2_lean", rng), 15))
        episodes.append((make_episode(np.zeros(n)), sample_intent("negative|N3_near", rng), None))
        episodes.append((make_episode(np.zeros(n)), sample_intent("toward_camera|S2_lean", rng), None))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_dataset(tmp / "bench_a", episodes)
            build_dataset(tmp / "bench_b", episodes[:2], seed=2)
            out = tmp / "split_001"
            split.main(["--dataset", str(tmp / "bench_a"), "--dataset", str(tmp / "bench_b"), "--seed", "5", "--output", str(out)])
            result = json.loads((out / "benchmark_split.json").read_text(encoding="utf-8"))
            strata = result["strata"]
            self.assertEqual(len(strata["toward_camera|S1_flat"]["dev"]) + len(strata["toward_camera|S1_flat"]["test"]), 11)
            self.assertEqual(len(strata["toward_camera|S1_flat"]["dev"]), 4)          # round(11/3)
            self.assertEqual((len(strata["lateral_outward|S2_lean"]["dev"]), len(strata["lateral_outward|S2_lean"]["test"])), (1, 2))
            self.assertEqual(len(strata["negative|N3_near"]["dev"]) + len(strata["negative|N3_near"]["test"]), 1)
            self.assertIn("negative|unintended", strata)
            dev_ids = {(e["dataset"], e["episode_id"]) for s in strata.values() for e in s["dev"]}
            test_ids = {(e["dataset"], e["episode_id"]) for s in strata.values() for e in s["test"]}
            self.assertFalse(dev_ids & test_ids)
            self.assertEqual(len(dev_ids | test_ids), 16)
            quota = {r["cell"]: r for r in result["quota"]}
            self.assertFalse(quota["toward_camera|S1_flat"]["met"])
            self.assertTrue(all(quota[c]["required"] for c in HARD_CELLS))
            self.assertEqual(result["datasets"][1]["episodes"], 2)
            report = (out / "REPORT_ZH.md").read_text(encoding="utf-8")
            self.assertIn("未达标", report)
            # Same seed -> same split; different seed -> different assignment somewhere.
            again = split.split_episodes(result["episodes"], 5, 1 / 3)
            self.assertEqual(again, strata)
            self.assertNotEqual(split.split_episodes(result["episodes"], 6, 1 / 3), strata)
            with self.assertRaises(FileExistsError):
                split.main(["--dataset", str(tmp / "bench_a"), "--seed", "5", "--output", str(out)])
            with self.assertRaises(ValueError):
                split.main(["--dataset", str(tmp / "bench_a"), "--seed", "5", "--output", str(tmp / "bench_a" / "inside")])
            with self.assertRaises(ValueError):
                split.split_episodes(result["episodes"], 5, 1.5)
            # Gallery copies original PNGs for every stratum without touching the datasets.
            gal = tmp / "gallery_001"
            gallery.main(["--split", str(out / "benchmark_split.json"), "--per-cell", "2", "--seed", "1", "--output", str(gal)])
            index = (gal / "index.md").read_text(encoding="utf-8")
            self.assertIn("toward_camera|S1_flat", index)
            copied = list(gal.rglob("*.png"))
            self.assertTrue(copied)
            first = sorted(gal.glob("toward_camera__S1_flat/*_onset_f*.png"))[0]
            self.assertEqual(first.read_bytes()[:3], b"PNG")
            self.assertIn("_onset_f000015", first.name)
            with self.assertRaises(FileExistsError):
                gallery.main(["--split", str(out / "benchmark_split.json"), "--output", str(gal)])

    def test_incomplete_or_old_datasets_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            build_dataset(tmp / "bench", [(make_episode(profile(30, 10, 90)), sample_intent("toward_camera|S1_flat", rng), 10)])
            manifest_path = tmp / "bench" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "interrupted"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                split.load_datasets([tmp / "bench"])
            manifest["status"], manifest["interaction_revision"] = "complete", 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                split.load_datasets([tmp / "bench"])
            meta_path = tmp / "bench" / "episode_0000" / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["controller"]["interaction_revision"] = 2
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_benchmark_episode(tmp / "bench" / "episode_0000")


if __name__ == "__main__":
    unittest.main()
