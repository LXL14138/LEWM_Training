"""numpy-only test of tools/bench_precheck_report.py on a synthetic benchmark dataset."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import bench_precheck_report as tool  # noqa: E402
from benchmark_plan import sample_intent  # noqa: E402
from wm_training.tests.test_danger_strata import make_episode, profile  # noqa: E402


def write_episode(root, eid, arrays, intent, hazard_step):
    folder = root / f"episode_{eid:04d}"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    record = {"recipe": intent["recipe"], "cell_intent": intent["cell"], "guard_index": intent["guard_index"], "lean": intent["lean"],
              "after_event": intent["after_event"], "park_green": intent["park_green"], "stop": intent["stop"], "params": {},
              "prologue": "none", "hazard_step": hazard_step, "hazard_phase": "BENCH_PUSH" if hazard_step is not None else None,
              "hazard_time_s": None, "hazard_tilt_deg": None, "stop_rule_reached": False, "stop_rule_value": None,
              "end_reason": "bench_done", "tip_angle_deg": 21.8, "intent": intent}
    meta = {"control_dt": 0.05, "num_states": len(arrays["unsafe"]), "status": "complete",
            "controller": {"interaction_revision": 3, "benchmark_intent": intent, "mode": "bench_push"},
            "outcome": {"benchmark": record, "stop_reason": "bench_done_settled", "ever_lifted": False}}
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"id": eid, "path": f"episode_{eid:04d}", "status": "complete"}


class PrecheckReportTests(unittest.TestCase):
    def test_hit_table_and_outputs(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bench"
            root.mkdir()
            n = 60
            toward_flat = sample_intent("toward_camera|S1_flat", rng)
            outward_s3 = sample_intent("lateral_outward|S3_just_over", rng)
            near = sample_intent("negative|N3_near", rng)
            rows = [write_episode(root, 0, make_episode(profile(n, 30, 90)), toward_flat, 30),
                    write_episode(root, 1, make_episode(profile(n, 30, 90)), outward_s3, 30),      # intended outward, fell toward camera
                    write_episode(root, 2, make_episode(np.zeros(n)), near, None),
                    {"id": 3, "path": "episode_0003", "status": "error"}]
            manifest = {"schema_version": 3, "status": "complete", "interaction_revision": 3, "requested_episodes": 4,
                        "benchmark_schedule": {"seed": 7, "sha256": "ab" * 32, "counts": {}, "attempts": 4}, "episodes": rows}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = tool.analyse(root)
            hit = result["hit_by_intent_cell"]
            self.assertEqual(hit["toward_camera|S1_flat"], {"attempts": 1, "hit": 1, "rate": 1.0, "measured": {"toward_camera|S1_flat": 1}})
            self.assertEqual((hit["lateral_outward|S3_just_over"]["hit"], hit["lateral_outward|S3_just_over"]["measured"]), (0, {"toward_camera|S1_flat": 1}))
            self.assertEqual(hit["negative|N3_near"]["hit"], 1)
            self.assertEqual(result["episodes_complete"], 3)
            self.assertEqual(result["by_recipe"][toward_flat["recipe"]]["hazard"], 1)
            output = Path(tmp) / "report_001"
            tool.main([str(root), "--output", str(output)])
            self.assertTrue((output / "precheck_report.json").exists())
            self.assertIn("命中率", (output / "REPORT_ZH.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                tool.main([str(root), "--output", str(output)])
            with self.assertRaises(FileExistsError):
                tool.main([str(root), "--output", str(root / "inside")])
            manifest["interaction_revision"] = 2
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                tool.analyse(root)


if __name__ == "__main__":
    unittest.main()
